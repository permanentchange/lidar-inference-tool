#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include "debug_runner.hpp"
#include "debug_report.hpp"
#include "debug_profile.hpp"
#include "bevfusion/debug.hpp"
#include "runtime_diagnostics.hpp"
namespace fs = std::filesystem;
namespace inference_tools {
namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
void Check(InferenceStatus status, const std::string& error) { if (status != INFERENCE_STATUS_OK) throw std::runtime_error(error); }
struct Detection {
  std::array<float,7> box{}; float score = 0; int label = 0; std::string name;
  std::optional<std::array<float,2>> velocity;
};
struct Result {
  std::vector<Detection> detections, decoded;
  std::vector<inference_internal::StageSample> stages;
  Json counts, cache;
};
struct Frame {
  std::vector<float> points; inference::BevFusionFrame fusion;
  inference_internal::PointCloudLoadInfo info;
};
Json Boxes(const std::vector<Detection>& detections, const std::string& backend) {
  Json::Array objects;
  for (const auto& d : detections) {
    Json::Array box; for (float v : d.box) box.emplace_back(v);
    Json j = Json::Object{{"box",box},{"score",d.score},{"label",d.label},{"label_name",d.name}};
    if (d.velocity) j["velocity"] = Json::Array{(*d.velocity)[0],(*d.velocity)[1]};
    objects.push_back(j);
  }
  return Json::Object{{"schema_version",2},{"count",uint64_t(objects.size())},{"box_z_origin","center"},
    {"result_source","cpp_api"},{"runtime_backend",backend},{"objects",objects}};
}
std::vector<Detection> Convert(const std::vector<inference::Detection>& source) {
  std::vector<Detection> out; for (const auto& x : source)
    out.push_back({x.box,x.score,x.label,x.label_name ? x.label_name : "",std::nullopt}); return out;
}
std::vector<Detection> Convert(const std::vector<inference::BevFusionDetection>& source) {
  std::vector<Detection> out; for (const auto& x : source)
    out.push_back({x.box,x.score,x.label,x.label_name,x.velocity}); return out;
}
class Backend {
 public:
  std::string name; int features = 5; std::string error;
  virtual ~Backend() = default;
  virtual Frame Load(const CliOptions&, const fs::path&) = 0;
  virtual Result Infer(const Frame&, bool profile) = 0;
  virtual void Dump(const Frame&, const fs::path&, bool decoded, bool internal) = 0;
};
void Arrays(const fs::path& dir, const std::string& prefix, const std::vector<Detection>& detections, bool velocity_channels = false) {
  std::vector<float> boxes, scores, velocities; std::vector<int32_t> labels; Json::Array names;
  bool has_velocity = velocity_channels || (!detections.empty() && detections.front().velocity.has_value());
  for (const auto& d : detections) {
    boxes.insert(boxes.end(),d.box.begin(),d.box.end()); scores.push_back(d.score); labels.push_back(d.label); names.emplace_back(d.name);
    if (has_velocity) velocities.insert(velocities.end(),d.velocity->begin(),d.velocity->end());
  }
  std::string error; size_t n=detections.size();
  Check(inference_internal::SaveFloatNpy((dir/(prefix+"_boxes.npy")).string(),boxes.data(),boxes.size(),{n,7},&error),error);
  Check(inference_internal::SaveFloatNpy((dir/(prefix+"_scores.npy")).string(),scores.data(),scores.size(),{n},&error),error);
  Check(inference_internal::SaveInt32Npy((dir/(prefix+"_labels.npy")).string(),labels.data(),labels.size(),{n},&error),error);
  if (has_velocity) Check(inference_internal::SaveFloatNpy((dir/(prefix+"_velocities.npy")).string(),velocities.data(),velocities.size(),{n,2},&error),error);
  WriteJson((dir/(prefix+"_label_names.json")).string(),names);
}
class LidarBackend final : public Backend {
  std::unique_ptr<inference::LidarInferenceRuntime> runtime;
  inference_internal::ToolConfig tool;
 public:
  explicit LidarBackend(const CliOptions& o) {
    Check(inference::LidarInferenceRuntime::CreateFromModelDir(o.model_dir,&runtime,&error,o.override_config_path),error);
    name=runtime->config().model.runtime_backend; features=runtime->config().model.num_point_features;
    Check(inference_internal::LoadToolConfigFromModelDir(o.model_dir,o.input_path,o.dump_dir,o.override_config_path,0,1,.2f,&tool,&error),error);
  }
  Frame Load(const CliOptions& o,const fs::path& p) override {
    Frame f; Check(inference_internal::LoadPcdFile(p.string(),features,runtime->config().runtime.max_points,
        &f.points,&error,&f.info,o.source_num_point_features),error); return f;
  }
  inference::PointCloudView View(const Frame& f) { inference::PointCloudView v; v.points=f.points.data(); v.point_count=f.points.size()/features; return v; }
  Result Infer(const Frame& f,bool profile) override {
    inference_internal::StageTimings t; t.profile=profile; std::vector<inference::Detection> d;
    Check(inference_internal::RuntimeDiagnostics::Lidar(*runtime,View(f),&d,&t),runtime->last_error());
    Result r; r.detections=Convert(d); r.stages=std::move(t.stages);
    r.counts=Json::Object{{"source_points",f.info.source_points},{"input_points",f.info.loaded_points},
      {"voxels",t.voxel_count},{"decoded_candidates",uint64_t(t.decoded_candidates)},{"final_detections",uint64_t(d.size())}};
    r.cache=Json::Object{}; return r;
  }
  void Dump(const Frame& f,const fs::path& out,bool decoded,bool internal) override {
    if (decoded) { std::vector<inference::Detection> d; Check(runtime->InferDecoded(View(f),&d),runtime->last_error()); Arrays(out,"decoded",Convert(d)); }
    if (internal) {
      inference_internal::DebugArtifacts debug; inference_internal::StageTimings t; t.profile=false; std::vector<inference::Detection> d;
      Check(inference_internal::RuntimeDiagnostics::Lidar(*runtime,View(f),&d,&t,&debug),runtime->last_error());
      auto config=tool; config.dump_dir=(out/"internal").string(); fs::create_directories(config.dump_dir);
      Check(inference_internal::SaveInternalAuxiliaryArtifacts(debug,config,&error),error);
    }
  }
};
class FusionBackend final : public Backend {
  inference::BevFusionInferenceRuntime runtime;
 public:
  explicit FusionBackend(const CliOptions& o):runtime(o.model_dir,o.override_config_path) { name="bevfusion"; features=runtime.config().num_point_features; }
  Frame Load(const CliOptions& o,const fs::path& p) override {
    Frame f; f.fusion=inference::LoadBevFusionFrame(p.string(),runtime.config(),o.image_input=="raw",o.source_num_point_features); return f;
  }
  Result Infer(const Frame& f,bool profile) override {
    auto x=inference_internal::RuntimeDiagnostics::Fusion(runtime,f.fusion,profile);
    Result r; r.detections=Convert(x.detections); r.decoded=Convert(x.decoded_detections);
    for (auto& t:x.timings) r.stages.push_back({t.name,t.host_ms,t.gpu_ms,t.name!="nms" && t.name!="output_pack"});
    r.counts=Json::Object{{"source_points",uint64_t(f.fusion.source_points)},{"input_points",uint64_t(x.input_points)},
      {"filtered_points",uint64_t(x.filtered_points)},{"used_points",uint64_t(x.used_points)},
      {"range_rejected_points",uint64_t(x.input_points-x.filtered_points)},
      {"truncated_points",uint64_t(x.filtered_points-x.used_points)},{"discarded_voxels",uint64_t(x.discarded_voxels)},
      {"voxels",uint64_t(x.voxels)},{"depth_collisions",uint64_t(x.depth_collisions)},
      {"decoded_candidates",uint64_t(x.decoded_detections.size())},{"final_detections",uint64_t(x.detections.size())}};
    r.cache=Json::Object{{"geometry_cache_hit",x.geometry_cache_hit}}; return r;
  }
  void Dump(const Frame& f,const fs::path& out,bool decoded,bool internal) override {
    if (!decoded && !internal) return;
    auto x=inference_internal::RuntimeDiagnostics::Fusion(runtime,f.fusion,false,internal);
    if (decoded) Arrays(out,"decoded",Convert(x.decoded_detections),true);
    if (internal) { fs::create_directories(out/"internal"); for(auto& t:x.tensors) inference::SaveBevFusionNpy((out/"internal"/(t.name+".npy")).string(),t); }
  }
};
std::vector<fs::path> Inputs(const CliOptions& o,ToolKind kind,bool fusion) {
  auto valid=[fusion](const fs::path& p) { auto e=p.extension().string(); return fusion ? e==".yaml"||e==".yml"||e==".json" : e==".pcd"||e==".bin"; };
  std::vector<fs::path> frames;
  if(fs::is_directory(o.input_path)) {
    if(kind==ToolKind::kPerf) throw std::runtime_error("perf_debug_tool requires a single input file");
    for(auto& e:fs::directory_iterator(o.input_path)) if(e.is_regular_file()&&valid(e.path())) frames.push_back(e.path());
    SortBevFusionManifests(&frames);
  } else if(fs::is_regular_file(o.input_path)&&valid(o.input_path)) frames.emplace_back(o.input_path);
  if(frames.empty()) throw std::runtime_error(fusion ? "BEVFusion input requires YAML/JSON manifests" : "LiDAR input requires PCD/BIN files");
  return frames;
}
Json Timestamp(const fs::path& p) {
  const auto s=p.stem().string(); if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos) return nullptr;
  try { return uint64_t(std::stoull(s)); } catch(...) { return nullptr; }
}
}  // namespace
int RunDebugTool(int argc,char** argv,ToolKind kind) {
  CliOptions o; std::string error;
  auto parsed=ParseCliOptions(argc,argv,kind,&o,&error);
  if(parsed==ParseResult::kHelp) { PrintUsage(std::cout,argv[0],kind); return 0; }
  if(parsed!=ParseResult::kOk) { std::cerr<<error<<'\n'; PrintUsage(std::cerr,argv[0],kind); return 1; }
  try {
    auto cfg=YAML::LoadFile((fs::path(o.model_dir)/"model.yaml").string());
    auto backend_name=cfg["model"]["runtime_backend"] ? cfg["model"]["runtime_backend"].as<std::string>() : "pillar_end2end";
    if(backend_name!="bevfusion" && backend_name!="pillar_end2end" && backend_name!="centerpoint_voxel") throw std::runtime_error("Unknown runtime_backend: "+backend_name);
    bool fusion=backend_name=="bevfusion";
    if(!fusion && o.image_input_explicit) throw std::runtime_error("--image-input applies only to BEVFusion");
    auto frames=Inputs(o,kind,fusion); const bool sequence=fs::is_directory(o.input_path);
    auto start=Clock::now(); std::unique_ptr<Backend> backend;
    if(fusion) backend=std::make_unique<FusionBackend>(o); else backend=std::make_unique<LidarBackend>(o);
    double init_ms=Ms(start); fs::create_directories(o.dump_dir);
    int device=0; cudaDeviceProp properties{}; cudaGetDevice(&device); cudaGetDeviceProperties(&properties,device);
    Json::Array sequence_manifest; std::ofstream sequence_output;
    if(sequence && o.dump_result) sequence_output.open(fs::path(o.dump_dir)/"sequence_detections.jsonl");
    for(size_t index=0;index<frames.size();++index) {
      const auto& path=frames[index]; fs::path out=o.dump_dir;
      if(sequence) { std::ostringstream id; id<<std::setfill('0')<<std::setw(6)<<index<<'_'<<path.stem().string(); out/="frames"; out/=id.str(); }
      fs::create_directories(out);
      start=Clock::now(); auto frame=backend->Load(o,path); double load_ms=Ms(start);
      std::vector<double> totals; Result result; size_t cache_misses=0;
      RunProfilePass(o.warmup_iters,o.profile_iters,[&]() {
        auto begin=Clock::now(); auto sample=backend->Infer(frame,false);
        double elapsed=Ms(begin); return std::make_pair(std::move(sample),elapsed);
      },[&](auto sample) {
        result=std::move(sample.first); totals.push_back(sample.second);
        if(fusion && !std::get<bool>(std::get<Json::Object>(result.cache.value).at("geometry_cache_hit").value)) ++cache_misses;
      });
      auto total=ComputeStats(totals);
      struct Stage { std::vector<double> host,gpu; };
      std::map<std::string,Stage> stages; std::vector<std::string> stage_order; size_t stage_cache_misses=0;
      if(kind==ToolKind::kPerf) {
        RunProfilePass(o.warmup_iters,o.profile_iters,[&]() { return backend->Infer(frame,true); },[&](const Result& sample) {
          if(fusion && !std::get<bool>(std::get<Json::Object>(sample.cache.value).at("geometry_cache_hit").value)) ++stage_cache_misses;
          for(auto& t:sample.stages) {
            if(!stages.count(t.name)) stage_order.push_back(t.name);
            stages[t.name].host.push_back(t.host_ms); if(t.has_gpu) stages[t.name].gpu.push_back(t.gpu_ms);
          }
        });
      }
      Json::Array stage_json;
      for(auto& name:stage_order) { auto& s=stages[name]; stage_json.emplace_back(Json::Object{{"name",name},
        {"host_ms",ComputeStats(s.host).json()},{"gpu_ms",s.gpu.empty()?Json(nullptr):ComputeStats(s.gpu).json()}}); }
      Json stage_profile=nullptr;
      if(kind==ToolKind::kPerf) stage_profile=Json::Object{{"collection","separate_pass"},{"warmup",o.warmup_iters},
        {"iterations",o.profile_iters},{"geometry_cache_misses",uint64_t(stage_cache_misses)},{"stages",stage_json}};
      Json timing=Json::Object{{"schema_version",2},{"unit","ms"},
        {"setup",Json::Object{{"runtime_init_ms",init_ms},{"input_load_ms",load_ms}}},
        {"total_infer_ms",total.json()},{"warmup",o.warmup_iters},{"iterations",o.profile_iters},
        {"total_collection","normal_inference_without_stage_events"},{"geometry_cache_misses",uint64_t(cache_misses)},
        {"stage_profile",stage_profile}};
      WriteJson((out/"timings.json").string(),timing);
      if(o.dump_result) {
        WriteJson((out/"result.json").string(),Boxes(result.detections,backend->name)); Arrays(out,"final",result.detections,fusion);
        const auto& points=fusion?frame.fusion.points:frame.points;
        Check(inference_internal::SaveFloatNpy((out/"input_points.npy").string(),points.data(),points.size(),{points.size()/backend->features,size_t(backend->features)},&error),error);
      }
      backend->Dump(frame,out,o.dump_decoded,o.dump_internal);
      Json::Array artifacts{Json("timings.json")};
      auto record=[&](const fs::path& file) { if(fs::is_regular_file(out/file)) artifacts.emplace_back(file.string()); };
      if(o.dump_result) {
        for(const auto* file:{"result.json","input_points.npy","final_boxes.npy","final_scores.npy","final_labels.npy","final_label_names.json"}) record(file);
        if(fusion) record("final_velocities.npy");
      }
      if(o.dump_decoded) {
        for(const auto* file:{"decoded_boxes.npy","decoded_scores.npy","decoded_labels.npy","decoded_label_names.json"}) record(file);
        if(fusion) record("decoded_velocities.npy");
      }
      if(o.dump_internal) for(auto& e:fs::recursive_directory_iterator(out/"internal"))
        if(e.is_regular_file()) record(fs::relative(e.path(),out));
      auto timestamp=Timestamp(path);
      Json seconds=nullptr;
      if(std::holds_alternative<uint64_t>(timestamp.value)) seconds=double(std::get<uint64_t>(timestamp.value))/1e9;
      Json meta=Json::Object{{"schema_version",2},{"mode",kind==ToolKind::kPerf?"perf":"infer"},{"runtime_backend",backend->name},
        {"architecture",fusion?"centerhead":"lidar"},{"model_dir",fs::absolute(o.model_dir).string()},
        {"input_path",fs::absolute(path).string()},{"frame_id",path.stem().string()},{"timestamp_ns",timestamp},{"timestamp_seconds",seconds},
        {"counts",result.counts},{"image_input",fusion?Json(o.image_input):Json(nullptr)},
        {"device",Json::Object{{"visible_index",device},{"name",properties.name}}},
        {"options",Json::Object{{"warmup",o.warmup_iters},{"profile_iters",o.profile_iters},{"dump_result",o.dump_result},
          {"dump_decoded",o.dump_decoded},{"dump_internal",o.dump_internal},{"override_config",o.override_config_path}}},
        {"artifacts",artifacts}};
      WriteJson((out/"metadata.json").string(),meta);
      std::cout<<(kind==ToolKind::kPerf?"Performance":"Inference")<<" debug run complete.\n"
        <<"backend: "<<backend->name<<"\ninput_file: "<<path.string()<<"\ncounts: "<<result.counts
        <<"\nruntime_init_ms: "<<init_ms<<"\ninput_load_ms: "<<load_ms
        <<"\nTotal inference (normal call):\n"<<std::left<<std::setw(28)<<"stage"<<std::setw(7)<<"clock"<<std::right
        <<std::setw(6)<<"n"<<std::setw(10)<<"mean"<<std::setw(10)<<"min"<<std::setw(10)<<"max"<<std::setw(10)<<"p50"<<std::setw(10)<<"p90"<<'\n';
      PrintStats("total_infer_ms","host",total);
      if(kind==ToolKind::kPerf) {
        std::cout<<"Stage analysis (separate pass; ms; host includes submission/waits; GPU spans are not additive wall time):\n";
        for(auto& name:stage_order) { auto& s=stages[name]; PrintStats(name,"host",ComputeStats(s.host)); if(!s.gpu.empty()) PrintStats(name,"cuda",ComputeStats(s.gpu)); }
      }
      std::cout<<"dump_dir: "<<out.string()<<'\n';
      Json entry=Json::Object{{"frame_index",uint64_t(index)},{"frame_id",path.stem().string()},{"timestamp_ns",timestamp},{"timestamp_seconds",seconds},
        {"input_path",fs::absolute(path).string()},{"frame_dir",fs::relative(out,o.dump_dir).string()}};
      sequence_manifest.push_back(entry);
      if(sequence_output.is_open()) { entry["result"]=Boxes(result.detections,backend->name); sequence_output<<entry<<'\n'; }
    }
    if(sequence) WriteJson((fs::path(o.dump_dir)/"sequence_manifest.json").string(),Json::Object{{"schema_version",2},{"frames",sequence_manifest}});
    return 0;
  } catch(const std::exception& e) { std::cerr<<"Inference tool: "<<e.what()<<'\n'; return 1; }
}
}  // namespace inference_tools
