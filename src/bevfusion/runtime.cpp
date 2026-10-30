#include "runtime_diagnostics.hpp"
#include <NvInfer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <onnx-parser.hpp>

#include "internal.hpp"
#include "postprocess_config.hpp"
#include "trt_engine.hpp"
#include "pillar_postprocess/pillar_postprocess.hpp"

namespace inference {
namespace {
struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* message) noexcept override {
    if (s <= Severity::kWARNING)
      std::cerr << "[BEVFusion TensorRT] " << message << '\n';
  }
};
/// Own a fixed-shape TensorRT session with linear FP16 or FP32 bindings.
struct Session {
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<std::unique_ptr<bev::Buffer>> buffers;
  std::vector<void*> bindings;
  std::map<std::string, int> names;
  explicit Session(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime || bytes.empty())
      throw std::runtime_error("Cannot load engine: " + path);
    engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine) throw std::runtime_error("Cannot deserialize engine: " + path);
    context.reset(engine->createExecutionContext());
    if (!context) throw std::runtime_error("Cannot create context: " + path);
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      auto type = engine->getBindingDataType(i);
      if ((type != nvinfer1::DataType::kFLOAT &&
           type != nvinfer1::DataType::kHALF) ||
          engine->getBindingFormat(i) != nvinfer1::TensorFormat::kLINEAR ||
          engine->getBindingVectorizedDim(i) != -1)
        throw std::runtime_error("Expected linear FP16/FP32 engine bindings: " + path);
      auto dims = engine->getBindingDimensions(i);
      size_t n = 1;
      for (int j = 0; j < dims.nbDims; ++j) {
        if (dims.d[j] <= 0)
          throw std::runtime_error("Dynamic/empty binding unsupported: " +
                                   path);
        n *= dims.d[j];
      }
      buffers.emplace_back(new bev::Buffer);
      buffers.back()->resize(n * (type == nvinfer1::DataType::kHALF ? 2 : 4));
      bindings.push_back(buffers.back()->data);
      names.emplace(engine->getBindingName(i), i);
    }
  }
  int index(const std::string& name) const {
    auto it = names.find(name);
    if (it == names.end())
      throw std::runtime_error("Engine missing binding: " + name);
    return it->second;
  }
  size_t count(const std::string& name) const {
    auto d = engine->getBindingDimensions(index(name));
    size_t n = 1;
    for (int j = 0; j < d.nbDims; ++j) n *= d.d[j];
    return n;
  }
  void* data(const std::string& name) { return bindings[index(name)]; }
  bev::ScalarType type(const std::string& name) const {
    return engine->getBindingDataType(index(name)) == nvinfer1::DataType::kHALF
               ? bev::ScalarType::Float16 : bev::ScalarType::Float32;
  }
  void require(const std::string& name, const std::vector<size_t>& shape,
               bool input) const {
    int i = index(name);
    auto d = engine->getBindingDimensions(i);
    bool ok =
        size_t(d.nbDims) == shape.size() && engine->bindingIsInput(i) == input;
    for (int j = 0; j < d.nbDims && j < int(shape.size()); ++j)
      ok = ok && size_t(d.d[j]) == shape[j];
    if (!ok)
      throw std::runtime_error("Engine binding shape/direction mismatch: " +
                               name);
  }
  void run(cudaStream_t stream) {
    if (!context->enqueueV2(bindings.data(), stream, nullptr))
      throw std::runtime_error("TensorRT enqueue failed");
  }
  void connect(const std::string& name, void* pointer) {
    bindings[index(name)] = pointer;
  }
  BevFusionTensor tensor(const std::string& name, cudaStream_t stream) {
    BevFusionTensor t;
    t.name = name;
    auto d = engine->getBindingDimensions(index(name));
    for (int j = 0; j < d.nbDims; ++j) t.shape.push_back(d.d[j]);
    t.values.resize(count(name));
    bev::Buffer converted;
    void* source = data(name);
    if (type(name) == bev::ScalarType::Float16) {
      converted.resize(t.values.size() * sizeof(float));
      bev::HalfToFloat(source, converted.ptr<float>(), t.values.size(), stream);
      source = converted.data;
    }
    bev::Check(cudaMemcpyAsync(t.values.data(), source, t.values.size() * 4,
                               cudaMemcpyDeviceToHost, stream));
    bev::Check(cudaStreamSynchronize(stream));
    return t;
  }
};
std::string GeometryKey(const BevFusionFrame& f) {
  std::string key(reinterpret_cast<const char*>(f.lidar_aug_matrix.data()), 64);
  for (const auto& c : f.cameras) {
    key.push_back(c.enabled);
    if (!c.enabled) continue;
    for (auto* m : {&c.cam2img, &c.cam2lidar, &c.img_aug_matrix})
      key.append(reinterpret_cast<const char*>(m->data()), 64);
  }
  return key;
}
}  // namespace
struct BevFusionInferenceRuntime::Impl {
  BevFusionConfig cfg;
  cudaStream_t stream = nullptr;
  std::unique_ptr<Session> camera, head;
  std::shared_ptr<spconv::Engine> sparse;
  bev::Kernels kernels;
  inference_internal::RuntimeConfig center_config;
  inference_internal::ResolvedHeadLayout center_layout;
  inference_internal::CenterPointHeadContract center_contract;
  inference_internal::GpuDecodedCandidates decoder;
  bev::Buffer bbox_buffer, cls_buffer, dir_buffer;
  std::string error;
  void check(InferenceStatus status) {
    if (status != INFERENCE_STATUS_OK) throw std::runtime_error(error);
  }
  void init_centerhead(size_t c, size_t gz, size_t gx, size_t gy) {
    center_config.point_cloud_range = cfg.point_cloud_range;
    // Use exactly the same decode/filter order as the LiDAR CenterPoint runtime.
    center_config.topk_before_range_filter = false;
    center_config.post_center_range = cfg.post_center_range;
    center_config.voxel_size = cfg.voxel_size;
    center_config.out_size_factor = cfg.out_size_factor;
    center_config.decode_mode = DecodeMode::kCenterPoint;
    center_config.yaw_encoding = YawEncoding::kSinCos;
    center_config.norm_bbox = cfg.norm_bbox;
    center_config.has_velocity = true;
    center_config.bbox_code_size = 9;
    center_config.bbox_max_num = cfg.bbox_max_num;
    center_config.score_threshold = cfg.score_threshold;
    center_config.label_count = cfg.label_names.size();
    center_config.task_label_ids = cfg.task_label_ids;
    for (auto& ids : cfg.task_label_ids) center_config.task_num_classes.push_back(ids.size());
    check(inference_internal::ResolveHeadLayout(center_config, &center_layout, &error));
    check(inference_internal::InspectCenterPointHead(center_config, center_layout, *head->engine, &center_contract, &error));
    if (head->names.size() != size_t(2 + center_contract.task_count * 6))
      throw std::runtime_error("Unexpected fusion CenterHead binding count");
    head->require("input", {1, 256, size_t(center_contract.height), size_t(center_contract.width)}, true);
    head->require("camera_bev", {1, c * gz, gx, gy}, true);
    if (head->type("camera_bev") != bev::ScalarType::Float16)
      throw std::runtime_error("Fusion camera BEV input must be FP16");
    size_t hw = size_t(center_contract.height) * center_contract.width;
    bbox_buffer.resize(center_contract.bbox_channels * hw * sizeof(float));
    cls_buffer.resize(center_contract.cls_channels * hw * sizeof(float));
    dir_buffer.resize(center_contract.dir_channels * hw * sizeof(float));
    inference_internal::BindCenterPointHeadOutputs(center_contract, bbox_buffer.ptr<float>(),
        cls_buffer.ptr<float>(), dir_buffer.ptr<float>(), &head->bindings);
    check(decoder.Initialize(center_config, center_layout, center_contract.task_count,
        center_contract.bbox_channels, center_contract.cls_channels, center_contract.dir_channels,
        center_contract.height, center_contract.width, &error));
  }
  std::vector<BevFusionDetection> detections(
      const inference_internal::CandidateDetections& candidates,
      const std::vector<float>& velocities, bool map_labels) {
    std::vector<BevFusionDetection> result;
    for (size_t i = 0; i < candidates.scores.size(); ++i) {
      BevFusionDetection det;
      det.label = candidates.labels[i];
      if (map_labels && !MapDetectionLabel(cfg.centerhead_postprocess, det.label, &det.label)) continue;
      const auto* name = map_labels
          ? LabelNameForLabel(cfg.label_names, cfg.centerhead_postprocess, det.label)
          : &cfg.label_names.at(det.label);
      if (name) det.label_name = *name;
      det.score = candidates.scores[i];
      std::copy_n(candidates.boxes.data() + i * 7, 7, det.box.data());
      const size_t source = candidates.source_indices.empty() ? i : candidates.source_indices.at(i);
      if (source < velocities.size() / 2)
        det.velocity = {velocities[source * 2], velocities[source * 2 + 1]};
      result.push_back(std::move(det));
    }
    return result;
  }
  std::string geometry_key;
  struct Span {
    cudaEvent_t start = nullptr, stop = nullptr;
    ~Span() {
      if (start) cudaEventDestroy(start);
      if (stop) cudaEventDestroy(stop);
    }
  };
  std::array<Span, 16> spans;

  explicit Impl(const std::string& path, const std::string& override_path)
      : cfg(LoadBevFusionConfig(path, override_path)) {
    bev::Check(cudaStreamCreate(&stream));
    try {
      camera.reset(new Session(cfg.camera_engine));
      head.reset(new Session(cfg.head_engine));
      if (camera->names.size() != 4)
        throw std::runtime_error("Unexpected BEVFusion camera binding count");
      const size_t nc = cfg.camera_names.size(), ih = cfg.image_size[0],
                   iw = cfg.image_size[1], fh = cfg.feature_size[0],
                   fw = cfg.feature_size[1], c = cfg.feature_channels,
                   d = cfg.depth_bins;
      camera->require("img", {nc, 3, ih, iw}, true);
      camera->require("depth", {nc, 1, ih, iw}, true);
      camera->require("camera_feature", {nc, fh, fw, c}, false);
      camera->require("camera_depth_weights", {nc, d, fh, fw}, false);
      if (camera->type("camera_feature") != camera->type("camera_depth_weights"))
        throw std::runtime_error("Camera features and depth weights must have the same dtype");
      size_t gx = std::lround((cfg.xbound[1] - cfg.xbound[0]) / cfg.xbound[2]),
             gy = std::lround((cfg.ybound[1] - cfg.ybound[0]) / cfg.ybound[2]),
             gz = std::lround((cfg.zbound[1] - cfg.zbound[0]) / cfg.zbound[2]);
      init_centerhead(c, gz, gx, gy);
      spconv::set_logger_level(spconv::LoggerLevel::Error);
      sparse = spconv::load_engine_from_onnx(cfg.sparse_model,
                                             spconv::Precision::Float16, false,
                                             false, false, 10000, stream);
      if (!sparse || sparse->num_input() != 1 || sparse->num_output() != 1)
        throw std::runtime_error("Cannot load sparse backbone: " +
                                 cfg.sparse_model);
    } catch (...) {
      cudaStreamDestroy(stream);
      stream = nullptr;
      throw;
    }
  }
  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
  }
  BevFusionResult infer(const BevFusionFrame& frame, bool debug, bool profile = true) {
    if (std::none_of(frame.cameras.begin(), frame.cameras.end(), [](const auto& c) { return c.enabled; }))
      throw std::runtime_error("At least one camera must be enabled");
    if (frame.num_point_features != cfg.num_point_features ||
        frame.cameras.size() != cfg.camera_names.size())
      throw std::runtime_error("Frame/model contract mismatch");
    for (size_t i = 0; i < frame.cameras.size(); ++i)
      if (frame.cameras[i].name != cfg.camera_names[i])
        throw std::runtime_error(
            "Frame camera order mismatch; use LoadBevFusionFrame");
    if (frame.points.empty() || frame.points.size() % cfg.num_point_features)
      throw std::runtime_error("Invalid/empty point buffer");
    if (!std::all_of(frame.points.begin(), frame.points.end(),
                     [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("Nonfinite point buffer");
    auto matrix_check = [](const std::array<float, 16>& matrix) {
      if (!std::all_of(matrix.begin(), matrix.end(),
                       [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("Nonfinite calibration matrix");
      bev::Inverse(matrix);
    };
    matrix_check(frame.lidar_aug_matrix);
    for (const auto& camera : frame.cameras)
      if (camera.enabled) for (auto* m : {&camera.cam2img, &camera.cam2lidar, &camera.lidar2img,
                      &camera.img_aug_matrix})
        matrix_check(*m);
    BevFusionResult result;
    size_t stage_index = 0;
    std::chrono::steady_clock::time_point stage_start;
    auto begin = [&](const std::string& name) {
      if (!profile) return;
      auto& span = spans[stage_index];
      if (!span.start) {
        bev::Check(cudaEventCreate(&span.start));
        bev::Check(cudaEventCreate(&span.stop));
      }
      result.timings.push_back({name, 0, 0});
      stage_start = std::chrono::steady_clock::now();
      bev::Check(cudaEventRecord(span.start, stream));
    };
    auto end = [&]() {
      if (!profile) return;
      bev::Check(cudaEventRecord(spans[stage_index].stop, stream));
      result.timings.back().host_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - stage_start)
              .count();
      ++stage_index;
    };
    result.input_points = frame.points.size() / cfg.num_point_features;
    begin("point_filter_upload");
    result.filtered_points =
        bev::FilterPoints(kernels, cfg, frame.points, stream);
    int n = std::min(size_t(cfg.max_points), result.filtered_points);
    result.used_points = n;
    end();
    begin("voxelize");
    bev::Voxelize(kernels, cfg, n, stream);
    result.voxels = kernels.voxel_count;
    result.discarded_voxels = kernels.total_voxel_count - kernels.voxel_count;
    end();
    begin("sparse_backbone");
    auto input = sparse->input(0);
    input->features().reference(kernels.features.data,
                                {kernels.voxel_count, cfg.num_point_features},
                                spconv::DataType::Float16);
    input->indices().reference(kernels.indices.data, {kernels.voxel_count, 4},
                               spconv::DataType::Int32);
    input->set_grid_size(
        std::vector<int>(cfg.sparse_shape.begin(), cfg.sparse_shape.end()));
    sparse->forward(stream);
    auto& sparse_output = sparse->output(0)->features();
    auto shape = sparse_output.shape;
    Session& lidar_session = *head;
    const char* lidar_binding = "input";
    auto ld = lidar_session.engine->getBindingDimensions(lidar_session.index(lidar_binding));
    std::vector<int64_t> expected(ld.d, ld.d + ld.nbDims), normalized;
    check(inference_internal::NormalizeCenterPointHeadInputShape(expected, shape, &normalized, &error));
    if (sparse_output.dtype() != spconv::DataType::Float16)
      throw std::runtime_error("Sparse baseline requires FP16 output");
    end();
    begin("sparse_bridge");
    if (lidar_session.type(lidar_binding) == bev::ScalarType::Float16) {
      lidar_session.connect(lidar_binding, sparse_output.ptr());
    } else {
      bev::HalfToFloat(sparse_output.ptr(), static_cast<float*>(lidar_session.data(lidar_binding)),
                       lidar_session.count(lidar_binding), stream);
    }
    end();
    begin("depth_projection");
    result.depth_collisions = bev::Depth(kernels, cfg, frame, n, stream,
                                         camera->type("depth"));
    camera->connect("depth", kernels.depth.data);
    end();
    begin("image_preprocess_upload");
    if (frame.raw_images) {
      for (const auto& c : frame.cameras)
        if (c.enabled && (c.width <= 0 || c.height <= 0 || c.resize_width <= 0 ||
            c.resize_height <= 0 ||
            c.rgb.size() != size_t(c.width) * c.height * 3))
          throw std::runtime_error(
              "Invalid decoded RGB image/preprocess metadata");
      bev::Preprocess(kernels, cfg, frame, camera->data("img"), stream,
                      camera->type("img"));
    } else {
      if (frame.images.size() != camera->count("img"))
        throw std::runtime_error("Preprocessed image size mismatch");
      void* destination = camera->data("img");
      if (camera->type("img") == bev::ScalarType::Float16) {
        kernels.image_float.resize(frame.images.size() * sizeof(float));
        destination = kernels.image_float.data;
      }
      bev::Check(cudaMemcpyAsync(destination, frame.images.data(),
                                 frame.images.size() * 4,
                                 cudaMemcpyHostToDevice, stream));
      if (camera->type("img") == bev::ScalarType::Float16)
        bev::FloatToHalf(kernels.image_float.ptr<float>(), camera->data("img"),
                          frame.images.size(), stream);
    }
    end();
    begin("geometry_cache");
    auto key = GeometryKey(frame);
    result.geometry_cache_hit = key == geometry_key;
    if (!result.geometry_cache_hit) {
      bev::Geometry(kernels, cfg, frame, stream);
      geometry_key = std::move(key);
    }
    end();
    begin("camera_backbone");
    camera->run(stream);
    bev::MaskCameraOutputs(cfg, frame, camera->data("camera_feature"),
        camera->data("camera_depth_weights"), camera->type("camera_feature"), stream);
    end();
    begin("bev_pool");
    bev::Pool(kernels, cfg, camera->data("camera_feature"),
              camera->data("camera_depth_weights"), stream,
              camera->type("camera_feature"), head->type("camera_bev"));
    end();
    begin("fusion_centerhead");
    head->connect("camera_bev", kernels.pooled.data);
    head->run(stream);
    end();
    begin("decode_download");

      inference_internal::CandidateDetections candidates, final;
      std::vector<float> velocities;
      check(decoder.Decode(bbox_buffer.ptr<float>(), cls_buffer.ptr<float>(), dir_buffer.ptr<float>(), stream, &error));
      check(decoder.CopyToHost(&candidates, stream, &error, &velocities));
      end();
      begin("nms");
      inference_internal::ToolConfig tool;
      tool.package.postprocess = cfg.centerhead_postprocess;
      auto postprocess = inference_internal::MakeToolPostprocessConfig(tool);
      // One call across all tasks, exactly as in the LiDAR CenterPoint backend.
      check(inference_internal::RunPostprocessNms(candidates, postprocess, center_config, &final, &error));
      end();
      begin("output_pack");
      result.decoded_detections = detections(candidates, velocities, false);
      result.detections = detections(final, velocities, true);
    
    end();
    if (debug) {
      auto save = [&](BevFusionTensor t, const std::string& name) {
        t.name = name;
        result.tensors.push_back(std::move(t));
      };
      save(camera->tensor("img", stream), "images");
      save(camera->tensor("depth", stream), "sparse_depth");
      save(camera->tensor("camera_feature", stream), "camera_feature");
      save(camera->tensor("camera_depth_weights", stream), "camera_depth_weights");

        save(head->tensor("input", stream), "sparse_bev");
        save(head->tensor("camera_bev", stream), "camera_bev_before_downsample");
        for (int i = 0; i < center_contract.task_count; ++i)
          for (const char* name : {"reg", "height", "dim", "rot", "vel", "hm"}) {
            auto key = std::string(name) + "_" + std::to_string(i);
            save(head->tensor(key, stream), key);
          }
      
      size_t geometry_count = cfg.camera_names.size() * cfg.depth_bins *
                              cfg.feature_size[0] * cfg.feature_size[1];
      std::vector<int> ranks(geometry_count), order(geometry_count);
      bev::Check(cudaMemcpyAsync(ranks.data(), kernels.ranks.data,
                                 geometry_count * 4, cudaMemcpyDeviceToHost,
                                 stream));
      bev::Check(cudaMemcpyAsync(order.data(), kernels.geometry_order.data,
                                 geometry_count * 4, cudaMemcpyDeviceToHost,
                                 stream));
      bev::Check(cudaStreamSynchronize(stream));
      BevFusionTensor rank_tensor;
      rank_tensor.name = "geometry_ranks";
      rank_tensor.shape = {cfg.camera_names.size(), size_t(cfg.depth_bins),
                           size_t(cfg.feature_size[0]),
                           size_t(cfg.feature_size[1])};
      rank_tensor.values.resize(geometry_count);
      for (size_t i = 0; i < geometry_count; ++i)
        rank_tensor.values[order[i]] =
            ranks[i] == std::numeric_limits<int>::max() ? -1 : float(ranks[i]);
      result.tensors.push_back(std::move(rank_tensor));
      BevFusionTensor vox;
      vox.name = "voxel_features";
      vox.shape = {size_t(kernels.voxel_count), size_t(cfg.num_point_features)};
      vox.values.resize(kernels.voxel_count * cfg.num_point_features);
      bev::Buffer temp;
      temp.resize(vox.values.size() * 4);
      bev::HalfToFloat(kernels.features.data, temp.ptr<float>(),
                       vox.values.size(), stream);
      bev::Check(cudaMemcpyAsync(vox.values.data(), temp.data,
                                 vox.values.size() * 4, cudaMemcpyDeviceToHost,
                                 stream));
      bev::Check(cudaStreamSynchronize(stream));
      result.tensors.push_back(std::move(vox));
      BevFusionTensor idx;
      idx.name = "voxel_indices";
      idx.shape = {size_t(kernels.voxel_count), 4};
      idx.values.resize(kernels.voxel_count * 4);
      std::vector<int> ints(idx.values.size());
      bev::Check(cudaMemcpyAsync(ints.data(), kernels.indices.data,
                                 ints.size() * 4, cudaMemcpyDeviceToHost,
                                 stream));
      bev::Check(cudaStreamSynchronize(stream));
      std::copy(ints.begin(), ints.end(), idx.values.begin());
      result.tensors.push_back(std::move(idx));
    }
    bev::Check(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < stage_index; ++i) {
      float ms = 0;
      bev::Check(cudaEventElapsedTime(&ms, spans[i].start, spans[i].stop));
      result.timings[i].gpu_ms = ms;
    }
    size_t free = 0, total = 0;
    if (profile) {
      bev::Check(cudaMemGetInfo(&free, &total));
      result.device_memory_used_bytes = total - free;
    }
    return result;
  }
};
BevFusionInferenceRuntime::BevFusionInferenceRuntime(
    const std::string& dir, const std::string& override_path)
    : impl_(new Impl(dir, override_path)) {}
BevFusionInferenceRuntime::~BevFusionInferenceRuntime() = default;
const BevFusionConfig& BevFusionInferenceRuntime::config() const {
  return impl_->cfg;
}
BevFusionResult BevFusionInferenceRuntime::Infer(const BevFusionFrame& frame,
                                                 bool debug) {
  return impl_->infer(frame, debug);
}
}  // namespace inference

inference::BevFusionResult inference_internal::RuntimeDiagnostics::Fusion(
    inference::BevFusionInferenceRuntime& runtime, const inference::BevFusionFrame& frame,
    bool profile, bool debug) { return runtime.impl_->infer(frame, debug, profile); }
