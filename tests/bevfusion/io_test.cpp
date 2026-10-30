#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

#include "bevfusion_inference.hpp"
namespace fs = std::filesystem;
using namespace inference;
void Expect(bool x, const char* text) {
  if (!x) throw std::runtime_error(text);
}
void Reject(const std::function<void()>& f) {
  bool fail = false;
  try {
    f();
  } catch (const std::exception&) {
    fail = true;
  }
  Expect(fail, "Malformed input accepted");
}
YAML::Node Identity() {
  return YAML::Load("[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]");
}
void Write(const fs::path& p, const YAML::Node& n) {
  std::ofstream f(p);
  f << n;
}
int main() {
  fs::path root = fs::temp_directory_path() /
                  ("bevfusion-io-test-" + std::to_string(getpid()));
  fs::create_directories(root);
  try {
    auto model = YAML::Load(R"(
model:
  runtime_backend: bevfusion
  architecture: centerpoint_bevfusion
  norm_bbox: true
  decode_mode: centerpoint_mmdeploy
  yaw_encoding: sin_cos
  task_num_classes: [1,1]
  task_label_ids: [[0],[1]]
  scn_onnx_path: sparse.onnx
  camera_engine_path: camera.engine
  vtransform_engine_path: vt.engine
  fuser_engine_path: fuser.engine
  head_engine_path: head.engine
  num_point_features: 5
  out_size_factor: 8
  bbox_code_size: 9
  has_velocity: true
  point_cloud_range: [-8,-8,-2,16,8,4]
  post_center_range: [-10,-10,-2,18,10,4]
  voxel_size: [0.05,0.05,0.2]
  sparse_shape: [480,320,31]
  label_names: [one,two]
  camera:
    num_cameras: 2
    camera_names: [a,b]
    image_size: [2,2]
    feature_size: [2,2]
    mean: [1,2,3]
    std: [1,1,1]
    feature_channels: 1
    depth_bins: 1
    xbound: [0,3,1]
    ybound: [0,2,1]
    zbound: [0,2,2]
    dbound: [1,2,1]
runtime: {max_points: 10, max_voxels: 8, min_voxels: 1, frame_num: 1, score_threshold: 0, max_points_per_voxel: 2, bbox_max_num: 500}
postprocess: {nms_type: iou_bev, nms_thresh: 0.2, nms_pre_maxsize: 100, nms_post_maxsize: 50, final_score_threshold: 0.3, label_score_thresh: [0.3,0.3], nms_label_groups: [{labels: [0,1], nms_thresh: 0.2, nms_pre_maxsize: 100, nms_post_maxsize: 50}]}
)");
    for (auto name : {"sparse.onnx", "camera.engine", "vt.engine",
                      "fuser.engine", "head.engine"}) {
      std::ofstream f(root / name);
      f << "placeholder";
    }
    Write(root / "model.yaml", model);
    auto obsolete = YAML::Clone(model);
    obsolete["model"]["architecture"] = "transfusion";
    Write(root / "model.yaml", obsolete);
    Reject([&] { LoadBevFusionConfig(root.string()); });
    Write(root / "model.yaml", model);
    auto cfg = LoadBevFusionConfig(root.string());
    Expect(IsBevFusionModel(root.string()), "backend dispatch");
    BevFusionTensor points{"points", {1, 5}, {.1, .2, .3, 4, 1}};
    SaveBevFusionNpy((root / "points.npy").string(), points);
    auto loaded = LoadBevFusionNpy((root / "points.npy").string());
    Expect(loaded.shape == points.shape && loaded.values == points.values,
           "NPY roundtrip");
    BevFusionTensor images{"images", {2, 3, 2, 2}, std::vector<float>(24)};
    std::fill(images.values.begin(), images.values.begin() + 12, 2);
    std::fill(images.values.begin() + 12, images.values.end(), 1);
    SaveBevFusionNpy((root / "images.npy").string(), images);
    YAML::Node manifest;
    manifest["point_cloud"] = "points.npy";
    manifest["preprocessed_images"] = "images.npy";
    manifest["num_point_features"] = 5;
    manifest["lidar_aug_matrix"] = Identity();
    for (auto name : {"b", "a"}) {
      YAML::Node camera;
      camera["name"] = name;
      for (auto key : {"cam2img", "cam2lidar", "lidar2img", "img_aug_matrix"})
        camera[key] = Identity();
      manifest["cameras"].push_back(camera);
    }
    Write(root / "frame.yaml", manifest);
    auto frame = LoadBevFusionFrame((root / "frame.yaml").string(), cfg, false);
    Expect(frame.cameras[0].name == "a" && frame.images[0] == 1 &&
               frame.images[12] == 2,
           "camera name reordering and image plane order");
    Reject(
        [&] { LoadBevFusionFrame((root / "frame.yaml").string(), cfg, true); });
    auto bad = YAML::Clone(manifest);
    bad["cameras"][1]["name"] = "b";
    Write(root / "frame.yaml", bad);
    Reject([&] {
      LoadBevFusionFrame((root / "frame.yaml").string(), cfg, false);
    });
    bad = YAML::Clone(manifest);
    bad["cameras"][0]["cam2img"][2][2] = 0;
    Write(root / "frame.yaml", bad);
    Reject([&] {
      LoadBevFusionFrame((root / "frame.yaml").string(), cfg, false);
    });
    auto wrong = YAML::Clone(model);
    wrong["model"]["camera"]["xbound"][2] = 0;
    Write(root / "model.yaml", wrong);
    Reject([&] { LoadBevFusionConfig(root.string()); });
    Write(root / "model.yaml", model);
    // Disabled cameras need neither image files nor usable calibration.
    auto masked = YAML::Clone(manifest);
    masked["cameras"][0]["enabled"] = false;
    masked["cameras"][0]["cam2img"] = YAML::Node();
    Write(root / "frame.yaml", masked);
    auto mask_frame = LoadBevFusionFrame((root / "frame.yaml").string(), cfg, false);
    Expect(mask_frame.cameras[0].enabled && !mask_frame.cameras[1].enabled,
           "camera validity follows names, not manifest order");
    masked["cameras"][1]["enabled"] = false;
    Write(root / "frame.yaml", masked);
    Reject([&] { LoadBevFusionFrame((root / "frame.yaml").string(), cfg, false); });
    auto aligned = YAML::Clone(model);
    aligned["model"]["architecture"] = "centerpoint_bevfusion";
    aligned["model"]["bbox_code_size"] = 9;
    aligned["model"]["norm_bbox"] = true;
    aligned["model"]["task_label_ids"] = YAML::Load("[[1],[0]]");
    aligned["model"].remove("vtransform_engine_path");
    aligned["model"].remove("fuser_engine_path");
    aligned["runtime"]["bbox_max_num"] = 500;
    aligned["postprocess"] = YAML::Load(R"({nms_type: iou_bev,
      nms_thresh: 0.2, nms_pre_maxsize: 1000, nms_post_maxsize: 55,
      final_score_threshold: 0.3, label_score_thresh: [0.35,0.45],
      nms_label_groups: [{labels: [0,1], nms_thresh: 0.3, nms_pre_maxsize: 500, nms_post_maxsize: 50}]})");
    Write(root / "model.yaml", aligned);
    auto aligned_cfg = LoadBevFusionConfig(root.string());
    Expect(aligned_cfg.task_label_ids[0][0] == 1,
           "two-engine architecture and noncontiguous task map");
    aligned["model"]["task_label_ids"] = YAML::Load("[[0],[0]]");
    Write(root / "model.yaml", aligned);
    Reject([&] { LoadBevFusionConfig(root.string()); });
    Write(root / "model.yaml", model);
    fs::remove(root / "head.engine");
    Reject([&] { LoadBevFusionConfig(root.string()); });
    // Corrupt a valid NPY to reject Fortran order and wrong scalar dtype.
    std::ifstream f(root / "images.npy", std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(f)), {});
    auto pos = bytes.find("False");
    Expect(pos != std::string::npos, "NPY fixture header");
    bytes.replace(pos, 5, "True ");
    std::ofstream g(root / "bad.npy", std::ios::binary);
    g.write(bytes.data(), bytes.size());
    g.close();
    Reject([&] { LoadBevFusionNpy((root / "bad.npy").string()); });
    fs::remove_all(root);
    std::cout << "BEVFusion configuration/input checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr << e.what() << '\n';
    return 1;
  }
}
