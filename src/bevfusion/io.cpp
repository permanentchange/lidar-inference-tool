#include <setjmp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <set>

#include "internal.hpp"
#include "postprocess_config.hpp"
#include "pillar_runtime.hpp"

extern "C" {
#include <jpeglib.h>
}

namespace fs = std::filesystem;
namespace inference {
namespace {
void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
template <class T, size_t N>
std::array<T, N> Array(const YAML::Node& node, const std::string& name) {
  Require(node.IsSequence() && node.size() == N, "Invalid shape: " + name);
  std::array<T, N> out{};
  for (size_t i = 0; i < N; ++i) {
    out[i] = node[i].as<T>();
    Require(std::isfinite(double(out[i])), "Nonfinite: " + name);
  }
  return out;
}
std::array<float, 16> Matrix(const YAML::Node& n, const std::string& name) {
  Require(n.IsSequence() && n.size() == 4, "Expected 4x4 matrix: " + name);
  std::array<float, 16> out{};
  for (int i = 0; i < 4; ++i) {
    auto row = Array<float, 4>(n[i], name);
    std::copy(row.begin(), row.end(), out.begin() + i * 4);
  }
  Require(std::abs(out[12]) + std::abs(out[13]) + std::abs(out[14]) < 1e-6f &&
              std::abs(out[15] - 1) < 1e-6f,
          "Expected affine 4x4 calibration: " + name);
  bev::Inverse(out);  // Reject singular/nonfinite calibration before CUDA work.
  return out;
}
std::string Path(const fs::path& root, const YAML::Node& node) {
  Require(node.IsScalar(), "Missing artifact path");
  fs::path p = node.as<std::string>();
  if (p.is_relative()) p = root / p;
  Require(fs::is_regular_file(p), "Missing file: " + p.string());
  return fs::absolute(p).string();
}
void Merge(YAML::Node target, const YAML::Node& source) {
  Require(source.IsMap(), "Override must be a mapping");
  for (const auto& kv : source) {
    auto k = kv.first.as<std::string>();
    if (kv.second.IsMap() && target[k].IsMap())
      Merge(target[k], kv.second);
    else
      target[k] = kv.second;
  }
}
struct JpegError {
  jpeg_error_mgr base;
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
};
void JpegFail(j_common_ptr c) {
  auto* e = reinterpret_cast<JpegError*>(c->err);
  (*c->err->format_message)(c, e->message);
  longjmp(e->jump, 1);
}
void ReadJpeg(BevFusionCamera& camera) {
  FILE* f = fopen(camera.image_path.c_str(), "rb");
  Require(f, "Cannot open JPEG: " + camera.image_path);
  jpeg_decompress_struct decoder{};
  JpegError error{};
  decoder.err = jpeg_std_error(&error.base);
  error.base.error_exit = JpegFail;
  if (setjmp(error.jump)) {
    jpeg_destroy_decompress(&decoder);
    fclose(f);
    throw std::runtime_error(error.message);
  }
  jpeg_create_decompress(&decoder);
  jpeg_stdio_src(&decoder, f);
  jpeg_save_markers(&decoder, JPEG_APP0 + 1, 0xffff);
  jpeg_read_header(&decoder, TRUE);
  // Pillow's input loader does not apply ImageOps.exif_transpose. Match it.
  decoder.out_color_space = JCS_RGB;
  decoder.dct_method = JDCT_ISLOW;
  decoder.do_fancy_upsampling = TRUE;
  jpeg_start_decompress(&decoder);
  camera.width = decoder.output_width;
  camera.height = decoder.output_height;
  if (camera.width <= 0 || camera.height <= 0 || camera.width > 16384 ||
      camera.height > 16384) {
    jpeg_destroy_decompress(&decoder);
    fclose(f);
    throw std::runtime_error("Invalid JPEG dimensions");
  }
  camera.rgb.resize(size_t(camera.width) * camera.height * 3);
  while (decoder.output_scanline < decoder.output_height) {
    JSAMPROW row =
        camera.rgb.data() + size_t(decoder.output_scanline) * camera.width * 3;
    jpeg_read_scanlines(&decoder, &row, 1);
  }
  jpeg_finish_decompress(&decoder);
  jpeg_destroy_decompress(&decoder);
  fclose(f);
}
}  // namespace
BevFusionTensor LoadBevFusionNpy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  Require(bool(f), "Cannot open NPY: " + path);
  char magic[8]{};
  f.read(magic, 8);
  Require(std::memcmp(magic, "\x93NUMPY", 6) == 0 &&
              (magic[6] == 1 || magic[6] == 2),
          "Unsupported NPY version: " + path);
  unsigned char length[4]{};
  int lenbytes = magic[6] == 1 ? 2 : 4;
  f.read(reinterpret_cast<char*>(length), lenbytes);
  uint32_t n = 0;
  for (int i = 0; i < lenbytes; ++i) n |= uint32_t(length[i]) << (8 * i);
  Require(n < 1024 * 1024, "NPY header too large: " + path);
  std::string header(n, ' ');
  f.read(header.data(), n);
  Require(header.find("'<f4'") != std::string::npos ||
              header.find("'=f4'") != std::string::npos,
          "NPY must contain little-endian float32: " + path);
  Require(std::regex_search(
              header, std::regex("['\"]fortran_order['\"]\\s*:\\s*False")),
          "NPY must be C contiguous: " + path);
  std::smatch match;
  Require(
      std::regex_search(header, match,
                        std::regex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)")),
      "Invalid NPY shape: " + path);
  BevFusionTensor out;
  out.name = fs::path(path).stem().string();
  std::string dims = match[1];
  Require(dims.find_first_not_of("0123456789, \t") == std::string::npos,
          "Invalid NPY dimensions: " + path);
  std::regex number("[0-9]+");
  size_t count = 1;
  for (std::sregex_iterator it(dims.begin(), dims.end(), number), end;
       it != end; ++it) {
    size_t d = std::stoull(it->str());
    Require(d <= 100000000 && count <= 100000000 / std::max(size_t(1), d),
            "NPY too large: " + path);
    out.shape.push_back(d);
    count *= d;
  }
  Require(!out.shape.empty(), "Scalar NPY unsupported: " + path);
  out.values.resize(count);
  f.read(reinterpret_cast<char*>(out.values.data()), count * sizeof(float));
  Require(bool(f) && f.peek() == std::char_traits<char>::eof(),
          "NPY size mismatch: " + path);
  Require(std::all_of(out.values.begin(), out.values.end(),
                      [](float x) { return std::isfinite(x); }),
          "Nonfinite NPY data: " + path);
  return out;
}
void SaveBevFusionNpy(const std::string& path, const BevFusionTensor& t) {
  std::string error;
  if (inference_internal::SaveFloatNpy(path, t.values.data(), t.values.size(),
                                       t.shape, &error) != INFERENCE_STATUS_OK)
    throw std::runtime_error(error);
}
bool IsBevFusionModel(const std::string& dir) {
  auto n = YAML::LoadFile((fs::path(dir) / "model.yaml").string());
  return n["model"] && n["model"]["runtime_backend"] &&
         n["model"]["runtime_backend"].as<std::string>() == "bevfusion";
}
BevFusionConfig LoadBevFusionConfig(const std::string& dir,
                                    const std::string& override_path) {
  auto n = YAML::LoadFile((fs::path(dir) / "model.yaml").string());
  if (!override_path.empty()) Merge(n, YAML::LoadFile(override_path));
  auto m = n["model"], c = m["camera"], r = n["runtime"], p = n["postprocess"];
  Require(m["runtime_backend"].as<std::string>() == "bevfusion",
          "Expected runtime_backend: bevfusion");
  Require(m["architecture"] && m["architecture"].as<std::string>() == "centerpoint_bevfusion",
          "BEVFusion requires the current CenterHead package (demo/bevfusion_exp20261028)");
  Require(m["bbox_code_size"].as<int>() == 9 && m["has_velocity"].as<bool>(),
          "BEVFusion requires velocity-enabled CenterHead code_size 9");
  Require(m["decode_mode"].as<std::string>() == "centerpoint_mmdeploy" &&
          m["yaw_encoding"].as<std::string>() == "sin_cos",
          "BEVFusion requires the common CenterPoint sin_cos decoder");
  BevFusionConfig o;
  o.architecture = "centerpoint_bevfusion";
  o.bev_pool_backend = r["bev_pool_backend"] ? r["bev_pool_backend"].as<std::string>() : "reference";
  Require(o.bev_pool_backend == "reference" || o.bev_pool_backend == "nvidia_pack10", "Unknown BEV pooling backend");
  o.model_dir = fs::absolute(dir).string();
  o.sparse_model = Path(dir, m["scn_onnx_path"]);
  o.camera_engine = Path(dir, m["camera_engine_path"]);
  o.head_engine = Path(dir, m["head_engine_path"]);
  o.num_point_features = m["num_point_features"].as<int>();
  o.out_size_factor = m["out_size_factor"].as<int>();
  o.max_points = r["max_points"].as<int>();
  o.max_voxels = r["max_voxels"].as<int>();
  o.max_points_per_voxel = r["max_points_per_voxel"].as<int>();
  o.point_cloud_range =
      Array<float, 6>(m["point_cloud_range"], "point_cloud_range");
  o.post_center_range =
      Array<float, 6>(m["post_center_range"], "post_center_range");
  o.voxel_size = Array<float, 3>(m["voxel_size"], "voxel_size");
  o.sparse_shape = Array<int, 3>(m["sparse_shape"], "sparse_shape");
  o.label_names = m["label_names"].as<std::vector<std::string>>();
  o.camera_names = c["camera_names"].as<std::vector<std::string>>();
  o.image_size = Array<int, 2>(c["image_size"], "image_size");
  o.feature_size = Array<int, 2>(c["feature_size"], "feature_size");
  o.mean = Array<float, 3>(c["mean"], "mean");
  o.std = Array<float, 3>(c["std"], "std");
  o.xbound = Array<float, 3>(c["xbound"], "xbound");
  o.ybound = Array<float, 3>(c["ybound"], "ybound");
  o.zbound = Array<float, 3>(c["zbound"], "zbound");
  o.dbound = Array<float, 3>(c["dbound"], "dbound");
  o.feature_channels = c["feature_channels"].as<int>();
  o.depth_bins = c["depth_bins"].as<int>();
  o.score_threshold = r["score_threshold"].as<float>();
  o.min_voxels = r["min_voxels"].as<int>();
  o.frame_num = r["frame_num"].as<int>();
  {
    o.task_label_ids = m["task_label_ids"].as<std::vector<std::vector<int>>>();
    o.bbox_max_num = r["bbox_max_num"].as<int>();
    o.norm_bbox = m["norm_bbox"].as<bool>();
    ModelConfig head_model;
    head_model.decode_mode = DecodeMode::kCenterPoint;
    head_model.label_names = o.label_names;
    head_model.task_label_ids = o.task_label_ids;
    head_model.task_num_classes = m["task_num_classes"].as<std::vector<int>>();
    Require(head_model.task_num_classes.size() == o.task_label_ids.size(), "Task count mismatch");
    std::set<int> ids;
    for (size_t task = 0; task < o.task_label_ids.size(); ++task) {
      Require(!o.task_label_ids[task].empty() &&
              size_t(head_model.task_num_classes[task]) == o.task_label_ids[task].size(),
              "Task class count mismatch");
      for (int id : o.task_label_ids[task])
        Require(id >= 0 && size_t(id) < o.label_names.size() && ids.insert(id).second,
                "Invalid/duplicate task label");
    }
    Require(ids.size() == o.label_names.size() && o.bbox_max_num > 0 && o.bbox_max_num <= 10000,
            "Invalid CenterHead label coverage/TopK limits");
    std::string error;
    auto& pp = o.centerhead_postprocess;
    Require(LoadPostprocessConfig(p, &pp, &error) == INFERENCE_STATUS_OK, error);
    Require(ValidatePostprocessConfig(head_model, pp, &error) == INFERENCE_STATUS_OK, error);
    Require(o.min_voxels > 0 && o.min_voxels <= o.max_voxels && o.frame_num == 1,
            "Invalid voxel limits or frame_num (only single-frame inference is supported)");
  }

  Require(
      o.num_point_features == 5 && o.max_points > 0 &&
          o.max_points <= 10000000 && o.max_voxels > 0 &&
          o.max_points_per_voxel > 0,
      "Invalid point/voxel limits (baseline requires five source features)");
  Require(o.max_voxels <= o.max_points && o.out_size_factor > 0 &&
              !o.label_names.empty(),
          "Invalid BEVFusion limits/classes");
  Require(!o.camera_names.empty() &&
              int(o.camera_names.size()) == c["num_cameras"].as<int>(),
          "Invalid camera count");
  Require(std::set<std::string>(o.camera_names.begin(), o.camera_names.end())
                  .size() == o.camera_names.size(),
          "Duplicate camera names");
  for (int i = 0; i < 3; ++i)
    Require(o.voxel_size[i] > 0 && o.sparse_shape[i] > 0 && o.std[i] > 0 &&
                o.point_cloud_range[i] < o.point_cloud_range[i + 3] &&
                o.post_center_range[i] <= o.post_center_range[i + 3],
            "Invalid range/voxel/std");
  for (auto a : {o.xbound, o.ybound, o.zbound, o.dbound})
    Require(a[2] > 0 && a[1] > a[0] &&
                std::abs((a[1] - a[0]) / a[2] -
                         std::round((a[1] - a[0]) / a[2])) < 1e-3,
            "Invalid view-transform bound");
  Require(o.image_size[0] > 1 && o.image_size[1] > 1 && o.feature_size[0] > 1 &&
              o.feature_size[1] > 1 && o.feature_channels > 0 &&
              o.depth_bins ==
                  int(std::round((o.dbound[1] - o.dbound[0]) / o.dbound[2])),
          "Invalid view-transform dimensions");
  Require(o.score_threshold >= 0 && std::isfinite(o.score_threshold),
          "Invalid score threshold");
  double cells = 1;
  for (auto b : {o.xbound, o.ybound, o.zbound})
    cells *= std::round((b[1] - b[0]) / b[2]);
  Require(cells * o.feature_channels <= 100000000 &&
              double(o.camera_names.size()) * o.feature_channels *
                      o.depth_bins * o.feature_size[0] * o.feature_size[1] <=
                  1000000000,
          "View-transform allocation too large");
  Require(double(o.max_points) * o.num_point_features <= 100000000,
          "Point allocation too large");
  return o;
}
BevFusionFrame LoadBevFusionFrame(const std::string& path,
                                  const BevFusionConfig& cfg, bool raw,
                                  int source_features) {
  auto n = YAML::LoadFile(path);
  fs::path root = fs::absolute(path).parent_path();
  BevFusionFrame out;
  out.manifest_path = fs::absolute(path).string();
  out.raw_images = raw;
  out.num_point_features = n["num_point_features"].as<int>();
  Require(out.num_point_features == cfg.num_point_features,
          "Point feature count differs from model");
  Require(source_features == 0 || source_features == out.num_point_features,
          "--source-num-point-features conflicts with manifest");
  std::string point_path = Path(root, n["point_cloud"]);
  if (fs::path(point_path).extension() == ".npy") {
    auto t = LoadBevFusionNpy(point_path);
    Require(t.shape.size() == 2 && t.shape[1] == size_t(cfg.num_point_features),
            "Expected points NPY [N,5]");
    out.points = std::move(t.values);
  } else {
    std::string error;
    if (inference_internal::LoadPcdFile(
            point_path, cfg.num_point_features, std::numeric_limits<int>::max(),
            &out.points, &error, nullptr,
            out.num_point_features) != INFERENCE_STATUS_OK)
      throw std::runtime_error(error);
  }
  out.source_points = out.points.size() / cfg.num_point_features;
  Require(out.source_points > 0, "Empty point cloud: " + point_path);
  out.lidar_aug_matrix = Matrix(n["lidar_aug_matrix"], "lidar_aug_matrix");
  auto cams = n["cameras"];
  Require(cams.IsSequence() && cams.size() == cfg.camera_names.size(),
          "Camera count mismatch");
  std::map<std::string, YAML::Node> byname;
  for (auto cam : cams)
    Require(byname.emplace(cam["name"].as<std::string>(), cam).second,
            "Duplicate manifest camera");
  std::vector<size_t> original_order;
  for (const auto& name : cfg.camera_names) {
    Require(byname.count(name), "Missing camera: " + name);
    auto c = byname.at(name);
    BevFusionCamera camera;
    camera.name = name;
    camera.enabled = !c["enabled"] || c["enabled"].as<bool>();
    auto identity = YAML::Load("[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]");
    camera.cam2img = Matrix(camera.enabled ? c["cam2img"] : identity, "cam2img");
    camera.cam2lidar = Matrix(camera.enabled ? c["cam2lidar"] : identity, "cam2lidar");
    camera.lidar2img = Matrix(camera.enabled ? c["lidar2img"] : identity, "lidar2img");
    camera.img_aug_matrix = Matrix(camera.enabled ? c["img_aug_matrix"] : identity, "img_aug_matrix");
    for (size_t i = 0; i < cams.size(); ++i)
      if (cams[i]["name"].as<std::string>() == name)
        original_order.push_back(i);
    if (raw && camera.enabled) {
      Require(bool(c["preprocess"]),
              "Raw JPEG preprocessing metadata missing; rerun "
              "scripts/prepare_bevfusion.py or use --image-input preprocessed");
      auto a = c["preprocess"];
      auto resize =
          Array<int, 2>(a["resize_size"], "resize_size [width,height]");
      auto crop = Array<int, 2>(a["crop_offset"], "crop_offset [left,top]");
      Require(a["interpolation"].as<std::string>() == "pillow_bicubic" &&
                  a["color_order"].as<std::string>() == "RGB",
              "Unsupported image preprocessing contract");
      camera.resize_width = resize[0];
      camera.resize_height = resize[1];
      camera.crop_left = crop[0];
      camera.crop_top = crop[1];
      Require(resize[0] > 0 && resize[1] > 0 && resize[0] <= 16384 &&
                  resize[1] <= 16384,
              "Invalid integer resize dimensions");
      camera.image_path = Path(root, c["image"]);
      ReadJpeg(camera);
      const std::string operation = a["operation"] ? a["operation"].as<std::string>() : "resize_crop";
      Require(operation == "resize_crop" || operation == "crop_resize", "Unsupported preprocessing operation");
      if (operation == "crop_resize") {
        camera.source_crop = Array<int, 4>(a["source_crop"], "source_crop");
        auto roi = camera.source_crop;
        Require(roi[0] >= 0 && roi[1] >= 0 && roi[2] > roi[0] && roi[3] > roi[1] &&
                roi[2] <= camera.width && roi[3] <= camera.height &&
                resize[0] == cfg.image_size[1] && resize[1] == cfg.image_size[0] && crop[0] == 0 && crop[1] == 0,
                "Invalid crop-before-resize ROI");

      }
    }
    out.cameras.push_back(std::move(camera));
  }
  Require(std::any_of(out.cameras.begin(), out.cameras.end(), [](const auto& cam) { return cam.enabled; }),
          "At least one camera must be enabled");
  if (!raw) {
    auto images = LoadBevFusionNpy(Path(root, n["preprocessed_images"]));
    const size_t plane = size_t(3) * cfg.image_size[0] * cfg.image_size[1];
    Require(images.shape == std::vector<size_t>{cfg.camera_names.size(), 3,
                                                size_t(cfg.image_size[0]),
                                                size_t(cfg.image_size[1])},
            "Expected images NPY [cameras,3,H,W]");
    out.images.resize(images.values.size());
    for (size_t i = 0; i < original_order.size(); ++i)
      std::copy_n(images.values.begin() + original_order[i] * plane, plane,
                  out.images.begin() + i * plane);
  }
  return out;
}
namespace bev {
std::array<float, 16> Inverse(const std::array<float, 16>& a) {
  double m[4][8]{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      m[i][j] = a[i * 4 + j];
      m[i][j + 4] = i == j;
    }
  for (int i = 0; i < 4; ++i) {
    int pivot = i;
    for (int j = i + 1; j < 4; ++j)
      if (std::abs(m[j][i]) > std::abs(m[pivot][i])) pivot = j;
    if (std::abs(m[pivot][i]) < 1e-12)
      throw std::runtime_error("Singular calibration matrix");
    for (int j = 0; j < 8; ++j) std::swap(m[i][j], m[pivot][j]);
    double s = m[i][i];
    for (int j = 0; j < 8; ++j) m[i][j] /= s;
    for (int k = 0; k < 4; ++k)
      if (k != i) {
        s = m[k][i];
        for (int j = 0; j < 8; ++j) m[k][j] -= s * m[i][j];
      }
  }
  std::array<float, 16> out{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) out[i * 4 + j] = float(m[i][j + 4]);
  return out;
}
}  // namespace bev
}  // namespace inference
