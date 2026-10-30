#include <cuda_fp16.h>

#include <cmath>
#include <algorithm>
#include <climits>
#include <iostream>
#include <stdexcept>

#include "internal.hpp"
#include "pillar_postprocess/pillar_postprocess.hpp"
using namespace inference;
void Expect(bool v, const char* text) {
  if (!v) throw std::runtime_error(text);
}
std::array<float, 16> Identity() {
  return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}
BevFusionConfig Config() {
  BevFusionConfig c;
  c.num_point_features = 5;
  c.max_points = 32;
  c.max_voxels = 8;
  c.max_points_per_voxel = 2;
  c.point_cloud_range = {-1, -1, -1, 1, 1, 1};
  c.voxel_size = {1, 1, 1};
  c.sparse_shape = {2, 2, 2};
  c.image_size = {2, 2};
  c.feature_size = {2, 2};
  c.camera_names = {"a", "b"};
  c.feature_channels = 1;
  c.depth_bins = 1;
  c.xbound = {0, 3, 1};
  c.ybound = {0, 2, 1};
  c.zbound = {0, 2, 2};
  c.dbound = {1, 2, 1};
  return c;
}
template <class T>
std::vector<T> Read(bev::Buffer& b, size_t n) {
  std::vector<T> x(n);
  bev::Check(
      cudaMemcpy(x.data(), b.data, n * sizeof(T), cudaMemcpyDeviceToHost));
  return x;
}
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
    std::cout << "SKIP: no CUDA device\n";
    return 77;
  }
  cudaStream_t stream;
  bev::Check(cudaStreamCreate(&stream));
  try {
    auto c = Config();
    bev::Kernels k;
    // First appearance ordering, inclusive lower range boundary, first K points and
    // mean.
    std::vector<float> points = {.2f, .2f, .2f, 2,   1, -.2f, -.2f, -.2f, 8,
                                 1,   .4f, .4f, .4f, 4, 1,    .8f,  .8f,  .8f,
                                 99,  1,   -1,  0,   0, 99,   1};
    k.points.resize(points.size() * 4);
    bev::Check(cudaMemcpyAsync(k.points.data, points.data(), points.size() * 4,
                               cudaMemcpyHostToDevice, stream));
    bev::Voxelize(k, c, 5, stream);
    bev::Check(cudaStreamSynchronize(stream));
    Expect(k.voxel_count == 3, "voxel count includes lower boundary");
    auto idx = Read<int>(k.indices, 12);
    Expect(idx == std::vector<int>({0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0}),
           "ZYX first appearance voxel ordering");
    auto mean = Read<__half>(k.features, 10);
    Expect(__half2float(mean[3]) == 3 && __half2float(mean[8]) == 8,
           "first K voxel mean");
    c.max_voxels = 1;
    bev::Voxelize(k, c, 5, stream);
    bev::Check(cudaStreamSynchronize(stream));
    Expect(k.voxel_count == 1, "max_voxels limit");
    // Non-square BEV, valid rank zero, truncate(-0.2)==0, per-camera
    // augmentation and the final exclusive interval must all contribute.
    c = Config();
    BevFusionFrame f;
    f.lidar_aug_matrix = Identity();
    f.cameras.resize(2);
    for (auto& camera : f.cameras) {
      camera.cam2img = Identity();
      camera.cam2lidar = Identity();
      camera.img_aug_matrix = Identity();
      camera.lidar2img = Identity();
    }
    f.cameras[1].cam2lidar[3] = -.2f;
    f.cameras[1].img_aug_matrix[3] = -1;
    bev::Geometry(k, c, f, stream);
    bev::Buffer feat, dep;
    std::vector<float> ones(8, 1);
    feat.resize(32);
    dep.resize(32);
    bev::Check(cudaMemcpyAsync(feat.data, ones.data(), 32,
                               cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(dep.data, ones.data(), 32,
                               cudaMemcpyHostToDevice, stream));
    bev::Pool(k, c, feat.ptr<float>(), dep.ptr<float>(), stream);
    bev::Check(cudaStreamSynchronize(stream));
    auto pool = Read<float>(k.pooled, 6);
    Expect(pool == std::vector<float>({2, 2, 2, 2, 0, 0}),
           "geometry/pooling asymmetric layout, camera augmentation and last "
           "interval");
    f.cameras[1].img_aug_matrix[3] = -2;
    bev::Geometry(k, c, f, stream);
    bev::Pool(k, c, feat.ptr<float>(), dep.ptr<float>(), stream);
    bev::Check(cudaStreamSynchronize(stream));
    pool = Read<float>(k.pooled, 6);
    Expect(pool == std::vector<float>({1, 1, 2, 2, 1, 1}),
           "geometry changes and invalid sentinel");
    f.cameras[1].img_aug_matrix[3] = -4;
    bev::Geometry(k, c, f, stream);
    bev::Pool(k, c, feat.ptr<float>(), dep.ptr<float>(), stream);
    bev::Check(cudaStreamSynchronize(stream));
    pool = Read<float>(k.pooled, 6);
    Expect(pool == std::vector<float>({1, 1, 1, 1, 0, 0}),
           "invalid geometry sentinel");
    f.cameras[0].img_aug_matrix = Identity();
    // Collision policy is last original point, with inverse lidar augmentation.
    c = Config();
    c.point_cloud_range = {-4, -4, 0, 4, 4, 4};
    c.camera_names = {"a"};
    f.cameras.resize(1);
    f.lidar_aug_matrix = Identity();
    f.lidar_aug_matrix[3] = 1;
    points = {1.5, .5, 1, 0, 1, 2, 1, 2, 0, 1};
    k.points.resize(points.size() * 4);
    bev::Check(cudaMemcpyAsync(k.points.data, points.data(), points.size() * 4,
                               cudaMemcpyHostToDevice, stream));
    auto collisions = bev::Depth(k, c, f, 2, stream);
    auto depth = Read<float>(k.depth, 4);
    Expect(collisions == 1 && depth[0] == 2,
           "last-source depth and inverse augmentation");
    collisions = bev::Depth(k, c, f, 2, stream, bev::ScalarType::Float16);
    auto half_depth = Read<__half>(k.depth, 4);
    Expect(collisions == 1 && __half2float(half_depth[0]) == 2,
           "FP16 depth preserves collision ownership and augmentation");
    f.cameras[0].enabled = false;
    f.cameras[0].img_aug_matrix = {};
    bev::Depth(k, c, f, 2, stream);
    Expect(Read<float>(k.depth, 4) == std::vector<float>(4, 0), "disabled camera has no projected depth");
    bev::Geometry(k, c, f, stream);
    auto ranks = Read<int>(k.ranks, 4);
    Expect(std::all_of(ranks.begin(), ranks.end(), [](int rank) { return rank == INT_MAX; }),
           "disabled geometry excluded from every interval");
    f.cameras[0].enabled = true;
    f.cameras[0].img_aug_matrix = Identity();
    // Empty ROI is a clear failure, not a spconv crash.
    points = {4, 0, 0, 0, 1};
    bev::Check(cudaMemcpyAsync(k.points.data, points.data(), points.size() * 4,
                               cudaMemcpyHostToDevice, stream));
    bool caught = false;
    try {
      bev::Voxelize(k, c, 1, stream);
    } catch (const std::runtime_error&) {
      caught = true;
    }
    Expect(caught, "empty ROI rejection");
    // Half arithmetic would lose the unit contribution between 2048 and -2048.
    // The pooling accumulator must stay FP32 and round only its final output.
    c = Config();
    c.camera_names = {"a"};
    c.feature_size = {1, 4};
    c.xbound = c.ybound = c.zbound = {0, 1, 1};
    std::vector<__half> half_feature = {__float2half(2048), __float2half(1),
                                       __float2half(-2048), __float2half(.3333f)};
    std::vector<__half> half_weight(4, __float2half(1));
    std::vector<int> order = {0, 1, 2, 3}, start = {0}, stop = {4};
    k.geometry_order.resize(16);
    k.pool_starts.resize(4);
    k.pool_ends.resize(4);
    feat.resize(8);
    dep.resize(8);
    bev::Check(cudaMemcpyAsync(k.geometry_order.data, order.data(), 16, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(k.pool_starts.data, start.data(), 4, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(k.pool_ends.data, stop.data(), 4, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(feat.data, half_feature.data(), 8, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(dep.data, half_weight.data(), 8, cudaMemcpyHostToDevice, stream));
    bev::Pool(k, c, feat.data, dep.data, stream, bev::ScalarType::Float16, bev::ScalarType::Float16);
    auto half_pool = Read<__half>(k.pooled, 1);
    float expected = 1 + __half2float(half_feature[3]);
    Expect(__half2float(half_pool[0]) == __half2float(__float2half(expected)),
           "FP16 pooling uses FP32 accumulation and one output rounding");
    bev::Pool(k, c, feat.data, dep.data, stream, bev::ScalarType::Float16, bev::ScalarType::Float32);
    auto float_pool = Read<float>(k.pooled, 1);
    Expect(float_pool[0] == expected, "FP16 camera output can feed an FP32 pooling boundary");
    // Non-square XY layout, empty cells, FP32 cancellation and the last interval.
    c = Config();
    c.camera_names = {"a"};
    c.feature_channels = 80;
    c.feature_size = {1, 5};
    std::vector<__half> tiled_features(5 * 80), tiled_depth(5, __float2half(1));
    for (int channel = 0; channel < 80; ++channel) {
      float values[] = {2048, 1, -2048, .3333f, 2 + channel / 16.f};
      for (int point = 0; point < 5; ++point)
        tiled_features[point * 80 + channel] = __float2half(values[point]);
    }
    order = {0, 1, 2, 3, 4};
    start = {0, -1, -1, -1, -1, 4};
    stop = {4, -1, -1, -1, -1, 5};
    k.geometry_order.resize(order.size() * 4);
    k.pool_starts.resize(start.size() * 4);
    k.pool_ends.resize(stop.size() * 4);
    feat.resize(tiled_features.size() * 2);
    dep.resize(tiled_depth.size() * 2);
    bev::Check(cudaMemcpyAsync(k.geometry_order.data, order.data(), order.size() * 4, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(k.pool_starts.data, start.data(), start.size() * 4, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(k.pool_ends.data, stop.data(), stop.size() * 4, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(feat.data, tiled_features.data(), tiled_features.size() * 2, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(dep.data, tiled_depth.data(), tiled_depth.size() * 2, cudaMemcpyHostToDevice, stream));
    bev::Pool(k, c, feat.data, dep.data, stream, bev::ScalarType::Float16, bev::ScalarType::Float16);
    auto baseline = Read<__half>(k.pooled, 6 * 80);
    c.bev_pool_backend = "nvidia_pack10";
    bev::Pool(k, c, feat.data, dep.data, stream, bev::ScalarType::Float16, bev::ScalarType::Float16);
    auto packed = Read<__half>(k.pooled, 6 * 80);
    for (int channel = 0; channel < 80; ++channel) {
      for (int cell = 0; cell < 6; ++cell)
        Expect(__half2float(packed[channel * 6 + cell]) == __half2float(baseline[channel * 6 + cell]), "tiled/reference pooling equality");
      Expect(__half2float(packed[channel * 6]) == __half2float(__float2half(1.3333f)), "tiled FP32 cancellation");
      Expect(__half2float(packed[channel * 6 + 5]) == __half2float(tiled_features[4 * 80 + channel]), "final pooling interval is exclusive and complete");
      Expect(__half2float(packed[channel * 6 + 1]) == 0, "empty pooled cell is zero");
    }
    c = Config();
    c.architecture = "centerpoint_bevfusion";
    points = {-1, -.2f, .2f, 0, 1, .2f, -.2f, -.2f, 0, 1};
    k.points.resize(points.size() * 4);
    bev::Check(cudaMemcpyAsync(k.points.data, points.data(), points.size() * 4, cudaMemcpyHostToDevice, stream));
    bev::Voxelize(k, c, 2, stream);
    Expect(Read<int>(k.indices, 8) == std::vector<int>({0,1,0,0, 0,0,0,1}),
           "CenterPoint ZYX order and inclusive lower ROI boundary");
    // An out-of-range highest-score box must still consume a TopK slot.
    inference_internal::RuntimeConfig decode_cfg;
    decode_cfg.topk_before_range_filter = true;
    decode_cfg.decode_mode = DecodeMode::kCenterPoint;
    decode_cfg.yaw_encoding = YawEncoding::kSinCos;
    decode_cfg.norm_bbox = true;
    decode_cfg.has_velocity = true;
    decode_cfg.bbox_code_size = 9;
    decode_cfg.task_num_classes = {1};
    decode_cfg.task_label_ids = {{0}};
    decode_cfg.label_count = 1;
    decode_cfg.bbox_max_num = 1;
    decode_cfg.out_size_factor = 1;
    decode_cfg.point_cloud_range = {0,0,-2,3,1,2};
    decode_cfg.post_center_range = decode_cfg.point_cloud_range;
    decode_cfg.voxel_size = {1,1,1};
    inference_internal::ResolvedHeadLayout layout;
    std::string error;
    Expect(inference_internal::ResolveHeadLayout(decode_cfg, &layout, &error) == INFERENCE_STATUS_OK, error.c_str());
    inference_internal::GpuDecodedCandidates gpu_decoder;
    std::vector<float> boxes(24, 0), heat = {10,9,8}, rotation = {0,0,0,1,1,1};
    boxes[0] = -10;
    boxes[6 * 3 + 1] = 2.5;
    boxes[7 * 3 + 1] = -3;
    bev::Buffer device_boxes, device_heat, device_rotation;
    device_boxes.resize(boxes.size() * 4);
    device_heat.resize(heat.size() * 4);
    device_rotation.resize(rotation.size() * 4);
    bev::Check(cudaMemcpyAsync(device_boxes.data, boxes.data(), boxes.size() * 4, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(device_heat.data, heat.data(), heat.size() * 4, cudaMemcpyHostToDevice, stream));
    bev::Check(cudaMemcpyAsync(device_rotation.data, rotation.data(), rotation.size() * 4, cudaMemcpyHostToDevice, stream));
    inference_internal::CandidateDetections candidates;
    std::vector<float> velocities;
    for (bool before_range : {true, false}) {
      decode_cfg.topk_before_range_filter = before_range;
      Expect(gpu_decoder.Initialize(decode_cfg, layout, 1, 8, 1, 2, 1, 3, &error) == INFERENCE_STATUS_OK, error.c_str());
      Expect(gpu_decoder.Decode(device_boxes.ptr<float>(), device_heat.ptr<float>(), device_rotation.ptr<float>(), stream, &error) == INFERENCE_STATUS_OK, error.c_str());
      Expect(gpu_decoder.CopyToHost(&candidates, stream, &error, &velocities) == INFERENCE_STATUS_OK, error.c_str());
      Expect(candidates.scores.size() == (before_range ? 0 : 1), "TopK/range ordering");
      if (!before_range)
        Expect(velocities == std::vector<float>({2.5f, -3.f}), "decoded velocity accompanies compact candidates");
    }
    std::vector<float> boundary = {1.0004f, -.3333f, 0, 65504};
    bev::Buffer source, half, restored;
    source.resize(16);
    half.resize(8);
    restored.resize(16);
    bev::Check(cudaMemcpyAsync(source.data, boundary.data(), 16, cudaMemcpyHostToDevice, stream));
    bev::FloatToHalf(source.ptr<float>(), half.data, 4, stream);
    bev::HalfToFloat(half.data, restored.ptr<float>(), 4, stream);
    auto converted = Read<float>(restored, 4);
    for (size_t i = 0; i < boundary.size(); ++i)
      Expect(converted[i] == __half2float(__float2half(boundary[i])),
             "external input conversion uses round-to-nearest FP16");
    bev::Check(cudaStreamDestroy(stream));
    std::cout << "BEVFusion CUDA semantic checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    cudaStreamDestroy(stream);
    std::cerr << e.what() << '\n';
    return 1;
  }
}
