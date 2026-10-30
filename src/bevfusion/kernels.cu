#include <cuda_fp16.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <climits>
#include <cmath>
#include <vector>

#include "internal.hpp"

namespace inference::bev {
namespace {
const std::array<float, 16>& ActiveMatrix(const BevFusionCamera& camera,
                                        const std::array<float, 16>& matrix) {
  static const std::array<float, 16> identity{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  return camera.enabled ? matrix : identity;
}
struct Params {
  int nf, np, nv, k, nx, ny, nz, ih, iw, fh, fw, nc, c, d, gx, gy, gz;
  float lo[3], hi[3], voxel[3], bound[3], step[3], dmin, dstep;
};
Params P(const BevFusionConfig& c) {
  Params p{};
  p.nf = c.num_point_features;
  p.np = c.max_points;
  p.nv = c.max_voxels;
  p.k = c.max_points_per_voxel;
  for (int i = 0; i < 3; ++i) {
    p.lo[i] = c.point_cloud_range[i];
    p.hi[i] = c.point_cloud_range[i + 3];
    p.voxel[i] = c.voxel_size[i];
  }
  p.nx = int(round((p.hi[0] - p.lo[0]) / p.voxel[0]));
  p.ny = int(round((p.hi[1] - p.lo[1]) / p.voxel[1]));
  p.nz = int(round((p.hi[2] - p.lo[2]) / p.voxel[2]));
  p.ih = c.image_size[0];
  p.iw = c.image_size[1];
  p.fh = c.feature_size[0];
  p.fw = c.feature_size[1];
  p.nc = c.camera_names.size();
  p.c = c.feature_channels;
  p.d = c.depth_bins;
  p.gx = int(round((c.xbound[1] - c.xbound[0]) / c.xbound[2]));
  p.gy = int(round((c.ybound[1] - c.ybound[0]) / c.ybound[2]));
  p.gz = int(round((c.zbound[1] - c.zbound[0]) / c.zbound[2]));
  for (int i = 0; i < 3; ++i) {
    auto b = i == 0 ? c.xbound : i == 1 ? c.ybound : c.zbound;
    p.bound[i] = b[0];
    p.step[i] = b[2];
  }
  p.dmin = c.dbound[0];
  p.dstep = c.dbound[2];
  return p;
}
__device__ bool Valid(const float* v, const Params& p) {
  for (int i = 0; i < 3; ++i)
    if (!(v[i] >= p.lo[i] && v[i] < p.hi[i])) return false;
  return true;
}
struct PointPredicate {
  const float* points;
  Params p;
  __device__ bool operator()(int i) const {
    return Valid(points + i * p.nf, p);
  }
};
__global__ void GatherPoints(const float* input, const int* ids, float* out,
                             int n, int nf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * nf) out[i] = input[ids[i / nf] * nf + i % nf];
}
__global__ void Keys(const float* points, uint64_t* keys, int n, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* v = points + i * p.nf;
  if (!Valid(v, p)) {
    keys[i] = ULLONG_MAX;
    return;
  }
  int x = int(floorf((v[0] - p.lo[0]) / p.voxel[0])),
      y = int(floorf((v[1] - p.lo[1]) / p.voxel[1])),
      z = int(floorf((v[2] - p.lo[2]) / p.voxel[2]));
  if (x >= p.nx || y >= p.ny || z >= p.nz) {
    keys[i] = ULLONG_MAX;
    return;
  }
  keys[i] = (uint64_t((x * p.ny + y) * p.nz + z) * p.np) + i;
}
struct StartPredicate {
  const uint64_t* keys;
  int np;
  __device__ bool operator()(int i) const {
    return keys[i] != ULLONG_MAX &&
           (i == 0 || keys[i] / np != keys[i - 1] / np);
  }
};
__global__ void First(const uint64_t* keys, const int* starts, int* first,
                      int groups, int np) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < groups) first[i] = keys[starts[i]] % np;
}
__global__ void Means(const float* points, const uint64_t* keys,
                      const int* starts, __half* features, int* indices,
                      int count, int n, Params p) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) return;
  int start = starts[g];
  uint64_t voxel = keys[start] / p.np;
  int used = 0;
  float sum[5] = {};
  for (int j = start;
       j < n && used < p.k && keys[j] != ULLONG_MAX && keys[j] / p.np == voxel;
       ++j, ++used) {
    int index = keys[j] % p.np;
    for (int c = 0; c < p.nf; ++c) sum[c] += points[index * p.nf + c];
  }
  for (int c = 0; c < p.nf; ++c)
    features[g * p.nf + c] = __float2half_rn(sum[c] / used);
  indices[g * 4] = 0;
  indices[g * 4 + 3] = voxel % p.nz;
  voxel /= p.nz;
  indices[g * 4 + 2] = voxel % p.ny;
  indices[g * 4 + 1] = voxel / p.ny;
  {
    int x = indices[g * 4 + 1];
    indices[g * 4 + 1] = indices[g * 4 + 3];
    indices[g * 4 + 3] = x;
  }
}
__device__ float3 Apply(const float* m, float3 v) {
  return make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3],
                     m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7],
                     m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11]);
}
__global__ void Project(const float* points, const float* matrices,
                        const int* enabled,
                        float* distances, int* pixels, int* owners,
                        unsigned long long* collisions, int n, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * p.nc) return;
  int point = i % n, cam = i / n;
  const float* v = points + point * p.nf;
  pixels[i] = -1;
  if (!enabled[cam]) return;
  if (!Valid(v, p)) return;
  float3 xyz = Apply(matrices, make_float3(v[0], v[1], v[2]));
  const float* mat = matrices + 16 + cam * 32;
  float3 q = Apply(mat, xyz);
  float depth = fminf(1e5f, fmaxf(1e-5f, q.z));
  q.x /= depth;
  q.y /= depth;
  q.z = depth;
  q = Apply(mat + 16, q);
  if (q.x >= 0 && q.y >= 0 && q.x < p.iw && q.y < p.ih) {
    int pixel = (cam * p.ih + int(q.y)) * p.iw + int(q.x);
    pixels[i] = pixel;
    distances[i] = depth;
    if (atomicMax(owners + pixel, point) >= 0) atomicAdd(collisions, 1ULL);
  }
}
template <class T>
__global__ void WriteDepth(const int* owners, const float* distances,
                           const int* pixels, T* depth, int n, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  int pix = pixels[i];
  if (pix >= 0 && owners[pix] == i % n) depth[pix] = T(distances[i]);
}
__device__ float3 Rotate(const float* m, float3 v) {
  float x = fmaf(m[2], v.z, fmaf(m[1], v.y, m[0] * v.x));
  float y = fmaf(m[6], v.z, fmaf(m[5], v.y, m[4] * v.x));
  float z = fmaf(m[10], v.z, fmaf(m[9], v.y, m[8] * v.x));
  return make_float3(x, y, z);
}
__global__ void MakeGeometry(const float* matrices, const int* enabled, int* rank, int* order,
                             int n, Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  order[i] = i;
  int x = i % p.fw, y = (i / p.fw) % p.fh, d = (i / (p.fw * p.fh)) % p.d,
      cam = i / (p.fw * p.fh * p.d);
  if (!enabled[cam]) { rank[i] = INT_MAX; return; }
  float3 q =
      make_float3(float(p.iw - 1) / float(p.fw - 1) * x,
                  float(p.ih - 1) / float(p.fh - 1) * y, p.dmin + d * p.dstep);
  const float* a = matrices + 16 + cam * 48;
  // Preserve the original operation association: undo post translation,
  // inverse post rotation, then (camera_rotation @ inverse_intrinsics) @ p.
  q.x -= a[3];
  q.y -= a[7];
  q.z -= a[11];
  q = Rotate(a, q);
  q.x *= q.z;
  q.y *= q.z;
  float combined[12] = {};
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col) {
      float value = 0;
      for (int j = 0; j < 3; ++j)
        value = fmaf(a[32 + row * 4 + j], a[16 + j * 4 + col], value);
      combined[row * 4 + col] = value;
    }
  q = Rotate(combined, q);
  q.x = __fadd_rn(q.x, a[35]);
  q.y = __fadd_rn(q.y, a[39]);
  q.z = __fadd_rn(q.z, a[43]);
  q = Rotate(matrices, q);
  q.x = __fadd_rn(q.x, matrices[3]);
  q.y = __fadd_rn(q.y, matrices[7]);
  q.z = __fadd_rn(q.z, matrices[11]);
  // PyTorch .long() truncates toward zero here, unlike voxel floor.
  int gx = int((q.x - p.bound[0]) / p.step[0]),
      gy = int((q.y - p.bound[1]) / p.step[1]),
      gz = int((q.z - p.bound[2]) / p.step[2]);
  rank[i] =
      (gx >= 0 && gx < p.gx && gy >= 0 && gy < p.gy && gz >= 0 && gz < p.gz)
          ? (gx * p.gy + gy) * p.gz + gz
          : INT_MAX;
}
__global__ void Intervals(const int* rank, int* starts, int* ends, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || rank[i] == INT_MAX) return;
  if (i == 0 || rank[i] != rank[i - 1]) starts[rank[i]] = i;
  if (i == n - 1 || rank[i] != rank[i + 1]) ends[rank[i]] = i + 1;
}
template <class In, class Out>
__global__ void PoolKernel(const int* order, const int* starts, const int* ends,
                           const In* feature, const In* depth, Out* out,
                           Params p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, cell = i / p.c, c = i % p.c;
  if (cell >= p.gx * p.gy * p.gz) return;
  float sum = 0;
  int start = starts[cell];
  if (start >= 0)
    for (int j = start; j < ends[cell]; ++j) {
      int q = order[j], hw = q % (p.fh * p.fw), d = (q / (p.fh * p.fw)) % p.d,
          cam = q / (p.fh * p.fw * p.d);
      sum += float(feature[(cam * p.fh * p.fw + hw) * p.c + c]) *
             float(depth[(cam * p.d + d) * p.fh * p.fw + hw]);
    }
  int z = cell % p.gz, xy = cell / p.gz;
  out[(c * p.gz + z) * p.gx * p.gy + xy] = Out(sum);
}
template <class In, class Out>
__global__ void Convert(const In* in, Out* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = Out(in[i]);
}

// Adapted from NVIDIA CUDA-BEVFusion camera-bevpool.cu (MIT).
// The original copyright and license are retained in NVIDIA-BEVPOOL-LICENSE.
struct __align__(4) HalfPack10 { __half values[10]; };
__global__ void PoolPack10(const int* order, const int* starts, const int* ends,
                          const __half* feature, const __half* depth,
                          __half* output, Params p) {
  int cell = blockIdx.x * blockDim.y + threadIdx.y;
  int channel = threadIdx.x * 10;
  if (cell >= p.gx * p.gy * p.gz) return;
  float sum[10] = {};
  int begin = starts[cell];
  if (begin >= 0) {
    for (int j = begin; j < ends[cell]; ++j) {
      int index = order[j], area = p.fh * p.fw;
      int cam = index / (p.d * area), pixel = index % area;
      float weight = __half2float(depth[index]);
      HalfPack10 values = *reinterpret_cast<const HalfPack10*>(feature + (cam * area + pixel) * p.c + channel);
#pragma unroll
      for (int c = 0; c < 10; ++c)
        sum[c] = fmaf(__half2float(values.values[c]), weight, sum[c]);
    }
  }
  int z = cell % p.gz, xy = cell / p.gz;
#pragma unroll
  for (int c = 0; c < 10; ++c)
    output[((channel + c) * p.gz + z) * p.gx * p.gy + xy] = __float2half(sum[c]);
}

template<class T>
__global__ void FillImage(T* output, int area, float a, float b, float c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < area * 3) output[i] = T(i < area ? a : i < area * 2 ? b : c);
}
// Pillow Resample.c uses 22-bit normalized coefficients and rounds to uint8
// after each separable pass. Host coefficients use double, as Pillow does.
struct Filter {
  std::vector<int> weights, bounds;
  int width;
};
double Cubic(double x) {
  x = std::abs(x);
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return (((-.5 * x + 2.5) * x - 4) * x) + 2;
  return 0;
}
Filter Coeff(int in, int out) {
  double scale = double(in) / out, support = 2 * std::max(1., scale);
  Filter f;
  f.width = int(std::ceil(support)) * 2 + 1;
  f.weights.resize(out * f.width);
  f.bounds.resize(out * 2);
  for (int i = 0; i < out; ++i) {
    double center = (i + .5) * scale;
    int first = std::max(0, int(center - support + .5)),
        last = std::min(in, int(center + support + .5));
    double total = 0;
    std::vector<double> w(last - first);
    for (int j = first; j < last; ++j)
      total += (w[j - first] = Cubic((j - center + .5) / std::max(1., scale)));
    f.bounds[2 * i] = first;
    f.bounds[2 * i + 1] = last - first;
    for (int j = 0; j < last - first; ++j) {
      double v = w[j] / total * (1 << 22);
      f.weights[i * f.width + j] = int(v < 0 ? v - .5 : v + .5);
    }
  }
  return f;
}
__global__ void Horizontal(const uint8_t* in, uint8_t* out, const int* bounds,
                           const int* weights, int kw, int iw, int ow, int h,
                           int left, int top) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ow * h * 3) return;
  int c = i % 3, x = i / 3 % ow, y = i / (ow * 3), sum = 1 << 21;
  int first = bounds[x * 2], n = bounds[x * 2 + 1];
  for (int j = 0; j < n; ++j)
    sum += in[((y + top) * iw + left + first + j) * 3 + c] * weights[x * kw + j];
  out[i] = min(255, max(0, sum >> 22));
}
template <class T>
__global__ void Vertical(const uint8_t* in, T* out, const int* bounds,
                         const int* weights, int kw, int rw, int rh, int ow,
                         int oh, int left, int top, float m0, float m1,
                         float m2, float s0, float s1, float s2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * ow * oh) return;
  int c = i / (ow * oh), x = i % ow + left, y = (i / ow) % oh + top;
  int value = 0;
  if (x >= 0 && x < rw && y >= 0 && y < rh) {
    int sum = 1 << 21;
    int first = bounds[y * 2], n = bounds[y * 2 + 1];
    for (int j = 0; j < n; ++j)
      sum += in[((first + j) * rw + x) * 3 + c] * weights[y * kw + j];
    value = min(255, max(0, sum >> 22));
  }
  out[i] = T((value - (c == 0   ? m0
                     : c == 1 ? m1
                              : m2)) /
           (c == 0   ? s0
            : c == 1 ? s1
                     : s2));
}
}  // namespace
void HalfToFloat(const void* in, float* out, size_t n, cudaStream_t stream) {
  Convert<<<(n + 255) / 256, 256, 0, stream>>>(static_cast<const __half*>(in),
                                               out, n);
  Check(cudaGetLastError());
}
// Convert only at an external FP32 input boundary; dense stages share FP16 buffers.
void FloatToHalf(const float* in, void* out, size_t n, cudaStream_t stream) {
  Convert<<<(n + 255) / 256, 256, 0, stream>>>(in, static_cast<__half*>(out), n);
  Check(cudaGetLastError());
}
size_t FilterPoints(Kernels& k, const BevFusionConfig& cfg,
                    const std::vector<float>& points, cudaStream_t stream) {
  Params p = P(cfg);
  int source = points.size() / p.nf;
  k.source_points.resize(points.size() * 4);
  k.valid_points.resize(size_t(source) * 4);
  Check(cudaMemcpyAsync(k.source_points.data, points.data(), points.size() * 4,
                        cudaMemcpyHostToDevice, stream));
  auto begin = thrust::make_counting_iterator(0);
  auto ids = k.valid_points.ptr<int>();
  int valid =
      thrust::copy_if(thrust::cuda::par.on(stream), begin, begin + source, ids,
                      PointPredicate{k.source_points.ptr<float>(), p}) -
      ids;
  int n = std::min(valid, cfg.max_points);
  if (!n)
    throw std::runtime_error(
        "No points inside the model's strict point-cloud range");
  k.points.resize(size_t(n) * p.nf * 4);
  GatherPoints<<<(n * p.nf + 255) / 256, 256, 0, stream>>>(
      k.source_points.ptr<float>(), ids, k.points.ptr<float>(), n, p.nf);
  Check(cudaGetLastError());
  return valid;
}
void Voxelize(Kernels& k, const BevFusionConfig& cfg, int n,
              cudaStream_t stream) {
  Params p = P(cfg);
  k.keys.resize(size_t(n) * 8);
  k.starts.resize(size_t(n) * 4);
  k.first.resize(size_t(n) * 4);
  auto keys = k.keys.ptr<uint64_t>();
  auto starts = k.starts.ptr<int>();
  auto first = k.first.ptr<int>();
  Keys<<<(n + 255) / 256, 256, 0, stream>>>(k.points.ptr<float>(), keys, n, p);
  auto exec = thrust::cuda::par.on(stream);
  thrust::sort(exec, keys, keys + n);
  auto iter = thrust::make_counting_iterator(0);
  auto end =
      thrust::copy_if(exec, iter, iter + n, starts, StartPredicate{keys, p.np});
  int groups = end - starts;
  k.total_voxel_count = groups;
  k.voxel_count = std::min(groups, p.nv);
  if (!k.voxel_count)
    throw std::runtime_error(
        "No points inside the model's strict point-cloud range");
  First<<<(groups + 255) / 256, 256, 0, stream>>>(keys, starts, first, groups,
                                                  p.np);
  thrust::sort_by_key(exec, first, first + groups, starts);
  k.features.resize(size_t(k.voxel_count) * p.nf * 2);
  k.indices.resize(size_t(k.voxel_count) * 16);
  Means<<<(k.voxel_count + 127) / 128, 128, 0, stream>>>(
      k.points.ptr<float>(), keys, starts, k.features.ptr<__half>(),
      k.indices.ptr<int>(), k.voxel_count, n, p);
  Check(cudaGetLastError());
}
size_t Depth(Kernels& k, const BevFusionConfig& cfg,
             const BevFusionFrame& frame, int n, cudaStream_t stream, ScalarType type) {
  Params p = P(cfg);
  std::vector<int> enabled;
  for (const auto& camera : frame.cameras) enabled.push_back(camera.enabled);
  k.camera_mask.resize(enabled.size() * sizeof(int));
  Check(cudaMemcpyAsync(k.camera_mask.data, enabled.data(), enabled.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
  std::vector<float> m;
  auto inv = Inverse(frame.lidar_aug_matrix);
  m.insert(m.end(), inv.begin(), inv.end());
  for (auto& camera : frame.cameras) {
    const auto& projection = ActiveMatrix(camera, camera.lidar2img);
    const auto& augmentation = ActiveMatrix(camera, camera.img_aug_matrix);
    m.insert(m.end(), projection.begin(), projection.end());
    m.insert(m.end(), augmentation.begin(), augmentation.end());
  }
  k.geometry.resize(m.size() * 4);
  Check(cudaMemcpyAsync(k.geometry.data, m.data(), m.size() * 4,
                        cudaMemcpyHostToDevice, stream));
  size_t size = size_t(p.nc) * p.ih * p.iw;
  k.depth.resize(size * ScalarBytes(type));
  k.owners.resize(size * 4);
  k.projection.resize(size_t(n) * p.nc * 4);
  k.pixels.resize(size_t(n) * p.nc * 4);
  k.collision_count.resize(8);
  Check(cudaMemsetAsync(k.depth.data, 0, size * ScalarBytes(type), stream));
  Check(cudaMemsetAsync(k.owners.data, 255, size * 4, stream));
  Check(cudaMemsetAsync(k.collision_count.data, 0, 8, stream));
  Project<<<(n * p.nc + 255) / 256, 256, 0, stream>>>(
      k.points.ptr<float>(), k.geometry.ptr<float>(), k.camera_mask.ptr<int>(), k.projection.ptr<float>(),
      k.pixels.ptr<int>(), k.owners.ptr<int>(),
      k.collision_count.ptr<unsigned long long>(), n, p);
  if (type == ScalarType::Float16)
    WriteDepth<<<(n * p.nc + 255) / 256, 256, 0, stream>>>(
        k.owners.ptr<int>(), k.projection.ptr<float>(), k.pixels.ptr<int>(),
        k.depth.ptr<__half>(), n, n * p.nc);
  else
    WriteDepth<<<(n * p.nc + 255) / 256, 256, 0, stream>>>(
        k.owners.ptr<int>(), k.projection.ptr<float>(), k.pixels.ptr<int>(),
        k.depth.ptr<float>(), n, n * p.nc);
  Check(cudaGetLastError());
  unsigned long long collisions = 0;
  Check(cudaMemcpyAsync(&collisions, k.collision_count.data, 8,
                        cudaMemcpyDeviceToHost, stream));
  Check(cudaStreamSynchronize(stream));
  return collisions;
}
void Geometry(Kernels& k, const BevFusionConfig& cfg,
              const BevFusionFrame& frame, cudaStream_t stream) {
  Params p = P(cfg);
  std::vector<int> enabled;
  for (const auto& camera : frame.cameras) enabled.push_back(camera.enabled);
  k.camera_mask.resize(enabled.size() * sizeof(int));
  Check(cudaMemcpyAsync(k.camera_mask.data, enabled.data(), enabled.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
  std::vector<float> m(frame.lidar_aug_matrix.begin(),
                       frame.lidar_aug_matrix.end());
  for (auto& camera : frame.cameras) {
    const auto& augmentation = ActiveMatrix(camera, camera.img_aug_matrix);
    auto inv = Inverse(augmentation);
    inv[3] = augmentation[3];
    inv[7] = augmentation[7];
    inv[11] = augmentation[11];
    m.insert(m.end(), inv.begin(), inv.end());
    inv = Inverse(ActiveMatrix(camera, camera.cam2img));
    inv[3] = inv[7] = inv[11] = 0;
    m.insert(m.end(), inv.begin(), inv.end());
    const auto& extrinsic = ActiveMatrix(camera, camera.cam2lidar);
    m.insert(m.end(), extrinsic.begin(), extrinsic.end());
  }
  k.geometry.resize(m.size() * 4);
  Check(cudaMemcpyAsync(k.geometry.data, m.data(), m.size() * 4,
                        cudaMemcpyHostToDevice, stream));
  int n = p.nc * p.d * p.fh * p.fw, cells = p.gx * p.gy * p.gz;
  k.ranks.resize(size_t(n) * 4);
  k.geometry_order.resize(size_t(n) * 4);
  k.pool_starts.resize(size_t(cells) * 4);
  k.pool_ends.resize(size_t(cells) * 4);
  Check(cudaMemsetAsync(k.pool_starts.data, 255, size_t(cells) * 4, stream));
  Check(cudaMemsetAsync(k.pool_ends.data, 255, size_t(cells) * 4, stream));
  MakeGeometry<<<(n + 255) / 256, 256, 0, stream>>>(
      k.geometry.ptr<float>(), k.camera_mask.ptr<int>(), k.ranks.ptr<int>(), k.geometry_order.ptr<int>(),
      n, p);
  thrust::stable_sort_by_key(thrust::cuda::par.on(stream), k.ranks.ptr<int>(),
                             k.ranks.ptr<int>() + n,
                             k.geometry_order.ptr<int>());
  Intervals<<<(n + 255) / 256, 256, 0, stream>>>(
      k.ranks.ptr<int>(), k.pool_starts.ptr<int>(), k.pool_ends.ptr<int>(), n);
  Check(cudaGetLastError());
  Check(cudaStreamSynchronize(stream));
}
/// Accumulate FP16 or FP32 camera tensors in FP32 and round only the output.
template <class In, class Out>
void PoolTyped(Kernels& k, Params p, const void* feature, const void* depth,
               cudaStream_t stream) {
  int n = p.gx * p.gy * p.gz * p.c;
  k.pooled.resize(size_t(n) * sizeof(Out));
  PoolKernel<<<(n + 127) / 128, 128, 0, stream>>>(
      k.geometry_order.ptr<int>(), k.pool_starts.ptr<int>(),
      k.pool_ends.ptr<int>(), static_cast<const In*>(feature),
      static_cast<const In*>(depth), k.pooled.ptr<Out>(), p);
  Check(cudaGetLastError());
}
void Pool(Kernels& k, const BevFusionConfig& cfg, const void* feature,
          const void* depth, cudaStream_t stream, ScalarType input_type,
          ScalarType output_type) {
  Params p = P(cfg);
  if (cfg.bev_pool_backend == "nvidia_pack10") {
    if (p.c != 80 || input_type != ScalarType::Float16 || output_type != ScalarType::Float16)
      throw std::runtime_error("nvidia_pack10 requires 80 channels and FP16 pooling I/O");
    int cells = p.gx * p.gy * p.gz;
    k.pooled.resize(size_t(cells) * p.c * sizeof(__half));
    PoolPack10<<<(cells + 15) / 16, dim3(8, 16), 0, stream>>>(
        k.geometry_order.ptr<int>(), k.pool_starts.ptr<int>(), k.pool_ends.ptr<int>(),
        static_cast<const __half*>(feature), static_cast<const __half*>(depth), k.pooled.ptr<__half>(), p);
    Check(cudaGetLastError());
    return;
  }
  if (input_type == ScalarType::Float16) {
    if (output_type == ScalarType::Float16)
      PoolTyped<__half, __half>(k, p, feature, depth, stream);
    else
      PoolTyped<__half, float>(k, p, feature, depth, stream);
  } else if (output_type == ScalarType::Float16) {
    PoolTyped<float, __half>(k, p, feature, depth, stream);
  } else {
    PoolTyped<float, float>(k, p, feature, depth, stream);
  }
}
/// Preserve Pillow resize/crop semantics and cast the final normalized pixels.
template <class T>
void PreprocessTyped(Kernels& k, const BevFusionConfig& cfg,
                const BevFusionFrame& frame, T* output,
                cudaStream_t stream) {
  int ow = cfg.image_size[1], oh = cfg.image_size[0];
  for (size_t i = 0; i < frame.cameras.size(); ++i) {
    const auto& c = frame.cameras[i];
    if (!c.enabled) {
      int area = ow * oh;
      FillImage<<<(area * 3 + 255) / 256, 256, 0, stream>>>(output + i * area * 3, area,
          -cfg.mean[0] / cfg.std[0], -cfg.mean[1] / cfg.std[1], -cfg.mean[2] / cfg.std[2]);
      Check(cudaGetLastError());
      continue;
    }
    int left = 0, top = 0, source_width = c.width, source_height = c.height;
    if (c.source_crop[2] || c.source_crop[3]) {
      left = c.source_crop[0]; top = c.source_crop[1];
      source_width = c.source_crop[2] - left;
      source_height = c.source_crop[3] - top;
      if (left < 0 || top < 0 || source_width <= 0 || source_height <= 0 ||
          left + source_width > c.width || top + source_height > c.height)
        throw std::runtime_error("Invalid source ROI");
    }
    auto h = Coeff(source_width, c.resize_width),
         v = Coeff(source_height, c.resize_height);
    k.rgb.resize(c.rgb.size());
    k.horizontal.resize(size_t(c.resize_width) * source_height * 3);
    k.weights.resize(std::max(h.weights.size(), v.weights.size()) * 4);
    k.bounds.resize(std::max(h.bounds.size(), v.bounds.size()) * 4);
    Check(cudaMemcpyAsync(k.rgb.data, c.rgb.data(), c.rgb.size(),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaMemcpyAsync(k.weights.data, h.weights.data(),
                          h.weights.size() * 4, cudaMemcpyHostToDevice,
                          stream));
    Check(cudaMemcpyAsync(k.bounds.data, h.bounds.data(), h.bounds.size() * 4,
                          cudaMemcpyHostToDevice, stream));
    int n = c.resize_width * source_height * 3;
    Horizontal<<<(n + 255) / 256, 256, 0, stream>>>(
        k.rgb.ptr<uint8_t>(), k.horizontal.ptr<uint8_t>(), k.bounds.ptr<int>(),
        k.weights.ptr<int>(), h.width, c.width, c.resize_width, source_height, left, top);
    Check(cudaMemcpyAsync(k.weights.data, v.weights.data(),
                          v.weights.size() * 4, cudaMemcpyHostToDevice,
                          stream));
    Check(cudaMemcpyAsync(k.bounds.data, v.bounds.data(), v.bounds.size() * 4,
                          cudaMemcpyHostToDevice, stream));
    n = ow * oh * 3;
    Vertical<<<(n + 255) / 256, 256, 0, stream>>>(
        k.horizontal.ptr<uint8_t>(), output + i * n, k.bounds.ptr<int>(),
        k.weights.ptr<int>(), v.width, c.resize_width, c.resize_height, ow, oh,
        c.crop_left, c.crop_top, cfg.mean[0], cfg.mean[1], cfg.mean[2],
        cfg.std[0], cfg.std[1], cfg.std[2]);
    Check(cudaGetLastError());
    Check(cudaStreamSynchronize(stream));  // host coefficient lifetime
  }
}
void Preprocess(Kernels& k, const BevFusionConfig& cfg,
                const BevFusionFrame& frame, void* output, cudaStream_t stream,
                ScalarType type) {
  if (type == ScalarType::Float16)
    PreprocessTyped(k, cfg, frame, static_cast<__half*>(output), stream);
  else
    PreprocessTyped(k, cfg, frame, static_cast<float*>(output), stream);
}

void MaskCameraOutputs(const BevFusionConfig& cfg, const BevFusionFrame& frame,
                       void* feature, void* depth, ScalarType type, cudaStream_t stream) {
  size_t area = size_t(cfg.feature_size[0]) * cfg.feature_size[1], bytes = ScalarBytes(type);
  for (size_t i = 0; i < frame.cameras.size(); ++i) {
    if (frame.cameras[i].enabled) continue;
    Check(cudaMemsetAsync(static_cast<char*>(feature) + i * area * cfg.feature_channels * bytes,
                         0, area * cfg.feature_channels * bytes, stream));
    Check(cudaMemsetAsync(static_cast<char*>(depth) + i * area * cfg.depth_bins * bytes,
                         0, area * cfg.depth_bins * bytes, stream));
  }
}
}  // namespace inference::bev
