#ifndef IOU3D_NMS_IMPL_H
#define IOU3D_NMS_IMPL_H

// 系统头文件：提供基本数据类型和指针定义
#include <cstdint>
#include <cstdlib>

// 前置声明宏：控制CUDA相关功能的编译
// 若需启用CUDA，需在编译时定义该宏（如 -DHAVE_CUDA_ENABLE）
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU版本计算BEV视角下的3D框IoU（鸟瞰图）
 * @param boxes_a_tensor 输入A框数据，形状为(N,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param boxes_b_tensor 输入B框数据，形状为(M,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param ans_iou_tensor 输出IoU结果，形状为(N,M)，存储A、B框两两之间的IoU值
 * @param num_a A框的数量（N）
 * @param num_b B框的数量（M）
 * @return int 执行状态，1表示成功
 */
int boxes_iou_bev_cpu(const float *boxes_a_tensor, 
                      const float *boxes_b_tensor, 
                      float *ans_iou_tensor, 
                      const int num_a, 
                      const int num_b);

/**
 * @brief GPU版本的3D NMS（非极大值抑制）实现
 * @note 编译时定义HAVE_CUDA_ENABLE则启用CUDA加速，否则退化为CPU模拟实现
 * @param boxes 输入检测框数据，形状为(N,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param keep 输出保留的框索引，数组长度需至少为N
 * @param nms_overlap_thresh NMS重叠度阈值（IoU阈值）
 * @param num_boxes 输入检测框的数量（N）【注：原代码中CUDA/非CUDA版本参数顺序有差异，已统一修正】
 * @return int 保留的框的数量
 */
int nms_gpu(const float *boxes, 
            long *keep, 
            float nms_overlap_thresh, 
            const int num_boxes);

/**
 * @brief IoU BEV NMS where boxes[0] uses a dedicated threshold when suppressing later boxes.
 */
int nms_gpu_with_self_car(const float *boxes,
                          long *keep,
                          float nms_overlap_thresh,
                          float self_car_overlap_thresh,
                          const int num_boxes);

/**
 * @brief GPU版本的BEV NMS，使用 overlap / min(area_a, area_b) 作为抑制度量
 * @note 编译时定义HAVE_CUDA_ENABLE则启用CUDA加速，否则退化为CPU模拟实现
 * @param boxes 输入检测框数据，形状为(N,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param keep 输出保留的框索引，数组长度需至少为N
 * @param nms_overlap_thresh NMS重叠度阈值
 * @param num_boxes 输入检测框的数量（N）
 * @return int 保留的框的数量
 */
int nms_normalized_gpu(const float *boxes,
                       long *keep,
                       float nms_overlap_thresh,
                       const int num_boxes);

/**
 * @brief IoU-min BEV NMS where boxes[0] uses a dedicated threshold.
 */
int nms_normalized_gpu_with_self_car(const float *boxes,
                                     long *keep,
                                     float nms_overlap_thresh,
                                     float self_car_overlap_thresh,
                                     const int num_boxes);

/**
 * @brief GPU版本的Circle NMS，按输入顺序贪心保留中心距离足够远的框
 * @note 编译时定义HAVE_CUDA_ENABLE则启用CUDA加速，否则退化为CPU模拟实现
 * @param boxes 输入检测框数据，形状为(N,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param keep 输出保留的框索引，数组长度需至少为N
 * @param distance_threshold 中心点平方距离阈值，与CenterPoint circle_nms保持一致
 * @param num_boxes 输入检测框的数量（N）
 * @return int 保留的框的数量
 */
int circle_nms_gpu(const float *boxes,
                   long *keep,
                   float distance_threshold,
                   const int num_boxes);

/**
 * @brief GPU版本计算BEV视角下的3D框重叠面积
 * @note 编译时定义HAVE_CUDA_ENABLE则启用CUDA加速，否则退化为CPU模拟实现
 * @param boxes_a 输入A框数据，形状为(N,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param boxes_b 输入B框数据，形状为(M,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param ans_overlap 输出重叠面积结果，形状为(N,M)
 * @param num_boxes_a A框的数量（N）
 * @param num_boxes_b B框的数量（M）
 * @return int 执行状态，1表示成功
 */
int boxes_overlap_bev_gpu(const float *boxes_a, 
                          const float *boxes_b, 
                          float *ans_overlap, 
                          const int num_boxes_a, 
                          const int num_boxes_b);

/**
 * @brief GPU版本计算BEV视角下的3D框并集面积
 * @note 编译时定义HAVE_CUDA_ENABLE则启用CUDA加速，否则退化为CPU模拟实现
 * @param boxes_a 输入A框数据，形状为(N,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param boxes_b 输入B框数据，形状为(M,7)，格式[x, y, z, dx, dy, dz, heading]
 * @param ans_union 输出并集面积结果，形状为(N,M)
 * @param num_boxes_a A框的数量（N）
 * @param num_boxes_b B框的数量（M）
 * @return int 执行状态，1表示成功
 */
int boxes_union_bev_gpu(const float *boxes_a, 
                        const float *boxes_b, 
                        float *ans_union, 
                        const int num_boxes_a, 
                        const int num_boxes_b);

#ifdef __cplusplus
}
#endif

#endif // IOU3D_NMS_IMPL_H
