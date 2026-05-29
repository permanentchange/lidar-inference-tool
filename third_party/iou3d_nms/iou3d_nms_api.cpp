#include <vector>
#include <cstring>
#ifdef HAVE_CUDA_ENABLE
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include "iou3d_nms_api.h"
#include "iou3d_cpu.h"
#include "iou3d_nms.h"

#include <iostream>


#ifdef HAVE_CUDA_ENABLE
#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      // fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

void nmsLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);
void circleNmsLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float distance_threshold);
void boxesoverlapLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap);
void boxesunionLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_union);
#endif

int boxes_iou_bev_cpu(const float *boxes_a_tensor, const float *boxes_b_tensor, float *ans_iou_tensor, const int num_a, const int num_b){
    // params boxes_a_tensor: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b_tensor: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_iou_tensor: (N, M)

    int num_boxes_a = num_a;
    int num_boxes_b = num_b;
    const float *boxes_a = boxes_a_tensor;
    const float *boxes_b = boxes_b_tensor;
    float *ans_iou = ans_iou_tensor;

    for (int i = 0; i < num_boxes_a; i++){
        for (int j = 0; j < num_boxes_b; j++){
            ans_iou[i * num_boxes_b + j] = iou_bev(boxes_a + i * 7, boxes_b + j * 7);
        }
    }
    return 1;
}

#ifdef HAVE_CUDA_ENABLE
int nms_gpu(const float *boxes, long *keep, float nms_overlap_thresh, const int num_boxes){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N);

    int boxes_num = num_boxes;

	  float *boxes_data_cuda = NULL;
	  cudaMalloc((void**)&boxes_data_cuda, num_boxes * 7 * sizeof(float));
	  cudaMemcpy(boxes_data_cuda, boxes, num_boxes * 7 * sizeof(float), cudaMemcpyHostToDevice);

    const float * boxes_data = boxes_data_cuda;
    long * keep_data = keep;

    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

//    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);
	  cudaFree(boxes_data_cuda);

    unsigned long long remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }
    if ( cudaSuccess != cudaGetLastError() ) {
        std::cerr<<"Error!";
    }

    return num_to_keep;
}

int circle_nms_gpu(const float *boxes, long *keep, float distance_threshold, const int num_boxes){
    int boxes_num = num_boxes;

    float *boxes_data_cuda = NULL;
    cudaMalloc((void**)&boxes_data_cuda, num_boxes * 7 * sizeof(float));
    cudaMemcpy(boxes_data_cuda, boxes, num_boxes * 7 * sizeof(float), cudaMemcpyHostToDevice);

    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    circleNmsLauncher(boxes_data_cuda, mask_data, boxes_num, distance_threshold);

    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);
    cudaFree(boxes_data_cuda);

    unsigned long long remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }
    if ( cudaSuccess != cudaGetLastError() ) {
        std::cerr<<"Error!";
    }

    return num_to_keep;
}

int boxes_overlap_bev_gpu(const float *boxes_a, const float *boxes_b, float *ans_overlap, const int num_boxes_a, const int num_boxes_b){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)

    float *boxes_a_data_cuda = NULL;
    float *boxes_b_data_cuda = NULL;
	cudaMalloc((void**)&boxes_a_data_cuda, num_boxes_a * 7 * sizeof(float));
	cudaMemcpy(boxes_a_data_cuda, boxes_a, num_boxes_a * 7 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&boxes_b_data_cuda, num_boxes_b * 7 * sizeof(float));
	cudaMemcpy(boxes_b_data_cuda, boxes_b, num_boxes_b * 7 * sizeof(float), cudaMemcpyHostToDevice);

    float *ans_overlap_data_cuda = NULL;
    cudaMalloc((void**)&ans_overlap_data_cuda, num_boxes_a * num_boxes_b * sizeof(float));

    int num_a = num_boxes_a;
    int num_b = num_boxes_b;

    const float * boxes_a_data = boxes_a_data_cuda;
    const float * boxes_b_data = boxes_b_data_cuda;
    float * ans_overlap_data = ans_overlap_data_cuda;

    boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_overlap_data);
    cudaMemcpy(ans_overlap, ans_overlap_data_cuda, num_boxes_a * num_boxes_b * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(boxes_a_data_cuda);
    cudaFree(boxes_b_data_cuda);
    cudaFree(ans_overlap_data_cuda);
    return 1;
}

int boxes_union_bev_gpu(const float *boxes_a, const float *boxes_b, float *ans_union, const int num_boxes_a, const int num_boxes_b){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)

    float *boxes_a_data_cuda = NULL;
    float *boxes_b_data_cuda = NULL;
	cudaMalloc((void**)&boxes_a_data_cuda, num_boxes_a * 7 * sizeof(float));
	cudaMemcpy(boxes_a_data_cuda, boxes_a, num_boxes_a * 7 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&boxes_b_data_cuda, num_boxes_b * 7 * sizeof(float));
	cudaMemcpy(boxes_b_data_cuda, boxes_b, num_boxes_b * 7 * sizeof(float), cudaMemcpyHostToDevice);

    float *ans_union_data_cuda = NULL;
    cudaMalloc((void**)&ans_union_data_cuda, num_boxes_a * num_boxes_b * sizeof(float));

    int num_a = num_boxes_a;
    int num_b = num_boxes_b;

    const float * boxes_a_data = boxes_a_data_cuda;
    const float * boxes_b_data = boxes_b_data_cuda;
    float * ans_union_data = ans_union_data_cuda;

    boxesunionLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_union_data);
    cudaMemcpy(ans_union, ans_union_data_cuda, num_boxes_a * num_boxes_b * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(boxes_a_data_cuda);
    cudaFree(boxes_b_data_cuda);
    cudaFree(ans_union_data_cuda);
    return 1;
}

#else

int nms_gpu(const float *boxes, long *keep, float nms_overlap_thresh, const int num_boxes){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N);

    int boxes_num = num_boxes;
    const float * boxes_data = boxes;
    long * keep_data = keep;

    std::vector<unsigned int> mask_cpu(boxes_num * boxes_num, 0);
    for (int i = 0; i < boxes_num; i++){
        for (int j = 0; j < boxes_num; j++){
            if (iou_bev(boxes_data + i * 7, boxes_data + j * 7) > nms_overlap_thresh) {
                mask_cpu[i * boxes_num + j] = 1;
            }
        }
    }

    unsigned int remv_cpu[boxes_num];
    memset(remv_cpu, 0, boxes_num * sizeof(unsigned int));

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        if (!remv_cpu[i]){
            keep_data[num_to_keep++] = i;
            for (int j = i + 1; j < boxes_num; j++){
                remv_cpu[j] |= mask_cpu[i * boxes_num + j];
            }
        }
    }

    return num_to_keep;
}

int circle_nms_gpu(const float *boxes, long *keep, float distance_threshold, const int num_boxes){
    std::vector<unsigned int> mask_cpu(num_boxes * num_boxes, 0);
    for (int i = 0; i < num_boxes; i++){
        for (int j = i + 1; j < num_boxes; j++){
            const float dx = boxes[i * 7 + 0] - boxes[j * 7 + 0];
            const float dy = boxes[i * 7 + 1] - boxes[j * 7 + 1];
            if (dx * dx + dy * dy <= distance_threshold) {
                mask_cpu[i * num_boxes + j] = 1;
            }
        }
    }

    std::vector<unsigned int> remv_cpu(num_boxes, 0);

    int num_to_keep = 0;

    for (int i = 0; i < num_boxes; i++){
        if (!remv_cpu[i]){
            keep[num_to_keep++] = i;
            for (int j = i + 1; j < num_boxes; j++){
                remv_cpu[j] |= mask_cpu[i * num_boxes + j];
            }
        }
    }

    return num_to_keep;
}

int boxes_overlap_bev_gpu(const float *boxes_a, const float *boxes_b, float *ans_overlap, const int num_boxes_a, const int num_boxes_b){
    const float *boxes_a_data = boxes_a;
    const float *boxes_b_data = boxes_b;
    float *ans_overlap_data = ans_overlap;

    for (int i = 0; i < num_boxes_a; i++){
        for (int j = 0; j < num_boxes_b; j++){
            ans_overlap_data[i * num_boxes_b + j] = overlap_bev(boxes_a_data + i * 7, boxes_b_data + j * 7);
        }
    }
    return 1;
}

int boxes_union_bev_gpu(const float *boxes_a, const float *boxes_b, float *ans_union, const int num_boxes_a, const int num_boxes_b){
    const float *boxes_a_data = boxes_a;
    const float *boxes_b_data = boxes_b;
    float *ans_union_data = ans_union;

    for (int i = 0; i < num_boxes_a; i++){
        for (int j = 0; j < num_boxes_b; j++){
            ans_union_data[i * num_boxes_b + j] = union_bev(boxes_a_data + i * 7, boxes_b_data + j * 7);
        }
    }
    return 1;
}

#endif
