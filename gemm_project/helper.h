#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cmath>

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "mma.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include <nvtx3/nvToolsExt.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = call;                                             \
        if(status_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,  \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

#define ceil_div(a, b) (((a) + (b) - 1) / (b))

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memory Bus Width: %d bits\n\
    max Threads Per Block: %d\n\
    max Threads Per Streaming MultiProcessor: %d\n\
    max Regs Per Block: %d\n\
    max Regs Per Streaming MultiProcessor: %d\n\
    total Global Mem: %zu MB\n\
    shared Mem Per Block: %zu KB\n\
    shared Mem Per Streaming Multiprocessor: %zu KB\n\
    total Const Mem: %zu KB\n\
    multi Processor Count: %d\n\
    Warp Size: %d\n\
    max warps per multiprocessor: %d\n\
    reg allocation unit size: %d\n\
    reg allocation granularity: %s\n\
    CUDA runtime shared mem overhead per block: %d B\n\
    warp allocation granularity: %d\n\n",
           deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
           props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
           props.regsPerBlock, props.regsPerMultiprocessor,
           props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
           props.multiProcessorCount, props.warpSize,
           props.maxThreadsPerMultiProcessor / props.warpSize,
           256, // reg allocation unit size
           "warp", // reg allocation granularity
           1024, // CUDA runtime shared mem overhead per block
           4 // warp allocation granularity
    );
}

void randomize_matrix(float *mat, int N) {
  struct timeval time {};
  
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);

  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %.6f, Is %.6f (Diff %.6f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

__global__ void convert_fp32_to_tf32(const float* input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = nvcuda::wmma::__float_to_tf32(input[idx]);
    }
}

void convert_to_tf32(float *tf32_A, float *tf32_B, float *A, float *B, int m, int n, int k) {
    dim3 blockDim(1024);
    dim3 gridDim(ceil_div(m * k, blockDim.x));
    convert_fp32_to_tf32 <<<gridDim, blockDim>>>(A, tf32_A, m * k);
    gridDim = {(ceil_div(k * n, blockDim.x))};
    convert_fp32_to_tf32 <<<gridDim, blockDim>>>(B, tf32_B, k * n);
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void convert_fp32_to_fp16(const float* input, __half* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

void convert_to_fp16(__half *half_A, __half *half_B, float *A, float *B, int m, int n, int k) {
    dim3 blockDim(1024);
    dim3 gridDim(ceil_div(m * k, blockDim.x));
    convert_fp32_to_fp16 <<<gridDim, blockDim>>>(A, half_A, m * k);
    gridDim = {(ceil_div(k * n, blockDim.x))};
    convert_fp32_to_fp16 <<<gridDim, blockDim>>>(B, half_B, k * n);
    CHECK_CUDA(cudaDeviceSynchronize());
}

