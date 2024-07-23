#pragma once

#include <cstdio>
#include "cuda_runtime.h"

template <const uint tile_size>
__global__ void global_coalesce(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    const int row = blockIdx.x * tile_size + (threadIdx.x / tile_size);
    const int col = blockIdx.y * tile_size + (threadIdx.x % tile_size);
    if(row < m && col < n) {
        float tmp = 0.0f;
        for(int i = 0; i < k; i++)
            tmp += A[row * k + i] * B[i * n + col];
        C[row * n + col] = alpha * tmp + beta * C[row * n + col];
    }
}