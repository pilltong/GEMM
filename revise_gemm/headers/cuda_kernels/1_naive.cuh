#pragma once

#include <cstdio>
#include "cuda_runtime.h"

__global__ void naive(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint row = blockDim.x * blockIdx.x + threadIdx.x;
    const uint col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < m && col < n) {
        float tmp = 0.0f;
        for(int i = 0; i < k; i++)
            tmp += A[row * k + i] * B[i * n + col];
        C[row * n + col] = alpha * tmp + beta * C[row * n + col];
    }
}