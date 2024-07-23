#pragma once

#include <cstdio>
#include "cuda_runtime.h"

template <const uint tile_size>
__global__ void shared_caching(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ float A_shared[tile_size * tile_size];
    __shared__ float B_shared[tile_size * tile_size];

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / tile_size;
    const int tcol = threadIdx.x % tile_size;

    // staring pointer of each matrcies
    A += brow * tile_size * k; // grid dimension = (m / tile_size) * (n / tile_size) => to recover the original shape, multiply tile_size
    B += bcol * tile_size;
    C += brow * tile_size * n + bcol * tile_size;

    float tmp = 0.0f;
    for(int i = 0; i < k; i += tile_size) {
        // load the data into shared memory
        A_shared[trow * tile_size + tcol] = A[trow * k + tcol];
        B_shared[trow * tile_size + tcol] = B[trow * n + tcol];
        // wait unit the whole threads in single thread block finish the data loading
        __syncthreads(); // unit of shared memory = thread block

        for(int j = 0; j < tile_size; j++)
            tmp += A_shared[trow * tile_size + j] * B_shared[j * tile_size + tcol];
        __syncthreads();

        A += tile_size;
        B += tile_size * n;
    }
    C[trow * n + tcol] = alpha * tmp + beta * C[trow * n + tcol];
}