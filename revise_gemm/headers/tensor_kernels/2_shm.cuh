#pragma once

#include <cstdio>
#include "mma.h"

using namespace nvcuda;

__global__ void shm_tf(const float *A, const float *B, float *C, int m, int n, int k, float alpha, float beta) {
    // Define WMMA fragment types for TF32 precision
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    // Initialize output fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // shared memory
    __shared__ float A_shared[16 * 8];
    __shared__ float B_shared[8 * 16];
    int A_idy = threadIdx.x / 8;
    int A_idx = threadIdx.x % 8;
    int B_idy = threadIdx.x / 16;
    int B_idx = threadIdx.x % 16;

    // warp index among total threads 
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    
    // index of each matrices' tile
    const int Crow = warp_id / (n / 16);
    const int Ccol = warp_id % (n / 16);
    int A_pos = Crow * (16 * k);
    int B_pos = Ccol * 16;
    
    //reinterpret_cast<float4 *>(&A_shared[A_trow * bk + A_tcol * 4])[0] = reinterpret_cast<float4 *>(&A[A_trow * k + A_tcol * 4])[0];
    //reinterpret_cast<float4 *>(&B_shared[B_trow * bn + B_tcol * 4])[0] = reinterpret_cast<float4 *>(&B[B_trow * n + B_tcol * 4])[0];

    for(int i = 0; i < k; i += 8) {
        A_shared[A_idy * 8 + A_idx] = A[A_pos + A_idy * k + A_idx];
        B_shared[B_idy * 16 + B_idx] = B[B_pos + B_idy * n + B_idx];
        __syncthreads();
        // Load the input matrices into fragments
        wmma::load_matrix_sync(a_frag, A_shared, 8);
        wmma::load_matrix_sync(b_frag, B_shared, 16);
        // Perform the matrix multiplication using Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    for(int i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] = alpha * c_frag.x[i] + beta * C[Crow * n + Ccol + i];

    // Store the result back to the output matrix
    wmma::store_matrix_sync(C + (Crow * n + Ccol) * 16, c_frag, n, wmma::mem_row_major);
}