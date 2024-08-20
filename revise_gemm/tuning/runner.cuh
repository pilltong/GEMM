#include "../headers/helpers.h"
#include "../headers/cuda_kernels.cuh"

// 각 커널을 실행하는 함수 선언
void run_naive_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    naive_fp<<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_coalesce_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    global_coalesce_fp<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_shared_caching_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    shared_caching_fp<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_1d_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 64;
    const uint bn = 64;
    const uint bk = 8;
    const uint tw = 8;
    dim3 blockDim((bm / tw) * bn);
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_1d_fp<bm, bn, bk, tw> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_2d_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_2d_fp<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_fp(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint k6_bm = 256;
    const uint k6_bn = 64;
    const uint k6_bk = 16;
    const uint k6_tw_m = 16;
    const uint k6_tw_n = 4;
    dim3 blockDim((k6_bm / k6_tw_m) * (k6_bn / k6_tw_n));
    dim3 gridDim(ceil_div(n, k6_bn), ceil_div(m, k6_bm));
    vectorized_fp<k6_bm, k6_bn, k6_bk, k6_tw_m, k6_tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_fp_revised(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint k7_bm = 128;
    const uint k7_bn = 128;
    const uint k7_bk = 8;
    const uint k7_tw_m = 8;
    const uint k7_tw_n = 8;
    dim3 blockDim((k7_bm / k7_tw_m) * (k7_bn / k7_tw_n));
    dim3 gridDim(ceil_div(n, k7_bn), ceil_div(m, k7_bm));
    vectorized_fp_revised<k7_bm, k7_bn, k7_bk, k7_tw_m, k7_tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_resolve_bank_conflict(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint k8_bm = 256;
    const uint k8_bn = 256;
    const uint k8_bk = 64;
    const uint k8_tw_m = 16;
    const uint k8_tw_n = 16;
    dim3 blockDim((k8_bm / k8_tw_m) * (k8_bn / k8_tw_n));
    dim3 gridDim(ceil_div(n, k8_bn), ceil_div(m, k8_bm));
    resolve_bank_conflict<k8_bm, k8_bn, k8_bk, k8_tw_m, k8_tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void runCublasFP32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n, 
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}