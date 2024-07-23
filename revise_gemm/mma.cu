#include "headers/helpers.h"
#include "headers/cuda_kernels.cuh"
#include "headers/tensor_kernels.cuh"

void run_naive(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    naive<<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_coalesce(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    global_coalesce<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_shared_caching(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    shared_caching<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_1d(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 32;
    const uint bn = 32;
    const uint bk = 8;
    const uint tw = 4;
    dim3 blockDim((bm / tw) * bn);
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_1d<bm, bn, bk, tw> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_blocking_2d(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 64;
    const uint bn = 64;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    blocking_2d<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_global_tf32(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    float *tf32_A, *tf32_B;
    CHECK_CUDA(cudaMalloc((void**)&tf32_A, sizeof(float) * m * k));
    CHECK_CUDA(cudaMalloc((void**)&tf32_B, sizeof(float) * k * n));
    convert_to_tf32(tf32_A, tf32_B, A, B, m, n, k);
    dim3 blockDim(16 * 16);
    dim3 gridDim(ceil_div(n * m, 16 * 16 * 8));
    global_tf32 <<<gridDim, blockDim>>>(tf32_A, tf32_B, C, m, n, k, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(tf32_A));
    CHECK_CUDA(cudaFree(tf32_B));
}

void run_global_fp16(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    __half *half_A, *half_B;
    cudaMalloc((void**)&half_A, sizeof(__half) * m * k);
    cudaMalloc((void**)&half_B, sizeof(__half) * k * n);
    convert_to_fp16(half_A, half_B, A, B, m, n, k);
    dim3 blockDim(16 * 16);
    dim3 gridDim(ceil_div(n * m, 16 * 16 * 8));
    global_fp16 <<<gridDim, blockDim>>>(half_A, half_B, C, m, n, k, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(half_A));
    CHECK_CUDA(cudaFree(half_B));
}

void runCublasTF32_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    float *tf32_A, *tf32_B;
    cudaMalloc((void**)&tf32_A, sizeof(float) * m * k);
    cudaMalloc((void**)&tf32_B, sizeof(float) * k * n);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(tf32_A));
    CHECK_CUDA(cudaFree(tf32_B));
}

void runCublasBF16_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasTF32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    float *tf32_A, *tf32_B;
    CHECK_CUDA(cudaMalloc((void**)&tf32_A, sizeof(float) * m * k));
    CHECK_CUDA(cudaMalloc((void**)&tf32_B, sizeof(float) * k * n));
    convert_to_tf32(tf32_A, tf32_B, A, B, m, n, k);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, tf32_B, CUDA_R_32F,
                n, tf32_A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(tf32_A));
    CHECK_CUDA(cudaFree(tf32_B));
}

void runCublasBF16(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasFP32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    //nvtxRangePushA("outer");
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    //nvtxRangePop();
}

void launch_kernel_with_option(int op, cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    switch(op) {
        case 0 :
            runCublasFP32(handle, A, B, C, m, n, k, alpha, beta);
            break;
        case 1 :
            run_naive(A, B, C, m, n, k, alpha, beta);
            break;
        case 2 :
            run_global_coalesce(A, B, C, m, n, k, alpha, beta);
            break;
        case 3 :
            run_shared_caching(A, B, C, m, n, k, alpha, beta);
            break;
        case 4 :
            run_blocking_1d(A, B, C, m, n, k, alpha, beta);
            break;
        case 5 :
            run_blocking_2d(A, B, C, m, n, k, alpha, beta);
            break;
        case 6 :
            run_global_tf32(A, B, C, m, n, k, alpha, beta);
            break;
        case 7 :
            run_global_fp16(A, B, C, m, n, k, alpha, beta);
            break;
    }
}

typedef struct _execution_result {
    float gflops;
    float time;
} result;

int main(int argc, char **argv) {
    // print the device information
    CudaDeviceInfo();

    // for checking the execution time
    float elapsed_time;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // create the cuBLAS handle
    cublasHandle_t handle;
    if(cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    // constant value for GEMM
    float alpha = 1.0, beta = 0.0;

    // host memory pointer
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    float *C_ref = nullptr; // for cuBLAS

    // device memory pointer
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    float *d_C_ref = nullptr; // for cuBLAS

    // define the various matrix size
    std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};
    long max_size;
    max_size = SIZE[SIZE.size() - 1];

    // allocate the memory as much as max_size
    // for small size matrices, only use part of whole memoryng problem
    // of not having proper bounds checking in the
    // host memory
    A = (float*)malloc(sizeof(float) * max_size * max_size);
    B = (float*)malloc(sizeof(float) * max_size * max_size);
    C = (float*)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float*)malloc(sizeof(float) * max_size * max_size);

    // devide memory
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * max_size * max_size));
    CHECK_CUDA(cudaMalloc((void**)&d_C_ref, sizeof(float) * max_size * max_size));

    // initialize the matrices
    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);

    // copy the host memory to device memory
    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    // number of kernels
    int op_num = 5;

    // for storing the gflops and elapsed_time
    result **exe_results = (result**)malloc(sizeof(result*) * (op_num + 1));
    for(int i = 0; i <= op_num; i++)
        exe_results[i] = (result*)malloc(sizeof(result) * SIZE.size());

    // repeat same kernel as 'repeat'
    int repeat = 100;

    // index for accessing the 'exe_results', tracking the matrix size
    int cnt = 0;

    // execute kernels from small size to largest size
    for(int size : SIZE) {
        long m, n, k;
        m = n = k = size;
        std::cout << "size : " << size << std::endl;

        // warm up the device and compare the result
        launch_kernel_with_option(0, handle, d_A, d_B, d_C_ref, m, n, k, alpha, beta);
        for(int i = 1; i <= op_num; i++) {
            std::cout << "This is op " << i << std::endl;
            launch_kernel_with_option(i, handle, d_A, d_B, d_C, m, n, k, alpha, beta);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(C_ref, d_C_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
            if(!verify_matrix(C_ref, C, n))
                std::cout << "Result is different" << std::endl; 
            else
                std::cout << "Result is same" << std::endl;
            free(C);
            free(C_ref);
            C = (float*)malloc(sizeof(float) * max_size * max_size);
            C_ref = (float*)malloc(sizeof(float) * max_size * max_size);
            CHECK_CUDA(cudaFree(d_C));
            CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * max_size * max_size));
        }
        // for checking the correct result, deallocate the memory 
        CHECK_CUDA(cudaFree(d_C_ref));
        CHECK_CUDA(cudaMalloc((void**)&d_C_ref, sizeof(float) * max_size * max_size));

        // for comparing the GFLOPS
        for(int i = 0; i <= op_num; i++) {
            CHECK_CUDA(cudaEventRecord(begin));
            for(int j = 0; j < repeat; j++) {
                if(i == 0)
                    launch_kernel_with_option(i, handle, d_A, d_B, d_C_ref, m, n, k, alpha, beta);
                else
                    launch_kernel_with_option(i, handle, d_A, d_B, d_C, m, n, k, alpha, beta);
            }
            CHECK_CUDA(cudaEventRecord(end));
            CHECK_CUDA(cudaEventSynchronize(begin));
            CHECK_CUDA(cudaEventSynchronize(end));
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, begin, end));
            elapsed_time /= 1000;

            long flops = 2 * m * n * k;
            printf("op : %d, Average time : (%7.6f)sec, performance : (%7.1f) GFLOPs/s\n", i, elapsed_time / repeat, (repeat * flops * 1e-9) / elapsed_time);
            fflush(stdout);
            exe_results[i][cnt].gflops = (repeat * flops * 1e-9) / elapsed_time;
            exe_results[i][cnt].time = elapsed_time;
        }
        printf("\n");
        cnt++;

        // for checking the correct result, deallocate the memory 
        CHECK_CUDA(cudaFree(d_C_ref));
        CHECK_CUDA(cudaMalloc((void**)&d_C_ref, sizeof(float) * max_size * max_size));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * max_size * max_size));
    }

    printf("execution finished\n\n");

    // display the result of each kernels
    for(int i = 0; i <= op_num; i++) {
        printf("op : %d\nperformance : ", i);
        for(int j = 0; j < SIZE.size(); j++)
            printf("%7.1f GFLOPs/s ", exe_results[i][j].gflops);
        printf("\n");
    }
    printf("\n");

    // summarize the execution results
    printf("kernel\tsize\tGFLOPs/s\tSpeed UP(time) relative to cuBLAS\tPerformance(Gflops) relative to cuBLAS\n");
    for(int i = 1; i <= op_num; i++) {
        for(int j = 0; j < SIZE.size(); j++) {
            printf("  %d\t%d\t%7.1f\t\t\t   %f\t\t\t\t   %9.6f%%\n", i, SIZE[j], exe_results[i][j].gflops, 
                                                exe_results[0][j].time / exe_results[i][j].time, (exe_results[i][j].gflops / exe_results[0][j].gflops) * 100);
        }
        printf("\n");
    }
    
    // deallocate the memory
    free(A);
    free(B);
    free(C);
    free(C_ref);
    for(int i = 0; i < op_num; i++)
        free(exe_results[i]);
    free(exe_results);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));

    // destroy the event variables
    CHECK_CUDA(cudaEventDestroy(begin));
    CHECK_CUDA(cudaEventDestroy(end));
    
    // destroy the cublas handle
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS destruction failed\n");
        return EXIT_FAILURE;
    }

    return 0;
}