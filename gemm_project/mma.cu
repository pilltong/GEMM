#include "helper.h"

using namespace nvcuda;

__global__ void sgemm_naive(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint col = blockDim.x * blockIdx.x + threadIdx.x;
    const uint row = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < m && col < n) {
        float tmp = 0.0f;
        for(int i = 0; i < k; i++)
            tmp += A[row * k + i] * B[i * n + col];
        C[row * n + col] = alpha * tmp + beta * C[row * n + col];
    }
}

void run_sgemm_naive(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    sgemm_naive<<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

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

void run_global_coalesce(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    global_coalesce<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

template <const uint tile_size>
__global__ void shared_cache_blocking(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
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

void run_shared_cache_blocking(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    dim3 blockDim(32 * 32);
    dim3 gridDim(ceil_div(n, 32), ceil_div(m, 32));
    shared_cache_blocking<32> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

template <const uint bm, const uint bn, const uint bk, const uint tw>
__global__ void blocking_1d(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ float A_shared[bm * bk];
    __shared__ float B_shared[bk * bn];
    float inter_result[tw] = {0.0f};

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / bn;
    const int tcol = threadIdx.x % bn;

    // coordinate of thread in thread block for each shared memory block
    const int A_trow = threadIdx.x / bk;
    const int A_tcol = threadIdx.x % bk;
    const int B_trow = threadIdx.x / bn;
    const int B_tcol = threadIdx.x % bn;

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // outer loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        A_shared[A_trow * bk + A_tcol] = A[A_trow * k + A_tcol];
        B_shared[B_trow * bn + B_tcol] = B[B_trow * n + B_tcol];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into vector where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            float tmp = B_shared[j * bn + tcol];
            // single thread multiply elements in the vector
            for(int k = 0; k < tw; k++)
                inter_result[k] += A_shared[(trow * tw + k) * bk + j] * tmp;
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }

    for(int k = 0; k < tw; k++)
        C[(trow * tw + k) * n + tcol] = alpha * inter_result[k] + beta * C[(trow * tw + k) * n + tcol];
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

template <const uint bm, const uint bn, const uint bk, const uint tw_m, const uint tw_n>
__global__ void blocking_2d(const float *__restrict A, const float *__restrict B, float *C, int m, int n, int k, float alpha, float beta) {
    __shared__ float A_shared[bm * bk];
    __shared__ float B_shared[bk * bn];
    float per_thread_result[tw_m * tw_n] = {0.0f};
    float reg_A[tw_m] = {0.0f};
    float reg_B[tw_n] = {0.0f};

    // coordinate of block in grid
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // coordinate of thread in thread block
    const int trow = threadIdx.x / (bn / tw_n);
    const int tcol = threadIdx.x % (bn / tw_n);

    // coordinate of thread in thread block for each shared memory block
    const int A_trow = threadIdx.x / bk;
    const int A_tcol = threadIdx.x % bk;
    const int B_trow = threadIdx.x / bn;
    const int B_tcol = threadIdx.x % bn;

    // total number of thread in thread block = total number of blocks in shared memory for each matrix
    const int total_num_thread = (bm * bn) / (tw_m * tw_n);
    const int A_row_num = total_num_thread / bk;
    const int B_row_num = total_num_thread / bn; 

    // staring pointer of each matrcies
    A += brow * bm * k; // grid dimension = (m / bm) * (n / bn) => to recover the original shape, mulitply bm
    B += bcol * bn;
    C += brow * bm * n + bcol * bn;

    // out most loop
    // divide whole matrix into blocks
    for(int i = 0; i < k; i += bk) {
        // load the data into shared memory
        for(int j = 0; j < bm; j += A_row_num)
            A_shared[(A_trow + j) * bk + A_tcol] = A[(A_trow + j) * k + A_tcol];
        for(int j = 0; j < bk; j += B_row_num)
            B_shared[(B_trow + j) * bn + B_tcol] = B[(B_trow + j) * n + B_tcol];
        // wait unit the whole threads in single thread block finish the data loading
        // unit of shared memory = thread block
        __syncthreads();
        
        // inner loop
        // divide block into small matrices where single thread responsible for 
        for(int j = 0; j < bk; j++) {
            // load data in shared memory into local register
            for(int k = 0; k < tw_m; k++)
                reg_A[k] = A_shared[(trow * tw_m + k) * bk + j];
            for(int k = 0; k < tw_n; k++)
                reg_B[k] = B_shared[j * bn + tcol * tw_n + k];
            // outer product two vectors to make matrix
            for(int inter_m = 0; inter_m < tw_m; inter_m++) {
                for(int inter_n = 0; inter_n < tw_n; inter_n++)
                    per_thread_result[inter_m * tw_n + inter_n] += reg_A[inter_m] * reg_B[inter_n];
            }
        }
        __syncthreads();

        // move onto next block
        A += bk;
        B += bk * n;
    }

    for(int i = 0; i < tw_m; i++) {
        for(int j = 0; j < tw_n; j++)
            C[(trow * tw_m + i) * n + tcol * tw_n + j] = alpha * per_thread_result[i * tw_n + j] + beta * C[(trow * tw_m + i) * n + tcol * tw_n + j];
    }
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

__global__ void wmma_tf32(const float *A, const float *B, float *C, int m, int n, int k, float alpha, float beta) {
    // Define WMMA fragment types for TF32 precision
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    // Initialize output fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // warp index among total threads 
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    
    // index of each matrices' tile
    const int Crow = warp_id / (n / 16);
    const int Ccol = warp_id % (n / 16);
    int Arow = Crow * (16 * k);
    int Bcol = Ccol * 16;
    
    for(int i = 0; i < k; i += 8) {
        // Load the input matrices into fragments
        wmma::load_matrix_sync(a_frag, A + Arow + i, k);
        wmma::load_matrix_sync(b_frag, B + Bcol + i * n, n);
        // Perform the matrix multiplication using Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    for(int i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] = alpha * c_frag.x[i] + beta * C[Crow * n + Ccol + i];

    // Store the result back to the output matrix
    wmma::store_matrix_sync(C + (Crow * n + Ccol) * 16, c_frag, n, wmma::mem_row_major);
}

void run_wmma_tf32(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    float *tf32_A;
    float *tf32_B;
    CHECK_CUDA(cudaMalloc((void**)&tf32_A, sizeof(float) * m * k));
    CHECK_CUDA(cudaMalloc((void**)&tf32_B, sizeof(float) * k * n));
    convert_to_tf32(tf32_A, tf32_B, A, B, m, n, k);
    dim3 blockDim(16 * 16);
    dim3 gridDim(ceil_div(n * m, 16 * 16 * 8));
    wmma_tf32 <<<gridDim, blockDim>>>(tf32_A, tf32_B, C, m, n, k, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(tf32_A));
    CHECK_CUDA(cudaFree(tf32_B));
}

__global__ void wmma_fp16(const half *A, const half *B, float *C, int m, int n, int k, float alpha, float beta) {
    // Define WMMA fragment types for TF32 precision
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize output fragment
    wmma::fill_fragment(c_frag, 0.0f);
    
    // warp index among total threads 
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    
    // index of each matrices' tile
    const int Crow = warp_id / (n / 16);
    const int Ccol = warp_id % (n / 16);
    int Arow = Crow * (16 * k);
    int Bcol = Ccol * 16;
    
    for(int i = 0; i < k; i += 16) {
        // Load the input matrices into fragments
        wmma::load_matrix_sync(a_frag, A + Arow + i, k);
        wmma::load_matrix_sync(b_frag, B + Bcol + i * n, n);
        // Perform the matrix multiplication using Tensor Cores
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    for(int i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] = alpha * c_frag.x[i] + beta * C[Crow * n + Ccol + i];

    // Store the result back to the output matrix
    wmma::store_matrix_sync(C + (Crow * n + Ccol) * 16, c_frag, n, wmma::mem_row_major);
}

void run_wmma_fp16(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    __half *half_A;
    __half *half_B;
    cudaMalloc((void**)&half_A, sizeof(__half) * m * k);
    cudaMalloc((void**)&half_B, sizeof(__half) * k * n);
    convert_to_fp16(half_A, half_B, A, B, m, n, k);
    dim3 blockDim(16 * 16);
    dim3 gridDim(ceil_div(n * m, 16 * 16 * 8));
    wmma_fp16 <<<gridDim, blockDim>>>(half_A, half_B, C, m, n, k, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(half_A));
    CHECK_CUDA(cudaFree(half_B));
}

void runCublasFP32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    // cuBLAS uses column-major order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp32 mode
    nvtxRangePushA("outer");
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    nvtxRangePop();
}

void runCublasBF16(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    // This runs cuBLAS with mixed precision (performing the mul with operands
    // downcast to bf16), which is ~4x faster
    
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
     
}

void runCublasTF32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    // This runs cuBLAS with mixed precision (performing the mul with operands
    // downcast to bf16), which is ~4x faster
    float *tf32_A;
    float *tf32_B;
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

void runCublasBF16_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    // This runs cuBLAS with mixed precision (performing the mul with operands
    // downcast to bf16), which is ~4x faster
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasTF32_with_TC(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    // This runs cuBLAS with mixed precision (performing the mul with operands
    // downcast to bf16), which is ~4x faster
    float *tf32_A;
    float *tf32_B;
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

void launch_kernel_with_option(int op, cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    switch(op) {
        case 0 :
            runCublasTF32_with_TC(handle, A, B, C, m, n, k, alpha, beta);
            break;
        case 1 :
            run_sgemm_naive(A, B, C, m, n, k, alpha, beta);
            break;
        case 2 :
            run_global_coalesce(A, B, C, m, n, k, alpha, beta);
            break;
        case 3 :
            run_shared_cache_blocking(A, B, C, m, n, k, alpha, beta);
            break;
        case 4 :
            run_blocking_1d(A, B, C, m, n, k, alpha, beta);
            break;
        case 5 :
            run_blocking_2d(A, B, C, m, n, k, alpha, beta);
            break;
        case 6 :
            run_wmma_tf32(A, B, C, m, n, k, alpha, beta);
            break;
        case 7 :
            run_wmma_fp16(A, B, C, m, n, k, alpha, beta);
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
    int op_num = 6;

    // for storing the gflops and elapsed_time
    result **exe_results = (result**)malloc(sizeof(result*) * (op_num + 1));
    for(int i = 0; i <= op_num; i++)
        exe_results[i] = (result*)malloc(sizeof(result) * SIZE.size());

    // repeat same kernel as 'repeat'
    int repeat = 10;

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