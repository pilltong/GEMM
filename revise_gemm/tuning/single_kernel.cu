#include "../headers/helpers.h"
#include "../headers/cuda_kernels.cuh"
#include "../headers/tensor_kernels.cuh"

void randomize_matrix_s(int N, float *M) {
	struct timeval time {};
	
	gettimeofday(&time, nullptr);
	srand(time.tv_usec);

	for (int i = 0; i < N; i++) {
		float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
		tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
		M[i] = tmp;
	}
}

bool verify_matrix_s(void *matRef, void *matOut, int N) {
	double diff = 0.0;
	int i;
	for (i = 0; i < N; i++) {
		diff = std::fabs(((float*)matRef)[i] - ((float*)matOut)[i]);
		if (diff > 0.001) {
			printf("Divergence! Should %.6f, Is %.6f (Diff %.6f) at %d\n",
				((float*)matRef)[i], ((float*)matOut)[i], diff, i);
			return false;
		}
	}
	return true;
}

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
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    vectorized_fp<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_vectorized_fp_revised(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    vectorized_fp_revised<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void run_resolve_bank_conflict(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    const uint bm = 128;
    const uint bn = 128;
    const uint bk = 8;
    const uint tw_m = 8;
    const uint tw_n = 8;
    dim3 blockDim((bm / tw_m) * (bn / tw_n));
    dim3 gridDim(ceil_div(n, bn), ceil_div(m, bm));
    resolve_bank_conflict<bm, bn, bk, tw_m, tw_n> <<<gridDim, blockDim>>>(A, B, C, m, n, k, alpha, beta);
}

void runCublasFP32(cublasHandle_t handle, float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, CUDA_R_32F,
                n, A, CUDA_R_32F, k, &beta, C, CUDA_R_32F, n, 
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

typedef struct _execution_result {
    float gflops;
    float time;
} result;

int main(int argc, char **argv) {
    // 실행할 커널을 선택하기 위해 커널 번호를 인자로 받습니다.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel_number>" << std::endl;
        return EXIT_FAILURE;
    }
    int kernel_number = std::atoi(argv[1]);

    // 디바이스 정보 출력
    CudaDeviceInfo();

    // CuBLAS 핸들 생성
    cublasHandle_t handle;
    if(cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        return EXIT_FAILURE;
    };

    // 행렬 곱셈 상수 값
    float alpha = 1.0, beta = 0.0;

    // 행렬 크기 설정 (예시: 4096x4096 크기의 행렬)
    long m = 4096, n = 4096, k = 4096;

    // repeats
    int repeat_time = 100;

    // 호스트 메모리 할당
    float *A = (float*)malloc(sizeof(float) * m * k);
    float *B = (float*)malloc(sizeof(float) * k * n);
    float *C = (float*)calloc(m * n, sizeof(float));
    float *C_ref = (float*)calloc(m * n, sizeof(float));

    randomize_matrix_s(m * k, A);
	randomize_matrix_s(k * n, B);

    // 디바이스 메모리 할당
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    float *d_C_ref = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * m * k));
	CHECK_CUDA(cudaMalloc((void**)&d_B, sizeof(float) * k * n));
	CHECK_CUDA(cudaMalloc((void**)&d_C, sizeof(float) * m * n));
	CHECK_CUDA(cudaMalloc((void**)&d_C_ref, sizeof(float) * m * n));

    CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice));

    // CUDA 이벤트 생성
    float elapsed_time1, elapsed_time2;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 커널 실행 및 시간 측정
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < repeat_time; i++)
        runCublasFP32(handle, d_A, d_B, d_C_ref, m, n, k, alpha, beta);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(start));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time1, start, stop));
    
    cudaMemcpy(C_ref, d_C_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < repeat_time; i++) {
        switch (kernel_number) {
            case 1 :
                run_naive_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 2 :
                run_global_coalesce_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 3 :
                run_shared_caching_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 4 :
                run_blocking_1d_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 5 :
                run_blocking_2d_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 6 :
                run_vectorized_fp(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 7:
                run_vectorized_fp_revised(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            case 8 :
                run_resolve_bank_conflict(d_A, d_B, d_C, m, n, k, alpha, beta);
                break;
            default:
                std::cerr << "Invalid kernel number." << std::endl;
                return EXIT_FAILURE;
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(start));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time2, start, stop));

    // 커널 실행 완료 후, 결과를 호스트 메모리로 복사
    CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    // GFLOPS 계산
    long flops = 2 * m * n * k;
    elapsed_time1 /= 1000;
    elapsed_time2 /= 1000;
    float gflops_ref = (repeat_time * flops * 1e-9) / elapsed_time1;
    float gflops_kernel = (repeat_time * flops * 1e-9) / elapsed_time2;

    // 결과 검증 (예시로 FP32 결과를 사용)
    if(!verify_matrix(C_ref, C, n)) {
        std::cout << "Result is different" << std::endl;
    } else {
        std::cout << "Result is correct" << std::endl;
    }

    // 성능 출력
    std::cout << "cuBLAS Execution Time : " << elapsed_time1 / repeat_time << " seconds" << std::endl;
    std::cout << "cuBLAS Performance : " << gflops_ref << " GFLOPS" << std::endl << std::endl;
    std::cout << "Kernel Execution Time : " << elapsed_time2 / repeat_time << " seconds" << std::endl;
    std::cout << "Kernel Performance : " << gflops_kernel << " GFLOPS" << std::endl;

    // 메모리 해제
    free(A);
    free(B);
    free(C);
    free(C_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));

    // CuBLAS 핸들 해제
    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS destruction failed\n");
        return EXIT_FAILURE;
    }

    // CUDA 이벤트 해제
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}