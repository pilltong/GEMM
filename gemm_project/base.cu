#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include "cuda_runtime.h"
#include "mma.h"
#include "cublas_v2.h"
#include <cuda_fp16.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = call;                                             \
        if(status_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,  \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)

// Check execution time for Linux based systems
#define CHECK_TIME_START(start) clock_gettime(CLOCK_MONOTONIC, &start)
#define CHECK_TIME_END(start, end, time) \
    clock_gettime(CLOCK_MONOTONIC, &end); \
    time = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6

template <typename T> 
struct Matrix {
    int Arow;
    int Acol;
    int Brow;
    int Bcol;
    int Crow;
    int Ccol;
    T *__restrict A;
    T *__restrict B;
    T *__restrict C;
};

template <typename T> 
void set_matrices(Matrix<T> &mat, int Ay, int Ax, int Bx) {
    mat.Arow = Ay;
    mat.Acol = Ax;
    mat.Brow = Ax;
    mat.Bcol = Bx;
    mat.Crow = Ay;
    mat.Ccol = Bx;
    mat.A = new T[Ay * Ax];
    mat.B = new T[Ax * Bx];
    mat.C = new T[Ay * Bx];

    std::default_random_engine gen(20240301);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for (int k = 0; k < Ay * Ax; k++)
        mat.A[k] = fran(gen);
    for (int k = 0; k < Ax * Bx; k++)
        mat.B[k] = fran(gen);
}

void initialize_A_and_B_hf(Matrix<float> &mat_s, Matrix<half> &mat_h) {
	mat_h.Arow = mat_s.Arow;
    mat_h.Acol = mat_s.Acol;
    mat_h.Brow = mat_s.Brow;
    mat_h.Bcol = mat_s.Bcol;
    mat_h.Crow = mat_s.Crow;
    mat_h.Ccol = mat_s.Ccol;
    
    mat_h.A = new half[mat_h.Arow * mat_h.Acol];
    mat_h.B = new half[mat_h.Brow * mat_h.Bcol];
    mat_h.C = new half[mat_h.Arow * mat_h.Bcol];

    for (int k = 0; k < mat_h.Arow * mat_h.Acol; k++) 
        mat_h.A[k] = (half)mat_s.A[k];
	for (int k = 0; k < mat_h.Brow * mat_h.Bcol; k++) 	
        mat_h.B[k] = (half)mat_s.B[k];
}

bool compare_two_matrices(float* M_host, const char* str_host, float* M_device, const char* str_device, int n_rows, int n_cols) {
    bool result = true;
    auto element = [&n_cols](float* M, int i, int j) {
        return *(M + i * n_cols + j);
    };

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            if (fabsf(element(M_host, i, j) - element(M_device, i, j)) / element(M_host, i, j) > 1.0e-6) {
                result = false;
            }
        }
    }
    return result;
}

void compare_two_matrices_flt_flt(float *A_flt_exact, float *A_flt_approx, int Arow, int Acol, float *ave_rel_error, float *max_rel_error) {
	double rel_error_sum = 0.0;
	float rel_error_max = 0.0f;
	for (int k = 0; k < Arow * Acol; k++) {
		float rel_error = fabsf(((float)A_flt_approx[k] - A_flt_exact[k]) / A_flt_exact[k]);
		rel_error_sum += rel_error;
		if (rel_error > rel_error_max)
			rel_error_max = rel_error;
	}
	*ave_rel_error = rel_error_sum / (Arow * Acol);
	*max_rel_error = rel_error_max;
}

void compare_two_matrices_flt_hf(float *A_flt, half *A_hf, int Arow, int Acol, float *ave_rel_error, float *max_rel_error) {
	double rel_error_sum = 0.0;
	float rel_error_max = 0.0f;
	for (int k = 0; k < Arow * Acol; k++) {
		float rel_error = fabsf(((float)A_hf[k] - A_flt[k]) / A_flt[k]);
		rel_error_sum += rel_error;
		if (rel_error > rel_error_max)
			rel_error_max = rel_error;
	}
	*ave_rel_error = rel_error_sum / (Arow * Acol);
	*max_rel_error = rel_error_max;
}

template <typename T>
void printf_matrix_element(T* X, int n_col, int i, int j) {
    fprintf(stdout, "^^^ M[%d][%d] = %f\n", i, j, (float)X[i * n_col + j]);
}

void print_computation_info(float _compute_time, int Arow, int Acol, int Bcol, int type_num) {
    fprintf(stdout, " = %f(ms) -----------------------------\n", _compute_time);
    //double flops = 2.0 * (double)Arow * (double)Acol * (double)Bcol;
    //double gflops = flops / (_compute_time * 1000000.0);
    //double gbytes = gflops * 6.0; // i.e. 12 bytes per term
    //fprintf(stdout, "*** Gflops = %.3f, Gbytes = %.3f\n", gflops, gbytes);
}

__global__ void matmul_GM(float* __restrict C, const float* __restrict A, const float* __restrict B, int Ay, int Ax, int Bx) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;   
    int ty = blockIdx.y * blockDim.y + threadIdx.y;   
    
    if (ty >= Ay || tx >= Bx)
        return;
    
    float csum = 0.0f;
    for (int k = 0; k < Ax; k++)
        csum += A[ty * Ax + k] * B[k * Bx + tx];
    
    C[ty * Bx + tx] = csum;
}

template <int TS> __global__ void matmul_SM(float* __restrict C, const float* __restrict A, const float* __restrict B, int Ay, int Ax, int Bx) {
    __shared__ float Atile[TS][TS];
    __shared__ float Btile[TS][TS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ocx = blockDim.x * blockIdx.x;
    int ocy = blockDim.y * blockIdx.y;

    int ax = tx;
    int ay = ty + ocy;
    int bx = tx + ocx;
    int by = ty;
    
    float csum = 0.0f;
    for(int i = 0; i < Ax / TS; i++) {
        Atile[ty][tx] = A[ay * Ax + ax];
        Btile[ty][tx] = B[by * Bx + bx];
        __syncthreads();

        for(int k = 0; k < TS; k++)
            csum += Atile[ty][k] * Btile[k][tx];
        __syncthreads();
        
        ax += TS;
        by += TS;
    }
    C[ay * Bx + bx] = csum;
}

template <int TS, int WPT, int RTS> __global__ void matmul_SM_MWPT(float* __restrict C, const float* __restrict A, const float* __restrict B, int Ay, int Ax, int Bx) {
    __shared__ float Atile[TS][TS];
    __shared__ float Btile[TS][TS];
    float accum[WPT];

    for(int i = 0; i < WPT; i++)
        accum[i] = 0.0f;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ocx = blockDim.x * blockIdx.x;
    int ocy = WPT * (blockDim.y * blockIdx.y);

    int ax = tx;
    int ay = ty + ocy;
    int bx = tx + ocx;
    int by = ty;

    for(int t = 0; t < Ax / TS; t++) {
        for(int w = 0; w < WPT; w++) {
            Atile[ty + w * RTS][tx] = A[(ay + w * RTS) * Ax + ax];
            Btile[ty + w * RTS][tx] = B[(by + w * RTS) * Bx + bx];
        }
        __syncthreads();

        for(int k = 0; k < TS; k++) {
            float tmp = Btile[k][tx];
            for(int w = 0; w < WPT; w++)
                accum[w] += Atile[ty + w * RTS][k] * tmp;
        }
        __syncthreads();

        ax += TS;
        by += TS;
    }

    for(int w = 0; w < WPT; w++)
        C[(ay + w * RTS) * Bx + bx] = accum[w];
}

__global__ void matmul_TC_GM(float *__restrict C, const half *__restrict A, const half *__restrict B, int Ay, int Ax, int Bx) {
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;

    int cx = warp % (Bx / 16);
    int cy = warp / (Bx / 16);

    int Atile_pos = cy * 16 * Ax;
    int Btile_pos = cx * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for(int k = 0; k < Ax / 16; k++) {
        wmma::load_matrix_sync(a_frag, &A[Atile_pos], Ax);
        wmma::load_matrix_sync(b_frag, &B[Btile_pos], Bx);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        Atile_pos += 16;
        Btile_pos += 16 * Bx;
    }
    wmma::store_matrix_sync(&C[(cy * Bx + cx) * 16], c_frag, Bx, wmma::mem_row_major);
}

__global__ void matmul_TC_SM(float *__restrict C, const half *__restrict A, const half *__restrict B, int Ay, int Ax, int Bx) {
    __shared__ half as[256];
    __shared__ half bs[8][256];

    if(blockDim.x != 256)
        return;
    
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;

    int cx = warp % (Bx / 16);
    int cy = warp / (Bx / 16);

    int Atile_pos = cy * 16 * Ax;
    int Btile_pos = cx * 16;

    int wb = threadIdx.x / 32;
    int trw = threadIdx.x % 32;
    int txw = trw % 16;
    int tyw = trw / 16;

    int idx = threadIdx.x % 16;
    int idy = threadIdx.x / 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for(int k = 0; k < Ax / 16; k++) {
        as[idy * 16 + idx] = A[Atile_pos + idy * Ax + idx];
        __syncthreads();
        for(int p = 0; p < 8; p++)
            bs[wb][p * 32 + tyw * 16 + txw] = B[p * 2 * Bx + Btile_pos + tyw * Bx + txw];
        //__syncwarp();
        wmma::load_matrix_sync(a_frag, &as[0], 16);
        wmma::load_matrix_sync(b_frag, &bs[wb][0], 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
        Atile_pos += 16;
        Btile_pos += 16 * Bx;
    }
    wmma::store_matrix_sync(&C[(cy * Bx + cx) * 16], c_frag, Bx, wmma::mem_row_major);
}

//template <typename T>
void MM_HOST(Matrix<float> &type_1, Matrix<float> &type_2, Matrix<float> &type_3, Matrix<half> &type_1_h, Matrix<half> &type_2_h, Matrix<half> &type_3_h) {
    struct timespec _start, _end;
    float _compute_time;
    float ave_rel_error, max_rel_error;
    float *A, *B, *C;
    int Ay, Ax, Bx;
    
    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[1] Host Time(double) Matrix Type 1");
    A = type_1.A;
    B = type_1.B;
    C = type_1.C;
    Ay = type_1.Arow;
    Ax = type_1.Acol;
    Bx = type_1.Bcol;
    CHECK_TIME_START(_start);
    for (int i = 0; i < Ay; i++) {
        for (int j = 0; j < Bx; j++) {
            double tmp = 0.0f;
            for (int k = 0; k < Ax; k++)
                tmp += A[i * Ax + k] * B[k * Bx + j];
            C[i * Bx + j] = (float)tmp;
        }
    }
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 1);
    compare_two_matrices_flt_hf(type_1.A, type_1_h.A, Ay, Ax, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_A_flt and h_A_hf] average = %f, maximum = %f\n", ave_rel_error, max_rel_error);
    compare_two_matrices_flt_hf(type_1.B, type_1_h.B, Ax, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_B_flt and h_B_hf] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[1] Host Time(double) Matrix Type 2");
    A = type_2.A;
    B = type_2.B;
    C = type_2.C;
    Ay = type_2.Arow;
    Ax = type_2.Acol;
    Bx = type_2.Bcol;
    CHECK_TIME_START(_start);
    for (int i = 0; i < Ay; i++) {
        for (int j = 0; j < Bx; j++) {
            double tmp = 0.0f;
            for (int k = 0; k < Ax; k++)
                tmp += A[i * Ax + k] * B[k * Bx + j];
            C[i * Bx + j] = (float)tmp;
        }
    }
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 2);
    compare_two_matrices_flt_hf(type_2.A, type_2_h.A, Ay, Ax, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_A_flt and h_A_hf] average = %f, maximum = %f\n", ave_rel_error, max_rel_error);
    compare_two_matrices_flt_hf(type_2.B, type_2_h.B, Ax, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_B_flt and h_B_hf] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[1] Host Time(double) Matrix Type 3");
    A = type_3.A;
    B = type_3.B;
    C = type_3.C;
    Ay = type_3.Arow;
    Ax = type_3.Acol;
    Bx = type_3.Bcol;
    CHECK_TIME_START(_start);
    for (int i = 0; i < Ay; i++) {
        for (int j = 0; j < Bx; j++) {
            double tmp = 0.0f;
            for (int k = 0; k < Ax; k++)
                tmp += A[i * Ax + k] * B[k * Bx + j];
            C[i * Bx + j] = (float)tmp;
        }
    }
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 3);
    compare_two_matrices_flt_hf(type_3.A, type_3_h.A, Ay, Ax, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_A_flt and h_A_hf] average = %f, maximum = %f\n", ave_rel_error, max_rel_error);
    compare_two_matrices_flt_hf(type_3.B, type_3_h.B, Ax, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_B_flt and h_B_hf] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);
}

template <typename T>
void MM_DEVICE_GM(Matrix<T> &type_1, Matrix<T> &type_2, Matrix<T> &type_3) {
    struct timespec _start, _end;
    float _compute_time;
    T *d_A, *d_B, *d_C, *h_C_device;
    int Ay, Ax, Bx;
    float ave_rel_error, max_rel_error;
    unsigned int tilex = 32, tiley = 16;
    dim3 blockDim, gridDim;

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[2] GPU time(CUDA Cores/float) Matrix Type 1");
    Ay = type_1.Arow;
    Ax = type_1.Acol;
    Bx = type_1.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_1.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_1.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, tiley, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay + blockDim.y - 1) / blockDim.y, 1};

    // dummy
    matmul_GM<<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start);
    matmul_GM<<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 1);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_1.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Initialize device memory of C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[2] GPU time(CUDA Cores/float) Matrix Type 2");
    Ay = type_2.Arow;
    Ax = type_2.Acol;
    Bx = type_2.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_2.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_2.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, tiley, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay + blockDim.y - 1) / blockDim.y, 1};

    CHECK_TIME_START(_start);
    matmul_GM<<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 2);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_2.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);    // // Define WMMA fragment types for TF32 precision
    // wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    // wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    // wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    // // Initialize output fragment
    // wmma::fill_fragment(c_frag, 0.0f);

    // const int brow = blockIdx.y * 16;
    // const int bcol = blockIdx.x * 16;
    
    // for(int i = 0; i < k; i += 8) {
    //     // Load the input matrices into fragments
    //     wmma::load_matrix_sync(a_frag, A + brow * k + i, k);
    //     wmma::load_matrix_sync(b_frag, B + i * n + bcol, n);
    //     // Perform the matrix multiplication using Tensor Cores
    //     wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    // }
    
    // for(int i = 0; i < c_frag.num_elements; i++)
    //     c_frag.x[i] = alpha * c_frag.x[i] + beta * C[brow * n + bcol + i];

    // // Store the result back to the output matrix
    // wmma::store_matrix_sync(C + brow * n + bcol, c_frag, n, wmma::mem_row_major);f(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_3.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_3.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, tiley, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay + blockDim.y - 1) / blockDim.y, 1};

    CHECK_TIME_START(_start);
    matmul_GM<<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 3);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_3.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);
}

template <typename T>
void MM_DEVICE_SM(Matrix<T> &type_1, Matrix<T> &type_2, Matrix<T> &type_3) {
    struct timespec _start, _end;
    float _compute_time;
    T *d_A, *d_B, *d_C, *h_C_device;
    int Ay, Ax, Bx;
    float ave_rel_error, max_rel_error;
    unsigned int tilex = 32;
    dim3 blockDim, gridDim;

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[3] GPU time(CUDA Cores/float/shared memory) Matrix Type 1");
    
    Ay = type_1.Arow;
    Ax = type_1.Acol;
    Bx = type_1.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_1.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_1.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, tilex, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay + blockDim.y - 1) / blockDim.y, 1};

    // dummy
    matmul_SM<32> <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start);
    matmul_SM<32> <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 1);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_1.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float/shared memory)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Initialize device memory of C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[3] GPU time(CUDA Cores/float/shared memory) Matrix Type 2");
    
    Ay = type_2.Arow;
    Ax = type_2.Acol;
    Bx = type_2.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_2.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_2.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, tilex, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay + blockDim.y - 1) / blockDim.y, 1};

    CHECK_TIME_START(_start);
    matmul_SM<32> <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 2);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_2.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float/shared memory)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[3] GPU time(CUDA Cores/float/shared memory) Matrix Type 3");
    
    Ay = type_3.Arow;
    Ax = type_3.Acol;
    Bx = type_3.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_3.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_3.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, tilex, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay + blockDim.y - 1) / blockDim.y, 1};

    CHECK_TIME_START(_start);
    matmul_SM<32> <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 3);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_3.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float/shared memory)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);
}

template <typename T>
void MM_DEVICE_SM_MWPT(Matrix<T> &type_1, Matrix<T> &type_2, Matrix<T> &type_3) {
    struct timespec _start, _end;
    float _compute_time;
    T *d_A, *d_B, *d_C, *h_C_device;
    int Ay, Ax, Bx;
    float ave_rel_error, max_rel_error;
    unsigned int tilex = 32, WPT = 8, RTS = tilex / WPT;
    dim3 blockDim, gridDim;

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[4] GPU time(CUDA Cores/float/shared memory/More-Work-per-Thread) Matrix Type 1");
    
    Ay = type_1.Arow;
    Ax = type_1.Acol;
    Bx = type_1.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_1.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_1.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, RTS, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay / WPT + blockDim.y - 1) / blockDim.y, 1};

    // dummy
    matmul_SM_MWPT<32, 8, 4> <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start);
    matmul_SM_MWPT<32, 8, 4> <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 1);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_1.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float/shared memory/More-Work-per-Thread)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Initialize device memory of C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[4] GPU time(CUDA Cores/float/shared memory/More-Work-per-Thread) Matrix Type 2");
    
    Ay = type_2.Arow;
    Ax = type_2.Acol;
    Bx = type_2.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_2.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_2.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, RTS, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay / WPT + blockDim.y - 1) / blockDim.y, 1};

    CHECK_TIME_START(_start);
    matmul_SM_MWPT<32, 8, 4> <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 2);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_2.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float/shared memory/More-Work-per-Thread)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[4] GPU time(CUDA Cores/float/shared memory/More-Work-per-Thread) Matrix Type 3");
    
    Ay = type_3.Arow;
    Ax = type_3.Acol;
    Bx = type_3.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_3.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_3.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);

    blockDim = {tilex, RTS, 1};
    gridDim = {(Bx + blockDim.x - 1) / blockDim.x, (Ay / WPT + blockDim.y - 1) / blockDim.y, 1};

    CHECK_TIME_START(_start);
    matmul_SM_MWPT<32, 8, 4> <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 3);

    h_C_device = new T[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_3.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float/shared memory/More-Work-per-Thread)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);
}

template <typename T>
void MM_DEVICE_TC_GM(float *type_1_h_C, float *type_2_h_C, float *type_3_h_C, Matrix<T> &type_1, Matrix<T> &type_2, Matrix<T> &type_3) {
    struct timespec _start, _end;
    float _compute_time;
    T *d_A, *d_B;
    float *d_C, *h_C_device;
    int Ay, Ax, Bx;
    float ave_rel_error, max_rel_error;
    dim3 blockDim, gridDim;

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[5] GPU time(Tensor Cores/half) Matrix Type 1");
    
    Ay = type_1.Arow;
    Ax = type_1.Acol;
    Bx = type_1.Bcol;
    CHECK_CUDA(cudaMalloc((void**)&d_A, Ay * Ax * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, Ax * Bx * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, type_1.A, Ay * Ax * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, type_1.B, Ax * Bx * sizeof(half), cudaMemcpyHostToDevice));

    blockDim = {256, 1, 1};
    gridDim = {Ay * Bx / (8 * blockDim.x), 1, 1};

    // dummy
    matmul_TC_GM <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start);
    matmul_TC_GM <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 1);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_1_h_C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(Tensor Cores/half)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Initialize device memory of C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[5] GPU time(Tensor Cores/half) Matrix Type 2");
    
    Ay = type_2.Arow;
    Ax = type_2.Acol;
    Bx = type_2.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(half));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(half));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_2.A, Ay * Ax * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_2.B, Ax * Bx * sizeof(half), cudaMemcpyHostToDevice);

    blockDim = {256, 1, 1};
    gridDim = {Ay * Bx / (8 * blockDim.x), 1, 1};

    CHECK_TIME_START(_start);
    matmul_TC_GM <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 2);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_2_h_C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(Tensor Cores/half)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);

    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[5] GPU time(Tensor Cores/half) Matrix Type 3");
    
    Ay = type_3.Arow;
    Ax = type_3.Acol;
    Bx = type_3.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(half));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(half));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_3.A, Ay * Ax * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_3.B, Ax * Bx * sizeof(half), cudaMemcpyHostToDevice);

    blockDim = {256, 1, 1};
    gridDim = {Ay * Bx / (8 * blockDim.x), 1, 1};

    CHECK_TIME_START(_start);
    matmul_TC_GM <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 3);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_3_h_C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(Tensor Cores/half)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);

    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);
}

template <typename T>
void MM_DEVICE_TC_SM(float *type_1_h_C, float *type_2_h_C, float *type_3_h_C, Matrix<T> &type_1, Matrix<T> &type_2, Matrix<T> &type_3) {
    struct timespec _start, _end;
    float _compute_time;
    T *d_A, *d_B;
    float *d_C, *h_C_device;
    int Ay, Ax, Bx;
    float ave_rel_error, max_rel_error;
    dim3 blockDim, gridDim;

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[6] GPU time(Tensor Cores/half/shared memory) Matrix Type 1");
    
    Ay = type_1.Arow;
    Ax = type_1.Acol;
    Bx = type_1.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(half));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(half));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_1.A, Ay * Ax * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_1.B, Ax * Bx * sizeof(half), cudaMemcpyHostToDevice);

    blockDim = {256, 1, 1};
    gridDim = {Ay * Bx / (8 * blockDim.x), 1, 1};

    // dummy
    matmul_TC_SM <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start);
    matmul_TC_SM <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 1);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_1_h_C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(Tensor Cores/half/shared memory)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);

    // Initialize device memory of C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[6] GPU time(Tensor Cores/half/shared memory) Matrix Type 2");
    
    Ay = type_2.Arow;
    Ax = type_2.Acol;
    Bx = type_2.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(half));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(half));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_2.A, Ay * Ax * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_2.B, Ax * Bx * sizeof(half), cudaMemcpyHostToDevice);

    blockDim = {256, 1, 1};
    gridDim = {Ay * Bx / (8 * blockDim.x), 1, 1};

    CHECK_TIME_START(_start);
    matmul_TC_SM <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 2);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_2_h_C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(Tensor Cores/half/shared memory)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);

    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[6] GPU time(Tensor Cores/half/shared memory) Matrix Type 3");
    
    Ay = type_3.Arow;
    Ax = type_3.Acol;
    Bx = type_3.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(half));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(half));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_3.A, Ay * Ax * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_3.B, Ax * Bx * sizeof(half), cudaMemcpyHostToDevice);

    blockDim = {256, 1, 1};
    gridDim = {Ay * Bx / (8 * blockDim.x), 1, 1};

    CHECK_TIME_START(_start);
    matmul_TC_SM <<<gridDim, blockDim>>>(d_C, d_A, d_B, Ay, Ax, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 3);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_3_h_C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(Tensor Cores/half/shared memory)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);

    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);
}

template <typename T>
void MM_DEVICE_CUBLAS(Matrix<T> &type_1, Matrix<T> &type_2, Matrix<T> &type_3) {
    struct timespec _start, _end;
    float _compute_time;
    T *d_A, *d_B;
    float *d_C, *h_C_device;
    int Ay, Ax, Bx;
    float ave_rel_error, max_rel_error;
    float alpha = 1.0, beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[7] GPU time(cuBlas/float) Matrix Type 1");

    Ay = type_1.Arow;
    Ax = type_1.Acol;
    Bx = type_1.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_1.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_1.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);
    
    // dummy
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, Bx, Ay, Ax, &alpha, d_B, Bx, d_A, Ax, &beta, d_C, Bx);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start);
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, Bx, Ay, Ax, &alpha, d_B, Bx, d_A, Ax, &beta, d_C, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 1);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_1.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(cuBlas/float)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[7] GPU time(cuBlas/float) Matrix Type 2");

    Ay = type_2.Arow;
    Ax = type_2.Acol;
    Bx = type_2.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_2.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_2.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);
    
    // dummy
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, Bx, Ay, Ax, &alpha, d_B, Bx, d_A, Ax, &beta, d_C, Bx);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start);
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, Bx, Ay, Ax, &alpha, d_B, Bx, d_A, Ax, &beta, d_C, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 2);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_2.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(cuBlas/float)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    //------------------------------------------------------------------------------------------------------------//
    fprintf(stdout, "[7] GPU time(cuBlas/float) Matrix Type 3");

    Ay = type_3.Arow;
    Ax = type_3.Acol;
    Bx = type_3.Bcol;
    cudaMalloc((void**)&d_A, Ay * Ax * sizeof(float));
    cudaMalloc((void**)&d_B, Ax * Bx * sizeof(float));
    cudaMalloc((void**)&d_C, Ay * Bx * sizeof(float));
    cudaMemcpy(d_A, type_3.A, Ay * Ax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, type_3.B, Ax * Bx * sizeof(float), cudaMemcpyHostToDevice);
    
    // dummy
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, Bx, Ay, Ax, &alpha, d_B, Bx, d_A, Ax, &beta, d_C, Bx);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start);
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, Bx, Ay, Ax, &alpha, d_B, Bx, d_A, Ax, &beta, d_C, Bx);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _compute_time);
    print_computation_info(_compute_time, Ay, Ax, Bx, 3);

    h_C_device = new float[Ay * Bx];
    cudaMemcpy(h_C_device, d_C, Ay * Bx * sizeof(float), cudaMemcpyDeviceToHost);

    compare_two_matrices_flt_flt(type_3.C, h_C_device, Ay, Bx, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(cuBlas/float)] average = %f, maximum = %f\n\n",
		ave_rel_error, max_rel_error);
    
    // Free device memory of d_A, d_B, d_C, h_C_device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C_device);

    cublasDestroy(cublasH);
}

int main(int argc, char *argv[]) {
    Matrix<float> type_1;
    Matrix<float> type_2;
    Matrix<float> type_3;
    
    Matrix<half> type_1_h;
    Matrix<half> type_2_h;
    Matrix<half> type_3_h;

    set_matrices(type_1, 1024, 2048, 512);
    set_matrices(type_2, 2048, 1024, 1024);
    set_matrices(type_3, 4096, 4096, 4096);
    
    initialize_A_and_B_hf(type_1, type_1_h);
    initialize_A_and_B_hf(type_2, type_2_h);
    initialize_A_and_B_hf(type_3, type_3_h);

    fprintf(stdout, "Matrix Type 1 : A = %dx%d, B = %dx%d, C = %dx%d\n", type_1.Arow, type_1.Acol, type_1.Brow, type_1.Bcol, type_1.Crow, type_1.Ccol);
    fprintf(stdout, "Matrix Type 2 : A = %dx%d, B = %dx%d, C = %dx%d\n", type_2.Arow, type_2.Acol, type_2.Brow, type_2.Bcol, type_2.Crow, type_2.Ccol);
    fprintf(stdout, "Matrix Type 3 : A = %dx%d, B = %dx%d, C = %dx%d\n\n", type_3.Arow, type_3.Acol, type_3.Brow, type_3.Bcol, type_3.Crow, type_3.Ccol);

    MM_HOST(type_1, type_2, type_3, type_1_h, type_2_h, type_3_h);

    MM_DEVICE_GM(type_1, type_2, type_3);

    MM_DEVICE_SM(type_1, type_2, type_3);

    MM_DEVICE_SM_MWPT(type_1, type_2, type_3);

    MM_DEVICE_TC_GM(type_1.C, type_2.C, type_3.C, type_1_h, type_2_h, type_3_h);
    
    MM_DEVICE_TC_SM(type_1.C, type_2.C, type_3.C, type_1_h, type_2_h, type_3_h);

    MM_DEVICE_CUBLAS(type_1, type_2, type_3);

    return 0;
}