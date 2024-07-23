# GEMM

Implement cuBLAS-like Performance Kernels with Tensor Cores

1. CUDA Cores
    1) Naive
 	2) Global Memory Coalescing
 	3) Shared Memory Caching
 	4) 1D Blocking
 	5) 2D Blocking
 	6) Vectorized Memory Access
 	7) Autotuning
 	8) Warp-tiling

2. Tensor Cores(WMMA API)
 	1) cuBLAS with TF32
 	2) cuBLAS with BF32
 	3) cuBLAS with fp16
 	4) 2D Blcking with Global Memory
 	5) 2D Blcking with Shared Memory
 	6) on-going
