#pragma once

#include "cuda_kernels/1_naive.cuh"
#include "cuda_kernels/2_global_coalesce.cuh"
#include "cuda_kernels/3_shared_caching.cuh"
#include "cuda_kernels/4_blocking_1d.cuh"
#include "cuda_kernels/5_blocking_2d.cuh"