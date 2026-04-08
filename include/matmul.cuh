#pragma once

#include <cuda_runtime.h>

// Stage 1: Naive kernel - one thread per output element
void run_kernel_01(float* d_A, float* d_B, float* d_C, int M, int N, int K);

// Stage 2: Global memory coalescing - remap threadIdx.x to column dimension
void run_kernel_02(float* d_A, float* d_B, float* d_C, int M, int N, int K);

// Stage 3: Shared memory tiling - load tiles into SMEM to reduce global reads
void run_kernel_03(float* d_A, float* d_B, float* d_C, int M, int N, int K);

// Stage 4: 1D block tiling - each thread computes TM output elements (vertical strip)
void run_kernel_04(float* d_A, float* d_B, float* d_C, int M, int N, int K);

// Stage 5: 2D block tiling - each thread computes TM x TN sub-block via outer product
void run_kernel_05(float* d_A, float* d_B, float* d_C, int M, int N, int K);

// Stage 6: Vectorized memory access - use float4 for 128-bit loads/stores
void run_kernel_06(float* d_A, float* d_B, float* d_C, int M, int N, int K);
