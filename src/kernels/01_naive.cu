/*
 * Stage 1: Naive Matrix Multiplication
 *
 * One thread computes one output element C[row][col].
 * Each thread iterates over the K dimension, performing a dot product
 * of row `row` of A and column `col` of B.
 *
 * Problem: For matrix A (row-major), consecutive threads (same warp) have
 * different `row` values, so they read elements A[row][k], A[row+1][k], ...
 * which are N floats apart in memory -> uncoalesced global reads.
 *
 * Grid:  ((N+31)/32, (M+31)/32)
 * Block: (32, 32)
 */

#include "matmul.cuh"

__global__ void kernel_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // threadIdx.x -> column direction within block (but mapped to row here)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void run_kernel_01(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    kernel_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
