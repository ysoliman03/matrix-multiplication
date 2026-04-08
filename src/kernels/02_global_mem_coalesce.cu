/*
 * Stage 2: Global Memory Coalescing
 *
 * Same algorithm as naive, but the thread-to-output mapping is changed so
 * that consecutive threads in a warp (same threadIdx.x, adjacent x positions)
 * map to adjacent COLUMNS of C.
 *
 * Key insight:
 *   Naive:    threadIdx.x -> row  => warp reads A[0..31][k] = stride N => uncoalesced
 *   Coalesced: threadIdx.x -> col => warp reads B[k][0..31] and C[row][0..31]
 *              which are contiguous in memory => coalesced 128-byte transactions.
 *
 * A reads are still strided (each thread reads the same row of A), but B and C
 * accesses are now fully coalesced, which typically gives 2-4x speedup.
 *
 * Grid:  ((N+31)/32, (M+31)/32)
 * Block: (32, 32)
 */

#include "matmul.cuh"

__global__ void kernel_coalesce(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Swap: threadIdx.x maps to column, threadIdx.y maps to row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

void run_kernel_02(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    kernel_coalesce<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
