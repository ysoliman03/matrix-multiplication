/*
 * Stage 3: Shared Memory Tiling
 *
 * Divide the K dimension into tiles of size TILE_SIZE (32).
 * Each thread block cooperatively loads a TILE_SIZE x TILE_SIZE sub-block
 * of A and B into shared memory, then all threads compute partial dot products
 * from shared memory before moving to the next tile.
 *
 * Benefit: Each global memory element is loaded once per tile and reused
 * TILE_SIZE times from fast shared memory (1-2 cycle latency vs ~200 cycles
 * for uncached global memory). This reduces global memory traffic by ~TILE_SIZE.
 *
 * Shared memory bank conflicts: We pad As by 1 to avoid bank conflicts when
 * reading down columns (As[i][tx] -> same bank for all i without padding).
 *
 * Grid:  ((N+TILE-1)/TILE, (M+TILE-1)/TILE)
 * Block: (TILE_SIZE, TILE_SIZE)
 */

#include "matmul.cuh"

#define TILE_SIZE 32

__global__ void kernel_shared_mem(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // +1 padding on As to avoid shared memory bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A: thread (ty, tx) loads A[row][t*TILE + tx]
        int aCol = t * TILE_SIZE + tx;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        // Load tile of B: thread (ty, tx) loads B[t*TILE + ty][col]
        int bRow = t * TILE_SIZE + ty;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        // Compute partial dot product from shared memory
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void run_kernel_03(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    kernel_shared_mem<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
