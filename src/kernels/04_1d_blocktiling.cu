/*
 * Stage 4: 1D Block Tiling (Thread Coarsening)
 *
 * Each thread computes TM output elements — a vertical strip of TM rows
 * in the same column of the output tile. This is "thread coarsening":
 * instead of one thread per output element, one thread handles TM elements.
 *
 * Why this helps:
 *   - More arithmetic per thread -> better instruction-level parallelism
 *   - TM accumulators stay in registers across the K loop
 *   - Amortizes the cost of loading B (each B element is reused TM times)
 *   - Reduces total thread count, reducing scheduling overhead
 *
 * Parameters:
 *   BM = 64  (block tile rows)
 *   BN = 64  (block tile cols)
 *   BK = 8   (K-dimension tile)
 *   TM = 8   (rows computed per thread)
 *
 * Block: (BN, BM/TM) = (64, 8) = 512 threads... actually (BN, BM/TM)
 * Grid:  ((N+BN-1)/BN, (M+BM-1)/BM)
 */

#include "matmul.cuh"

#define BM4 64
#define BN4 64
#define BK4 8
#define TM4 8

__global__ void kernel_1d_blocktiling(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[BM4][BK4];
    __shared__ float Bs[BK4][BN4];

    // Thread index within the block
    int threadCol = threadIdx.x;             // 0..BN-1
    int threadRow = threadIdx.y;             // 0..BM/TM-1

    // Starting row/col of this thread's output strip
    int rowStart = blockIdx.y * BM4 + threadRow * TM4;
    int col      = blockIdx.x * BN4 + threadCol;

    // TM accumulators in registers
    float acc[TM4] = {0.0f};

    // Total threads in block = BN * (BM/TM)
    int totalThreads = BN4 * (BM4 / TM4);
    int threadId = threadRow * BN4 + threadCol;

    int numTiles = (K + BK4 - 1) / BK4;

    for (int t = 0; t < numTiles; ++t) {
        // Cooperatively load As[BM][BK] and Bs[BK][BN] into shared memory
        // Each thread loads multiple elements to fill the tiles
        for (int i = threadId; i < BM4 * BK4; i += totalThreads) {
            int r = i / BK4;
            int c = i % BK4;
            int globalRow = blockIdx.y * BM4 + r;
            int globalCol = t * BK4 + c;
            As[r][c] = (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
        }
        for (int i = threadId; i < BK4 * BN4; i += totalThreads) {
            int r = i / BN4;
            int c = i % BN4;
            int globalRow = t * BK4 + r;
            int globalCol = blockIdx.x * BN4 + c;
            Bs[r][c] = (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
        }

        __syncthreads();

        // Compute: each thread iterates over BK dimension, accumulating TM results
        for (int k = 0; k < BK4; ++k) {
            float bVal = Bs[k][threadCol];
            #pragma unroll
            for (int rm = 0; rm < TM4; ++rm) {
                acc[rm] += As[threadRow * TM4 + rm][k] * bVal;
            }
        }

        __syncthreads();
    }

    // Write TM results back to global memory
    #pragma unroll
    for (int rm = 0; rm < TM4; ++rm) {
        int outRow = rowStart + rm;
        if (outRow < M && col < N) {
            C[outRow * N + col] = acc[rm];
        }
    }
}

void run_kernel_04(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    dim3 block(BN4, BM4 / TM4);  // (64, 8)
    dim3 grid((N + BN4 - 1) / BN4, (M + BM4 - 1) / BM4);
    kernel_1d_blocktiling<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
