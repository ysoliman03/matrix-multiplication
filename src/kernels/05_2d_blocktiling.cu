/*
 * Stage 5: 2D Block Tiling
 *
 * Each thread computes a TM x TN sub-block of C using an outer product
 * approach: load TM elements from A tile and TN elements from B tile into
 * registers, then update TM*TN accumulators.
 *
 * Why this is fast:
 *   - TM*TN = 64 accumulations per BK iteration with only TM+TN = 16 loads
 *   - Arithmetic intensity ≈ TM*TN/(TM+TN) = 4x better than 1D tiling
 *   - All TM*TN accumulators live in registers -> no extra memory traffic
 *   - 256 threads per block, each doing 64 MACs per BK step
 *
 * Parameters:
 *   BM = 128  BN = 128  BK = 16
 *   TM = 8    TN = 8
 *   Threads per block: (BM/TM) * (BN/TN) = 16 * 16 = 256
 *
 * Loading pattern: the 256 threads cooperatively load:
 *   As[BM][BK] = 128*16 = 2048 floats  (8 floats per thread)
 *   Bs[BK][BN] = 16*128 = 2048 floats  (8 floats per thread)
 */

#include "matmul.cuh"

#define BM5  128
#define BN5  128
#define BK5  16
#define TM5  8
#define TN5  8

__global__ void kernel_2d_blocktiling(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Shared memory tiles — padding to avoid bank conflicts
    __shared__ float As[BM5][BK5 + 1];
    __shared__ float Bs[BK5][BN5 + 1];

    // 2D thread index within the output tile
    int threadCol = threadIdx.x % (BN5 / TN5);  // 0..15
    int threadRow = threadIdx.x / (BN5 / TN5);  // 0..15

    // Starting global position of this thread's TM x TN output block
    int rowStart = blockIdx.y * BM5 + threadRow * TM5;
    int colStart = blockIdx.x * BN5 + threadCol * TN5;

    // TM * TN accumulators
    float acc[TM5][TN5] = {{0.0f}};

    // Registers for A and B fragments
    float regA[TM5];
    float regB[TN5];

    // Number of threads in block and this thread's linear ID
    int totalThreads = BM5 / TM5 * BN5 / TN5;  // 256
    int threadId = threadIdx.x;

    int numTiles = (K + BK5 - 1) / BK5;

    for (int t = 0; t < numTiles; ++t) {
        // Load As[BM][BK]: 256 threads each load BM*BK/256 = 8 elements
        for (int i = threadId; i < BM5 * BK5; i += totalThreads) {
            int r = i / BK5;
            int c = i % BK5;
            int gRow = blockIdx.y * BM5 + r;
            int gCol = t * BK5 + c;
            As[r][c] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }

        // Load Bs[BK][BN]: 256 threads each load BK*BN/256 = 8 elements
        for (int i = threadId; i < BK5 * BN5; i += totalThreads) {
            int r = i / BN5;
            int c = i % BN5;
            int gRow = t * BK5 + r;
            int gCol = blockIdx.x * BN5 + c;
            Bs[r][c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();

        // Compute outer products: iterate over BK dimension
        for (int k = 0; k < BK5; ++k) {
            // Load TM elements of A column into registers
            #pragma unroll
            for (int rm = 0; rm < TM5; ++rm) {
                regA[rm] = As[threadRow * TM5 + rm][k];
            }
            // Load TN elements of B row into registers
            #pragma unroll
            for (int rn = 0; rn < TN5; ++rn) {
                regB[rn] = Bs[k][threadCol * TN5 + rn];
            }
            // Outer product: TM * TN MACs
            #pragma unroll
            for (int rm = 0; rm < TM5; ++rm) {
                #pragma unroll
                for (int rn = 0; rn < TN5; ++rn) {
                    acc[rm][rn] += regA[rm] * regB[rn];
                }
            }
        }

        __syncthreads();
    }

    // Write TM x TN results back to global memory
    #pragma unroll
    for (int rm = 0; rm < TM5; ++rm) {
        #pragma unroll
        for (int rn = 0; rn < TN5; ++rn) {
            int outRow = rowStart + rm;
            int outCol = colStart + rn;
            if (outRow < M && outCol < N) {
                C[outRow * N + outCol] = acc[rm][rn];
            }
        }
    }
}

void run_kernel_05(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    dim3 block(BM5 / TM5 * BN5 / TN5);  // 256 threads, 1D block
    dim3 grid((N + BN5 - 1) / BN5, (M + BM5 - 1) / BM5);
    kernel_2d_blocktiling<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
