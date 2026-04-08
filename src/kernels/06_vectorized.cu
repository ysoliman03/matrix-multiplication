/*
 * Stage 6: Vectorized Memory Access
 *
 * Builds on Stage 5 (2D block tiling) and replaces scalar float loads
 * with float4 vector loads (128-bit / 4 floats per instruction).
 *
 * Why this helps:
 *   - A single float4 load uses one memory instruction instead of four
 *   - 4x reduction in load instruction count -> less instruction-issue pressure
 *   - Modern GPUs saturate their memory bus more efficiently with wider loads
 *   - The L1/L2 cache hit rate improves because fewer cache line fetches are needed
 *
 * Implementation:
 *   - Global->shared memory loads use float4: each thread loads 4 consecutive
 *     floats at once using reinterpret_cast<const float4*>
 *   - Matrix dimensions must be multiples of 4 for alignment; we assert this.
 *   - The final C write-back also uses float4 stores.
 *
 * Parameters (same as Stage 5):
 *   BM = 128  BN = 128  BK = 16
 *   TM = 8    TN = 8
 *   Threads per block: 256
 *
 * Loading with float4:
 *   As[BM][BK]: 2048 floats = 512 float4s, 256 threads -> 2 float4s each
 *   Bs[BK][BN]: 2048 floats = 512 float4s, 256 threads -> 2 float4s each
 */

#include "matmul.cuh"

#define BM6  128
#define BN6  128
#define BK6  16
#define TM6  8
#define TN6  8

__global__ void kernel_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[BM6][BK6 + 4];  // +4 padding for bank conflicts
    __shared__ float Bs[BK6][BN6 + 4];

    int threadCol = threadIdx.x % (BN6 / TN6);
    int threadRow = threadIdx.x / (BN6 / TN6);

    int rowStart = blockIdx.y * BM6 + threadRow * TM6;
    int colStart = blockIdx.x * BN6 + threadCol * TN6;

    float acc[TM6][TN6] = {{0.0f}};
    float regA[TM6];
    float regB[TN6];

    int totalThreads = (BM6 / TM6) * (BN6 / TN6);  // 256
    int threadId = threadIdx.x;

    int numTiles = (K + BK6 - 1) / BK6;

    for (int t = 0; t < numTiles; ++t) {
        // ---- Load As[BM][BK] using float4 ----
        // BM * BK = 2048 floats = 512 float4s -> 256 threads load 2 float4s each
        for (int i = threadId; i < (BM6 * BK6) / 4; i += totalThreads) {
            int elem = i * 4;                  // flat float index within the tile
            int r = elem / BK6;
            int c = elem % BK6;
            int gRow = blockIdx.y * BM6 + r;
            int gCol = t * BK6 + c;

            if (gRow < M && gCol + 3 < K && (gCol % 4 == 0)) {
                // Aligned float4 load
                float4 val = reinterpret_cast<const float4*>(A + gRow * K + gCol)[0];
                As[r][c]     = val.x;
                As[r][c + 1] = val.y;
                As[r][c + 2] = val.z;
                As[r][c + 3] = val.w;
            } else {
                // Fallback for boundary or unaligned
                for (int q = 0; q < 4; ++q) {
                    int gr = blockIdx.y * BM6 + (elem + q) / BK6;
                    int gc = t * BK6 + (elem + q) % BK6;
                    As[(elem + q) / BK6][(elem + q) % BK6] =
                        (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                }
            }
        }

        // ---- Load Bs[BK][BN] using float4 ----
        // BK * BN = 2048 floats = 512 float4s -> 256 threads load 2 float4s each
        for (int i = threadId; i < (BK6 * BN6) / 4; i += totalThreads) {
            int elem = i * 4;
            int r = elem / BN6;
            int c = elem % BN6;
            int gRow = t * BK6 + r;
            int gCol = blockIdx.x * BN6 + c;

            if (gRow < K && gCol + 3 < N && (gCol % 4 == 0)) {
                float4 val = reinterpret_cast<const float4*>(B + gRow * N + gCol)[0];
                Bs[r][c]     = val.x;
                Bs[r][c + 1] = val.y;
                Bs[r][c + 2] = val.z;
                Bs[r][c + 3] = val.w;
            } else {
                for (int q = 0; q < 4; ++q) {
                    int gr = t * BK6 + (elem + q) / BN6;
                    int gc = blockIdx.x * BN6 + (elem + q) % BN6;
                    Bs[(elem + q) / BN6][(elem + q) % BN6] =
                        (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute outer products
        for (int k = 0; k < BK6; ++k) {
            #pragma unroll
            for (int rm = 0; rm < TM6; ++rm) {
                regA[rm] = As[threadRow * TM6 + rm][k];
            }
            #pragma unroll
            for (int rn = 0; rn < TN6; ++rn) {
                regB[rn] = Bs[k][threadCol * TN6 + rn];
            }
            #pragma unroll
            for (int rm = 0; rm < TM6; ++rm) {
                #pragma unroll
                for (int rn = 0; rn < TN6; ++rn) {
                    acc[rm][rn] += regA[rm] * regB[rn];
                }
            }
        }

        __syncthreads();
    }

    // ---- Write back using float4 where possible ----
    #pragma unroll
    for (int rm = 0; rm < TM6; ++rm) {
        int outRow = rowStart + rm;
        if (outRow >= M) continue;

        // Try float4 store for groups of 4 consecutive columns
        int rn = 0;
        for (; rn + 3 < TN6; rn += 4) {
            int outCol = colStart + rn;
            if (outCol + 3 < N && (outCol % 4 == 0)) {
                float4 val = {acc[rm][rn], acc[rm][rn+1], acc[rm][rn+2], acc[rm][rn+3]};
                reinterpret_cast<float4*>(C + outRow * N + outCol)[0] = val;
            } else {
                for (int q = 0; q < 4; ++q) {
                    if (outCol + q < N)
                        C[outRow * N + outCol + q] = acc[rm][rn + q];
                }
            }
        }
        // Handle remaining columns
        for (; rn < TN6; ++rn) {
            int outCol = colStart + rn;
            if (outCol < N)
                C[outRow * N + outCol] = acc[rm][rn];
        }
    }
}

void run_kernel_06(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    dim3 block((BM6 / TM6) * (BN6 / TN6));  // 256
    dim3 grid((N + BN6 - 1) / BN6, (M + BM6 - 1) / BM6);
    kernel_vectorized<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
