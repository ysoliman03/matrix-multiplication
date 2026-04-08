#include "cublas_ref.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

static cublasHandle_t g_handle = nullptr;

void cublas_init()
{
    if (g_handle) return;
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed: %d\n", status);
        exit(1);
    }
}

void cublas_destroy()
{
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

/*
 * cuBLAS uses column-major storage, but our matrices are row-major.
 * For row-major C = A * B, we exploit the identity:
 *   C^T = B^T * A^T
 * cuBLAS computes: C = alpha * op(A) * op(B) + beta * C
 * So we call: cublasSgemm with A and B swapped, both as CUBLAS_OP_N,
 * which computes:  C_col = B_col * A_col  (in column-major)
 * which in row-major is:  C_row = A_row * B_row  ✓
 *
 * Dimensions for the swapped call:
 *   "m" = N (rows of B^T = cols of B)
 *   "n" = M (cols of A^T = rows of A)
 *   "k" = K
 *   lda = N (leading dim of B in row-major = N)
 *   ldb = K (leading dim of A in row-major = K)
 *   ldc = N (leading dim of C in row-major = N)
 */
void run_cublas(float* d_A, float* d_B, float* d_C, int M, int N, int K)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t status = cublasSgemm(
        g_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,   // B is first arg (N x K in column-major view)
        d_A, K,   // A is second arg (K x M in column-major view)
        &beta,
        d_C, N    // C is (N x M in column-major) = (M x N row-major) ✓
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed: %d\n", status);
        exit(1);
    }
}
