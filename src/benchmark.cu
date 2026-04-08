#include "benchmark.cuh"
#include "matmul.cuh"
#include "cublas_ref.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

static void fill_random(float* h, int n)
{
    for (int i = 0; i < n; ++i)
        h[i] = (float)rand() / RAND_MAX;
}

// ─────────────────────────────────────────────
// GPU info
// ─────────────────────────────────────────────

void print_gpu_info()
{
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("=== GPU Info ===\n");
    printf("  Device:            %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Memory:            %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Memory Bus Width:  %d-bit\n", prop.memoryBusWidth);
    printf("  Multiprocessors:   %d\n", prop.multiProcessorCount);
    printf("================\n\n");
}

// ─────────────────────────────────────────────
// Timing helper
// ─────────────────────────────────────────────

typedef void (*KernelFn)(float*, float*, float*, int, int, int);

static float time_kernel(KernelFn fn, float* dA, float* dB, float* dC,
                          int M, int N, int K,
                          int warmup, int iters)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < warmup; ++i)
        fn(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        fn(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

static float time_cublas(float* dA, float* dB, float* dC,
                          int M, int N, int K,
                          int warmup, int iters)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i)
        run_cublas(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        run_cublas(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

// ─────────────────────────────────────────────
// Correctness verification
// ─────────────────────────────────────────────

static bool verify(float* dRef, float* dTest, int M, int N, float tol)
{
    int sz = M * N;
    float* hRef  = (float*)malloc(sz * sizeof(float));
    float* hTest = (float*)malloc(sz * sizeof(float));

    CUDA_CHECK(cudaMemcpy(hRef,  dRef,  sz * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hTest, dTest, sz * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    float maxErr = 0.0f;
    float maxRef = 0.0f;

    for (int i = 0; i < sz; ++i) {
        float ref = hRef[i];
        float err = fabsf(hTest[i] - ref);
        if (fabsf(ref) > maxRef) maxRef = fabsf(ref);
        if (err > maxErr) maxErr = err;
    }

    // relative error
    float relErr = (maxRef > 1e-8f) ? maxErr / maxRef : maxErr;
    if (relErr > tol) {
        fprintf(stderr, "  VERIFICATION FAILED: max_rel_err=%.6f (tol=%.4f)\n",
                relErr, tol);
        ok = false;
    }

    free(hRef);
    free(hTest);
    return ok;
}

// ─────────────────────────────────────────────
// Main benchmark driver
// ─────────────────────────────────────────────

static const int VERIFY_SIZE = 256;
static const int WARMUP = 10;
static const int ITERS  = 100;

struct KernelEntry {
    const char* name;
    KernelFn    fn;
};

static const KernelEntry KERNELS[] = {
    {"naive",             run_kernel_01},
    {"global_coalesce",   run_kernel_02},
    {"shared_mem_tiling", run_kernel_03},
    {"1d_blocktiling",    run_kernel_04},
    {"2d_blocktiling",    run_kernel_05},
    {"vectorized",        run_kernel_06},
};
static const int NUM_KERNELS = sizeof(KERNELS) / sizeof(KERNELS[0]);

static const int SIZES[] = {512, 1024, 2048, 4096};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

void run_benchmark()
{
    cublas_init();
    print_gpu_info();

    printf("kernel,size,time_ms,gflops,pct_cublas\n");

    for (int si = 0; si < NUM_SIZES; ++si) {
        int SZ = SIZES[si];
        int M = SZ, N = SZ, K = SZ;
        long long flops = 2LL * M * N * K;

        // Allocate device memory
        float *dA, *dB, *dC_ref, *dC_test;
        CUDA_CHECK(cudaMalloc(&dA,      (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB,      (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC_ref,  (size_t)M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC_test, (size_t)M * N * sizeof(float)));

        // Initialize with random data on CPU, then copy to GPU
        float* hA = (float*)malloc((size_t)M * K * sizeof(float));
        float* hB = (float*)malloc((size_t)K * N * sizeof(float));
        fill_random(hA, M * K);
        fill_random(hB, K * N);
        CUDA_CHECK(cudaMemcpy(dA, hA, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));
        free(hA);
        free(hB);

        // cuBLAS reference timing
        float cublas_ms = time_cublas(dA, dB, dC_ref, M, N, K, WARMUP, ITERS);
        double cublas_gflops = (double)flops / (cublas_ms * 1e6);
        printf("cublas,%d,%.4f,%.2f,100.00%%\n", SZ, cublas_ms, cublas_gflops);

        // Run cuBLAS once more to get a clean reference for verification
        run_cublas(dA, dB, dC_ref, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Allocate smaller buffers for verification
        float *vdA, *vdB, *vdC_ref, *vdC_test;
        int VM = VERIFY_SIZE, VN = VERIFY_SIZE, VK = VERIFY_SIZE;
        CUDA_CHECK(cudaMalloc(&vdA,      (size_t)VM * VK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&vdB,      (size_t)VK * VN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&vdC_ref,  (size_t)VM * VN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&vdC_test, (size_t)VM * VN * sizeof(float)));

        float* vhA = (float*)malloc((size_t)VM * VK * sizeof(float));
        float* vhB = (float*)malloc((size_t)VK * VN * sizeof(float));
        fill_random(vhA, VM * VK);
        fill_random(vhB, VK * VN);
        CUDA_CHECK(cudaMemcpy(vdA, vhA, (size_t)VM * VK * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vdB, vhB, (size_t)VK * VN * sizeof(float), cudaMemcpyHostToDevice));
        free(vhA);
        free(vhB);

        // cuBLAS reference on verification size
        run_cublas(vdA, vdB, vdC_ref, VM, VN, VK);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Each custom kernel
        for (int ki = 0; ki < NUM_KERNELS; ++ki) {
            const KernelEntry& ke = KERNELS[ki];

            // Verify correctness at VERIFY_SIZE
            CUDA_CHECK(cudaMemset(vdC_test, 0, (size_t)VM * VN * sizeof(float)));
            ke.fn(vdA, vdB, vdC_test, VM, VN, VK);
            CUDA_CHECK(cudaDeviceSynchronize());

            bool pass = verify(vdC_ref, vdC_test, VM, VN, 1e-3f);
            fprintf(stderr, "  [%s @ %dx%d] Correctness: %s\n",
                    ke.name, SZ, SZ, pass ? "PASS" : "FAIL");

            if (!pass) {
                printf("%s,%d,NaN,NaN,NaN\n", ke.name, SZ);
                continue;
            }

            // Benchmark at full size
            CUDA_CHECK(cudaMemset(dC_test, 0, (size_t)M * N * sizeof(float)));
            float ms = time_kernel(ke.fn, dA, dB, dC_test, M, N, K, WARMUP, ITERS);
            double gflops = (double)flops / (ms * 1e6);
            double pct    = gflops / cublas_gflops * 100.0;

            printf("%s,%d,%.4f,%.2f,%.2f%%\n",
                   ke.name, SZ, ms, gflops, pct);
            fflush(stdout);
        }

        // Free
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC_ref));
        CUDA_CHECK(cudaFree(dC_test));
        CUDA_CHECK(cudaFree(vdA));
        CUDA_CHECK(cudaFree(vdB));
        CUDA_CHECK(cudaFree(vdC_ref));
        CUDA_CHECK(cudaFree(vdC_test));
    }

    cublas_destroy();
}
