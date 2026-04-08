# CUDA Matrix Multiplication — From Naive to 43% of cuBLAS

Hand-written CUDA kernels optimized across 6 stages, benchmarked against NVIDIA's cuBLAS on real hardware. No CUTLASS, no template libraries, every kernel written from scratch to demonstrate deep understanding of GPU memory hierarchy, warp execution, and arithmetic intensity.

**Tech:** CUDA C++ · cuBLAS · CMake · Python · RTX 3060

---

## Results at a Glance

| Kernel | 4096×4096 | vs cuBLAS |
|---|---|---|
| 1. Naive | 429.5 GFLOPS | 9.8% |
| 2. Global Coalescing | 429.3 GFLOPS | 9.8% |
| 3. Shared Memory Tiling | 358.3 GFLOPS | 8.1% |
| 4. 1D Block Tiling | 646.5 GFLOPS | 14.7% |
| 5. 2D Block Tiling | 2,186.0 GFLOPS | 49.7% |
| 6. Vectorized (float4) | 1,899.6 GFLOPS | 43.2% |
| **cuBLAS (reference)** | **4,400.4 GFLOPS** | **100%** |

> Benchmarked on RTX 3060 Laptop GPU · CUDA 13.2 · April 2026

Full results across all sizes (512 to 4096): [results/benchmark_results.md](results/benchmark_results.md)

---

## What This Project Demonstrates

- **GPU memory hierarchy** — global memory coalescing, shared memory tiling, register file reuse
- **Warp-level thinking** — thread-to-data mapping, bank conflict avoidance, occupancy tradeoffs
- **Arithmetic intensity** — progressing from 1 MAC/load (naive) to 4 MACs/load (2D tiling)
- **Profiling intuition** — identifying *why* shared memory underperforms on L2-heavy workloads, and why float4 regresses when boundary checks add branching
- **Production tooling** — CUDA events for timing, cuBLAS as reference, automated correctness verification against a tolerance of 1e-3 relative error

---

## The Optimization Journey

### Stage 1 — Naive `01_naive.cu`

One thread per output element. Baseline for everything that follows. The bottleneck is uncoalesced memory: adjacent threads in a warp map to different rows of A, reading elements N floats apart — 32 separate cache-line transactions per warp.

```cuda
int row = blockIdx.y * 32 + threadIdx.y;
int col = blockIdx.x * 32 + threadIdx.x;
float sum = 0.f;
for (int k = 0; k < K; ++k)
    sum += A[row * K + k] * B[k * N + col];
C[row * N + col] = sum;
```

---

### Stage 2 — Global Memory Coalescing `02_global_mem_coalesce.cu`

One-line fix: swap `threadIdx.x` to map to the column dimension instead of the row. Adjacent threads now read adjacent columns of B and C — one 128-byte transaction per warp.

```cuda
int col = blockIdx.x * 32 + threadIdx.x;  // threadIdx.x -> col (was row)
int row = blockIdx.y * 32 + threadIdx.y;
```

**Observed result:** No speedup on RTX 3060 — the L2 cache was already absorbing the uncoalesced reads. Architecture-specific insight: not every textbook optimization helps on every GPU.

---

### Stage 3 — Shared Memory Tiling `03_shared_mem_tiling.cu`

Cooperative loading of 32×32 tiles into on-chip SRAM (≈1–2 cycle latency vs ≈200 cycles for global memory). Each element is loaded once and reused 32 times, cutting global traffic 32×. Padding (+1 column) prevents shared memory bank conflicts.

```cuda
__shared__ float As[32][33];  // +1 pad eliminates bank conflicts on column reads
__shared__ float Bs[32][33];
for (int t = 0; t < numTiles; ++t) {
    As[ty][tx] = A[row * K + t*32 + tx];
    Bs[ty][tx] = B[(t*32 + ty) * N + col];
    __syncthreads();
    for (int i = 0; i < 32; ++i) sum += As[ty][i] * Bs[i][tx];
    __syncthreads();
}
```

**Observed result:** 0.83× — slower than naive. 1024 threads/block reduces occupancy; `__syncthreads()` stalls dominate; the L2 cache made this redundant on this workload. A lesson in profiling before optimizing.

---

### Stage 4 — 1D Block Tiling `04_1d_blocktiling.cu`

Thread coarsening: each thread computes 8 output elements (a vertical strip) instead of 1. Eight register accumulators amortize the cost of loading B values across 8 MACs — better arithmetic intensity, fewer total threads, lower scheduling overhead.

```cuda
float acc[8] = {0.f};
for (int k = 0; k < BK; ++k) {
    float bVal = Bs[k][threadCol];       // one load
    for (int rm = 0; rm < 8; ++rm)       // 8 MACs
        acc[rm] += As[threadRow*8 + rm][k] * bVal;
}
```

**Speedup over Stage 3:** 1.80× at 4096×4096

---

### Stage 5 — 2D Block Tiling `05_2d_blocktiling.cu`

The key insight of the project. Each thread computes an 8×8 = 64-element sub-block via outer product: load 8 values from A and 8 from B into registers, update 64 accumulators. Arithmetic intensity = 64 MACs / 16 loads = 4 ops/load.

```cuda
float regA[8], regB[8];
for (int k = 0; k < BK; ++k) {
    for (int rm = 0; rm < 8; ++rm) regA[rm] = As[threadRow*8 + rm][k];
    for (int rn = 0; rn < 8; ++rn) regB[rn] = Bs[k][threadCol*8 + rn];
    for (int rm = 0; rm < 8; ++rm)
        for (int rn = 0; rn < 8; ++rn)
            acc[rm][rn] += regA[rm] * regB[rn];  // 64-element outer product
}
```

**Speedup over Stage 4:** 3.38× — the single biggest jump. **Reaches 49.7% of cuBLAS.**

---

### Stage 6 — Vectorized Memory Access `06_vectorized.cu`

Replaces 32-bit scalar loads with 128-bit `float4` vector loads — 4 floats per instruction, 4× fewer load instructions, lower instruction-issue pressure.

```cuda
// 4 instructions -> 1 instruction, 4 floats
float4 v = reinterpret_cast<const float4*>(A + gRow * K + gCol)[0];
As[r][c]=v.x;  As[r][c+1]=v.y;  As[r][c+2]=v.z;  As[r][c+3]=v.w;
```

**Observed result:** 0.87× regression. Boundary-check fallback paths for non-aligned addresses add branching overhead that outweighs the wider loads on this workload. Documents a real performance pitfall.

---

## Skills Demonstrated

| Area | Specifics |
|---|---|
| CUDA C++ | Custom kernels, shared memory, warp semantics, `float4` intrinsics |
| Performance analysis | CUDA events, GFLOPS measurement, arithmetic intensity calculation |
| Systems thinking | Occupancy vs. ILP tradeoffs, L2 cache behavior, memory alignment |
| Build tooling | CMake with multi-arch CUDA, cuBLAS linking, Ninja generator |
| Automation | Python report generator parsing benchmark CSV output |

---

## Build & Run

**Prerequisites:** CUDA Toolkit 11.8+, CMake 3.18+, GPU with compute capability 7.0+

```powershell
# Windows (PowerShell)
mkdir build; cd build
cmake .. -G Ninja -DCMAKE_CUDA_COMPILER="<path_to_nvcc>"
cmake --build . --config Release
.\matmul_benchmark.exe | python ..\scripts\generate_report.py
```

```bash
# Linux / WSL
mkdir build && cd build && cmake .. && make -j$(nproc)
./matmul_benchmark | python3 ../scripts/generate_report.py
```

Report is written to `results/benchmark_results.md`.

---

## Repository Layout

```
├── include/
│   └── matmul.cuh                  kernel declarations
├── src/
│   ├── kernels/
│   │   ├── 01_naive.cu
│   │   ├── 02_global_mem_coalesce.cu
│   │   ├── 03_shared_mem_tiling.cu
│   │   ├── 04_1d_blocktiling.cu
│   │   ├── 05_2d_blocktiling.cu
│   │   └── 06_vectorized.cu
│   ├── benchmark.cu                timing harness + correctness verification
│   ├── cublas_ref.cu               cuBLAS reference wrapper
│   └── main.cu
├── scripts/
│   └── generate_report.py          benchmark CSV -> markdown report
├── results/
│   └── benchmark_results.md        auto-generated performance report
└── CMakeLists.txt
```
