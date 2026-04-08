cmake_minimum_required(VERSION 3.18)
project(matmul_optimization CUDA CXX)

# CUDA architectures: Turing (75), Ampere (80, 86), Ada (89), Hopper (90)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -allow-unsupported-compiler")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(CUDAToolkit REQUIRED)

set(KERNEL_SOURCES
    src/kernels/01_naive.cu
    src/kernels/02_global_mem_coalesce.cu
    src/kernels/03_shared_mem_tiling.cu
    src/kernels/04_1d_blocktiling.cu
    src/kernels/05_2d_blocktiling.cu
    src/kernels/06_vectorized.cu
)

add_executable(matmul_benchmark
    src/main.cu
    src/benchmark.cu
    src/cublas_ref.cu
    ${KERNEL_SOURCES}
)

target_include_directories(matmul_benchmark PRIVATE include)
target_link_libraries(matmul_benchmark PRIVATE CUDA::cublas CUDA::cudart)
set_target_properties(matmul_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
