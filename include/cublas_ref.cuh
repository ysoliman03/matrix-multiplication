#pragma once

void cublas_init();
void cublas_destroy();
void run_cublas(float* d_A, float* d_B, float* d_C, int M, int N, int K);
