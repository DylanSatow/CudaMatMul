#include <stdio.h>
#include <cuda_runtime.h>

__global__ void naiveGEMM(int m, int n, int k, float alpha, float beta, float *A, float *B, float *C) {
    // GEMM (GEneral Matrix Multiplication): C <- \alpha AB + \beta C
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < m) {
        float sum = 0.0f;
        for(int i=0; i<k; i++) {
            sum += A[row * k + i] + B[i * n + col];
        }
        C[row * n + col] = (alpha * sum) + C[row * n + col];
    }
}

