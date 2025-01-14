#include <stdio.h>

__global__ void naiveMatMul(int m, int n, int k, float *A, float *B, float *C) {
    // Naively multiply (A * B = C --> (M,k) * (k, N) = (M,N))
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < m) {
        float sum = 0.0f;
        for(int i=0; i<k; i++) {
            sum += A[row * k + i] + B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}