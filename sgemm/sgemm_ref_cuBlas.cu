#include "params.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(x,y) ((x + y - 1) / y)
// #define ENABLE_CPU_GEMM

void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

int main() {
    // int M = 2048;
    // int N = 2048;
    // int K = 2048;

    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    // Allocate memory on the host
    float *A = (float *)malloc(sizeA);
    float *B = (float *)malloc(sizeB);

    float *C_cpu = (float *)malloc(sizeC);
    float *C_gpu = (float *)malloc(sizeC);

    if(A == NULL || B == NULL || C_cpu == NULL || C_gpu == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Initialize matrices A and B
    srand(time(NULL));
    
    // Initialize A with random values between -1 and 1
    for (int i = 0; i < M * K; i++) {
        A[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    }
    
    // Initialize B with random values between -1 and 1
    for (int i = 0; i < K * N; i++) {
        B[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);
    if (d_A == NULL || d_B == NULL || d_C == NULL) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return -1;
    }

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // Launch the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Init cuBlas
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create cuBLAS handle\n");
        return -1;
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    for(int i = 0; i < 3; i++) {
        cudaEventRecord(start);
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                    &alpha, d_A, M, d_B, K, &beta, d_C, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for GPU's SGEMM: %f ms\n", milliseconds);
    
    // Calculate throughput in GFLOPS
    double seconds = milliseconds / 1000.0;
    double operations = 2.0 * M * N * K;  // Multiply-add operations
    double throughput = operations / seconds / 1e9;
    printf("Throughput: %.2f GFLOPS\n", throughput);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Get the result back to the host
    cudaMemcpy(C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);

#ifdef ENABLE_CPU_GEMM
    // Perform CPU computation for verification
    clock_t start_cpu, end_cpu;
    start_cpu = clock();
    cpu_gemm(A, B, C_cpu, M, N, K);
    end_cpu = clock();

    double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("Time taken for CPU's SGEMM: %f seconds\n", cpu_time);
    
    // Calculate CPU throughput in GFLOPS
    // double operations = 2.0 * M * N * K;  // Multiply-add operations
    double cpu_throughput = operations / cpu_time / 1e9;
    printf("CPU Throughput: %.2f GFLOPS\n", cpu_throughput);


    // Verify the result
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-3) {
            fprintf(stderr, "Mismatch at index %d: CPU = %f, GPU = %f\n", i, C_cpu[i], C_gpu[i]);
            break;
        }
    }
#endif
    printf("SGEMM computation completed successfully.\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Free host memory
    free(A);
    free(B);

    free(C_cpu);
    free(C_gpu);

    cublasDestroy(handle);

    // Reset the device and exit
    cudaDeviceReset();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;

}