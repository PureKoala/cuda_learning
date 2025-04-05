#include "params.h"
#include <stdio.h>
#include <cuda_runtime.h>
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CEIL_DIV(x,y) ((x + y - 1) / y)
// #define ENABLE_CPU_GEMM

/*
    V2 using shared memory to perform SGEMM:
    -> Spilt A to blocks of size bm * K
    -> Spilt B to blocks of size K * bn
    -> Each thread block computes a block of C of size bm * bn

    -> Read  Count: bm * K * (N / bn) * (M / bm) + K * bn * (M / bm) * (N / bn) = KMN*(1/bm + 1/bn)
    -> Write Count: bm * bn * (M / bm) * (N / bn) = MN
*/

__global__ void sgemm_kernel(
    const float *A,
    const float *B,
    float *C,
    const int M,
    const int N,
    const int K
) {
    // int bm = blockDim.x; // Block size for A
    // int bn = blockDim.y; // Block size for B

    // Calculate the row and column index of the element to be computed
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int threadRow = tx / BLOCK_SIZE;
    int threadCol = tx % BLOCK_SIZE;

    float* A_ptr = (float*)A + BLOCK_SIZE * bx * K;
    float* B_ptr = (float*)B + BLOCK_SIZE * by;
    float* C_ptr = (float*)C + BLOCK_SIZE * bx * N + BLOCK_SIZE * by;

    // read a block of A and B to shared memory
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = &shared_mem[BLOCK_SIZE * BLOCK_SIZE];
    
    float tmp = 0.0f;
    for(int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
        // Load A and B into shared memory
        As[threadRow * BLOCK_SIZE + threadCol] = A_ptr[threadRow * K + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B_ptr[threadRow * N + threadCol];
        __syncthreads();

        A_ptr += BLOCK_SIZE;
        B_ptr += BLOCK_SIZE * N;

        for(int dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++) {
            tmp += As[threadRow * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + threadCol];
        }
        __syncthreads();
    }


    // Write the result to global memory
    C_ptr[threadRow * N + threadCol] = tmp;

}

#ifdef ENABLE_CPU_GEMM
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
#endif

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
    int randMax = 1000;
    // Seed the random number generator
    srand(time(NULL));
    
    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; i++) {
        A[i] = static_cast<float>(rand() % randMax);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = static_cast<float>(rand() % randMax);
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

    // Configure grid and block dimensions
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    size_t sharedMemSize = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float); // Shared memory size for A and B

    // Launch the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, M, N, K);
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