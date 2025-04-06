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

/*
    V4 impove BLOCK_SIZE and more Tile to perform SGEMM:
    -> Outer loop tile A to blocks of size BM * K
    -> Outer loop tile B to blocks of size K * BN
    -> Inner loop tile A to blocks of size BM * BN
    -> Each thread block computes a matrix of C of size TM * TN

    // TODO: recompute
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
    uint tx = threadIdx.x;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;

    uint threadRow = tx / (BN / TN);
    uint threadCol = tx % (BN / TN);

    uint innerRowA = tx / BK;
    uint innerColA = tx % BK;
    uint innerRowB = tx / BN;
    uint innerColB = tx % BN;

    uint totalResBlockTile = BM * BN;
    uint numResBlockTile = totalResBlockTile / (TM * TN);
    uint strideA = numResBlockTile / BK;
    uint strideB = numResBlockTile / BN;

    float* A_ptr = (float*)A + BM * bx * K;
    float* B_ptr = (float*)B + BN * by;
    float* C_ptr = (float*)C + BM * bx * N + BN * by;

    // read a block of A and B to shared memory
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = &shared_mem[BM * BK];
    
    float threadRes[TM * TN] = {0.0f};

    for(int bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // Load A and B into shared memory
        for(int loadOff = 0; loadOff < BM; loadOff += strideA) {
            As[(innerRowA + loadOff) * BK + innerColA] = 
                A_ptr[(innerRowA + loadOff) * K + innerColA];
        }
        for(int loadOff = 0; loadOff < BK; loadOff += strideB) {
            Bs[(innerRowB + loadOff) * BN + innerColB] =
                B_ptr[(innerRowB + loadOff) * N + innerColB];
        }

        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        for(int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for(int resIdxN = 0; resIdxN < TN; resIdxN++) {
                for(int resIdxM = 0; resIdxM < TM; resIdxM++) {
                    threadRes[resIdxM * TN + resIdxN] += 
                        As[(threadRow * TM + resIdxM) * BK + dotIdx] * 
                            Bs[dotIdx * BN + threadCol * TN + resIdxN];
                }
            }
        }
        
        __syncthreads();
    }


    // Write the result to global memory
    for(int resIdxM = 0; resIdxM < TM; resIdxM++) {
        for(int resIdxN = 0; resIdxN < TN; resIdxN++) {
            C_ptr[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = 
                threadRes[resIdxM * TN + resIdxN];
        }
    }

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
    // Seed the random number generator with current time
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

    // Configure grid and block dimensions
    dim3 blockDim(BM * BN / (TM * TN));
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    size_t sharedMemSize = (BM * BK + BN * BK) * sizeof(float); // Shared memory size for A and B

    // Launch the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int r = 1; r < 5; r++) {
        cudaEventRecord(start);
        sgemm_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, M, N, K);
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