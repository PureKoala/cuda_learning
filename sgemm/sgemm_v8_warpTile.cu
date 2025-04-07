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
    V8 using BlockTile / WarpTile / ThreadTile to perform SGEMM:
    -> BlockTile totally compute A(BM * K) and B(K * BN)
       -> Load A and B from GMEM into SMEM
       -> Loop over K to compute C(BM * BN)
       -> Using a single SM
       -> Each loop use A(BM * BK) and B(BK * BN)
    -> WarpTile totally compute A(BM * BK) and B(BK * BN)
       -> Load A and B from SMEM into warp-scheduler local registers
       -> Loop over BK to compute C(WM * WN) for each warp
       -> Using a single warp
    -> ThreadTile totally compute A(WM * BK) and B(BK * WN)
       -> Load A and B from warp-scheduler local registers into thread local registers
       -> Inner loop count is WMITER and WNITER


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
    

    // Calculate the row and column index of the element to be computed
    uint tx = threadIdx.x;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;

    // Put warp in BlockTile
    uint warpIdx = tx / 32;
    uint warpRow = warpIdx / (V8_BN / V8_WN);
    uint warpCol = warpIdx % (V8_BN / V8_WN);

    // Size of warp subtile
    constexpr uint V8_WMITER = (V8_WM * V8_WN) / (V8_WNITER * V8_TM * V8_TN * 32);
    constexpr uint WSUBM = V8_WM / V8_WMITER;
    constexpr uint WSUBN = V8_WN / V8_WNITER;

    // Put thread in warp subtile
    const uint threadIdxInWarp = tx % 32;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / V8_TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / V8_TN);

    // vector load params
    const uint innerRowA = tx / (V8_BK / 4);
    const uint innerColA = tx % (V8_BK / 4);
    const uint innerRowB = tx / (V8_BN / 4);
    const uint innerColB = tx % (V8_BN / 4);
    constexpr uint strideA = (4 * V8_NUM_THREADS) / V8_BK;
    constexpr uint strideB = (4 * V8_NUM_THREADS) / V8_BN;

    float* A_ptr = (float*)A + V8_BM * bx * K;
    float* B_ptr = (float*)B + V8_BN * by;
    float* C_ptr = (float*)C + (V8_BM * bx + V8_WM * warpRow) * N + V8_BN * by + V8_WN * warpCol;

    // read a block of A and B to shared memory
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = &shared_mem[V8_BM * V8_BK];
    
    float threadRes[V8_TM * V8_TN * V8_WMITER * V8_WNITER] = {0.0f};
    float A_reg[V8_TM * V8_WMITER] = {0.0f};
    float B_reg[V8_TN * V8_WNITER] = {0.0f};

    for(int bkIdx = 0; bkIdx < K; bkIdx += V8_BK) {

        // Load A and B into shared memory
        // Load A into shared memory using strideA
        for (int i = 0; i < V8_BM; i += strideA) {
            // if (innerRowA + i < V8_BM && innerColA * 4 < V8_BK && 
            //     bkIdx + innerColA * 4 < K && // Add bound check for K dimension
            //     V8_BM * bx + innerRowA + i < M) { // Add bound check for M dimension
                float4 tmp = reinterpret_cast<const float4*>(
                    &A_ptr[(innerRowA + i) * K + innerColA * 4]
                )[0];

                As[(innerColA * 4 + 0) * V8_BM + (innerRowA + i)] = tmp.x;
                As[(innerColA * 4 + 1) * V8_BM + (innerRowA + i)] = tmp.y;
                As[(innerColA * 4 + 2) * V8_BM + (innerRowA + i)] = tmp.z;
                As[(innerColA * 4 + 3) * V8_BM + (innerRowA + i)] = tmp.w;
            // }
        }

        // Load B into shared memory using strideB
        for (int i = 0; i < V8_BK; i += strideB) {
            // if (innerRowB + i < V8_BK && innerColB * 4 < V8_BN && 
            //     bkIdx + innerRowB + i < K && // Add bound check for K dimension
            //     V8_BN * by + innerColB * 4 < N) { // Add bound check for N dimension
                reinterpret_cast<float4*>(&Bs[(innerRowB + i) * V8_BN + innerColB * 4])[0] = 
                    reinterpret_cast<const float4*>(
                    &B_ptr[(innerRowB + i) * N + innerColB * 4]
                    )[0];
            // }
        }

        __syncthreads();

        A_ptr += V8_BK;
        B_ptr += V8_BK * N;

        for(int dotIdx = 0; dotIdx < V8_BK; dotIdx++) {

            // Load shared memory data into registers first with boundary checks
            for(int mIter = 0; mIter < V8_WMITER; mIter++) {
                for(int i = 0; i < V8_TM; i++) {
                    int a_idx = dotIdx * V8_BM + warpRow * V8_WM + mIter * WSUBM + threadRowInWarp * V8_TM + i;
                    // if (a_idx < V8_BM * V8_BK) { // Ensure index is within shared memory bounds
                        A_reg[mIter * V8_TM + i] = As[a_idx];
                    // }
                }
            }

            for(int nIter = 0; nIter < V8_WNITER; nIter++) {
                for(int i = 0; i < V8_TN; i++) {
                    int b_idx = dotIdx * V8_BN + warpCol * V8_WN + nIter * WSUBN + threadColInWarp * V8_TN + i;
                    // if (b_idx < V8_BK * V8_BN) { // Ensure index is within shared memory bounds
                        B_reg[nIter * V8_TN + i] = Bs[b_idx];
                    // }
                }
            }

            for(int mIter = 0; mIter < V8_WMITER; mIter++) {
                for(int nIter = 0; nIter < V8_WNITER; nIter++) {
                    for(int resIdxM = 0; resIdxM < V8_TM; resIdxM++) {
                        for(int resIdxN = 0; resIdxN < V8_TN; resIdxN++) {
                            threadRes[mIter * V8_WNITER * V8_TN * V8_TM + nIter * V8_TN * V8_TM + resIdxM * V8_TN + resIdxN] += 
                                A_reg[mIter * V8_TM + resIdxM] * B_reg[nIter * V8_TN + resIdxN];
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }


    // Write the result to global memory with boundary checks
    for(int mIter = 0; mIter < V8_WMITER; mIter++) {
        for(int nIter = 0; nIter < V8_WNITER; nIter++) {
            for(int resIdxM = 0; resIdxM < V8_TM; resIdxM++) {
                // Check if we're within the matrix bounds for M dimension
                // if (V8_BM * bx + V8_WM * warpRow + mIter * WSUBM + threadRowInWarp * V8_TM + resIdxM < M) {
                    for(int resIdxN = 0; resIdxN < V8_TN; resIdxN += 4) {
                        // Check if we're within the matrix bounds for N dimension
                        // if (V8_BN * by + V8_WN * warpCol + nIter * WSUBN + threadColInWarp * V8_TN + resIdxN + 3 < N) {
                            reinterpret_cast<float4*>(
                                &C_ptr[(mIter * WSUBM + threadRowInWarp * V8_TM + resIdxM) * N + 
                                    (nIter * WSUBN + threadColInWarp * V8_TN + resIdxN)]
                            )[0] = 
                                reinterpret_cast<float4*>(
                                    &threadRes[mIter * V8_WNITER * V8_TN * V8_TM + nIter * V8_TN * V8_TM +
                                        resIdxM * V8_TN + resIdxN]
                                )[0];
                        // }
                    }
                // } else {
                //     printf("Thread out of bounds for block.x: %d, block.y: %d, mIter: %d, nIter: %d, resIdxM: %d\n", 
                //         bx, by, mIter, nIter, resIdxM);
                // }
            }
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
    
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount returned error %s (code %d)\n", 
                cudaGetErrorString(error), error);
        return -1;
    }
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices detected\n");
        return -1;
    }
    
    // Reset the device to clear any previous errors
    cudaDeviceReset();
    
    // Select the first device
    cudaSetDevice(0);
    
    // Check if the selected device is available
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using GPU: %s\n", deviceProp.name);
    
    // Check maximum shared memory per block
    size_t sharedMemSize = (V8_BM * V8_BK + V8_BN * V8_BK) * sizeof(float);
    printf("Required shared memory: %zu bytes\n", sharedMemSize);
    printf("Maximum shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        fprintf(stderr, "Error: Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n", 
                sharedMemSize, deviceProp.sharedMemPerBlock);
        return -1;
    }
    
    // Calculate and display register usage estimate (rough estimate)
    constexpr uint V8_WMITER = (V8_WM * V8_WN) / (V8_WNITER * V8_TM * V8_TN * 32);
    int estimatedRegistersPerThread = V8_TM * V8_TN * V8_WMITER * V8_WNITER + 
                                      V8_TM * V8_WMITER + 
                                      V8_TN * V8_WNITER + 10; // +10 for other variables
    printf("Estimated registers per thread: %d\n", estimatedRegistersPerThread);
    printf("Maximum registers per thread: %d\n", deviceProp.regsPerBlock / V8_NUM_THREADS);
    
    if (estimatedRegistersPerThread * V8_NUM_THREADS > deviceProp.regsPerBlock) {
        fprintf(stderr, "Warning: Estimated register usage may exceed device capabilities\n");
    }

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
    dim3 blockDim(V8_NUM_THREADS);
    dim3 gridDim(CEIL_DIV(M, V8_BM), CEIL_DIV(N, V8_BN)); // Fixed: was incorrectly using V8_BM for both dimensions
    // size_t sharedMemSize = (V8_BM * V8_BK + V8_BN * V8_BK) * sizeof(float); // Shared memory size for A and B

    // Launch the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up run
    sgemm_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, M, N, K);
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(kernelError));
        return -1;
    }
    cudaDeviceSynchronize();
    
    // Actual timed run
    cudaEventRecord(start);
    sgemm_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, M, N, K);
    kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(kernelError));
        return -1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for GPU's SGEMM: %f ms\n", milliseconds);
    
    if (milliseconds < 0.001) {
        fprintf(stderr, "Warning: Kernel execution time suspiciously low. Check for silent failures.\n");
    }
    
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

    return 0;

}