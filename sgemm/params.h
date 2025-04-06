#ifndef PARAMS_H
#define PARAMS_H

// #define ENABLE_CPU_GEMM

// Global matrix dimension parameters for SGEMM (Single-precision General Matrix Multiply)
// For operation C = A * B
// A is an M x K matrix
// B is a K x N matrix
// C is an M x N matrix

#ifndef ENABLE_CPU_GEMM
    const uint M = 4096; // Number of rows in matrices A and C
    const uint K = 4096; // Number of columns in matrix A / rows in matrix B
    const uint N = 4096; // Number of columns in matrices B and C
#else
    const uint M = 1024; // Number of rows in matrices A and C
    const uint K = 1024; // Number of columns in matrix A / rows in matrix B
    const uint N = 1024; // Number of columns in matrices B and C
#endif

// Block size for tiling
const uint BLOCK_SIZE = 32; // Size of the square blocks used in the tiled matrix multiplication

const uint BM = 128;
const uint BN = 128;
const uint BK = 8;
const uint TM = 8;
const uint TN = 8;

// Params for V8
const uint V8_NUM_THREADS = 128;
constexpr uint V8_NUM_WARPS = V8_NUM_THREADS / 32;
const uint V8_BM = 128;
const uint V8_BN = 128;
const uint V8_BK = 16;
const uint V8_WM = 64;
const uint V8_WN = 64;
const uint V8_WNITER = 4;
const uint V8_TM = 8;
const uint V8_TN = 4;


#endif // PARAMS_H