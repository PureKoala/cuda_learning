#ifndef PARAMS_H
#define PARAMS_H

// Global matrix dimension parameters for SGEMM (Single-precision General Matrix Multiply)
// For operation C = A * B
// A is an M x K matrix
// B is a K x N matrix
// C is an M x N matrix

const int M = 4096; // Number of rows in matrices A and C
const int K = 4096; // Number of columns in matrix A / rows in matrix B
const int N = 4096; // Number of columns in matrices B and C

// Block size for tiling
const int BLOCK_SIZE = 32; // Size of the square blocks used in the tiled matrix multiplication

const int BM = 64;
const int BN = 64;
const int BK = 8;
const int TM = 8;

#endif // PARAMS_H