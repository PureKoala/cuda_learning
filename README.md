# cuda_learning

## Test results

### SGEMM Performance Results
| Program Type | Throughput (GFLOPS) | Throughput Percentage (%) | Execution Time (ms) |
|:------------:|:-------------------:|:-------------------------:|:-------------------:|
| cuBlas Ref | 18848.16 | 100.00 | 7.291904 |
| GMEM | 289.02 | 1.53 | 475.536743 |
| GMEM Coalesce | 3046.64 | 16.16 | 45.111713 |
| SMEM | 5364.99 | 28.46 | 25.617760 |
| 1D BlockTiling (BM=BN=64) | 9955.72 | 52.82 | 13.805024 |
| 2D BlockTiling (BM=BN=128) | 11079.56 | 58.78 | 12.404736 |
| 2D BlockTiling + Regfile (BM=BN=128) | 11047.64 | 58.61 | 12.440576 |
| Transpose & Vectorization | 15043.46 | 79.81 | 9.136128 |



#### Configures
- Matrix dimensions: M = N = K = 4096
- BLOCK_SIZR: 32
