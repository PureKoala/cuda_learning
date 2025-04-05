# cuda_learning

## Test results

### SGEMM Performance Results

| Program Type | Throughput (GFLOPS) | Execution Time (ms) |
|--------------|---------------------|---------------------|
| cuBlas Red | 18848.16 | 7.291904 |
| GMEM | 289.02 | 475.536743 |
| GMEM Coalesce | 3046.64 | 45.111713 |
| SMEM | 5364.99 | 25.617760 |
| 1D BlockTiling | 9955.72 | 13.805024 |
| CUDA Register Blocking | | |
| CUDA Fully Optimized | | |

## Notes
- Matrix dimensions: M = N = K = 4096
- BLOCK_SIZR: 32
