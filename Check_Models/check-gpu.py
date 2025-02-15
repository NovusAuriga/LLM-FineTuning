#!/usr/bin/env python3
import torch
import time
import numpy

n = 4096

for i in range(100):
# Create random matrices on GPU
    A = torch.randn((n, n), device="cuda", dtype=torch.float16)
    B = torch.randn((n, n), device="cuda", dtype=torch.float16)

    # Warm-up (optional): run a dummy matmul to “warm up” the GPU
    #_ = torch.matmul(A, B)  
    # Synchronize before timing
    torch.cuda.synchronize()  

    # Start the monotonic clock
    start_time = time.monotonic()  

    # Actual multiplication
    C = torch.matmul(A, B)

    # Ensure all CUDA kernels have completed
    torch.cuda.synchronize()  

    end_time = time.monotonic()
    elapsed = end_time - start_time

    # Compute number of floating-point operations:
    # For an (n x n) x (n x n), that’s 2 * n^3 FLOPs (multiplications + additions).
    flops = 2.0 * (n ** 3)

    # Convert to TFLOPS: (flops / elapsed_time) / 1e12
    tflops = (flops / elapsed) / 1e12

    print(f"Throughput:   {tflops:.3f} TFLOPS")
