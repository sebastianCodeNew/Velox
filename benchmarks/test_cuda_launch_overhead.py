"""
Benchmark Suite for Challenger 2: Hardware & CUDA Kernel Execution
Empirical verification of:
1. CUDA Driver launch overhead and serialization across N launches.
2. GPU execution time vs CPU submission time for micro-kernels (simulating 80-layer Transformer decode steps).
3. Host-bound GPU starvation bubbles when kernel compute duration < CPU launch latency.
"""

import time
import numpy as np
from numba import cuda

@cuda.jit
def dummy_norm_kernel(x, y, eps, n):
    """Simulates a lightweight memory-bandwidth bound layer norm / RMSNorm kernel (B=1, D=4096)"""
    idx = cuda.grid(1)
    if idx < n:
        # RMSNorm elementwise operation
        val = x[idx]
        y[idx] = val * 0.5 + eps

@cuda.jit
def dummy_gemv_kernel(A, x, y, M, N):
    """Simulates a small GEMV or projection kernel for decode step B=1"""
    row = cuda.grid(1)
    if row < M:
        acc = 0.0
        for col in range(N):
            acc += A[row, col] * x[col]
        y[row] = acc

def benchmark_cuda_launch_overhead():
    print("=" * 70)
    print("CHALLENGE 1: CUDA Driver Launch Overhead & Serialization Test")
    print("=" * 70)
    
    device = cuda.get_current_device()
    print(f"Active Device: {device.name}")
    print(f"Compute Capability: {device.compute_capability}")
    
    # Setup small tensors simulating B=1 decode step (D=4096)
    D = 4096
    h_x = np.ones(D, dtype=np.float32)
    h_y = np.zeros(D, dtype=np.float32)
    
    d_x = cuda.to_device(h_x)
    d_y = cuda.to_device(h_y)
    
    threadsperblock = 256
    blockspergrid = (D + threadsperblock - 1) // threadsperblock
    
    # Warmup
    for _ in range(50):
        dummy_norm_kernel[blockspergrid, threadsperblock](d_x, d_y, 1e-5, D)
    cuda.synchronize()
    
    # Test varying number of kernel launches: 10, 100, 400, 800 (representing 1 to 80 layers)
    launch_counts = [10, 100, 400, 800]
    
    for num_launches in launch_counts:
        # Measure CPU submission time (time to issue all kernel launches without waiting for GPU)
        t_sub_start = time.perf_counter()
        for _ in range(num_launches):
            dummy_norm_kernel[blockspergrid, threadsperblock](d_x, d_y, 1e-5, D)
        t_sub_end = time.perf_counter()
        
        # Measure total time until GPU completes all launches
        cuda.synchronize()
        t_sync_end = time.perf_counter()
        
        cpu_submission_total_ms = (t_sub_end - t_sub_start) * 1000.0
        cpu_sub_per_kernel_us = (cpu_submission_total_ms * 1000.0) / num_launches
        total_wallclock_ms = (t_sync_end - t_sub_start) * 1000.0
        
        print(f"Launches: {num_launches:3d} | "
              f"CPU Submission Total: {cpu_submission_total_ms:6.2f} ms | "
              f"Per-Kernel CPU Launch: {cpu_sub_per_kernel_us:5.2f} us | "
              f"Total Wallclock (Sync): {total_wallclock_ms:6.2f} ms")
        
    print("-" * 70)
    print("Measuring isolated single-kernel GPU execution time via CUDA Events...")
    
    start_event = cuda.event()
    end_event = cuda.event()
    
    # Run 100 individual iterations with CUDA Events to measure GPU-only kernel execution time
    gpu_times_ms = []
    for _ in range(100):
        start_event.record()
        dummy_norm_kernel[blockspergrid, threadsperblock](d_x, d_y, 1e-5, D)
        end_event.record()
        end_event.synchronize()
        gpu_times_ms.append(cuda.event_elapsed_time(start_event, end_event))
        
    mean_gpu_kernel_us = np.mean(gpu_times_ms) * 1000.0
    print(f"Individual Kernel GPU Execution Time: {mean_gpu_kernel_us:.2f} us")
    print(f"For 800 kernels, Pure GPU Execution: {mean_gpu_kernel_us * 800 / 1000.0:.2f} ms")
    print("=" * 70)

if __name__ == "__main__":
    benchmark_cuda_launch_overhead()
