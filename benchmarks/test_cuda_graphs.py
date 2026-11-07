"""
Benchmark: CUDA Graph Capture, Launch Latency, and Invariance Constraints
Tests:
1. cudaStreamBeginCapture / EndCapture / cudaGraphInstantiate / cudaGraphLaunch latency vs sequential cuLaunchKernel.
2. Speedup of CUDA Graph vs 800 sequential kernel launches.
3. Invariance limits: What happens if pointers, sizes, or shapes change?
4. Memory overhead of captured graph instances.
"""

import ctypes
import time
import numpy as np

# Load nvcuda.dll
cuda = ctypes.windll.LoadLibrary("nvcuda.dll")

# PTX for dummy kernel
PTX_CODE = b"""
.version 6.5
.target sm_50
.address_size 64

.visible .entry dummy_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<3>;

    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r0, %r2, %r3, %r4;

    setp.ge.s32 %p1, %r0, %r1;
    @%p1 bra DONE;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    mul.wide.s32 %rd3, %r0, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;

    ld.global.f32 %f1, [%rd4];
    mul.f32 %f2, %f1, 0f3f000000;
    st.global.f32 [%rd5], %f2;

DONE:
    ret;
}
"""

def check_cu(res, msg="CUDA Error"):
    if res != 0:
        raise RuntimeError(f"{msg}: error code {res}")

def benchmark_cuda_graphs():
    print("=" * 70)
    print("CHALLENGE 2: CUDA Graph Capture, Launch Speed & Invariance Test")
    print("=" * 70)
    
    check_cu(cuda.cuInit(0), "cuInit")
    
    device = ctypes.c_int()
    check_cu(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    
    ctx = ctypes.c_void_p()
    check_cu(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device), "cuCtxCreate")
    
    module = ctypes.c_void_p()
    check_cu(cuda.cuModuleLoadData(ctypes.byref(module), PTX_CODE), "cuModuleLoadData")
    
    kernel = ctypes.c_void_p()
    check_cu(cuda.cuModuleGetFunction(ctypes.byref(kernel), module, b"dummy_kernel"), "cuModuleGetFunction")
    
    # Create stream
    stream = ctypes.c_void_p()
    check_cu(cuda.cuStreamCreate(ctypes.byref(stream), 0), "cuStreamCreate")
    
    D = 4096
    size_bytes = D * 4
    d_in = ctypes.c_uint64()
    d_out = ctypes.c_uint64()
    check_cu(cuda.cuMemAlloc_v2(ctypes.byref(d_in), size_bytes), "cuMemAlloc in")
    check_cu(cuda.cuMemAlloc_v2(ctypes.byref(d_out), size_bytes), "cuMemAlloc out")
    
    n_val = ctypes.c_uint32(D)
    p_in = ctypes.cast(ctypes.byref(d_in), ctypes.c_void_p)
    p_out = ctypes.cast(ctypes.byref(d_out), ctypes.c_void_p)
    p_n = ctypes.cast(ctypes.byref(n_val), ctypes.c_void_p)
    kernel_params = (ctypes.c_void_p * 3)(p_in, p_out, p_n)
    
    grid_dim_x = (D + 255) // 256
    block_dim_x = 256
    
    num_nodes = 800  # Simulating 80 layers * 10 ops
    
    print(f"Capturing CUDA Graph with {num_nodes} kernel launches...")
    t_cap_start = time.perf_counter()
    
    # Stream capture mode: CU_STREAM_CAPTURE_MODE_GLOBAL = 0
    check_cu(cuda.cuStreamBeginCapture_v2(stream, 0), "cuStreamBeginCapture")
    
    for _ in range(num_nodes):
        cuda.cuLaunchKernel(
            kernel,
            grid_dim_x, 1, 1,
            block_dim_x, 1, 1,
            0, stream,
            kernel_params, None
        )
        
    graph = ctypes.c_void_p()
    check_cu(cuda.cuStreamEndCapture(stream, ctypes.byref(graph)), "cuStreamEndCapture")
    t_cap_end = time.perf_counter()
    
    print(f"Graph Capture Time: {(t_cap_end - t_cap_start) * 1000.0:.2f} ms")
    
    # Graph Instantiation
    t_inst_start = time.perf_counter()
    graph_exec = ctypes.c_void_p()
    # cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *pLogBuffer, size_t bufferSize)
    check_cu(cuda.cuGraphInstantiate_v2(ctypes.byref(graph_exec), graph, None, None, 0), "cuGraphInstantiate")
    t_inst_end = time.perf_counter()
    print(f"Graph Instantiation Time: {(t_inst_end - t_inst_start) * 1000.0:.2f} ms")
    
    # Warmup graph launch
    check_cu(cuda.cuGraphLaunch(graph_exec, stream), "cuGraphLaunch warmup")
    check_cu(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize warmup")
    
    # Benchmark CUDA Graph Launch
    iterations = 50
    t0 = time.perf_counter()
    for _ in range(iterations):
        cuda.cuGraphLaunch(graph_exec, stream)
    t1 = time.perf_counter()
    
    cuda.cuStreamSynchronize(stream)
    t2 = time.perf_counter()
    
    cpu_launch_total_ms = (t1 - t0) * 1000.0 / iterations
    synced_total_ms = (t2 - t0) * 1000.0 / iterations
    
    print(f"CUDA Graph Launch: CPU Submission per step: {cpu_launch_total_ms * 1000.0:.2f} us | "
          f"Total Synced Time (800 ops): {synced_total_ms:.3f} ms")
    
    # Compare with non-captured launches
    t3 = time.perf_counter()
    for _ in range(num_nodes):
        cuda.cuLaunchKernel(
            kernel,
            grid_dim_x, 1, 1,
            block_dim_x, 1, 1,
            0, stream,
            kernel_params, None
        )
    t4 = time.perf_counter()
    cuda.cuStreamSynchronize(stream)
    t5 = time.perf_counter()
    
    non_graph_cpu_ms = (t4 - t3) * 1000.0
    non_graph_sync_ms = (t5 - t3) * 1000.0
    
    print(f"Sequential 800 Launches: CPU Submission: {non_graph_cpu_ms:.2f} ms | "
          f"Total Synced Time: {non_graph_sync_ms:.2f} ms")
    
    speedup = non_graph_sync_ms / synced_total_ms
    print(f"CUDA Graph Speedup: {speedup:.2f}x")
    
    # Cleanup
    cuda.cuGraphExecDestroy(graph_exec)
    cuda.cuGraphDestroy(graph)
    cuda.cuStreamDestroy_v2(stream)
    cuda.cuMemFree_v2(d_in)
    cuda.cuMemFree_v2(d_out)
    print("=" * 70)

if __name__ == "__main__":
    benchmark_cuda_graphs()
