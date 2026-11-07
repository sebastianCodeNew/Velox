"""
Benchmark: Raw CUDA Driver cuLaunchKernel via ctypes (direct nvcuda.dll invocation)
Measures the pure C-API driver submission latency without PyTorch/Numba framework wrappers.
"""

import ctypes
import time
import numpy as np

# Load nvcuda.dll
cuda = ctypes.windll.LoadLibrary("nvcuda.dll")

# Function prototypes
# CUresult cuInit(unsigned int Flags)
# CUresult cuDeviceGet(CUdevice *device, int ordinal)
# CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
# CUresult cuModuleLoadData(CUmodule *module, const void *image)
# CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
# CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
# CUresult cuMemFree_v2(CUdeviceptr dptr)
# CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, Y, Z, blockDimX, Y, Z, sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
# CUresult cuCtxSynchronize(void)

# Minimal PTX for a dummy kernel (RMSNorm / copy / no-op)
# Target sm_50+ compatible PTX
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
    mul.f32 %f2, %f1, 0f3f000000; // * 0.5
    st.global.f32 [%rd5], %f2;

DONE:
    ret;
}
"""

def check_cu(res, msg="CUDA Error"):
    if res != 0:
        raise RuntimeError(f"{msg}: error code {res}")

def benchmark_raw_driver_launch():
    print("=" * 70)
    print("BENCHMARK: Raw CUDA Driver C-API cuLaunchKernel Latency")
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
    
    # Allocate device buffers
    D = 4096
    size_bytes = D * 4
    d_in = ctypes.c_uint64()
    d_out = ctypes.c_uint64()
    check_cu(cuda.cuMemAlloc_v2(ctypes.byref(d_in), size_bytes), "cuMemAlloc in")
    check_cu(cuda.cuMemAlloc_v2(ctypes.byref(d_out), size_bytes), "cuMemAlloc out")
    
    n_val = ctypes.c_uint32(D)
    
    # Kernel parameters array (pointers to arguments)
    arg_in = ctypes.c_void_p(d_in.value)
    arg_out = ctypes.c_void_p(d_out.value)
    
    # Void* array of pointers
    p_in = ctypes.cast(ctypes.byref(d_in), ctypes.c_void_p)
    p_out = ctypes.cast(ctypes.byref(d_out), ctypes.c_void_p)
    p_n = ctypes.cast(ctypes.byref(n_val), ctypes.c_void_p)
    
    kernel_params = (ctypes.c_void_p * 3)(p_in, p_out, p_n)
    
    grid_dim_x = (D + 255) // 256
    block_dim_x = 256
    
    # Warmup
    for _ in range(100):
        check_cu(cuda.cuLaunchKernel(
            kernel,
            grid_dim_x, 1, 1,
            block_dim_x, 1, 1,
            0, None,
            kernel_params, None
        ), "cuLaunchKernel warmup")
    check_cu(cuda.cuCtxSynchronize(), "cuCtxSynchronize warmup")
    
    # Benchmark varying launches
    for num_launches in [10, 100, 400, 800]:
        t0 = time.perf_counter()
        for _ in range(num_launches):
            cuda.cuLaunchKernel(
                kernel,
                grid_dim_x, 1, 1,
                block_dim_x, 1, 1,
                0, None,
                kernel_params, None
            )
        t1 = time.perf_counter()
        
        cuda.cuCtxSynchronize()
        t2 = time.perf_counter()
        
        cpu_sub_ms = (t1 - t0) * 1000.0
        cpu_per_launch_us = (cpu_sub_ms * 1000.0) / num_launches
        total_sync_ms = (t2 - t0) * 1000.0
        
        print(f"Launches: {num_launches:3d} | "
              f"Raw cuLaunchKernel CPU Time: {cpu_sub_ms:6.3f} ms | "
              f"Per-Launch Driver Latency: {cpu_per_launch_us:5.2f} us | "
              f"Total Synced Time: {total_sync_ms:6.3f} ms")
        
    cuda.cuMemFree_v2(d_in)
    cuda.cuMemFree_v2(d_out)
    print("=" * 70)

if __name__ == "__main__":
    benchmark_raw_driver_launch()
