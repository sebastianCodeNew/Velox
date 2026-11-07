"""
Empirical Benchmark: Measuring exact C-FFI boundary and dispatch overhead.
Uses CFFI to measure calling a C function from Python with pointer unboxing and struct manipulation.
"""

import time
import cffi

ffi = cffi.FFI()
ffi.cdef("""
    typedef struct {
        void* data_ptr;
        int64_t sizes[4];
        int64_t strides[4];
        int dtype;
        int device;
    } MockTensor;

    void mock_kernel_launch(MockTensor* a, MockTensor* b, MockTensor* out);
""")

# Compile a minimal in-memory C library
C_CODE = """
typedef struct {
    void* data_ptr;
    long long sizes[4];
    long long strides[4];
    int dtype;
    int device;
} MockTensor;

void mock_kernel_launch(MockTensor* a, MockTensor* b, MockTensor* out) {
    // Simulates C++ kernel wrapper validation
    if (a->dtype != b->dtype || a->device != b->device) {
        return;
    }
}
"""

lib = ffi.verify(C_CODE)

def benchmark_cffi_dispatch(iterations=200000):
    t_a = ffi.new("MockTensor*")
    t_b = ffi.new("MockTensor*")
    t_out = ffi.new("MockTensor*")
    
    t_a.sizes[0] = 1; t_a.sizes[1] = 1; t_a.sizes[2] = 8192
    t_b.sizes[0] = 8192; t_b.sizes[1] = 8192
    t_a.device = 0; t_b.device = 0
    t_a.dtype = 1; t_b.dtype = 1
    
    start = time.perf_counter_ns()
    for _ in range(iterations):
        lib.mock_kernel_launch(t_a, t_b, t_out)
    end = time.perf_counter_ns()
    
    avg_ns = (end - start) / iterations
    return avg_ns / 1000.0

if __name__ == "__main__":
    us = benchmark_cffi_dispatch()
    print(f"CFFI boundary + C validation overhead per call: {us:.3f} microseconds")
    ops_80_layers = 1120
    print(f"80 layers (1,120 ops) total C-FFI overhead: {ops_80_layers * us / 1000.0:.3f} ms")
