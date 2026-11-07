"""
Benchmark: PTXAS Compilation Latency Test
Measures the invocation and compilation time of NVIDIA ptxas.exe on kernels of varying complexity.
Validates or refutes the diagnostic's claim: 'ptxas compilation takes between 200ms and 5.0 seconds per kernel'.
"""

import subprocess
import time
import os
import tempfile

def generate_complex_ptx(num_unrolls=200):
    """Generates synthetic PTX with heavy register usage and unrolled loops simulating FlashAttention/GEMM"""
    ptx_lines = [
        ".version 6.5",
        ".target sm_75",
        ".address_size 64",
        ".visible .entry complex_kernel(",
        "    .param .u64 in_ptr,",
        "    .param .u64 out_ptr,",
        "    .param .u32 n",
        ")",
        "{",
        "    .reg .b32 %r<100>;",
        "    .reg .b64 %rd<100>;",
        f"    .reg .f32 %f<{num_unrolls * 4 + 10}>;",
        "    .reg .pred %p<10>;",
        "    ld.param.u64 %rd1, [in_ptr];",
        "    ld.param.u64 %rd2, [out_ptr];",
        "    mov.f32 %f0, 0f00000000;"
    ]
    
    # Generate hundreds of unrolled FMA operations with dependencies
    for i in range(1, num_unrolls * 4):
        prev = i - 1
        ptx_lines.append(f"    fma.rn.f32 %f{i}, %f{prev}, 0f3f800000, 0f40000000;")
        
    ptx_lines.extend([
        "    st.global.f32 [%rd2], %f1;",
        "    ret;",
        "}"
    ])
    return "\n".join(ptx_lines)

def benchmark_ptxas():
    print("=" * 70)
    print("CHALLENGE 3: NVIDIA PTXAS Compilation Latency Test")
    print("=" * 70)
    
    complexities = [
        ("Simple Kernel (10 ops)", 10),
        ("Medium Kernel (100 ops)", 100),
        ("Heavy Fused Kernel (500 ops)", 500),
        ("Extreme Fused Kernel (1500 ops)", 1500)
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ptx_path = os.path.join(tmpdir, "test.ptx")
        cubin_path = os.path.join(tmpdir, "test.cubin")
        
        for name, num_ops in complexities:
            ptx_code = generate_complex_ptx(num_ops)
            with open(ptx_path, "w") as f:
                f.write(ptx_code)
                
            cmd = ["ptxas", "-O3", "-arch=sm_75", ptx_path, "-o", cubin_path]
            
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                res = subprocess.run(cmd, capture_output=True, text=True)
                t1 = time.perf_counter()
                if res.returncode != 0:
                    print(f"ptxas error: {res.stderr}")
                    break
                times.append((t1 - t0) * 1000.0)
                
            mean_time = sum(times) / len(times) if times else 0.0
            min_time = min(times) if times else 0.0
            print(f"{name:35s} | Mean: {mean_time:6.2f} ms | Min: {min_time:6.2f} ms")
            
    print("=" * 70)

if __name__ == "__main__":
    benchmark_ptxas()
