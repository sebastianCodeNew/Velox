@echo off
setlocal enabledelayedexpansion

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\nvcc.exe"

call %VCVARS% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] MSVC initialization failed
    exit /b 1
)

cd /d C:\Sebastian\Project\Velox\benchmarks\micro_phase2

echo ============================================================================
echo Compiling and Running Step 11 Isolated Micro-Benchmark Suite...
echo ============================================================================

echo [1/5] Compiling and running Bench 1 (Single vs Multi-SM)...
%NVCC% -O3 -arch=sm_86 -std=c++14 bench1_single_vs_multi_sm.cu -o bench1.exe
if %ERRORLEVEL% neq 0 exit /b 1
bench1.exe

echo [2/5] Compiling and running Bench 2 (Strided vs Coalesced float4 GEMV)...
%NVCC% -O3 -arch=sm_86 -std=c++14 bench2_gemv_coalescing.cu -o bench2.exe
if %ERRORLEVEL% neq 0 exit /b 1
bench2.exe

echo [3/5] Compiling and running Bench 3 (Single-Thread vs Parallel Tree LM Head)...
%NVCC% -O3 -arch=sm_86 -std=c++14 bench3_parallel_lm_head.cu -o bench3.exe
if %ERRORLEVEL% neq 0 exit /b 1
bench3.exe

echo [4/5] Compiling and running Bench 4 (Kernel Launch Overhead vs CUDA Graphs)...
%NVCC% -O3 -arch=sm_86 -std=c++14 bench4_kernel_launch_overhead.cu -o bench4.exe
if %ERRORLEVEL% neq 0 exit /b 1
bench4.exe

echo [5/5] Compiling and running Bench 5 (CPU AVX2+FMA3 Multi-Thread GEMV)...
cl.exe /nologo /O2 /arch:AVX2 /openmp /Fe:bench5.exe bench5_cpu_avx2_gemv.c
if %ERRORLEVEL% neq 0 exit /b 1
bench5.exe

echo ============================================================================
echo ALL 5 STEP 11 MICRO-BENCHMARKS COMPLETED SUCCESSFULLY
echo ============================================================================
