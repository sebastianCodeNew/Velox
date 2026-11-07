@echo off
setlocal enabledelayedexpansion

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\nvcc.exe"

call %VCVARS% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] MSVC initialization failed
    exit /b 1
)

cd /d C:\Sebastian\Project\Velox\tests\adversarial_challenge_phase2_m1_c2

echo ============================================================================
echo Running Challenger 2 Adversarial Stress Suite...
echo ============================================================================

echo [1/4] Compiling and running Adversarial Probe 1 (L2 Eviction vs GDDR6 Streaming)...
%NVCC% -O3 -arch=sm_86 -std=c++14 test_l2_eviction_and_bandwidth.cu -o adv_test1.exe
if %ERRORLEVEL% neq 0 exit /b 1
adv_test1.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo [2/4] Compiling and running Adversarial Probe 2 (Little's Law CPU Limits)...
cl.exe /nologo /O2 /arch:AVX2 /openmp /Fe:adv_test2.exe test_cpu_littles_law_and_concurrency.c
if %ERRORLEVEL% neq 0 exit /b 1
adv_test2.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo [3/4] Compiling and running Adversarial Probe 3 (Zero Runtime Allocations Verification)...
%NVCC% -O3 -arch=sm_86 -std=c++14 test_zero_runtime_allocations.cu -o adv_test3.exe
if %ERRORLEVEL% neq 0 exit /b 1
adv_test3.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo [4/4] Compiling and running Adversarial Probe 4 (Split-K Numerical Parity Stress)...
%NVCC% -O3 -arch=sm_86 -std=c++14 test_splitk_numerical_stress.cu -o adv_test4.exe
if %ERRORLEVEL% neq 0 exit /b 1
adv_test4.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo ============================================================================
echo ALL 4 CHALLENGER 2 ADVERSARIAL STRESS PROBES PASSED 100%%
echo ============================================================================
