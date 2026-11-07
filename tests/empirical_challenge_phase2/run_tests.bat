@echo off
setlocal enabledelayedexpansion

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\nvcc.exe"

call %VCVARS% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] MSVC initialization failed
    exit /b 1
)

cd /d C:\Sebastian\Project\Velox\tests\empirical_challenge_phase2

echo ============================================================================
echo Compiling and Running Step 11B Empirical Challenge Verification Suite...
echo ============================================================================

echo [1/4] Compiling and running Test 1 (Warp float4 Shuffle GEMV)...
%NVCC% -O3 -arch=sm_86 -std=c++14 test1_warp_gemv_float4_shuffle.cu -o test1.exe
if %ERRORLEVEL% neq 0 exit /b 1
test1.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo [2/4] Compiling and running Test 2 (Split-K FlashDecoding Parity)...
%NVCC% -O3 -arch=sm_86 -std=c++14 test2_split_k_flashdecoding_parity.cu -o test2.exe
if %ERRORLEVEL% neq 0 exit /b 1
test2.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo [3/4] Compiling and running Test 3 (Marlin LOP3 0xEA Dequantization)...
%NVCC% -O3 -arch=sm_86 -std=c++14 test3_marlin_lop3_dequant.cu -o test3.exe
if %ERRORLEVEL% neq 0 exit /b 1
test3.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo [4/4] Compiling and running Test 4 (AVX2+FMA3 4-Row Register Blocking)...
cl.exe /nologo /O2 /arch:AVX2 /Fe:test4.exe test4_avx2_fma3_register_blocking.c
if %ERRORLEVEL% neq 0 exit /b 1
test4.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo ============================================================================
echo ALL 4 EMPIRICAL CHALLENGE TESTS PASSED 100%%
echo ============================================================================
