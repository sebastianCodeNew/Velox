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
echo Compiling and Running Phase II Adversarial Challenge Verification Suite...
echo ============================================================================

echo [1/4] Compiling and running Adversarial Test 1 (Warp float4 Shuffle Stress)...
%NVCC% -O3 -arch=sm_86 -std=c++14 adversarial_test1_gemv.cu -o adv_test1.exe
if %ERRORLEVEL% neq 0 exit /b 1
adv_test1.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo.
echo [2/4] Compiling and running Adversarial Test 2 (Split-K FlashDecoding Stress)...
%NVCC% -O3 -arch=sm_86 -std=c++14 adversarial_test2_splitk.cu -o adv_test2.exe
if %ERRORLEVEL% neq 0 exit /b 1
adv_test2.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo.
echo [3/4] Compiling and running Adversarial Test 3 (Marlin LOP3 16-Nibble Dequant Stress)...
%NVCC% -O3 -arch=sm_86 -std=c++14 adversarial_test3_marlin.cu -o adv_test3.exe
if %ERRORLEVEL% neq 0 exit /b 1
adv_test3.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo.
echo [4/4] Compiling and running Adversarial Test 4 (AVX2+FMA3 Unaligned Dimension Stress)...
cl.exe /nologo /O2 /arch:AVX2 /Fe:adv_test4.exe adversarial_test4_avx2.c
if %ERRORLEVEL% neq 0 exit /b 1
adv_test4.exe
if %ERRORLEVEL% neq 0 exit /b 1

echo.
echo ============================================================================
echo ALL 4 ADVERSARIAL CHALLENGE TESTS PASSED 100%%
echo ============================================================================
