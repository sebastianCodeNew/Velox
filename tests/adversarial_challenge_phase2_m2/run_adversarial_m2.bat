@echo off
setlocal enabledelayedexpansion

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\nvcc.exe"

call %VCVARS% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] MSVC initialization failed
    exit /b 1
)

cd /d C:\Sebastian\Project\Velox\tests\adversarial_challenge_phase2_m2

echo ============================================================================
echo [1/2] Compiling CUDA Blueprint with NVCC -arch=sm_86 and ptxas -v...
echo ============================================================================
%NVCC% -O3 -arch=sm_86 -Xptxas="-v" -std=c++14 test_adversarial_cuda_blueprint.cu -o test_adversarial_cuda_blueprint.exe
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo.
echo ============================================================================
echo [2/2] Running CUDA Blueprint Verification Executable on RTX 3050 GPU...
echo ============================================================================
test_adversarial_cuda_blueprint.exe
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Execution failed!
    exit /b 1
)

echo.
echo [COMPLETE] CUDA Blueprint Test Completed Successfully.
