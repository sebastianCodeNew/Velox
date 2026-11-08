@echo off
setlocal enabledelayedexpansion

echo ============================================================================
echo Velox Milestone 2 Standalone C99 and CUDA Runner Build System
echo ============================================================================

set ROOT=%~dp0
if exist "%ROOT%src\runners" (
    set RUNNERS_DIR=%ROOT%src\runners
) else (
    set RUNNERS_DIR=%ROOT%
    set ROOT=%ROOT%..\..\
)

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\nvcc.exe"

if not exist %VCVARS% (
    echo [ERROR] MSVC vcvars64.bat not found at %VCVARS%
    exit /b 1
)

if not exist %NVCC% (
    echo [ERROR] CUDA nvcc.exe not found at %NVCC%
    exit /b 1
)

echo [1/4] Initializing 64-bit MSVC Environment...
call %VCVARS% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to initialize MSVC vcvars64.bat
    exit /b 1
)

echo [2/5] Emitting latest C99 and CUDA source from .vlx models...
if exist "%ROOT%target\release\veloxc.exe" (
    "%ROOT%target\release\veloxc.exe" --emit-c "%ROOT%examples\tinystories_smollm.vlx" -o "%RUNNERS_DIR%\tinystories_smollm.c"
    "%ROOT%target\release\veloxc.exe" --emit-cuda "%ROOT%examples\tinystories_smollm.vlx" -o "%RUNNERS_DIR%\tinystories_smollm.cu"
    "%ROOT%target\release\veloxc.exe" --emit-cuda "%ROOT%examples\k2_horizon_3_7b.vlx" -o "%RUNNERS_DIR%\k2_horizon_3_7b.cu"
)

echo [3/5] Compiling velox_c99_runner.exe (MSVC cl.exe /O2 /openmp /arch:AVX2)...
pushd "%RUNNERS_DIR%"
cl.exe /nologo /O2 /openmp /arch:AVX2 /I. /Fe:velox_c99_runner.exe velox_c99_runner.c
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to compile velox_c99_runner.c
    popd
    exit /b 1
)
echo [OK] Built %RUNNERS_DIR%\velox_c99_runner.exe

echo [4/5] Compiling velox_cuda_runner.exe for SmolLM-135M (NVCC -O3 -arch=sm_86)...
%NVCC% -O3 -arch=sm_86 -I. -o velox_cuda_runner.exe velox_cuda_runner.cu
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to compile velox_cuda_runner.cu
    popd
    exit /b 1
)
echo [OK] Built %RUNNERS_DIR%\velox_cuda_runner.exe

echo [5/5] Compiling velox_cuda_runner_k2.exe for K2-Horizon-3.7B (NVCC -O3 -arch=sm_86)...
%NVCC% -O3 -arch=sm_86 -I. -o velox_cuda_runner_k2.exe velox_cuda_runner_k2.cu
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to compile velox_cuda_runner_k2.cu
    popd
    exit /b 1
)
echo [OK] Built %RUNNERS_DIR%\velox_cuda_runner_k2.exe
popd

echo ============================================================================
echo Build Successful!
echo To run C99 Host CPU runner (SmolLM-135M):
echo   %RUNNERS_DIR%\velox_c99_runner.exe --weights smollm-135m.safetensors
echo To run CUDA RTX 3050 GPU runner (SmolLM-135M):
echo   %RUNNERS_DIR%\velox_cuda_runner.exe --weights smollm-135m.safetensors
echo To run CUDA RTX 3050 GPU runner (K2-Horizon-3.7B):
echo   %RUNNERS_DIR%\velox_cuda_runner_k2.exe --steps 16
echo ============================================================================
