$clPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
$env:PATH = "$clPath;$env:PATH"
$incPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\include"
$libPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\lib\x64"
$winSdkInc = "C:\Program Files (x86)\Windows Kits\10\Include"
$winSdkLib = "C:\Program Files (x86)\Windows Kits\10\Lib"
$sdkVer = (Get-ChildItem $winSdkInc -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name
$env:INCLUDE = "$incPath;$winSdkInc\$sdkVer\ucrt;$winSdkInc\$sdkVer\um;$winSdkInc\$sdkVer\shared"
$env:LIB = "$libPath;$winSdkLib\$sdkVer\ucrt\x64;$winSdkLib\$sdkVer\um\x64"
Set-Location "C:\Sebastian\Project\Velox"
nvcc -o velox_cuda_runner_k2.exe src/runners/velox_cuda_runner_k2.cu -arch=sm_86 -O3 --use_fast_math -Xcompiler "/EHsc /O2" -Xptxas -v

# Sync the freshly built runner into src\runners\ — the canonical location
# resolved by veloxc's CLI backend dispatcher (prevents stale-binary drift:
# the CLI would otherwise execute an outdated, pre-fix runner copy).
if (Test-Path "velox_cuda_runner_k2.exe") {
    Copy-Item "velox_cuda_runner_k2.exe" "src\runners\velox_cuda_runner_k2.exe" -Force
    Write-Host "[OK] Synced velox_cuda_runner_k2.exe -> src\runners\"
}

