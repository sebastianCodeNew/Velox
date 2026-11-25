$clPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
$env:PATH = "$clPath;$env:PATH"
$incPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\include"
$libPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\lib\x64"
$winSdkInc = "C:\Program Files (x86)\Windows Kits\10\Include"
$winSdkLib = "C:\Program Files (x86)\Windows Kits\10\Lib"
$sdkVer = (Get-ChildItem $winSdkInc -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name
$env:INCLUDE = "$incPath;$winSdkInc\$sdkVer\ucrt;$winSdkInc\$sdkVer\um;$winSdkInc\$sdkVer\shared"
$env:LIB = "$libPath;$winSdkLib\$sdkVer\ucrt\x64;$winSdkLib\$sdkVer\um\x64"
Set-Location $PSScriptRoot
nvcc -o velox_cuda_runner_k2.exe src/runners/velox_cuda_runner_k2.cu -arch=sm_86 -O3 --use_fast_math -Xcompiler "/EHsc /O2" -Xptxas -v
if ($LASTEXITCODE -ne 0) {
    Write-Error "nvcc failed with exit code $LASTEXITCODE; refusing to sync a stale runner"
    exit $LASTEXITCODE
}

# Sync the freshly built runner into src\runners\ — the canonical location
# resolved by veloxc's CLI backend dispatcher (prevents stale-binary drift:
# the CLI would otherwise execute an outdated, pre-fix runner copy).
if (-not (Test-Path "velox_cuda_runner_k2.exe")) {
    Write-Error "nvcc reported success but velox_cuda_runner_k2.exe is missing"
    exit 1
}
$srcTime = (Get-Item "velox_cuda_runner_k2.exe").LastWriteTime
try {
    Copy-Item "velox_cuda_runner_k2.exe" "src\runners\velox_cuda_runner_k2.exe" -Force -ErrorAction Stop
} catch {
    Write-Error "failed to sync runner into src\runners (is a chat session still running?): $_"
    exit 1
}
$dstTime = (Get-Item "src\runners\velox_cuda_runner_k2.exe").LastWriteTime
if ($dstTime -lt $srcTime) {
    Write-Error "synced runner is older than the build output; src\runners copy is stale"
    exit 1
}
Write-Host "[OK] Synced velox_cuda_runner_k2.exe -> src\runners\"

