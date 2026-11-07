# build.ps1 - Native Rust Build & Deployment Helper for Velox Compiler Frontend
[CmdletBinding()]
param(
    [switch]$Release = $true
)

$ErrorActionPreference = "Stop"

$cargoBin = "C:\Users\user\.cargo\bin"
if (Test-Path $cargoBin) {
    if (-not ($env:Path -split ';' -contains $cargoBin)) {
        $env:Path = "$cargoBin;$env:Path"
    }
}

Write-Host "Building Velox Compiler Frontend (veloxc)..." -ForegroundColor Cyan

if ($Release) {
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Cargo release build failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    $targetExe = "target\release\veloxc.exe"
} else {
    cargo build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Cargo debug build failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    $targetExe = "target\debug\veloxc.exe"
}

if (Test-Path $targetExe) {
    Copy-Item -Path $targetExe -Destination "veloxc.exe" -Force
    Write-Host "Build complete: C:\Sebastian\Project\Velox\veloxc.exe" -ForegroundColor Green
} else {
    Write-Error "Expected build artifact not found at $targetExe"
    exit 1
}
