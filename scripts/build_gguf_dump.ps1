# Builds scripts\test_gguf_kv_dump.exe — GGUF header metadata dumper (MSVC x64).
# Mirrors build_runner.ps1 toolchain pinning (VS2019 BuildTools, Windows SDK).
$ErrorActionPreference = 'Stop'

$clRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133"
$binDir = "$clRoot\bin\Hostx64\x64"
$incDir = "$clRoot\include"
$libDir = "$clRoot\lib\x64"

$winSdkInc = "C:\Program Files (x86)\Windows Kits\10\Include"
$winSdkLib = "C:\Program Files (x86)\Windows Kits\10\Lib"
$sdkVer = (Get-ChildItem $winSdkInc -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name

$env:PATH    = "$binDir;$env:PATH"
$env:INCLUDE = "$incDir;$winSdkInc\$sdkVer\ucrt;$winSdkInc\$sdkVer\um;$winSdkInc\$sdkVer\shared"
$env:LIB     = "$libDir;$winSdkLib\$sdkVer\ucrt\x64;$winSdkLib\$sdkVer\um\x64"

Set-Location "C:\Sebastian\Project\Velox"
cl /nologo /W4 /O2 scripts\test_gguf_kv_dump.c /Fe:scripts\test_gguf_kv_dump.exe
