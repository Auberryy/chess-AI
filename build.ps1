# Windows build script for Chess AI
# Requires: Visual Studio 2019/2022, CUDA Toolkit, CMake

param(
    [string]$BuildType = "Release",
    [string]$LibTorchPath = "C:\libtorch",
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

Write-Host "Chess AI Build Script" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan

# Check prerequisites
Write-Host "`nChecking prerequisites..." -ForegroundColor Yellow

# Check CMake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: CMake not found. Please install CMake 3.18+" -ForegroundColor Red
    exit 1
}
$cmakeVersion = cmake --version | Select-Object -First 1
Write-Host "  CMake: $cmakeVersion" -ForegroundColor Green

# Check CUDA
if (-not (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")) {
    Write-Host "WARNING: CUDA Toolkit not found at default location" -ForegroundColor Yellow
}

# Check LibTorch
if (-not (Test-Path $LibTorchPath)) {
    Write-Host "`nLibTorch not found at $LibTorchPath" -ForegroundColor Yellow
    Write-Host "Downloading LibTorch (CUDA 11.8)..." -ForegroundColor Yellow
    
    $url = "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.0%2Bcu118.zip"
    $zipPath = "$env:TEMP\libtorch.zip"
    
    Invoke-WebRequest -Uri $url -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath "C:\" -Force
    Remove-Item $zipPath
    
    Write-Host "LibTorch downloaded and extracted to $LibTorchPath" -ForegroundColor Green
}
Write-Host "  LibTorch: $LibTorchPath" -ForegroundColor Green

# Download Stockfish if not present
$stockfishPath = ".\stockfish"
if (-not (Test-Path "$stockfishPath\stockfish.exe")) {
    Write-Host "`nDownloading Stockfish..." -ForegroundColor Yellow
    
    New-Item -ItemType Directory -Path $stockfishPath -Force | Out-Null
    $stockfishUrl = "https://github.com/official-stockfish/Stockfish/releases/download/sf_16/stockfish-windows-x86-64-avx2.zip"
    $stockfishZip = "$env:TEMP\stockfish.zip"
    
    Invoke-WebRequest -Uri $stockfishUrl -OutFile $stockfishZip
    Expand-Archive -Path $stockfishZip -DestinationPath $stockfishPath -Force
    
    # Move exe to correct location
    $exePath = Get-ChildItem -Path $stockfishPath -Recurse -Filter "stockfish*.exe" | Select-Object -First 1
    if ($exePath) {
        Move-Item $exePath.FullName "$stockfishPath\stockfish.exe" -Force
    }
    
    Remove-Item $stockfishZip
    Write-Host "Stockfish downloaded to $stockfishPath" -ForegroundColor Green
}

# Clean build directory if requested
if ($Clean -and (Test-Path "build")) {
    Write-Host "`nCleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "build"
}

# Create build directory
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

# Configure
Write-Host "`nConfiguring CMake..." -ForegroundColor Yellow
Push-Location "build"

try {
    cmake .. `
        -DCMAKE_BUILD_TYPE=$BuildType `
        -DCMAKE_PREFIX_PATH="$LibTorchPath" `
        -G "Visual Studio 17 2022" `
        -A x64

    if ($LASTEXITCODE -ne 0) {
        Write-Host "CMake configuration failed!" -ForegroundColor Red
        exit 1
    }

    # Build
    Write-Host "`nBuilding ($BuildType)..." -ForegroundColor Yellow
    cmake --build . --config $BuildType --parallel

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }

    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Executable: build\$BuildType\chess_ai.exe" -ForegroundColor Cyan

}
finally {
    Pop-Location
}

# Create run script
$runScript = @"
@echo off
set PATH=$LibTorchPath\lib;%PATH%
cd /d "%~dp0"
build\$BuildType\chess_ai.exe %*
"@
Set-Content -Path "run.bat" -Value $runScript

Write-Host "`nTo run the AI, use: .\run.bat --train" -ForegroundColor Cyan
