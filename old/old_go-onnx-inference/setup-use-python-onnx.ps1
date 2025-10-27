# Setup script for Go + ONNX Runtime using existing Python installation

Write-Host "🚀 Setting up Go + ONNX Runtime (using Python installation)" -ForegroundColor Green
Write-Host ""

# Locate ONNX Runtime from Python environment
$pythonOnnxPath = "D:\Projects\webcodecstest\.venv312\Lib\site-packages\onnxruntime"
$onnxCapi = Join-Path $pythonOnnxPath "capi"
$onnxInclude = Join-Path $pythonOnnxPath "capi\include"

Write-Host "Checking ONNX Runtime in Python environment..." -ForegroundColor Yellow
if (Test-Path $onnxCapi) {
    Write-Host "✅ Found ONNX Runtime at: $onnxCapi" -ForegroundColor Green
}
else {
    Write-Host "❌ ONNX Runtime not found in Python environment!" -ForegroundColor Red
    exit 1
}

# Check for required DLLs
Write-Host ""
Write-Host "Checking for CUDA support..." -ForegroundColor Yellow
$cudaDll = Join-Path $onnxCapi "onnxruntime_providers_cuda.dll"
if (Test-Path $cudaDll) {
    Write-Host "✅ CUDA provider found" -ForegroundColor Green
}
else {
    Write-Host "⚠️  CUDA provider not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setting environment variables..." -ForegroundColor Yellow

# Set ONNXRUNTIME_DIR to Python's ONNX installation
[System.Environment]::SetEnvironmentVariable('ONNXRUNTIME_DIR', $onnxCapi, 'User')
Write-Host "✅ Set ONNXRUNTIME_DIR=$onnxCapi" -ForegroundColor Green

# Enable CGO
[System.Environment]::SetEnvironmentVariable('CGO_ENABLED', '1', 'User')
Write-Host "✅ Set CGO_ENABLED=1" -ForegroundColor Green

# Add ONNX Runtime to PATH
$currentPath = [System.Environment]::GetEnvironmentVariable('PATH', 'User')
if ($currentPath -notlike "*$onnxCapi*") {
    $newPath = $currentPath + ";$onnxCapi"
    [System.Environment]::SetEnvironmentVariable('PATH', $newPath, 'User')
    Write-Host "✅ Added $onnxCapi to PATH" -ForegroundColor Green
}
else {
    Write-Host "✅ ONNX Runtime already in PATH" -ForegroundColor Green
}

# Also set for current session
$env:ONNXRUNTIME_DIR = $onnxCapi
$env:CGO_ENABLED = "1"
$env:PATH = "$env:PATH;$onnxCapi"

Write-Host ""
Write-Host "Checking for MinGW (required for CGO)..." -ForegroundColor Yellow
$gccTest = Get-Command gcc -ErrorAction SilentlyContinue
if ($null -eq $gccTest) {
    Write-Host "⚠️  MinGW (GCC) not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You need MinGW-w64 to compile Go with CGO." -ForegroundColor Yellow
    Write-Host "Install options:" -ForegroundColor Cyan
    Write-Host "  1. Chocolatey: choco install mingw" -ForegroundColor White
    Write-Host "  2. Download: https://www.mingw-w64.org/" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne 'y') {
        exit 0
    }
}
else {
    Write-Host "✅ MinGW found: $($gccTest.Path)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Installing Go dependencies..." -ForegroundColor Yellow

# Install dependencies
go get github.com/yalue/onnxruntime_go
go mod tidy

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
}
else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-Host "Environment configured to use:" -ForegroundColor Cyan
Write-Host "  ONNX Runtime: $onnxCapi" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  IMPORTANT: Restart your terminal for PATH changes to take effect!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then try:" -ForegroundColor Cyan
Write-Host "  go run ./cmd/simple-test/main.go" -ForegroundColor White
Write-Host "  go run ./cmd/benchmark/main.go" -ForegroundColor White
Write-Host ""
