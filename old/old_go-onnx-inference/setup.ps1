# Setup script for Go + ONNX Runtime on Windows

Write-Host "🚀 Setting up Go + ONNX Runtime for Lip Sync Inference" -ForegroundColor Green
Write-Host ""

# Check if Go is installed
Write-Host "Checking for Go installation..." -ForegroundColor Yellow
$goVersion = go version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Go is not installed!" -ForegroundColor Red
    Write-Host "Please install Go from: https://go.dev/dl/" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Go is installed: $goVersion" -ForegroundColor Green
Write-Host ""

# Check if GCC is installed (for CGO)
Write-Host "Checking for GCC (MinGW)..." -ForegroundColor Yellow
$gccVersion = gcc --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  GCC not found. You need MinGW-w64 for CGO." -ForegroundColor Yellow
    Write-Host "Install with: choco install mingw" -ForegroundColor Yellow
    Write-Host "Or download from: https://www.mingw-w64.org/" -ForegroundColor Yellow
    Write-Host ""
}
else {
    Write-Host "✅ GCC is installed" -ForegroundColor Green
    Write-Host ""
}

# Check for ONNX Runtime
Write-Host "Checking for ONNX Runtime..." -ForegroundColor Yellow
$onnxDir = "C:\onnxruntime-gpu"
if (Test-Path $onnxDir) {
    Write-Host "✅ ONNX Runtime found at: $onnxDir" -ForegroundColor Green
}
else {
    Write-Host "⚠️  ONNX Runtime not found at: $onnxDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please download ONNX Runtime GPU from:" -ForegroundColor Yellow
    Write-Host "https://github.com/microsoft/onnxruntime/releases/latest" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Look for: onnxruntime-win-x64-gpu-*.zip" -ForegroundColor Yellow
    Write-Host "Extract to: C:\onnxruntime-gpu\" -ForegroundColor Yellow
    Write-Host ""
    
    $response = Read-Host "Do you want me to help you download it? (y/n)"
    if ($response -eq 'y') {
        Write-Host "Opening download page in browser..." -ForegroundColor Green
        Start-Process "https://github.com/microsoft/onnxruntime/releases/latest"
        Write-Host ""
        Write-Host "After downloading and extracting, run this script again." -ForegroundColor Yellow
        exit 0
    }
}
Write-Host ""

# Set environment variables
Write-Host "Setting environment variables..." -ForegroundColor Yellow

# Set ONNXRUNTIME_DIR
$currentOnnxDir = [System.Environment]::GetEnvironmentVariable('ONNXRUNTIME_DIR', 'User')
if ($currentOnnxDir -ne $onnxDir) {
    [System.Environment]::SetEnvironmentVariable('ONNXRUNTIME_DIR', $onnxDir, 'User')
    Write-Host "✅ Set ONNXRUNTIME_DIR=$onnxDir" -ForegroundColor Green
}
else {
    Write-Host "✅ ONNXRUNTIME_DIR already set" -ForegroundColor Green
}

# Enable CGO
$currentCgo = [System.Environment]::GetEnvironmentVariable('CGO_ENABLED', 'User')
if ($currentCgo -ne '1') {
    [System.Environment]::SetEnvironmentVariable('CGO_ENABLED', '1', 'User')
    Write-Host "✅ Set CGO_ENABLED=1" -ForegroundColor Green
}
else {
    Write-Host "✅ CGO_ENABLED already set" -ForegroundColor Green
}

# Add ONNX Runtime to PATH
$currentPath = [System.Environment]::GetEnvironmentVariable('PATH', 'User')
$onnxLibPath = "$onnxDir\lib"
if ($currentPath -notlike "*$onnxLibPath*") {
    $newPath = $currentPath + ";$onnxLibPath"
    [System.Environment]::SetEnvironmentVariable('PATH', $newPath, 'User')
    Write-Host "✅ Added $onnxLibPath to PATH" -ForegroundColor Green
}
else {
    Write-Host "✅ ONNX Runtime already in PATH" -ForegroundColor Green
}

Write-Host ""
Write-Host "Installing Go dependencies..." -ForegroundColor Yellow

# Initialize Go module if needed
if (-not (Test-Path "go.mod")) {
    go mod init go-onnx-inference
}

# Install dependencies
go get github.com/yalue/onnxruntime_go
go mod tidy

Write-Host "✅ Dependencies installed" -ForegroundColor Green
Write-Host ""

# Try to build
Write-Host "Testing build..." -ForegroundColor Yellow
$buildResult = go build -o test-build.exe ./cmd/simple-test/main.go 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Remove-Item test-build.exe -ErrorAction SilentlyContinue
}
else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    Write-Host $buildResult -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above and:" -ForegroundColor Yellow
    Write-Host "1. Ensure MinGW-w64 is installed" -ForegroundColor Yellow
    Write-Host "2. Ensure ONNX Runtime is downloaded and extracted" -ForegroundColor Yellow
    Write-Host "3. Restart your terminal/VS Code for environment variables to take effect" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Restart your terminal or VS Code" -ForegroundColor White
Write-Host "2. Run: go run ./cmd/simple-test/main.go" -ForegroundColor White
Write-Host "3. Run: go run ./cmd/benchmark/main.go" -ForegroundColor White
Write-Host ""
Write-Host "For more information, see:" -ForegroundColor Cyan
Write-Host "- SETUP.md" -ForegroundColor White
Write-Host "- README.md" -ForegroundColor White
Write-Host "- PERFORMANCE_COMPARISON.md" -ForegroundColor White
Write-Host ""
