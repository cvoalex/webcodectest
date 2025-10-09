# Quick installer for protoc on Windows

$ErrorActionPreference = "Stop"

Write-Host "🔧 Installing protoc for Windows..." -ForegroundColor Cyan
Write-Host ""

# Configuration
$PROTOC_VERSION = "25.1"
$PROTOC_ZIP = "protoc-$PROTOC_VERSION-win64.zip"
$PROTOC_URL = "https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOC_VERSION/$PROTOC_ZIP"
$INSTALL_DIR = "C:\protoc"
$TEMP_DIR = "$env:TEMP\protoc_install"

# Create temp directory
Write-Host "📁 Creating temp directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null

# Download protoc
Write-Host "📥 Downloading protoc $PROTOC_VERSION..." -ForegroundColor Yellow
Write-Host "   URL: $PROTOC_URL"
$zipPath = Join-Path $TEMP_DIR $PROTOC_ZIP

try {
    Invoke-WebRequest -Uri $PROTOC_URL -OutFile $zipPath -UseBasicParsing
    Write-Host "✅ Downloaded successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Download failed: $_" -ForegroundColor Red
    exit 1
}

# Extract
Write-Host "📦 Extracting to $INSTALL_DIR..." -ForegroundColor Yellow
if (Test-Path $INSTALL_DIR) {
    Write-Host "   Removing existing installation..."
    Remove-Item -Path $INSTALL_DIR -Recurse -Force
}

Expand-Archive -Path $zipPath -DestinationPath $INSTALL_DIR -Force
Write-Host "✅ Extracted successfully" -ForegroundColor Green

# Add to PATH (current session)
Write-Host "Adding to PATH..." -ForegroundColor Yellow
$env:PATH = "$INSTALL_DIR\bin;$env:PATH"

# Verify
Write-Host "Verifying installation..." -ForegroundColor Yellow
try {
    $version = & protoc --version
    Write-Host "   $version" -ForegroundColor Green
} catch {
    Write-Host "❌ Verification failed" -ForegroundColor Red
    exit 1
}

# Clean up
Write-Host "Cleaning up..." -ForegroundColor Yellow
Remove-Item -Path $TEMP_DIR -Recurse -Force

Write-Host ""
Write-Host "SUCCESS: protoc installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: PATH updated for THIS SESSION ONLY" -ForegroundColor Yellow
Write-Host ""
Write-Host "To make it permanent, add to System PATH:" -ForegroundColor Cyan
Write-Host "   $INSTALL_DIR\bin" -ForegroundColor White
Write-Host ""
Write-Host "Or run this as Administrator:" -ForegroundColor Cyan
Write-Host '   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\protoc\bin", "Machine")' -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Install Go plugins:" -ForegroundColor White
Write-Host "   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest" -ForegroundColor Gray
Write-Host "   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest" -ForegroundColor Gray
Write-Host "2. Generate proto files:" -ForegroundColor White
Write-Host "   .\generate_proto.bat" -ForegroundColor Gray
Write-Host "3. Build proxy:" -ForegroundColor White
Write-Host "   .\build.bat" -ForegroundColor Gray
Write-Host ""
