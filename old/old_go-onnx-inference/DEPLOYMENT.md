# Go + ONNX Inference - Deployment Guide

## 📋 Overview

This guide covers everything needed to deploy the Python-free lip-sync inference system to production servers.

---

## 🎯 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11 or Windows Server 2019+
- **GPU**: NVIDIA RTX 4090 (or any CUDA-compatible GPU)
- **CUDA**: Version 12.x (automatically used via ONNX Runtime)
- **Memory**: 8GB+ RAM recommended
- **Disk Space**: ~500MB for runtime components

### No Python Required! ✅
This deployment does **NOT** require:
- ❌ Python runtime
- ❌ pip or conda
- ❌ Virtual environments
- ❌ PyTorch or TensorFlow
- ❌ Python packages

---

## 📦 What You Need to Deploy

### 1. Your Application Binary
- **File**: `your_service.exe` (or `main.exe`, `benchmark.exe`, etc.)
- **Size**: ~5-15MB (depending on your application)
- **Build**: Compile with Go 1.24+ and CGO enabled

### 2. ONNX Runtime Libraries (Required)

Download from: https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0

**File**: `onnxruntime-win-x64-gpu-1.22.0.zip` (~313MB download)

**Required DLLs** (extract from the zip):
```
lib/
├── onnxruntime.dll                     (~13MB) - Core runtime
├── onnxruntime_providers_cuda.dll      (~305MB) - CUDA acceleration
└── onnxruntime_providers_shared.dll    (~21KB) - Provider interface
```

### 3. Your ONNX Model
- **File**: `99.onnx` (or your model filename)
- **Location**: `fast_service/models/default_model/models/99.onnx`
- **Size**: ~46MB (your model size may vary)

### 4. CUDA Runtime (Usually Pre-installed)
- CUDA 12.x libraries (typically already on systems with NVIDIA drivers)
- If missing, install: **NVIDIA CUDA Toolkit 12.x**
  - Download: https://developer.nvidia.com/cuda-downloads

---

## 🚀 Step-by-Step Deployment

### Option A: Simple Deployment (Single Server)

#### Step 1: Create Deployment Folder
```powershell
# On target server
New-Item -Path "C:\LipSyncService" -ItemType Directory
cd C:\LipSyncService
```

#### Step 2: Copy Files
```
C:\LipSyncService\
├── service.exe                         # Your Go application
├── 99.onnx                            # Your ONNX model
└── lib\
    ├── onnxruntime.dll
    ├── onnxruntime_providers_cuda.dll
    └── onnxruntime_providers_shared.dll
```

#### Step 3: Set Environment Variables (Optional)
```powershell
# Add ONNX Runtime DLLs to PATH (or place DLLs next to .exe)
$env:PATH = "C:\LipSyncService\lib;$env:PATH"

# Or set permanently:
[Environment]::SetEnvironmentVariable(
    "PATH", 
    "C:\LipSyncService\lib;$env:PATH", 
    [EnvironmentVariableTarget]::Machine
)
```

#### Step 4: Run
```powershell
cd C:\LipSyncService
.\service.exe
```

**That's it!** Your service is running. 🎉

---

### Option B: Production Deployment (Multiple Servers)

#### 1. Prepare Deployment Package
On your build machine:

```powershell
# Create deployment package
New-Item -Path ".\deploy-package" -ItemType Directory

# Copy compiled executable
Copy-Item ".\go-onnx-inference\cmd\your-service\service.exe" ".\deploy-package\"

# Copy ONNX Runtime DLLs
New-Item -Path ".\deploy-package\lib" -ItemType Directory
Copy-Item "C:\onnxruntime-1.22.0\lib\onnxruntime.dll" ".\deploy-package\lib\"
Copy-Item "C:\onnxruntime-1.22.0\lib\onnxruntime_providers_cuda.dll" ".\deploy-package\lib\"
Copy-Item "C:\onnxruntime-1.22.0\lib\onnxruntime_providers_shared.dll" ".\deploy-package\lib\"

# Copy model
Copy-Item ".\fast_service\models\default_model\models\99.onnx" ".\deploy-package\"

# Create archive
Compress-Archive -Path ".\deploy-package\*" -DestinationPath ".\lipsync-service-v1.0.zip"
```

#### 2. Deploy to Servers
```powershell
# On each target server
# Upload lipsync-service-v1.0.zip

# Extract
Expand-Archive -Path ".\lipsync-service-v1.0.zip" -DestinationPath "C:\LipSyncService"

# Add to PATH
[Environment]::SetEnvironmentVariable(
    "PATH", 
    "C:\LipSyncService\lib;$env:PATH", 
    [EnvironmentVariableTarget]::Machine
)

# Run service
cd C:\LipSyncService
.\service.exe
```

#### 3. Automate with PowerShell Script

Create `deploy.ps1`:
```powershell
# Deployment script for LipSync Service

param(
    [string]$ServicePath = "C:\LipSyncService",
    [string]$PackagePath = ".\lipsync-service-v1.0.zip"
)

Write-Host "🚀 Deploying LipSync Service..." -ForegroundColor Green

# Stop existing service if running
Stop-Process -Name "service" -ErrorAction SilentlyContinue

# Create directory
New-Item -Path $ServicePath -ItemType Directory -Force | Out-Null

# Extract package
Expand-Archive -Path $PackagePath -DestinationPath $ServicePath -Force

# Set PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::Machine)
$libPath = Join-Path $ServicePath "lib"
if ($currentPath -notlike "*$libPath*") {
    [Environment]::SetEnvironmentVariable(
        "PATH", 
        "$libPath;$currentPath", 
        [EnvironmentVariableTarget]::Machine
    )
    Write-Host "✅ Added to system PATH: $libPath" -ForegroundColor Green
}

# Test run
Write-Host "🧪 Testing service..." -ForegroundColor Yellow
cd $ServicePath
.\service.exe --test

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host "📁 Location: $ServicePath"
Write-Host "▶️  Run: cd $ServicePath; .\service.exe"
```

Run deployment:
```powershell
.\deploy.ps1
```

---

## 🔧 Building from Source

If you need to build the application yourself:

### Prerequisites for Building
1. **Go 1.24+**: https://go.dev/dl/
2. **TDM-GCC 10.3.0** (for CGO): https://jmeubank.github.io/tdm-gcc/
3. **ONNX Runtime 1.22.0**: https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0

### Build Steps

```powershell
# 1. Install Go
# Download and install from https://go.dev/dl/

# 2. Install TDM-GCC
# Download tdm64-gcc-10.3.0-2.exe and install to C:\TDM-GCC-64

# 3. Download ONNX Runtime
Invoke-WebRequest -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-gpu-1.22.0.zip" -OutFile "onnxruntime.zip"
Expand-Archive -Path "onnxruntime.zip" -DestinationPath "C:\"
Rename-Item "C:\onnxruntime-win-x64-gpu-1.22.0" "C:\onnxruntime-1.22.0"

# 4. Set build environment
$env:PATH = "C:\onnxruntime-1.22.0\lib;C:\TDM-GCC-64\bin;$env:PATH"
$env:CGO_CFLAGS = "-IC:\onnxruntime-1.22.0\include"
$env:CGO_LDFLAGS = "-LC:\onnxruntime-1.22.0\lib -lonnxruntime"
$env:CGO_ENABLED = "1"

# 5. Build your application
cd D:\Projects\webcodecstest\go-onnx-inference
go build -ldflags="-extldflags=-static-libgcc" -o service.exe .\cmd\your-service\

# Done! service.exe is ready for deployment
```

---

## ✅ Verification Checklist

After deployment, verify everything works:

### 1. Check Files
```powershell
cd C:\LipSyncService

# Should see:
ls
# service.exe
# 99.onnx
# lib\onnxruntime.dll
# lib\onnxruntime_providers_cuda.dll
# lib\onnxruntime_providers_shared.dll
```

### 2. Check GPU
```powershell
# Verify NVIDIA GPU is detected
nvidia-smi

# Should show your RTX 4090 and CUDA version
```

### 3. Test Run
```powershell
# Run simple test
.\service.exe --test

# Expected output:
# ✅ CUDA execution provider enabled
# ✅ Model loaded successfully
# ✅ Inference successful!
```

### 4. Check Performance
```powershell
# Run benchmark
.\benchmark.exe

# Expected results:
# Average time: ~5.4 ms
# FPS: ~184
```

---

## 🐛 Troubleshooting

### Error: "onnxruntime.dll not found"
**Solution**: Add DLL directory to PATH or copy DLLs next to .exe
```powershell
$env:PATH = "C:\LipSyncService\lib;$env:PATH"
# Or copy all DLLs to the same folder as service.exe
```

### Error: "CUDA initialization failed"
**Solution**: Install NVIDIA drivers and CUDA toolkit
```powershell
# Check driver
nvidia-smi

# If not found, install from:
# https://www.nvidia.com/Download/index.aspx
```

### Error: "Failed to load model"
**Solution**: Check model path is correct
```powershell
# Verify model exists
Test-Path "C:\LipSyncService\99.onnx"

# Update path in code or use absolute path
```

### Error: "API version mismatch"
**Solution**: Ensure ONNX Runtime version matches
```
Go library v1.21.0 → Requires ONNX Runtime 1.22.0 (API version 22)
```

### Performance is slow
**Solutions**:
1. Verify CUDA is enabled: Look for "✅ CUDA execution provider enabled"
2. Check GPU usage: `nvidia-smi` should show GPU activity
3. Warm up: First inference is slow (CUDA initialization)
4. Check GPU isn't being used by other processes

---

## 📊 Resource Requirements

### Per-Server Requirements
| Resource | Amount | Notes |
|----------|--------|-------|
| CPU | 2+ cores | Low CPU usage when GPU is used |
| RAM | 8GB+ | Model loaded in memory |
| GPU VRAM | 2GB+ | Model + working memory |
| Disk | 500MB | Runtime + model |
| Network | 10Mbps+ | For gRPC communication |

### Scaling Estimates
| Servers | Total Disk | Deployment Time | Concurrent Capacity |
|---------|------------|-----------------|---------------------|
| 1       | 500MB      | 2 min           | ~180 FPS            |
| 10      | 5GB        | 20 min          | ~1,800 FPS          |
| 100     | 50GB       | 3 hours         | ~18,000 FPS         |

---

## 🔐 Security Considerations

1. **File Permissions**: Restrict access to service directory
   ```powershell
   icacls C:\LipSyncService /grant Administrators:F /t
   ```

2. **Firewall Rules**: Open ports for gRPC service
   ```powershell
   New-NetFirewallRule -DisplayName "LipSync Service" -Direction Inbound -Port 50051 -Protocol TCP -Action Allow
   ```

3. **Service Account**: Run as dedicated service account (not Administrator)

4. **Model Protection**: Encrypt or restrict access to .onnx model file

---

## 🎯 Performance Expectations

### Expected Performance (RTX 4090)
- **Inference Time**: ~5.4ms per frame
- **Throughput**: ~184 FPS
- **Latency**: <10ms total (including data transfer)
- **GPU Utilization**: ~40-60% during inference
- **Memory Usage**: ~2GB GPU VRAM, ~500MB system RAM

### Scaling Characteristics
- **Linear scaling**: 10 servers = 10x capacity
- **No Python overhead**: Instant startup
- **Consistent performance**: No GC pauses or interpreter overhead
- **Resource efficient**: Lower memory than Python

---

## 📞 Support & Next Steps

### If You Need Help
1. Check logs: Enable verbose logging in your application
2. Verify GPU: Run `nvidia-smi` to check GPU status
3. Test inference: Run simple test to isolate issues
4. Check versions: Ensure ONNX Runtime 1.22.0 is used

### Integration with Your System
- See `SETUP.md` for development setup
- See `RESULTS.md` for performance benchmarks
- See `README.md` for API documentation
- See example code in `cmd/simple-test/` and `cmd/benchmark/`

### Production Monitoring
Consider monitoring:
- Inference latency (should stay ~5ms)
- GPU temperature and utilization
- Memory usage
- Request queue depth
- Error rates

---

## 🎉 Summary

**You can now deploy with just:**
1. ✅ One executable (`service.exe`)
2. ✅ Three DLLs (ONNX Runtime)
3. ✅ One model file (`99.onnx`)

**No Python. No complexity. Just run.** 🚀

**Deployment time per server: ~2 minutes**
**Performance: 184 FPS (5.4ms per frame)**
**Scaling: Linear, no overhead**

You're ready for production! 🎯
