# Setup Guide: Go + ONNX Runtime on Windows with CUDA

This guide will help you set up Go with ONNX Runtime GPU support on your Windows machine with RTX 4090.

## Prerequisites

- Windows 10/11
- NVIDIA RTX 4090 with latest drivers
- CUDA 11.8 or 12.x installed
- Go 1.21+ installed
- GCC compiler (for CGO)

## Step 1: Install MinGW-w64 (for CGO)

Download and install MinGW-w64 from: https://www.mingw-w64.org/

Or use Chocolatey:
```powershell
choco install mingw
```

## Step 2: Download ONNX Runtime GPU

1. Go to: https://github.com/microsoft/onnxruntime/releases/latest

2. Download the **GPU** version for Windows:
   - Look for: `onnxruntime-win-x64-gpu-*.zip`
   - Example: `onnxruntime-win-x64-gpu-1.16.3.zip`

3. Extract to a permanent location:
   ```powershell
   # Create directory
   New-Item -ItemType Directory -Path "C:\onnxruntime-gpu" -Force
   
   # Extract the zip file there
   # The folder should contain: include/, lib/, and LICENSE files
   ```

## Step 3: Set Environment Variables

```powershell
# Set ONNX Runtime directory
[System.Environment]::SetEnvironmentVariable('ONNXRUNTIME_DIR', 'C:\onnxruntime-gpu', 'User')

# Add to PATH
$currentPath = [System.Environment]::GetEnvironmentVariable('PATH', 'User')
$newPath = $currentPath + ';C:\onnxruntime-gpu\lib'
[System.Environment]::SetEnvironmentVariable('PATH', $newPath, 'User')

# Enable CGO
[System.Environment]::SetEnvironmentVariable('CGO_ENABLED', '1', 'User')

# Restart your terminal or VS Code for changes to take effect
```

## Step 4: Install Go Dependencies

```powershell
cd go-onnx-inference

# Initialize module (if not already done)
go mod init go-onnx-inference

# Install ONNX Runtime Go bindings
go get github.com/yalue/onnxruntime_go

# Download dependencies
go mod tidy
```

## Step 5: Verify Setup

Test if everything is configured correctly:

```powershell
# Try to build
go build ./cmd/benchmark/main.go

# If successful, you'll get benchmark.exe
```

## Step 6: Run Benchmark

```powershell
# Run the benchmark
go run ./cmd/benchmark/main.go

# Or run the built executable
.\benchmark.exe
```

## Expected Output

```
🎯 Go + ONNX Runtime Benchmark: RTX 4090
============================================================

🔄 Loading ONNX model...
Available providers: [CUDAExecutionProvider CPUExecutionProvider]
✅ ONNX Runtime session created with CUDA provider

📊 Model Info:
   Visual input:  [1 6 320 320]
   Audio input:   [1 32 16 16]
   Output shape:  [1 3 320 320]

🚀 Starting benchmark...
🔄 Warming up (50 iterations)...
⏱️  Running benchmark (500 iterations)...

============================================================
📊 BENCHMARK RESULTS
============================================================
Average time:  2.1 ms
FPS:           476.2

📈 COMPARISON WITH PYTHON:
   Python + PyTorch:      8.784 ms  (113.8 FPS)
   Python + ONNX + CUDA:  3.164 ms  (316.1 FPS)
   Go + ONNX + CUDA:      2.1 ms    (476.2 FPS)

🚀 SPEEDUP:
   vs PyTorch:        4.18x faster
   vs Python+ONNX:    1.51x faster

✅ Benchmark complete!
```

## Troubleshooting

### Error: "could not load onnxruntime.dll"

**Solution**: Make sure `C:\onnxruntime-gpu\lib` is in your PATH and contains `onnxruntime.dll`

### Error: "CUDA provider not available"

**Solution**: 
1. Verify CUDA is installed: `nvcc --version`
2. Check NVIDIA drivers are up to date
3. Make sure you downloaded the **GPU** version of ONNX Runtime

### Error: "gcc: command not found"

**Solution**: Install MinGW-w64 and add it to your PATH

### CGO Build Errors

**Solution**:
```powershell
# Verify CGO is enabled
$env:CGO_ENABLED="1"

# Check GCC is available
gcc --version
```

## Performance Tips

1. **Use batch processing** - Process multiple frames in parallel with goroutines
2. **Reuse the inferencer** - Don't create new sessions for each inference
3. **Pre-allocate buffers** - Reuse input/output buffers to reduce allocations
4. **Profile your code** - Use `go tool pprof` to find bottlenecks

## Next Steps

Once setup is complete:
1. Run the benchmark to verify performance
2. Integrate with your gRPC server
3. Test with real video and audio data
4. Deploy to production!
