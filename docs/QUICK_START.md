# 🚀 Quick Start Guide - Real-Time Lip-Sync System

## TL;DR - Get Running in 5 Minutes

### 1. Prerequisites
- NVIDIA GPU (RTX 4090 recommended)
- ONNX Runtime 1.22.0: Download from https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0
- Extract to `C:/onnxruntime-1.22.0/`

### 2. Build Everything
```powershell
# Build Inference Server
cd D:\Projects\webcodecstest\go-inference-server
go build -o inference-server.exe ./cmd/server

# Build Compositing Server
cd D:\Projects\webcodecstest\go-compositing-server
go build -o compositing-server.exe ./cmd/server
go build -o test-client.exe test_client.go
```

### 3. Run the System
```powershell
# Option A: Use automated script
cd D:\Projects\webcodecstest
.\run-separated-test.ps1

# Option B: Manual start
# Terminal 1:
cd D:\Projects\webcodecstest\go-inference-server
.\inference-server.exe

# Terminal 2:
cd D:\Projects\webcodecstest\go-compositing-server
.\compositing-server.exe

# Terminal 3:
cd D:\Projects\webcodecstest\go-compositing-server
.\test-client.exe
```

### 4. Check Results
- Output frames: `D:\Projects\webcodecstest\go-compositing-server\test_output\`
- Look for: `batch_1_frame_0.jpg`, etc.
- Check console for performance metrics

---

## Configuration Quick Reference

### Key Settings

**Inference Server** (`go-inference-server/config.yaml`):
```yaml
server:
  worker_count_per_gpu: 8      # More = more concurrent clients
  
onnx:
  library_path: "C:/onnxruntime-1.22.0/lib/onnxruntime.dll"  # CRITICAL!
  
models:
  sanders:  # Add your model here
    model_path: "path/to/model.onnx"
```

**Compositing Server** (`go-compositing-server/config.yaml`):
```yaml
output:
  jpeg_quality: 75             # Lower = faster (65-85 recommended)
  
capacity:
  background_cache_frames: 600 # Higher = faster, more RAM
  
models:
  sanders:  # Add your model here
    background_dir: "path/to/frames/"
    crop_rects_path: "path/to/crop_rects.json"
    preload_backgrounds: true  # TRUE for best performance!
```

**Test Client** (`go-compositing-server/test_client.go`):
```go
const (
    batchSize  = 24    // Increase for better throughput (max 50)
    modelID    = "sanders"  // Change to your model name
)
```

---

## Performance Tuning Cheat Sheet

### Goal: Higher Throughput
```yaml
# Inference server
worker_count_per_gpu: 16  # More workers

# Test client
batchSize = 32  # Larger batches
```

### Goal: Lower Latency
```yaml
# Compositing server
output:
  jpeg_quality: 65  # Faster encoding
  
# Test client
batchSize = 8  # Smaller batches
```

### Goal: Save RAM
```yaml
# Compositing server
capacity:
  background_cache_frames: 100  # Less cache
  
models:
  sanders:
    preload_backgrounds: false  # Lazy load (slower but less RAM)
```

### Goal: More Users
```yaml
# Inference server
worker_count_per_gpu: 16  # More concurrent workers

# Scale horizontally
# Deploy multiple inference servers behind load balancer
```

---

## Common Commands

### Monitor GPU
```bash
nvidia-smi -l 1  # Update every 1 second
```

### Check Servers are Running
```powershell
Get-Process | Where-Object { $_.ProcessName -like "*inference*" -or $_.ProcessName -like "*compositing*" }
```

### Stop Everything
```powershell
Stop-Process -Name "inference-server" -Force
Stop-Process -Name "compositing-server" -Force
```

### Rebuild After Config Change
```powershell
# Inference server: Just restart (config is read at startup)
# Compositing server: Just restart (config is read at startup)  
# Test client: Rebuild if you changed constants
cd go-compositing-server
go build -o test-client.exe test_client.go
```

---

## Troubleshooting Quick Fixes

### "CUDA out of memory"
```yaml
# Reduce models or batch size
capacity:
  max_models_per_gpu: 500  # Was 1000
```

### "gRPC message too large"
```yaml
# Both servers:
server:
  max_message_size_mb: 200  # Was 100
  
# Test client:
maxMsgSize = 200 * 1024 * 1024
```

### Slow compositing (> 100ms)
```yaml
# Enable preload
models:
  sanders:
    preload_backgrounds: true
    
# Lower quality
output:
  jpeg_quality: 65
```

### Model not found
```yaml
# Check paths in config
models:
  sanders:
    model_path: "CORRECT/PATH/TO/model.onnx"  # Use absolute path!
    background_dir: "CORRECT/PATH/TO/frames/"
```

---

## Adding a New Model (User)

### 1. Prepare Files
```
minimal_server/models/new_user/
├── checkpoint/
│   └── model_best.onnx      # Your ONNX model
├── frames/                   # Background frames
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
└── crop_rects.json          # [[x1,y1,x2,y2], ...]
```

### 2. Update Inference Server Config
```yaml
# go-inference-server/config.yaml
models:
  new_user:
    model_path: "d:/path/to/minimal_server/models/new_user/checkpoint/model_best.onnx"
```

### 3. Update Compositing Server Config
```yaml
# go-compositing-server/config.yaml
models:
  new_user:
    background_dir: "d:/path/to/minimal_server/models/new_user/frames"
    crop_rects_path: "d:/path/to/minimal_server/models/new_user/crop_rects.json"
    num_frames: 523  # Number of background frames
    preload_backgrounds: true
```

### 4. Test
```go
// Change in test_client.go
const modelID = "new_user"  // Was "sanders"
```

Rebuild and run!

---

## Performance Expectations

### Single RTX 4090
- **Per client**: ~40 FPS
- **Max concurrent clients**: 8
- **Total throughput**: ~320 FPS

### Latency Breakdown (Batch 24)
- Inference: 400ms (67%)
- Compositing: 60ms (10%)
- gRPC: 140ms (23%)
- **Total: ~600ms** (~25ms per frame)

### Scaling
- **2 GPUs**: ~640 FPS
- **4 GPUs**: ~1280 FPS
- **8 GPUs**: ~2560 FPS

---

## System Requirements

### Minimum
- GPU: RTX 3090 (24GB)
- RAM: 16 GB
- CPU: 8 cores
- Storage: 100 GB SSD

### Recommended (Production)
- GPU: RTX 4090 (24GB) or A100
- RAM: 128 GB (or 2TB for 11,000 models)
- CPU: 32 cores
- Storage: 1 TB NVMe SSD
- Network: 10 Gbps

---

## File Locations Reference

| Component | Location |
|-----------|----------|
| Inference Server Binary | `go-inference-server/inference-server.exe` |
| Inference Server Config | `go-inference-server/config.yaml` |
| Compositing Server Binary | `go-compositing-server/compositing-server.exe` |
| Compositing Server Config | `go-compositing-server/config.yaml` |
| Test Client Binary | `go-compositing-server/test-client.exe` |
| Test Output | `go-compositing-server/test_output/` |
| Models | `minimal_server/models/` |
| ONNX Runtime | `C:/onnxruntime-1.22.0/` |

---

## Health Check

### Quick Test
```powershell
# Start servers, then:
cd go-compositing-server
.\test-client.exe

# Should see:
# ✅ Connected successfully
# ✅ Server Status: Healthy
# 🚀 Running 5 batches...
# [Success output with timing]
```

### Expected Output
```
Batch 2/5: GPU=0, frames=24
  ⚡ Inference:    400 ms
  🎨 Compositing:  60 ms
  📊 Total:        600 ms
  💾 Saved 24 frames to test_output/
```

If you see this, **everything is working!** 🎉

---

## Need More Help?

- **Full Documentation**: `PRODUCTION_GUIDE.md`
- **Architecture**: `ARCHITECTURE.md`
- **Technical Details**: `REALTIME_LIPSYNC_SYSTEM.md`

---

**Last Updated:** October 24, 2025  
**Quick Start Version:** 1.0.0
