# Quick Start Guide - Real-Time Lip Sync System

Get up and running in 5 minutes!

## Prerequisites

✅ **NVIDIA GPU** (RTX 2060 or better, 6GB+ VRAM)
✅ **Python 3.12+** with CUDA support
✅ **Git** (for cloning if needed)
✅ **Windows** (Linux/Mac: slight command adjustments needed)

---

## Option 1: Single Server (Development)

**Best for:** Testing, single user, development

### Step 1: Install Dependencies
```bash
cd D:\Projects\webcodecstest\minimal_server
pip install -r requirements.txt
```

### Step 2: Generate Protocol Buffers
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto
```

### Step 3: Start gRPC Server
```bash
python optimized_grpc_server.py
```

**Wait for:** `✅ All models loaded and ready!` (takes ~8 seconds)

### Step 4: Start Go Proxy
```bash
cd ..\grpc-websocket-proxy
.\build.bat
.\run.bat
```

### Step 5: Open Browser
Navigate to: **http://localhost:8086/**

Click **"Start"** → **"Auto-play"**

**Expected:** 55-65 FPS, 17-22ms latency

---

## Option 2: Multi-Process (Production with RTX 6000 Ada)

**Best for:** Production, multiple users, RTX 6000 Ada GPU

### Step 1: Install Dependencies (Same as Option 1)
```bash
cd D:\Projects\webcodecstest\minimal_server
pip install -r requirements.txt
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto
```

### Step 2: Start Multiple gRPC Servers
```powershell
cd minimal_server
.\start_multi_grpc.ps1 -NumProcesses 4
```

**Wait for:** All servers to show `✅ All models loaded and ready!` (~35-40 seconds total)

You'll see:
```
Starting gRPC server 1 on port 50051...
Starting gRPC server 2 on port 50052...
Starting gRPC server 3 on port 50053...
Starting gRPC server 4 on port 50054...
```

### Step 3: Start Multi-Backend Proxy
```bash
cd ..\grpc-websocket-proxy
.\build.bat
.\run_multi.bat 4 50051
```

You'll see:
```
📍 Backend configuration:
   [1] localhost:50051
   [2] localhost:50052
   [3] localhost:50053
   [4] localhost:50054

✅ Connected to 4/4 gRPC servers
⚖️  Load balancing: Round-robin across 4 backends
```

### Step 4: Open Browser
Navigate to: **http://localhost:8086/**

Click **"Start"** → **"Auto-play"**

**Expected (RTX 6000 Ada):** 180-220 FPS aggregate, 17-22ms latency

---

## Option 3: Direct WebSocket (Simplest)

**Best for:** Quick testing without gRPC/Go

### Step 1: Install Dependencies
```bash
cd minimal_server
pip install -r requirements.txt
```

### Step 2: Start WebSocket Server
```bash
python optimized_websocket_server.py
```

### Step 3: Open Browser Client
Open: `realtime-lipsync-binary.html` in your browser

**Expected:** 55-65 FPS, 15-18ms latency

---

## Verification

### Check if it's working:

1. **Browser shows video playback** ✅
2. **FPS counter shows 50+ FPS** ✅
3. **Latency < 25ms** ✅
4. **No console errors** ✅

### Common Issues:

**Problem:** "Connection refused"
**Solution:** Make sure gRPC server is running (`python optimized_grpc_server.py`)

**Problem:** "Models not found"
**Solution:** Run model pre-rendering: `python demo_dynamic_loading.py`

**Problem:** "CUDA out of memory"
**Solution:** Reduce backend count: `.\start_multi_grpc.ps1 -NumProcesses 2`

---

## Performance Testing

### Test Single Server:
```bash
cd minimal_server
python optimized_grpc_client.py --port 50051 --count 100
```

Expected output:
```
✅ Total: 100 requests in 1.72s
✅ Throughput: 58.1 FPS
✅ Avg latency: 17.2ms
```

### Test Multi-Process (if using Option 2):
```bash
python test_multi_process.py --ports 50051 50052 50053 50054
```

Expected output (RTX 6000 Ada):
```
📈 Total throughput: 216.4 FPS
📈 Speedup: 3.7x
✅ EXCELLENT! Your RTX 6000 Ada has great multi-process support!
```

---

## What's Next?

### For Development:
- Modify models in `model/unet_328.py`
- Test with Python client: `optimized_grpc_client.py`
- Monitor GPU: `nvidia-smi dmon -s um`

### For Production:
- Read: `minimal_server/RTX6000_OPTIMIZATION_GUIDE.md`
- Scale backends: `.\start_multi_grpc.ps1 -NumProcesses 8`
- Monitor health: `curl http://localhost:8086/health`

### For Understanding:
- Architecture: `PROJECT_SUMMARY.md`
- GPU parallelism: `GPU_PARALLELISM_GUIDE.md`
- Python GIL: `CONCURRENCY_GUIDE.md`

---

## Commands Cheat Sheet

### Start Single gRPC Server:
```bash
cd minimal_server
python optimized_grpc_server.py
```

### Start Multi-Process (4 servers):
```powershell
cd minimal_server
.\start_multi_grpc.ps1 -NumProcesses 4
```

### Stop All Servers:
```powershell
Stop-Process -Id (Get-Content minimal_server\grpc_processes.txt)
```

### Start Go Proxy (Single Backend):
```bash
cd grpc-websocket-proxy
.\run.bat
```

### Start Go Proxy (Multi-Backend):
```bash
cd grpc-websocket-proxy
.\run_multi.bat 4 50051
```

### Check Health:
```bash
curl http://localhost:8086/health
```

### Test Performance:
```bash
cd minimal_server
python test_multi_process.py --ports 50051 50052 50053 50054
```

### Monitor GPU:
```bash
nvidia-smi dmon -s um -c 30
```

---

## Architecture Diagram

```
┌──────────┐
│ Browser  │
│  Client  │
└────┬─────┘
     │ WebSocket Binary
     │ (24-40 KB JPEG)
     ▼
┌────────────────┐
│   Go Proxy     │
│ Load Balancer  │
│  (Port 8086)   │
└────┬───────────┘
     │ gRPC (Round-Robin)
     │
     ├──────────┬──────────┬──────────┐
     ▼          ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Python  │ │ Python  │ │ Python  │ │ Python  │
│  gRPC   │ │  gRPC   │ │  gRPC   │ │  gRPC   │
│ :50051  │ │ :50052  │ │ :50053  │ │ :50054  │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │
     └───────────┴───────────┴───────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  RTX 6000 Ada   │
            │  142 SMs        │
            │  96 GB VRAM     │
            └─────────────────┘
```

---

## Troubleshooting

### Server won't start

**Error:** `ModuleNotFoundError: No module named 'grpc'`

**Solution:**
```bash
pip install grpcio grpcio-tools
```

---

### Go proxy won't build

**Error:** `package github.com/gorilla/websocket is not in GOROOT`

**Solution:**
```bash
cd grpc-websocket-proxy
go mod download
```

---

### Low FPS (< 30)

**Check:**
1. GPU utilization: `nvidia-smi`
   - Should be 85-95%
2. CPU usage: Task Manager
   - Should be < 50% per core
3. Network latency:
   - Check browser console for warnings

**Solutions:**
- Close other GPU applications
- Reduce video quality
- Use binary protocol (not JSON)

---

### Multi-process no speedup

**Symptom:** 4 processes = 65 FPS (should be 200+ on RTX 6000 Ada)

**Diagnosis:**
```bash
nvidia-smi --query-gpu=name --format=csv
```

**If RTX 2060-4090 (Consumer):**
- This is expected (time-slicing limitation)
- Use batching instead: Edit `optimized_grpc_server.py`, set `BATCH_SIZE = 4`

**If RTX 6000 Ada:**
- Check driver version: `nvidia-smi`
- Need driver 535+ for good multi-process support
- Update: https://www.nvidia.com/download/index.aspx

---

## Need Help?

📖 **Full Documentation:** `minimal_server/PROJECT_SUMMARY.md`

🔧 **GPU Issues:** `minimal_server/GPU_PARALLELISM_GUIDE.md`

⚡ **Performance:** `minimal_server/RTX6000_OPTIMIZATION_GUIDE.md`

🐍 **Python GIL:** `minimal_server/CONCURRENCY_GUIDE.md`

🌐 **Multi-Backend:** `grpc-websocket-proxy/README_MULTI_BACKEND.md`

---

## Success Criteria

✅ **Single server:** 55-65 FPS, 17-22ms latency
✅ **Multi-process (RTX 6000 Ada):** 180-220 FPS with 4 servers
✅ **Multi-process (Consumer GPU):** 60-70 FPS (use batching instead!)
✅ **Browser playback:** Smooth video, no dropped frames
✅ **Health check:** Returns `{"healthy": true}`

**Enjoy your real-time lip-sync system! 🎉**
