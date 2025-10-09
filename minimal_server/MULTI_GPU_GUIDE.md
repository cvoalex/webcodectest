# Multi-GPU Deployment Guide

Complete guide for deploying across 1-8 GPUs with flexible process configuration.

## Quick Start

### Single GPU (Development):
```powershell
.\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 4
# 4 servers, ports 50051-50054
# Expected: 120-200 FPS
```

### Dual GPU (Small Production):
```powershell
.\start_multi_gpu.ps1 -NumGPUs 2 -ProcessesPerGPU 6
# 12 servers, ports 50051-50062
# Expected: 350-700 FPS
```

### Quad GPU (Medium Production):
```powershell
.\start_multi_gpu.ps1 -NumGPUs 4 -ProcessesPerGPU 6
# 24 servers, ports 50051-50074
# Expected: 700-1,400 FPS
```

### Octa GPU (Large Production):
```powershell
.\start_multi_gpu.ps1 -NumGPUs 8 -ProcessesPerGPU 6
# 48 servers, ports 50051-50098
# Expected: 1,400-2,800 FPS
```

---

## Performance Expectations

### Consumer GPUs (RTX 2060-4090)

**Architecture:** Time-slicing (processes share GPU sequentially)

| GPUs | Processes/GPU | Total Servers | Expected FPS | Speedup |
|------|---------------|---------------|--------------|---------|
| 1 | 1 | 1 | 58 | 1.0x |
| 1 | 4 | 4 | 90-140 | 1.5-2.4x |
| 1 | 6 | 6 | 100-160 | 1.7-2.7x |
| 2 | 4 | 8 | 180-280 | 3.1-4.8x |
| 2 | 6 | 12 | 200-320 | 3.4-5.5x |
| 4 | 4 | 16 | 360-560 | 6.2-9.7x |
| 4 | 6 | 24 | 400-640 | 6.9-11.0x |
| 8 | 4 | 32 | 720-1,120 | 12.4-19.3x |
| 8 | 6 | 48 | 800-1,280 | 13.8-22.1x |

**Key Insight:** Speedup comes primarily from multiple GPUs, not multi-process on single GPU.

---

### Professional GPUs (RTX 6000 Ada, A100, H100)

**Architecture:** True concurrency (with MPS) or better time-slicing (without MPS)

#### Without MPS (Windows):
| GPUs | Processes/GPU | Total Servers | Expected FPS | Speedup |
|------|---------------|---------------|--------------|---------|
| 1 | 1 | 1 | 58 | 1.0x |
| 1 | 4 | 4 | 140-180 | 2.4-3.1x |
| 1 | 6 | 6 | 180-240 | 3.1-4.1x |
| 1 | 8 | 8 | 210-280 | 3.6-4.8x |
| 2 | 6 | 12 | 360-480 | 6.2-8.3x |
| 4 | 6 | 24 | 720-960 | 12.4-16.6x |
| 8 | 6 | 48 | 1,440-1,920 | 24.8-33.1x |

#### With MPS (Linux/WSL2):
| GPUs | Processes/GPU | Total Servers | Expected FPS | Speedup |
|------|---------------|---------------|--------------|---------|
| 1 | 1 | 1 | 58 | 1.0x |
| 1 | 4 | 4 | 180-220 | 3.1-3.8x |
| 1 | 6 | 6 | 240-300 | 4.1-5.2x |
| 1 | 8 | 8 | 280-380 | 4.8-6.6x |
| 2 | 6 | 12 | 480-600 | 8.3-10.3x |
| 4 | 6 | 24 | 960-1,200 | 16.6-20.7x |
| 8 | 6 | 48 | 1,920-2,400 | 33.1-41.4x |
| 8 | 8 | 64 | 2,240-2,880 | 38.6-49.7x |

**Key Insight:** Multi-process works well on professional GPUs even without MPS.

---

## Configuration Guidelines

### How Many Processes Per GPU?

#### Consumer GPUs:
```powershell
# Minimal overhead, some gain
.\start_multi_gpu.ps1 -ProcessesPerGPU 2

# Moderate gain (recommended)
.\start_multi_gpu.ps1 -ProcessesPerGPU 4

# Diminishing returns beyond 4
.\start_multi_gpu.ps1 -ProcessesPerGPU 6
```

**Recommendation:** 4 processes per consumer GPU

#### Professional GPUs (Without MPS):
```powershell
# Good baseline
.\start_multi_gpu.ps1 -ProcessesPerGPU 4

# Sweet spot (recommended)
.\start_multi_gpu.ps1 -ProcessesPerGPU 6

# High throughput
.\start_multi_gpu.ps1 -ProcessesPerGPU 8
```

**Recommendation:** 6 processes per professional GPU on Windows

#### Professional GPUs (With MPS on Linux):
```powershell
# Good performance
.\start_multi_gpu.ps1 -ProcessesPerGPU 6 -EnableMPS

# Excellent throughput
.\start_multi_gpu.ps1 -ProcessesPerGPU 8 -EnableMPS

# Maximum throughput
.\start_multi_gpu.ps1 -ProcessesPerGPU 12 -EnableMPS
```

**Recommendation:** 8 processes per professional GPU on Linux with MPS

---

## Memory Planning

### VRAM Requirements Per GPU:

**Per Model Instance:**
- Model weights: ~100 MB
- Overhead: ~50 MB
- **Total per process:** ~150 MB

**Examples:**

| GPU VRAM | Max Processes | Recommended | Headroom |
|----------|---------------|-------------|----------|
| 6 GB | 40 | 8 | Safe |
| 8 GB | 53 | 10 | Safe |
| 12 GB | 80 | 12 | Safe |
| 16 GB | 106 | 16 | Safe |
| 24 GB | 160 | 20 | Safe |
| 48 GB | 320 | 40 | Plenty |

**20 Models Across System:**
- Disk space: ~1.4 GB (pre-rendered frames + videos)
- System RAM: ~4 GB (video cache)
- VRAM per GPU: Same as single model (~150 MB per process)

---

## Complete Deployment Examples

### Example 1: Developer Workstation (1× RTX 4090)
```powershell
# Start 4 servers on single GPU
.\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 4

# Wait 35 seconds for initialization...

# Start proxy
cd ..\grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 4

# Open browser: http://localhost:8086/
```

**Expected:** 90-140 FPS, handles 3-5 concurrent users

---

### Example 2: Small Studio (2× RTX 6000 Ada)
```powershell
# Start 6 servers per GPU = 12 total
.\start_multi_gpu.ps1 -NumGPUs 2 -ProcessesPerGPU 6

# Wait 90 seconds (12 servers × 8s delay)...

# Start proxy
cd ..\grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 12

# Test performance
cd ..\minimal_server
python test_multi_process.py --port-range 50051-50062
```

**Expected:** 360-480 FPS, handles 10-15 concurrent users

---

### Example 3: Production Server (4× RTX 6000 Ada)
```powershell
# Start 6 servers per GPU = 24 total
.\start_multi_gpu.ps1 -NumGPUs 4 -ProcessesPerGPU 6

# Wait 3-4 minutes (24 servers × 8s delay)...

# Start proxy
cd ..\grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 24

# Monitor GPUs
nvidia-smi dmon -s um -c 100
```

**Expected:** 720-960 FPS, handles 25-35 concurrent users

---

### Example 4: Large-Scale Production (8× RTX 6000 Ada)
```powershell
# Start 6 servers per GPU = 48 total
.\start_multi_gpu.ps1 -NumGPUs 8 -ProcessesPerGPU 6

# Wait 6-7 minutes (48 servers × 8s delay)...

# Start proxy with all backends
cd ..\grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 48

# Check health
curl http://localhost:8086/health
```

**Expected:** 1,440-1,920 FPS, handles 50-70 concurrent users

---

## Enabling MPS (Linux/WSL2 Only)

### What is MPS?

**Multi-Process Service (MPS)** allows true concurrent kernel execution on professional GPUs.

**Benefits:**
- 3-7x speedup on single GPU (vs 1.5-2.5x without MPS)
- Better GPU utilization (80-95% vs 60-75%)
- Lower context switching overhead

**Limitations:**
- **Linux/WSL2 only** (not supported on Windows native)
- Requires Volta+ architecture (all RTX 6000, A100, H100)
- Requires root or specific permissions

### Setup MPS on Linux:

```bash
# Start MPS daemon (one-time per boot)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # All GPUs
sudo nvidia-cuda-mps-control -d

# Verify MPS is running
ps aux | grep mps

# Now start your servers
./start_multi_gpu.ps1 -NumGPUs 8 -ProcessesPerGPU 8 -EnableMPS
```

### Setup MPS on WSL2:

```bash
# In WSL2 Ubuntu
wsl

# Start MPS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-cuda-mps-control -d

# Run servers in WSL2
cd /mnt/d/Projects/webcodecstest/minimal_server
python optimized_grpc_server.py --port 50051 &
python optimized_grpc_server.py --port 50052 &
# ... etc
```

---

## Monitoring & Health Checks

### GPU Utilization:

```bash
# Real-time monitoring (30 samples)
nvidia-smi dmon -s um -c 30

# Watch memory usage
watch -n 1 nvidia-smi

# Per-GPU utilization
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory --format=csv -l 5
```

**Expected Utilization:**

| Configuration | SM Util | Memory | Notes |
|---------------|---------|---------|-------|
| 1 GPU, 1 process | 18-25% | 150 MB | Baseline |
| 1 GPU, 4 processes (consumer) | 60-75% | 600 MB | Time-sliced |
| 1 GPU, 4 processes (professional, no MPS) | 65-80% | 600 MB | Better |
| 1 GPU, 4 processes (professional, MPS) | 75-90% | 600 MB | Excellent |

### Process Health:

```powershell
# Check running servers
Get-Process python | Where-Object {$_.MainWindowTitle -like "*gRPC*"}

# Check PIDs from file
Get-Content grpc_processes.txt | ForEach-Object {
    Get-Process -Id $_ -ErrorAction SilentlyContinue
}

# Stop all servers
Stop-Process -Id (Get-Content grpc_processes.txt)
```

### Proxy Health Check:

```bash
# Check backend status
curl http://localhost:8086/health

# Pretty print JSON
curl http://localhost:8086/health | python -m json.tool
```

**Response:**
```json
{
  "healthy": true,
  "total_backends": 48,
  "healthy_count": 47,
  "backends": [
    {
      "address": "localhost:50051",
      "healthy": true,
      "total_requests": 1245,
      "error_count": 0
    },
    ...
  ]
}
```

---

## Troubleshooting

### Problem: Low Speedup on Professional GPU

**Symptoms:** 4 processes only gives 1.5x speedup (should be 3x)

**Diagnosis:**
```bash
# Check if MPS is available
nvidia-smi --query-gpu=name,driver_version --format=csv

# Check GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Solutions:**
1. **Update drivers:** Need 535+ for best multi-process support
2. **Enable MPS:** Use Linux/WSL2 with MPS enabled
3. **Check GPU mode:** Should not be in EXCLUSIVE_PROCESS mode without MPS

---

### Problem: CUDA Out of Memory

**Symptoms:** Server crashes with "CUDA out of memory"

**Diagnosis:**
```bash
# Check VRAM usage
nvidia-smi

# Check per-process memory
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

**Solutions:**
1. **Reduce processes per GPU:**
   ```powershell
   .\start_multi_gpu.ps1 -ProcessesPerGPU 4  # Instead of 6
   ```

2. **Use fewer GPUs:**
   ```powershell
   .\start_multi_gpu.ps1 -NumGPUs 4  # Instead of 8
   ```

3. **Check for memory leaks:**
   - Restart servers periodically
   - Monitor memory over time

---

### Problem: Servers Not Starting

**Symptoms:** PowerShell windows close immediately

**Diagnosis:**
```powershell
# Check if port is already in use
Test-NetConnection -ComputerName localhost -Port 50051

# Check Python errors
python optimized_grpc_server.py --port 50051
```

**Solutions:**
1. **Kill existing processes:**
   ```powershell
   Stop-Process -Name python -Force
   ```

2. **Use different port range:**
   ```powershell
   .\start_multi_gpu.ps1 -BasePort 60051
   ```

3. **Check Python environment:**
   ```powershell
   python -c "import grpc; print(grpc.__version__)"
   ```

---

### Problem: Uneven Load Distribution

**Symptoms:** Some GPUs at 90%, others at 20%

**Diagnosis:**
```bash
# Monitor per-GPU utilization
nvidia-smi dmon -s u
```

**Causes:**
- Model-specific requests (users prefer certain models)
- Network routing issues
- Some backends marked unhealthy

**Solutions:**
1. **Check proxy health:**
   ```bash
   curl http://localhost:8086/health
   ```

2. **Restart unhealthy backends:** Find PIDs from health check, restart those processes

3. **Verify round-robin:** Check proxy logs for even distribution

---

## Scaling Guidelines

### Number of Users vs Configuration

| Concurrent Users | GPUs | Processes/GPU | Total FPS | Notes |
|------------------|------|---------------|-----------|-------|
| 1-3 | 1 | 4 | 120-180 | Dev/testing |
| 3-10 | 1 | 6 | 180-300 | Small demo |
| 10-20 | 2 | 6 | 360-600 | Small production |
| 20-40 | 4 | 6 | 720-1,200 | Medium production |
| 40-80 | 8 | 6 | 1,440-2,400 | Large production |
| 80-150 | 8 | 8 | 1,920-3,200 | Enterprise |

**Formula:** Concurrent Users ≈ Total FPS / 30 (assuming 30 FPS per user)

---

## Cost-Benefit Analysis

### Consumer GPU Scaling (RTX 4090):

| Config | Cost | FPS | FPS/$ | Best For |
|--------|------|-----|-------|----------|
| 1× RTX 4090 | $1,600 | 100-160 | 0.063-0.100 | Development |
| 2× RTX 4090 | $3,200 | 200-320 | 0.063-0.100 | Small studio |
| 4× RTX 4090 | $6,400 | 400-640 | 0.063-0.100 | Medium studio |

**Linear scaling** - Speedup comes from multiple GPUs

### Professional GPU Scaling (RTX 6000 Ada):

| Config | Cost | FPS | FPS/$ | Best For |
|--------|------|-----|-------|----------|
| 1× RTX 6000 | $6,800 | 180-300 | 0.026-0.044 | Small production |
| 2× RTX 6000 | $13,600 | 360-600 | 0.026-0.044 | Medium production |
| 4× RTX 6000 | $27,200 | 720-1,200 | 0.026-0.044 | Large production |
| 8× RTX 6000 | $54,400 | 1,440-2,400 | 0.026-0.044 | Enterprise |

**Super-linear scaling** - Multi-process + multiple GPUs

**Key Insight:** Professional GPUs provide better per-GPU throughput but higher cost. Consumer GPUs need more GPUs for same throughput.

---

## Next Steps

1. **Start with 1 GPU, 4 processes** - Baseline test
2. **Measure actual FPS** - Use `test_multi_process.py`
3. **Scale up gradually** - Add GPUs or processes based on needs
4. **Monitor health** - Use `nvidia-smi` and `/health` endpoint
5. **Optimize** - Adjust processes per GPU based on results

**Good luck with your deployment! 🚀**
