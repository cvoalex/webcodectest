# Deployment Scenarios

Quick reference for common deployment scenarios with exact commands and expected performance.

## Scenario 1: Development (Single GPU, Single Process)

**Use Case:** Development, testing, debugging  
**Hardware:** Any NVIDIA GPU with 4GB+ VRAM  
**Performance:** 40-50 FPS  

```powershell
# Start server
python optimized_grpc_server.py --port 50051

# Start proxy (in another terminal)
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 1

# Open browser
# Navigate to http://localhost:8086/
```

**Expected:**
- Startup: 8-10 seconds
- Latency: 25-30ms
- Throughput: 40-50 FPS
- Concurrent users: 5-10

---

## Scenario 2: Small Production (Single GPU, Multi-Process)

**Use Case:** Small business, prototype, <100 users  
**Hardware:** RTX 4090, RTX 3090, or RTX 6000 Ada (24-48GB VRAM)  
**Performance:** 180-300 FPS  

```powershell
# Start 6 processes on GPU 0
.\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 6

# Start proxy
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 6
```

**Expected:**
- Startup: ~45 seconds (6 processes)
- Latency: 17-22ms
- Throughput: 180-300 FPS
- Concurrent users: 50-100

---

## Scenario 3: Medium Production (2-4 GPUs)

**Use Case:** Growing business, 100-500 users  
**Hardware:** 2-4 × RTX 4090 or RTX 6000 Ada  
**Performance:** 720-1,200 FPS  

```powershell
# Start 4 GPUs × 6 processes = 24 servers
.\start_multi_gpu.ps1 -NumGPUs 4 -ProcessesPerGPU 6

# Start proxy
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 24
```

**Expected:**
- Startup: ~3 minutes (24 processes)
- Latency: 17-22ms
- Throughput: 720-1,200 FPS
- Concurrent users: 200-400

---

## Scenario 4: Large Production (8 GPUs)

**Use Case:** Large enterprise, high traffic, 500-1,000+ users  
**Hardware:** 8 × RTX 6000 Ada (48GB each)  
**Performance:** 1,440-2,400 FPS  

```powershell
# Start 8 GPUs × 6 processes = 48 servers
.\start_multi_gpu.ps1 -NumGPUs 8 -ProcessesPerGPU 6

# Start proxy
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 48
```

**Expected:**
- Startup: ~6-7 minutes (48 processes)
- Latency: 17-22ms
- Throughput: 1,440-2,400 FPS
- Concurrent users: 500-1,000+

---

## Scenario 5: Linux with MPS (Maximum Performance)

**Use Case:** Maximum performance on professional GPUs  
**Hardware:** Linux/WSL2 with RTX 6000 Ada, A100, or H100  
**Performance:** 4-7× single-process baseline  

```bash
# Enable MPS
sudo nvidia-cuda-mps-control -d

# Start with MPS enabled
./start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 8 -EnableMPS

# Start proxy
cd ../grpc-websocket-proxy
./proxy --start-port 50051 --num-servers 8
```

**Expected (Single GPU with MPS):**
- Startup: ~60 seconds (8 processes)
- Latency: 15-20ms
- Throughput: 300-400 FPS
- Concurrent users: 100-150

**Expected (8 GPUs with MPS):**
- Startup: ~8 minutes (64 processes)
- Latency: 15-20ms
- Throughput: 2,400-3,200 FPS
- Concurrent users: 800-1,200

---

## Scenario 6: Multi-Model Service (20+ Models)

**Use Case:** Multiple avatars, celebrities, characters  
**Hardware:** Any setup from Scenarios 1-5  
**Performance:** Same as base scenario (no penalty!)  

```powershell
# 1. Prepare models (one-time)
python batch_prepare_models.py --workers 4

# 2. Start servers (same as any scenario)
.\start_multi_gpu.ps1 -NumGPUs 4 -ProcessesPerGPU 6

# 3. Start proxy
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 24
```

**Storage Requirements:**
- 10 models: 500 MB disk, 1 GB VRAM
- 20 models: 1 GB disk, 2 GB VRAM
- 50 models: 2.5 GB disk, 5 GB VRAM

**Performance:** Same as base scenario - models are free!

---

## Scenario 7: Cloud Deployment (AWS, Azure, GCP)

**Use Case:** Cloud-based service with autoscaling  
**Hardware:** Cloud GPU instances  
**Performance:** Depends on instance type  

### AWS (p4d.24xlarge - 8 × A100 40GB)

```bash
# Install dependencies
sudo apt update
sudo apt install -y nvidia-driver-535 ffmpeg
pip install -r requirements.txt

# Enable MPS
sudo nvidia-cuda-mps-control -d

# Start service
./start_multi_gpu.ps1 -NumGPUs 8 -ProcessesPerGPU 6 -EnableMPS

# Start proxy
cd ../grpc-websocket-proxy
./proxy --start-port 50051 --num-servers 48
```

**Cost:** ~$32/hour (~$23,000/month)  
**Performance:** 2,400-3,200 FPS  
**Users:** 800-1,200 concurrent  

### Azure (Standard_ND96asr_v4 - 8 × A100 40GB)

Same as AWS setup.

**Cost:** ~$27/hour (~$19,500/month)  
**Performance:** 2,400-3,200 FPS  
**Users:** 800-1,200 concurrent  

### GCP (a2-ultragpu-8g - 8 × A100 40GB)

Same as AWS setup.

**Cost:** ~$28/hour (~$20,200/month)  
**Performance:** 2,400-3,200 FPS  
**Users:** 800-1,200 concurrent  

---

## Scenario 8: Development with Consumer GPU

**Use Case:** Budget development, testing  
**Hardware:** GTX 1080, RTX 2060, RTX 3060 (8-12GB VRAM)  
**Performance:** 60-120 FPS  

```powershell
# Start 3 processes (don't overload consumer GPU)
.\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 3

# Start proxy
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 3
```

**Expected:**
- Startup: ~25 seconds (3 processes)
- Latency: 25-35ms
- Throughput: 60-120 FPS
- Concurrent users: 20-40

---

## Scenario 9: Docker Deployment

**Use Case:** Containerized deployment, Kubernetes  
**Hardware:** Any NVIDIA GPU with Docker + nvidia-docker2  
**Performance:** Depends on hardware  

```bash
# Build image
docker build -t lipsync-server:latest .

# Run single container (1 GPU)
docker run --gpus device=0 -p 50051:50051 lipsync-server:latest

# Run multiple containers (4 GPUs, 6 processes each)
for gpu in 0 1 2 3; do
  for proc in 0 1 2 3 4 5; do
    port=$((50051 + gpu*6 + proc))
    docker run -d \
      --gpus device=$gpu \
      -p $port:50051 \
      --name lipsync-gpu${gpu}-proc${proc} \
      lipsync-server:latest
  done
done

# Start proxy
docker run -d -p 8086:8086 lipsync-proxy:latest --num-servers 24
```

---

## Scenario 10: Edge Deployment (Jetson)

**Use Case:** Edge AI, embedded systems, low latency  
**Hardware:** NVIDIA Jetson AGX Orin (32GB), Jetson Orin NX  
**Performance:** 20-40 FPS  

```bash
# Install dependencies (ARM)
sudo apt install -y ffmpeg
pip3 install -r requirements.txt

# Start single process (Jetson has limited resources)
python3 optimized_grpc_server.py --port 50051

# Start proxy
cd ../grpc-websocket-proxy
go run main.go --start-port 50051 --num-servers 1
```

**Expected:**
- Startup: 15-20 seconds
- Latency: 30-40ms
- Throughput: 20-40 FPS
- Concurrent users: 3-5
- Power: 15-60W

---

## Performance Comparison Table

| Scenario | GPUs | Processes | FPS | Latency | Users | Cost/Hour |
|----------|------|-----------|-----|---------|-------|-----------|
| Dev Single | 1 | 1 | 50 | 30ms | 10 | $0.50 |
| Small Prod | 1 | 6 | 240 | 20ms | 80 | $1.00 |
| Medium Prod | 4 | 24 | 960 | 20ms | 320 | $4.00 |
| Large Prod | 8 | 48 | 1,920 | 20ms | 640 | $8.00 |
| MPS Single | 1 | 8 | 320 | 18ms | 100 | $2.50 |
| MPS Large | 8 | 64 | 2,560 | 18ms | 850 | $20.00 |
| Consumer | 1 | 3 | 90 | 28ms | 30 | $0.30 |
| Cloud (AWS) | 8 | 48 | 2,400 | 18ms | 800 | $32.00 |
| Edge (Jetson) | 1 | 1 | 30 | 35ms | 5 | $0.00* |

*Jetson runs on device, no cloud cost

---

## Monitoring Commands

### Check GPU Usage

```bash
# Real-time monitoring
nvidia-smi -l 1

# Detailed per-process
nvidia-smi pmon -s um -c 10

# GPU utilization only
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1
```

### Check Server Health

```powershell
# Check all servers
for ($port=50051; $port -le 50098; $port++) {
    curl "http://localhost:$port/health" 2>&1 | Out-Null
    if ($?) { Write-Host "✅ Port $port" } else { Write-Host "❌ Port $port" }
}
```

### Check Proxy Statistics

```bash
# View load balancing stats
curl http://localhost:8086/stats
```

### Monitor Latency

```bash
# Test end-to-end latency
python test_multi_process.py --ports 50051-50098 --num-requests 100
```

---

## Troubleshooting by Scenario

### Scenario 1-2: Out of Memory on Single GPU

**Solution:** Reduce processes per GPU
```powershell
.\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 4
```

### Scenario 3-4: Uneven GPU Load

**Solution:** Check GPU assignment
```bash
nvidia-smi pmon -s um
# Should show even distribution across GPUs
```

### Scenario 5: MPS Not Working

**Solution:** Verify MPS is running
```bash
nvidia-cuda-mps-control -d
ps aux | grep mps
```

### Scenario 6: Model Not Loading

**Solution:** Check frame directory
```bash
ls data/model_name/
# Should have frame_0000.jpg to frame_3304.jpg
```

### Scenario 7-9: Network Latency

**Solution:** Check network configuration
```bash
# Test internal latency
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8086/health
```

### Scenario 10: Jetson Power Throttling

**Solution:** Increase power mode
```bash
sudo nvpmodel -m 0  # Max performance
sudo jetson_clocks  # Max clocks
```

---

## Quick Decision Tree

```
Need < 50 FPS?
  └─ Scenario 1 (Single process)

Need 50-300 FPS?
  ├─ Consumer GPU? → Scenario 8 (3 processes)
  └─ Professional GPU? → Scenario 2 (6 processes)

Need 300-1,000 FPS?
  └─ Scenario 3 (2-4 GPUs)

Need 1,000-2,000 FPS?
  └─ Scenario 4 (8 GPUs)

Need > 2,000 FPS?
  └─ Scenario 5 (8 GPUs + MPS)

Need multiple models?
  └─ Scenario 6 (Add to any above)

Cloud deployment?
  └─ Scenario 7 (AWS/Azure/GCP)

Containerized?
  └─ Scenario 9 (Docker/Kubernetes)

Edge device?
  └─ Scenario 10 (Jetson)
```

---

## Next Steps

1. **Choose your scenario** based on requirements
2. **Follow the exact commands** for that scenario
3. **Monitor performance** using provided commands
4. **Scale up or down** by adjusting GPU/process count
5. **Add models** using Scenario 6 guide

For detailed guides, see:
- [QUICKSTART.md](../QUICKSTART.md) - Basic setup
- [MULTI_GPU_GUIDE.md](MULTI_GPU_GUIDE.md) - Multi-GPU deployment
- [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md) - Multiple models
