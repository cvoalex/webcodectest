# Multi-Backend Load Balancing Proxy

This Go proxy now supports load balancing across multiple gRPC backend servers using a round-robin algorithm.

## Features

✅ **Round-Robin Load Balancing**: Distributes requests evenly across all healthy backends
✅ **Automatic Health Checking**: Marks backends as unhealthy on errors, retries on success
✅ **Flexible Configuration**: Support both explicit addresses or port ranges
✅ **Per-Backend Statistics**: Track total requests and error counts for each backend
✅ **Health Check Endpoint**: Monitor backend status via HTTP `/health`

## Quick Start

### 1. Start Multiple gRPC Servers

```powershell
cd ..\minimal_server
powershell -File .\start_multi_grpc.ps1 -NumProcesses 4
```

This will start 4 gRPC servers on ports 50051, 50052, 50053, 50054.

Wait 30-35 seconds for all servers to load models.

### 2. Start the Load-Balancing Proxy

**Option A: Using port range (recommended)**

```bash
# Connect to 4 servers starting at port 50051
.\proxy.exe --start-port 50051 --num-servers 4

# Or using the batch file
.\run_multi.bat 4 50051
```

**Option B: Using explicit addresses**

```bash
.\proxy.exe --grpc-addrs "localhost:50051,localhost:50052,localhost:50053,localhost:50054"
```

### 3. Open the Browser Client

Navigate to: `http://localhost:8086/`

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ws-port` | 8086 | WebSocket server port |
| `--start-port` | 50051 | First gRPC server port (for range mode) |
| `--num-servers` | 1 | Number of gRPC servers (for range mode) |
| `--grpc-addrs` | - | Comma-separated explicit addresses |

### Examples

**Single backend (default):**
```bash
.\proxy.exe --start-port 50051
```

**4 backends on sequential ports:**
```bash
.\proxy.exe --start-port 50051 --num-servers 4
# Connects to: 50051, 50052, 50053, 50054
```

**8 backends starting at port 50060:**
```bash
.\proxy.exe --start-port 50060 --num-servers 8
# Connects to: 50060, 50061, 50062, 50063, 50064, 50065, 50066, 50067
```

**Custom non-sequential ports:**
```bash
.\proxy.exe --grpc-addrs "localhost:50051,localhost:50055,localhost:50099"
```

## Load Balancing Behavior

### Round-Robin Algorithm

1. Proxy maintains a circular list of backends
2. Each request goes to the next backend in the list
3. Automatically skips unhealthy backends
4. Wraps around to the beginning after the last backend

### Health Management

**Marking Unhealthy:**
- Any gRPC error marks backend as unhealthy
- Connection timeouts
- Model loading failures

**Marking Healthy:**
- Successful response marks backend as healthy
- Backends can recover automatically

**Health Check Endpoint:**
```bash
curl http://localhost:8086/health
```

**Response:**
```json
{
  "healthy": true,
  "total_backends": 4,
  "healthy_count": 3,
  "backends": [
    {
      "address": "localhost:50051",
      "healthy": true,
      "total_requests": 45,
      "error_count": 0
    },
    {
      "address": "localhost:50052",
      "healthy": false,
      "total_requests": 12,
      "error_count": 3
    },
    ...
  ]
}
```

## Expected Performance

### Single Server Baseline
- ~58 FPS
- ~17ms latency

### Multi-Server with RTX 6000 Ada (96GB)

| Backends | Expected Throughput | Speedup | Use Case |
|----------|-------------------|---------|----------|
| 1 | 58 FPS | 1.0x | Single user testing |
| 2 | 115 FPS | 2.0x | 2-3 concurrent users |
| 4 | 180-220 FPS | 3-4x | 5-10 concurrent users |
| 6 | 250-350 FPS | 4-6x | 10-20 concurrent users |
| 8 | 300-450 FPS | 5-8x | 20-30 concurrent users |
| 12 | 400-650 FPS | 7-11x | 30-50 concurrent users |

**Note:** Actual speedup depends on:
- GPU model (RTX 6000 Ada has better multi-process support)
- VRAM capacity (96GB allows many concurrent models)
- Professional vs consumer drivers
- SM utilization (142 SMs on RTX 6000 Ada)

### Consumer GPUs (RTX 2060-4090)
- **Time-slicing only** - No throughput improvement
- Multi-process still useful for fault isolation
- Consider batching instead for throughput

## Testing

### Test Multi-Process Performance

```bash
cd ..\minimal_server
python test_multi_process.py --ports 50051 50052 50053 50054
```

This runs two tests:
1. **Concurrent Load**: All servers hit simultaneously
2. **Round-Robin**: Simulates normal load balancing

Output shows:
- Per-server throughput
- Total aggregate FPS
- Speedup vs single server
- Latency statistics

### Manual Testing

**Test specific backend:**
```bash
python optimized_grpc_client.py --port 50052
```

**Test round-robin from browser:**
1. Open DevTools Console
2. Watch log messages showing `[localhost:50051]`, `[localhost:50052]`, etc.
3. Verify requests alternate between backends

## Monitoring

### Proxy Logs

Watch the console output for:
```
✅ sanders frame 42 [localhost:50051]: gRPC=16ms total=18ms size=24576 bytes
✅ sanders frame 43 [localhost:50052]: gRPC=17ms total=19ms size=24576 bytes
✅ sanders frame 44 [localhost:50053]: gRPC=15ms total=17ms size=24576 bytes
```

The `[localhost:5005X]` shows which backend handled each request.

### GPU Monitoring

**Watch GPU utilization:**
```bash
nvidia-smi dmon -s um -c 30
```

**Expected with 4 processes on RTX 6000 Ada:**
- SM Utilization: 60-85% (each process uses ~25% of 142 SMs)
- Memory: 400-600 MB (4 × 100 MB per model + overhead)
- Power: 150-250W

**On consumer GPUs (time-slicing):**
- SM Utilization: 85-95% (same as single process)
- Multiple processes share the same GPU time

## Troubleshooting

### "No backends available"

**Problem:** All backends marked unhealthy

**Solutions:**
1. Check gRPC servers are running:
   ```bash
   curl http://localhost:50051/health  # Should fail, gRPC doesn't support HTTP/1
   python optimized_grpc_client.py --port 50051  # Should succeed
   ```

2. Check server startup logs for model loading errors

3. Restart unhealthy servers:
   ```powershell
   # In PowerShell
   Stop-Process -Id (Get-Content grpc_processes.txt)
   .\start_multi_grpc.ps1 -NumProcesses 4
   ```

### "Connection refused"

**Problem:** Backend not listening on expected port

**Solutions:**
1. Verify server processes:
   ```bash
   netstat -an | findstr "50051"  # Should show LISTENING
   ```

2. Check if processes are running:
   ```powershell
   Get-Process python | Where-Object {$_.MainWindowTitle -like "*gRPC*"}
   ```

3. Check firewall settings (usually not an issue for localhost)

### Poor Performance

**Symptoms:**
- Total FPS barely higher than single server
- `nvidia-smi` shows 95%+ utilization with minimal improvement

**Analysis:**
- You likely have a consumer GPU (RTX 2060-4090)
- Time-slicing provides no throughput gain
- GPU already maxed at 85-95% utilization

**Solutions:**
1. Use batching instead:
   ```python
   # In optimized_grpc_server.py
   BATCH_SIZE = 4  # Process 4 frames at once
   ```
   Expected gain: 6-8x throughput (300-400 FPS)

2. Keep multi-process for fault isolation but don't expect speedup

3. Upgrade to data center GPU (A100, H100, RTX 6000 Ada) for true multi-process gains

## Architecture

```
Browser (WebSocket Binary) → Go Proxy (Load Balancer) → gRPC Servers (4-20 instances)
                                      ↓
                              Round-Robin Selection
                                      ↓
                         [50051] [50052] [50053] [50054]
                            ↓       ↓       ↓       ↓
                          RTX 6000 Ada (142 SMs, 96GB VRAM)
```

Each gRPC server:
- Loads models once at startup (~8 seconds)
- Runs inference on GPU (15-18ms)
- Handles ~58 FPS when alone
- Uses ~100 MB VRAM per model

Proxy:
- No GPU usage (CPU only)
- Adds 1-4ms latency overhead
- Handles 100+ concurrent WebSocket connections
- Tracks backend health automatically

## Next Steps

1. **Benchmark your GPU**: Run `test_multi_process.py` to measure actual speedup
2. **Tune backend count**: Start with 4, scale up if speedup > 3x
3. **Monitor production**: Use `/health` endpoint for alerting
4. **Add batching**: If you need >600 FPS, combine batching + multi-process
5. **Consider caching**: For repeated frames, add Redis cache layer

## Related Documentation

- `../minimal_server/RTX6000_OPTIMIZATION_GUIDE.md` - RTX 6000 Ada specific tuning
- `../minimal_server/GPU_PARALLELISM_GUIDE.md` - Understanding GPU time-slicing vs MPS
- `../minimal_server/CONCURRENCY_GUIDE.md` - Python GIL and multi-process strategies
- `ARCHITECTURE.md` - Overall system design
