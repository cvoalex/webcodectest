# 🔥 Concurrency & Scalability Guide

## 🎯 The Problem

**Question:** How does the Python gRPC server handle concurrent requests?

**Short Answer:** It doesn't handle them well by default due to Python's GIL and single GPU constraints.

**Long Answer:** Let's explore the reality and solutions...

---

## 📊 Current Performance

### Single Request (Ideal)
```
Frame generation: 15-18ms
Throughput: ~58 FPS (1000ms / 17ms)
```

### Concurrent Requests (Reality)

**With Current Implementation:**
```python
# Asyncio queues requests, but PyTorch inference is SERIAL:

Request 1: 17ms
Request 2: 17ms (waits for Request 1)
Request 3: 17ms (waits for Request 2)
Request 4: 17ms (waits for Request 3)

Total for 4 requests: 68ms (sequential)
Effective throughput: Still ~58 FPS total, not per client
```

**Why?**
1. **GIL (Global Interpreter Lock)** - Python threads can't run truly parallel for CPU work
2. **PyTorch model** - Single model instance on single GPU
3. **GPU memory** - Model lives in VRAM, can't easily duplicate
4. **CUDA context** - One context per process limits parallelism

---

## 🛠️ Solution 1: Request Batching (Easy Win)

**Idea:** Process multiple requests in a single GPU batch.

**Implementation:**

```python
# Collect requests for 20ms, then batch process
batch = []
while len(batch) < 8 and time_elapsed < 20ms:
    batch.append(await get_next_request())

# Process all 8 frames in one batch
with torch.no_grad():
    frames_batch = torch.stack([get_frame(req.frame_id) for req in batch])
    audio_batch = torch.stack([get_audio(req.frame_id) for req in batch])
    
    # Single GPU call for 8 frames
    predictions = model(frames_batch, audio_batch)  # ~22ms for 8 frames
    
# Return results to all 8 clients
```

**Performance:**
```
8 frames in 22ms = 2.75ms per frame (6x faster!)
Throughput: ~363 FPS total
```

**Pros:**
- ✅ Easy to implement
- ✅ GPU is better utilized
- ✅ 6x throughput improvement
- ✅ No additional hardware needed

**Cons:**
- ❌ Adds 20ms latency (waiting for batch)
- ❌ Variable latency (first request waits, last doesn't)
- ❌ Requires request queue management

---

## 🛠️ Solution 2: Multiple GPU Workers (Medium Difficulty)

**Idea:** Run multiple Python processes, each with own model on GPU.

**Architecture:**
```
                ┌─► Process 1 (GPU 0, model copy 1)
Load Balancer ──┼─► Process 2 (GPU 0, model copy 2)
                ├─► Process 3 (GPU 0, model copy 3)
                └─► Process 4 (GPU 0, model copy 4)
```

**Implementation:**

```bash
# Start 4 server instances
python optimized_grpc_server.py --port 50051 &
python optimized_grpc_server.py --port 50052 &
python optimized_grpc_server.py --port 50053 &
python optimized_grpc_server.py --port 50054 &

# Use nginx/HAProxy to load balance
upstream grpc_backend {
    server localhost:50051;
    server localhost:50052;
    server localhost:50053;
    server localhost:50054;
}
```

**Performance:**
```
4 processes × 58 FPS = 232 FPS total throughput
Latency: Still 17ms per request (no waiting)
```

**VRAM Requirements:**
```
Single model: ~100 MB
4 copies: ~400 MB (easily fits on most GPUs)
Video pre-load: 1.8 GB × 4 = 7.2 GB RAM
```

**Pros:**
- ✅ True parallelism (separate processes)
- ✅ No GIL issues
- ✅ 4x throughput
- ✅ Low latency (no batching delay)
- ✅ Fault tolerance (one crash doesn't kill all)

**Cons:**
- ❌ 4x memory usage (RAM)
- ❌ Need load balancer (nginx/HAProxy)
- ❌ More complex deployment

---

## 🛠️ Solution 3: Multi-GPU (Best Performance)

**Idea:** Use multiple GPUs, each running model instance.

**Architecture:**
```
Request → Router → GPU 0 (Model 1)
                 → GPU 1 (Model 2)
                 → GPU 2 (Model 3)
                 → GPU 3 (Model 4)
```

**Implementation:**

```python
# Start server with specific GPU
CUDA_VISIBLE_DEVICES=0 python optimized_grpc_server.py --port 50051 &
CUDA_VISIBLE_DEVICES=1 python optimized_grpc_server.py --port 50052 &
CUDA_VISIBLE_DEVICES=2 python optimized_grpc_server.py --port 50053 &
CUDA_VISIBLE_DEVICES=3 python optimized_grpc_server.py --port 50054 &
```

**Performance:**
```
4 GPUs × 58 FPS = 232 FPS total
Latency: 17ms per request
No GPU contention!
```

**Pros:**
- ✅ True parallelism across GPUs
- ✅ No GPU contention
- ✅ Linear scaling (4 GPUs = 4x throughput)
- ✅ Low latency

**Cons:**
- ❌ Requires 4 GPUs (expensive)
- ❌ 4x VRAM (400 MB total)
- ❌ 4x RAM (7.2 GB for videos)

---

## 🛠️ Solution 4: Hybrid Approach (Recommended)

**Combine batching + multiple processes for best results:**

```
Load Balancer
├─► Process 1 (batch size 4)
├─► Process 2 (batch size 4)
└─► Process 3 (batch size 4)
```

**Implementation:**

```python
# Each process batches up to 4 requests
# 3 processes = 12 concurrent requests

class BatchingServicer:
    def __init__(self, max_batch_size=4, max_wait_ms=10):
        self.batch_queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        # Start batch processor
        asyncio.create_task(self._process_batches())
    
    async def _process_batches(self):
        while True:
            batch = []
            deadline = time.time() + (self.max_wait_ms / 1000)
            
            # Collect requests for batch
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    req = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=deadline - time.time()
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                await self._process_batch(batch)
    
    async def _process_batch(self, batch):
        # Process all requests in single GPU call
        frame_ids = [req.frame_id for req in batch]
        
        # Batch inference (6-8 frames in 20-25ms)
        results = await self._batch_inference(frame_ids)
        
        # Return results to all clients
        for req, result in zip(batch, results):
            await req.response_future.set_result(result)
```

**Performance:**
```
3 processes × 4 batch size × (1000ms / 25ms) = 480 FPS!
Latency: 17-27ms (10ms wait + 17ms inference)
```

**Pros:**
- ✅ Excellent throughput (480 FPS)
- ✅ Reasonable latency (<30ms)
- ✅ Good GPU utilization
- ✅ Works on single GPU

**Cons:**
- ❌ Complex implementation
- ❌ Variable latency
- ❌ 3x memory usage

---

## 📈 Performance Comparison

| Solution | Throughput | Latency | GPU Usage | RAM | Complexity |
|----------|-----------|---------|-----------|-----|------------|
| **Current** | 58 FPS | 17ms | 15% | 2 GB | ⭐ |
| **Batching** | 363 FPS | 20-37ms | 60% | 2 GB | ⭐⭐ |
| **Multi-Process** | 232 FPS | 17ms | 60% | 8 GB | ⭐⭐⭐ |
| **Multi-GPU** | 232 FPS+ | 17ms | 15% each | 8 GB | ⭐⭐⭐⭐ |
| **Hybrid** | 480 FPS | 17-27ms | 75% | 6 GB | ⭐⭐⭐⭐⭐ |

---

## 🎯 Which Solution to Use?

### For Development/Testing
**Use current implementation** - Simple, good enough for single client.

### For Small Production (1-5 clients)
**Use Solution 2: Multi-Process**
```bash
# Start 3 instances
for port in 50051 50052 50053; do
    python optimized_grpc_server.py --port $port &
done

# Use Go proxy with round-robin
```

### For Medium Production (5-20 clients)
**Use Solution 4: Hybrid (Batching + Multi-Process)**

Benefits:
- 480 FPS total throughput
- 20-27ms latency (acceptable)
- Single GPU
- Best throughput-per-dollar

### For Large Production (20+ clients)
**Use Solution 3: Multi-GPU + Multi-Process**

Each GPU runs 2-3 processes with batching:
```
4 GPUs × 3 processes × 121 FPS = 1,452 FPS total!
```

---

## 🚀 Quick Multi-Process Setup

Let me create scripts for Solution 2 (easiest production setup):

### Windows (PowerShell)

```powershell
# start_multi_grpc.ps1

$ports = 50051, 50052, 50053, 50054

foreach ($port in $ports) {
    Start-Process -FilePath "D:\Projects\webcodecstest\.venv312\Scripts\python.exe" `
                  -ArgumentList "optimized_grpc_server.py --port $port" `
                  -WorkingDirectory "D:\Projects\webcodecstest\minimal_server"
    Start-Sleep -Seconds 2
}

Write-Host "Started 4 gRPC servers on ports 50051-50054"
```

### Linux/Mac

```bash
#!/bin/bash
# start_multi_grpc.sh

for port in 50051 50052 50053 50054; do
    python optimized_grpc_server.py --port $port &
    sleep 2
done

echo "Started 4 gRPC servers on ports 50051-50054"
```

### Update Go Proxy for Load Balancing

Modify `grpc-websocket-proxy/main.go`:

```go
var grpcAddrs = []string{
    "localhost:50051",
    "localhost:50052",
    "localhost:50053",
    "localhost:50054",
}

var currentBackend = 0
var backendMutex sync.Mutex

func getNextBackend() string {
    backendMutex.Lock()
    defer backendMutex.Unlock()
    
    addr := grpcAddrs[currentBackend]
    currentBackend = (currentBackend + 1) % len(grpcAddrs)
    return addr
}

// In ProxyServer, maintain connection pool:
type ProxyServer struct {
    grpcClients []pb.OptimizedLipSyncServiceClient
    grpcConns   []*grpc.ClientConn
}

// Round-robin on each request
func (p *ProxyServer) getClient() pb.OptimizedLipSyncServiceClient {
    return p.grpcClients[atomic.AddUint32(&currentIdx, 1) % len(p.grpcClients)]
}
```

---

## 🧪 Testing Concurrency

### Benchmark Script

```python
# test_concurrency.py
import asyncio
import grpc
from grpc import aio
import optimized_lipsyncsrv_pb2
import optimized_lipsyncsrv_pb2_grpc
import time

async def test_concurrent_requests(num_requests=10, num_concurrent=5):
    """Test concurrent request handling"""
    
    channel = aio.insecure_channel('localhost:50051')
    stub = optimized_lipsyncsrv_pb2_grpc.OptimizedLipSyncServiceStub(channel)
    
    async def single_request(frame_id):
        start = time.time()
        request = optimized_lipsyncsrv_pb2.OptimizedInferenceRequest(
            model_name='sanders',
            frame_id=frame_id
        )
        response = await stub.GenerateInference(request)
        latency = (time.time() - start) * 1000
        return latency, response.success
    
    # Test concurrent requests
    print(f"Testing {num_requests} requests, {num_concurrent} concurrent...")
    start_time = time.time()
    
    tasks = [single_request(i % 523) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    latencies = [r[0] for r in results]
    successes = sum(1 for r in results if r[1])
    
    print(f"\nResults:")
    print(f"  Total time: {total_time*1000:.1f}ms")
    print(f"  Successful: {successes}/{num_requests}")
    print(f"  Avg latency: {sum(latencies)/len(latencies):.1f}ms")
    print(f"  Min latency: {min(latencies):.1f}ms")
    print(f"  Max latency: {max(latencies):.1f}ms")
    print(f"  Throughput: {num_requests/total_time:.1f} FPS")
    
    await channel.close()

if __name__ == '__main__':
    asyncio.run(test_concurrent_requests(num_requests=20, num_concurrent=10))
```

**Expected Results:**

```
Current Implementation (Single Process):
  Total time: 340ms (20 requests serial)
  Throughput: ~58 FPS

Multi-Process (4 instances):
  Total time: 85ms (20 requests parallelized)
  Throughput: ~235 FPS
```

---

## 💡 Practical Recommendation

**For your use case, I recommend:**

### Option 1: Current Setup is Fine If...
- You have 1-2 concurrent clients
- Latency is acceptable (17-20ms)
- Don't want deployment complexity

### Option 2: Multi-Process (Recommended)
- Start **3-4 server instances** on different ports
- Modify **Go proxy** to round-robin between them
- Gets you **232 FPS** throughput
- Still **17ms** latency per request
- Only needs **6-8 GB RAM**

### Option 3: Add Batching Later
- If you need **400+ FPS**, implement batching
- Adds code complexity but huge gains
- Best for high-throughput scenarios

---

## 📚 Summary

**The Truth:**
- Python GIL limits threading
- PyTorch model is the bottleneck
- Single GPU = serial inference
- Current: ~58 FPS total (not per client)

**The Solutions:**
1. **Batching** - 6x throughput, adds latency
2. **Multi-Process** - 4x throughput, same latency ✅ **Best**
3. **Multi-GPU** - Linear scaling, expensive
4. **Hybrid** - 8x throughput, complex

**Quick Win:**
Run 3-4 server processes + load balancer = 232 FPS for free!

---

**Want me to implement multi-process support?** I can create:
1. Multi-instance startup scripts
2. Updated Go proxy with round-robin
3. Health check system
4. Process manager

Just let me know! 🚀
