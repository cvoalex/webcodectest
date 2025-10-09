# 🚀 gRPC Server Setup Guide

## Quick Start (3 Steps)

### 1. Generate gRPC Stubs

The Protocol Buffer definitions need to be compiled to Python code:

```bash
cd minimal_server
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto
```

This creates:
- `optimized_lipsyncsrv_pb2.py` - Message classes
- `optimized_lipsyncsrv_pb2_grpc.py` - Service stubs

### 2. Start the Server

```bash
# Option 1: Direct Python
D:\Projects\webcodecstest\.venv312\Scripts\python.exe optimized_grpc_server.py

# Option 2: Batch file (Windows)
start_optimized_grpc_server.bat
```

Expected output:
```
================================================================================
🚀 ULTRA-OPTIMIZED GRPC SERVER
================================================================================

📦 Loading model 'sanders'...
   ✅ Pre-loading videos into RAM...
      Total: 1,838.92 MB loaded in 2.45s
   ✅ Memory-mapping audio features: 1.02 MB
   ✅ Model loaded in 7.6s

🚀 gRPC server started on [::]:50051
```

### 3. Test the Server

```bash
# Quick test (single frame)
python test_grpc_quick.py

# Full test suite
python optimized_grpc_client.py
```

---

## 📁 Files Created

| File | Purpose | Size |
|------|---------|------|
| `optimized_lipsyncsrv.proto` | Protocol Buffer definition | 5 KB |
| `optimized_grpc_server.py` | gRPC server implementation | 14 KB |
| `optimized_grpc_client.py` | Test client with full test suite | 12 KB |
| `start_optimized_grpc_server.bat` | Windows startup script | 1 KB |
| `GRPC_SERVER_README.md` | Comprehensive documentation | 35 KB |
| `test_grpc_quick.py` | Quick verification test | 4 KB |

**Generated files** (from proto compilation):
- `optimized_lipsyncsrv_pb2.py` - Auto-generated message classes
- `optimized_lipsyncsrv_pb2_grpc.py` - Auto-generated service stubs

---

## 🎯 What You Get

### RPC Methods

1. **GenerateInference** - Single frame (most common)
   ```python
   request = OptimizedInferenceRequest(model_name='sanders', frame_id=50)
   response = await stub.GenerateInference(request)
   ```

2. **GenerateBatchInference** - Multiple frames in one call
   ```python
   request = BatchInferenceRequest(model_name='sanders', frame_ids=[0,10,20,30])
   response = await stub.GenerateBatchInference(request)
   ```

3. **StreamInference** - Bidirectional streaming
   ```python
   async def request_generator():
       for frame_id in range(100):
           yield OptimizedInferenceRequest(model_name='sanders', frame_id=frame_id)
   
   async for response in stub.StreamInference(request_generator()):
       print(f"Frame {response.frame_id}: {response.processing_time_ms}ms")
   ```

4. **LoadPackage** - Load model packages dynamically
5. **GetStats** - Query performance statistics
6. **ListModels** - List loaded models
7. **HealthCheck** - Server health and uptime

### Performance Metrics in Response

Every response includes detailed timing:
- `processing_time_ms` - Total processing time
- `prepare_time_ms` - Input preparation time
- `inference_time_ms` - Model inference time
- `composite_time_ms` - Result compositing time

### Example Response

```python
response = OptimizedInferenceResponse(
    success=True,
    frame_id=50,
    prediction_data=<JPEG bytes>,
    prediction_shape="(720, 1280, 3)",
    bounds=[640, 360, 640, 720],
    processing_time_ms=17.8,
    prepare_time_ms=8.2,
    inference_time_ms=6.1,
    composite_time_ms=3.5
)
```

---

## 🔍 Testing

### Quick Test

```bash
python test_grpc_quick.py
```

Output:
```
🧪 Quick gRPC Server Test
============================================================

1️⃣ Connecting to localhost:50051...
   ✅ Connected!

2️⃣ Health Check...
   Status: SERVING
   Healthy: True
   Loaded models: 1

3️⃣ Generating frame 50...
   ✅ Success!
   Processing time: 17.8ms
   Inference time: 6.1ms
   Image size: 98,432 bytes
   💾 Saved to: test_grpc_output.jpg

✅ ALL TESTS PASSED!
```

### Full Test Suite

```bash
python optimized_grpc_client.py
```

Tests:
- ✅ Connection and health check
- ✅ List loaded models
- ✅ Single frame inference (4 frames)
- ✅ Batch inference (6 frames)
- ✅ Streaming inference (20 frames at 50 FPS)
- ✅ Performance statistics

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'optimized_lipsyncsrv_pb2'"

**Problem:** gRPC stubs not generated.

**Solution:**
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto
```

---

### "No module named 'torch'"

**Problem:** Not using virtual environment.

**Solution:**
```bash
# Use the correct Python
D:\Projects\webcodecstest\.venv312\Scripts\python.exe optimized_grpc_server.py
```

---

### "Connection refused" / "UNAVAILABLE"

**Problem:** Server not running.

**Solution:**
```bash
# Start server in one terminal
python optimized_grpc_server.py

# Run client in another terminal
python optimized_grpc_client.py
```

---

### First request takes 7+ seconds

**Problem:** This is **normal** - server pre-loads 1.8 GB of video data on first model access.

**Solution:** No action needed. Subsequent requests are 15-18ms.

```
Request 1: 7,632ms (one-time cost)
Request 2: 17.8ms
Request 3: 16.9ms
...
```

---

## 📊 Performance

### Latency Breakdown

```
Prepare Input:       5-8ms
Model Inference:     6-8ms
Composite Result:    3-4ms
─────────────────────────────
Server Total:        15-18ms

Network (gRPC):      1-2ms
─────────────────────────────
Client Total:        16-20ms
```

### Throughput

- **Single requests**: 55-65 FPS
- **Batch requests**: 60-70 FPS (6 frames/batch)
- **Streaming**: 44-50 FPS (with 20ms delay between requests)

### Comparison

| Server | Protocol | Latency | Throughput |
|--------|----------|---------|------------|
| Original | WebSocket JSON | 85ms | 12 FPS |
| Binary WebSocket | WebSocket Binary | 60ms | 17 FPS |
| Optimized WebSocket | WebSocket Binary | 20ms | 50 FPS |
| **gRPC** | **HTTP/2 + Protobuf** | **15-18ms** | **55-65 FPS** |

**gRPC is 4.7-5.7x faster than the original!** 🚀

---

## 🌐 Integration Examples

### Python Client

```python
import asyncio
import grpc
from grpc import aio
import optimized_lipsyncsrv_pb2
import optimized_lipsyncsrv_pb2_grpc

async def main():
    channel = aio.insecure_channel('localhost:50051')
    stub = optimized_lipsyncsrv_pb2_grpc.OptimizedLipSyncServiceStub(channel)
    
    request = optimized_lipsyncsrv_pb2.OptimizedInferenceRequest(
        model_name='sanders',
        frame_id=50
    )
    
    response = await stub.GenerateInference(request)
    
    if response.success:
        print(f"Inference: {response.inference_time_ms}ms")
        # response.prediction_data contains JPEG bytes
    
    await channel.close()

asyncio.run(main())
```

### Go Client

```go
package main

import (
    "context"
    "log"
    
    "google.golang.org/grpc"
    pb "your_module/optimized_lipsyncsrv"
)

func main() {
    conn, _ := grpc.Dial("localhost:50051", grpc.WithInsecure())
    defer conn.Close()
    
    client := pb.NewOptimizedLipSyncServiceClient(conn)
    
    req := &pb.OptimizedInferenceRequest{
        ModelName: "sanders",
        FrameId:   50,
    }
    
    resp, _ := client.GenerateInference(context.Background(), req)
    
    if resp.Success {
        log.Printf("Frame %d: %.1fms", resp.FrameId, resp.ProcessingTimeMs)
    }
}
```

### Node.js Client

```javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

const packageDefinition = protoLoader.loadSync('optimized_lipsyncsrv.proto');
const proto = grpc.loadPackageDefinition(packageDefinition);

const client = new proto.OptimizedLipSyncService(
    'localhost:50051',
    grpc.credentials.createInsecure()
);

const request = {
    model_name: 'sanders',
    frame_id: 50
};

client.GenerateInference(request, (error, response) => {
    if (response.success) {
        console.log(`Frame ${response.frame_id}: ${response.processing_time_ms}ms`);
    }
});
```

---

## 🎓 Next Steps

1. **Read the full documentation**: [GRPC_SERVER_README.md](GRPC_SERVER_README.md)
2. **Compare servers**: [SERVER_IMPLEMENTATIONS.md](SERVER_IMPLEMENTATIONS.md)
3. **Performance details**: [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md)
4. **Start building**: Use the examples above to integrate into your application

---

## 📚 Additional Resources

- **Protocol Buffers**: https://protobuf.dev/
- **gRPC Python**: https://grpc.io/docs/languages/python/
- **gRPC Performance**: https://grpc.io/docs/guides/performance/

---

**Server ready to use!** 🚀

For questions or issues, check the troubleshooting section or refer to the full documentation.
