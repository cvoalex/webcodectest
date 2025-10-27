# Go Lipsync Server - Architecture Guide

## Overview

This project provides **three complete server implementations** for real-time lip-sync inference and compositing:

1. **🚀 Monolithic Server** (Recommended for most use cases)
2. **🔧 Separated Servers** (Compositing + Inference)
3. **🧪 Legacy Python Server** (Reference implementation)

## Quick Comparison

| Feature | Monolithic | Separated | Python |
|---------|-----------|-----------|--------|
| **Latency** | ✅ ~130ms | ~145ms | ~250ms |
| **Overhead** | ✅ 2-3ms | 10-15ms | High |
| **Deployment** | ✅ 1 binary | 2 binaries | 1 script |
| **Scalability** | Single machine | ✅ Horizontal | Limited |
| **Best for** | ✅ Performance | Distribution | Development |

## Architecture Decision Tree

```
START: What's your primary goal?

├─ Maximum Performance? (Lowest latency)
│  └─ 🚀 Use MONOLITHIC SERVER
│     Location: go-monolithic-server/
│     Latency: ~130ms
│     Overhead: 2-3ms
│
├─ Horizontal Scaling? (Multiple machines)
│  └─ 🔧 Use SEPARATED SERVERS
│     Location: go-compositing-server/ + go-inference-server/
│     Scale: Independent compositing/inference scaling
│     Latency: ~145ms
│
└─ Development/Prototyping?
   └─ 🧪 Use PYTHON SERVER (or Monolithic)
      Location: fast_service/
      Easy to modify, slower performance
```

## Detailed Comparison

### 1. Monolithic Server (Recommended ⭐)

**Location**: `go-monolithic-server/`

**Architecture**:
```
Single Process:
  Audio Processing → GPU Inference → Compositing → JPEG
  (Direct function calls, zero overhead)
```

**Pros**:
- ✅ **Lowest latency** (~130ms)
- ✅ **Minimal overhead** (2-3ms)
- ✅ **Simplest deployment** (1 binary, 1 config)
- ✅ **Lower cloud costs** (1 server instance)
- ✅ **Easier debugging** (single process)

**Cons**:
- ⚠️ Can't scale compositing independently
- ⚠️ Single point of failure

**Use When**:
- Running on single powerful machine
- Latency is critical (real-time requirements)
- Simpler operations preferred
- Cloud cost optimization important

**Quick Start**:
```powershell
cd go-monolithic-server
go build -o monolithic-server.exe ./cmd/server
.\monolithic-server.exe
```

📖 [Full Documentation](go-monolithic-server/README.md)  
🚀 [Quick Start Guide](go-monolithic-server/QUICKSTART.md)

---

### 2. Separated Servers (Scalable)

**Location**: `go-compositing-server/` + `go-inference-server/`

**Architecture**:
```
Two Processes:
  Client → Compositing Server (gRPC) → Inference Server → GPU
                    ↓
              Compositing CPU Work
```

**Pros**:
- ✅ **Horizontal scaling** (multiple compositing servers)
- ✅ **Resource isolation** (separate CPU/GPU limits)
- ✅ **Independent deployment** (update without full restart)
- ✅ **Load balancing** (distribute across inference servers)

**Cons**:
- ⚠️ Higher latency (~145ms)
- ⚠️ More overhead (10-15ms for gRPC)
- ⚠️ Complex deployment (2 binaries, 2 configs)
- ⚠️ Higher cloud costs (2 server instances)

**Use When**:
- Need horizontal scaling (>10 req/s)
- Multiple compositing servers share 1 inference server
- Kubernetes/cloud-native deployment
- Different scaling policies needed

**Quick Start**:
```powershell
# Terminal 1: Inference Server
cd go-inference-server
.\inference-server.exe

# Terminal 2: Compositing Server
cd go-compositing-server
.\compositing-server.exe
```

📖 [Compositing Server Docs](go-compositing-server/README.md)  
📖 [Inference Server Docs](go-inference-server/README.md)

---

### 3. Python Server (Legacy)

**Location**: `fast_service/`

**Architecture**:
```
Single Python Process:
  FastAPI/gRPC → PyTorch Inference → NumPy Compositing
```

**Pros**:
- ✅ Easy to modify (Python)
- ✅ Good for experimentation
- ✅ Original reference implementation

**Cons**:
- ⚠️ Slower (~250ms latency)
- ⚠️ GIL limitations
- ⚠️ Higher memory usage
- ⚠️ Slower than Go implementations

**Use When**:
- Rapid prototyping
- Algorithm development
- Need Python ecosystem tools
- Not concerned with performance

**Quick Start**:
```powershell
cd fast_service
python service.py
```

## Performance Benchmarks

### Latency (Batch Size 24)

| Server | Audio | Inference | Composite | Total | Overhead |
|--------|-------|-----------|-----------|-------|----------|
| **Monolithic** | 7ms | 120ms | 2ms | **129ms** | **2ms** |
| Separated | 7ms | 120ms | 2ms | 145ms | 16ms |
| Python | N/A | 180ms | 20ms | 250ms | 50ms |

### Throughput (Requests/Second)

| Server | Single Client | 4 Concurrent | 8 Concurrent |
|--------|--------------|--------------|--------------|
| **Monolithic** | **7.8** | **28** | **52** |
| Separated | 7.0 | 25 | 45 |
| Python | 4.0 | 12 | 18 |

### Memory Usage (Per Model)

| Server | Model Weights | Backgrounds | Runtime | Total |
|--------|--------------|-------------|---------|-------|
| **Monolithic** | 500MB | 250MB | 250MB | **1.0GB** |
| Separated | 500MB + 250MB | - | 300MB | 1.05GB |
| Python | 600MB | 300MB | 400MB | 1.3GB |

## Migration Paths

### From Python → Go Monolithic

**Benefits**: 2× faster, 30% less memory

1. Convert models to ONNX format
2. Build monolithic server
3. Update client to use gRPC (from HTTP)
4. Test with same inputs
5. Deploy and monitor

**Effort**: Medium (model conversion needed)

### From Separated → Monolithic

**Benefits**: 10% faster, simpler deployment

1. Merge config files
2. Update client port (50052 → 50053)
3. Stop both old servers
4. Start monolithic server
5. Test end-to-end

**Effort**: Low (config changes only)

### From Monolithic → Separated

**Benefits**: Enables horizontal scaling

1. Split config into 2 files
2. Start inference server first
3. Start compositing server
4. Update client to compositing server port
5. Test connection health

**Effort**: Low (split and deploy)

## File Structure

```
webcodecstest/
├── go-monolithic-server/          ⭐ RECOMMENDED
│   ├── cmd/server/main.go
│   ├── config.yaml
│   ├── README.md
│   ├── QUICKSTART.md
│   └── test_client.go
│
├── go-compositing-server/         (For horizontal scaling)
│   ├── cmd/server/main.go
│   ├── config.yaml
│   ├── README.md
│   └── test_client.go
│
├── go-inference-server/           (For horizontal scaling)
│   ├── cmd/server/main.go
│   ├── config.yaml
│   ├── README.md
│   └── audio/ (processing pipeline)
│
├── fast_service/                  (Legacy Python)
│   ├── service.py
│   ├── requirements.txt
│   └── README.md
│
├── ARCHITECTURE_COMPARISON.md     📊 Detailed comparison
└── ARCHITECTURE_GUIDE.md          📖 This file
```

## Common Components

All Go servers share:

### Audio Processing Pipeline
- **Location**: `audio/` package
- **Components**:
  - Mel-spectrogram processor (pure Go, gonum/fft)
  - Audio encoder (ONNX Runtime wrapper)
  - Sliding window extraction
- **Performance**: ~7ms per 640ms audio chunk
- **Compatibility**: Validated against Python reference (<0.001 error)

### Model Registry
- **Location**: `registry/` package
- **Features**:
  - Dynamic model loading/unloading
  - LFU eviction policy
  - Multi-GPU support
  - Memory tracking
  - Usage statistics

### ONNX Inference
- **Location**: `lipsyncinfer/` package
- **Runtime**: ONNX Runtime 1.22.0 with CUDA
- **Optimizations**:
  - Pre-allocated tensors
  - Zero-copy operations
  - GPU memory pooling
  - Batch processing

## Configuration

All servers use YAML configuration:

```yaml
server:
  port: ":5005X"              # 50053 (mono), 50052 (comp), 50051 (inf)
  max_message_size_mb: 100

gpus:
  enabled: true
  count: 1
  memory_gb_per_gpu: 24

models:
  sanders:
    model_path: "sanders/checkpoint/model_best.onnx"
    background_dir: "sanders/frames"  # Monolithic/Compositing only
    crop_rects_path: "sanders/crop_rects.json"
    num_frames: 523
    preload_backgrounds: true
```

## Testing

Each server includes a test client:

```powershell
# Build test client
go build -o test-client.exe test_client.go

# Run test
.\test-client.exe
```

**Test features**:
- Loads real audio from `aud.wav`
- Processes raw PCM audio (16kHz mono)
- Sends batches of 24 frames
- Saves JPEG output to `test_output/`
- Reports detailed performance metrics

## Monitoring

All servers provide:

1. **Health Check RPC**
   ```
   Health() → { healthy, loaded_models, gpu_info }
   ```

2. **Performance Logging**
   ```
   🎵 Audio processing: 7.45ms
   ⚡ Inference: 120.28ms
   🎨 Compositing: 2.15ms
   ```

3. **Model Statistics**
   ```
   GetModelStats() → usage counts, avg latency, memory
   ```

## Recommendations

### For Production

| Scenario | Recommended | Reason |
|----------|------------|--------|
| **Single cloud VM** | Monolithic | Lowest latency, lower cost |
| **Edge device** | Monolithic | Resource constrained |
| **Kubernetes cluster** | Separated | Better k8s integration |
| **Multi-region** | Separated | Distributed inference |
| **Development** | Monolithic | Simpler debugging |

### For Scaling

| Load | Architecture | Notes |
|------|-------------|-------|
| **< 10 req/s** | Monolithic | Single server sufficient |
| **10-50 req/s** | Separated (1 inf + 5 comp) | Scale compositing |
| **> 50 req/s** | Separated (Multi-GPU) | Multiple inference servers |

## Getting Started

**Recommended path for new users:**

1. **Start with Monolithic** (fastest to get running)
   ```powershell
   cd go-monolithic-server
   go build -o monolithic-server.exe ./cmd/server
   .\monolithic-server.exe
   ```

2. **Run the test client**
   ```powershell
   go build -o test-client.exe test_client.go
   .\test-client.exe
   ```

3. **Verify performance** (should see ~130ms latency)

4. **If you need scaling later**, migrate to Separated architecture

## Documentation

- 📖 [Monolithic Server README](go-monolithic-server/README.md)
- 🚀 [Quick Start Guide](go-monolithic-server/QUICKSTART.md)
- 📊 [Architecture Comparison](ARCHITECTURE_COMPARISON.md)
- 🎵 [Audio Integration](go-inference-server/AUDIO_INTEGRATION.md)
- 📝 [Test Client Guide](go-compositing-server/TEST_CLIENT_AUDIO_UPDATE.md)

## Support

Need help choosing? Ask yourself:

1. **Do you need >10 req/s?**
   - No → **Monolithic**
   - Yes → Continue to #2

2. **Do you need independent scaling of compositing vs inference?**
   - No → **Monolithic**
   - Yes → **Separated**

3. **Are you running on Kubernetes?**
   - No → **Monolithic**
   - Yes → **Separated**

**When in doubt, start with Monolithic.** It's faster, simpler, and you can migrate later if needed.

---

**All architectures are production-ready!** Choose based on your scaling needs. 🚀
