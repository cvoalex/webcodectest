# Monolithic Lipsync Server

## Overview

The monolithic server combines **inference** and **compositing** into a single Go process, eliminating inter-service communication overhead for maximum performance.

## Documentation

📚 **Comprehensive documentation available in [`docs/`](docs/):**

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[QUICK_TEST_GUIDE.md](docs/QUICK_TEST_GUIDE.md)** - Running tests and benchmarks
- **[AUTO_EXTRACTION.md](docs/AUTO_EXTRACTION.md)** - Automatic frame extraction feature
- **[IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md)** - Complete implementation guide
- **[PERFORMANCE_RESULTS.md](docs/PERFORMANCE_RESULTS.md)** - Performance benchmarks and analysis
- **[EXTRACTION_VS_CACHING.md](docs/EXTRACTION_VS_CACHING.md)** - Detailed caching strategy analysis
- **[AUDIO_PROCESSING_EXPLAINED.md](docs/AUDIO_PROCESSING_EXPLAINED.md)** - Audio pipeline documentation
- **[AUDIO_VALIDATION_RESULTS.md](docs/AUDIO_VALIDATION_RESULTS.md)** - Audio validation testing

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│          Monolithic Lipsync Server (Port 50053)         │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │   Audio     │  │    Model     │  │     Image      │ │
│  │ Processing  │  │   Registry   │  │   Registry     │ │
│  │  Pipeline   │  │              │  │                │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
│         │                 │                 │           │
│         ▼                 ▼                 ▼           │
│  ┌───────────────────────────────────────────────────┐ │
│  │         InferBatchComposite Handler               │ │
│  │                                                     │ │
│  │  1. Process raw audio → mel → features           │ │
│  │  2. Run GPU inference (ONNX Runtime + CUDA)       │ │
│  │  3. Composite mouth onto backgrounds              │ │
│  │  4. Encode to JPEG                                │ │
│  └───────────────────────────────────────────────────┘ │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Key Differences from Separated Architecture

### Separated (2 servers):
```
Client → Compositing Server (gRPC) → Inference Server (gRPC) → GPU
            ↓ (overhead: serialization, network, queue)
         Compositing CPU work
```

### Monolithic (1 server):
```
Client → Monolithic Server → GPU → Compositing → JPEG
            ↓ (direct function calls, zero overhead)
```

## Performance Benefits

| Metric | Separated | Monolithic | Improvement |
|--------|-----------|------------|-------------|
| **Inter-service latency** | ~3-5ms | 0ms | **100% eliminated** |
| **Serialization overhead** | 2× (gRPC in+out) | 1× (client gRPC only) | **50% reduced** |
| **Memory copies** | 3× | 1× | **67% reduced** |
| **Context switches** | Multiple | Single | **Faster** |
| **Total overhead** | ~10-15ms | ~2-3ms | **70-80% faster** |

## Features

✅ **Zero Inter-Service Communication** - All processing in single process  
✅ **Real-Time Audio Processing** - Mel-spectrogram + audio encoder in Go  
✅ **Multi-GPU Inference** - ONNX Runtime with CUDA acceleration  
✅ **Dynamic Model Loading** - Load models on-demand  
✅ **Background Caching** - Preload backgrounds into memory  
✅ **LFU Eviction** - Automatic model unloading when capacity reached  
✅ **JPEG Encoding** - Fast CPU-based image encoding  

## Configuration

### config.yaml

```yaml
server:
  port: ":50053"              # Different from other servers
  max_message_size_mb: 100
  worker_count_per_gpu: 8
  queue_size: 50

gpus:
  enabled: true
  count: 1
  memory_gb_per_gpu: 24
  assignment_strategy: "round-robin"

capacity:
  max_models: 40
  max_memory_gb: 20
  background_cache_frames: 600
  eviction_policy: "lfu"
  idle_timeout_minutes: 60

onnx:
  library_path: "C:/onnxruntime-1.22.0/lib/onnxruntime.dll"
  cuda_streams_per_worker: 2
  intra_op_threads: 4
  inter_op_threads: 2

output:
  format: "jpeg"
  jpeg_quality: 75

logging:
  log_inference_times: true
  log_level: "info"

models_root: "d:/Projects/webcodecstest/minimal_server/models"

models:
  sanders:
    model_path: "sanders/checkpoint/model_best.onnx"
    background_dir: "sanders/frames"
    crop_rects_path: "sanders/crop_rects.json"
    num_frames: 523
    preload_backgrounds: true
    preload_model: false
    preferred_gpu: 0
```

## Building

```powershell
cd go-monolithic-server
go mod tidy
go build -o monolithic-server.exe ./cmd/server
```

## Running

```powershell
.\monolithic-server.exe
```

**Output:**
```
================================================================================
🚀 Monolithic Lipsync Server (Inference + Compositing)
================================================================================
✅ Configuration loaded from config.yaml
   GPUs: 1 × 24GB
   Workers per GPU: 8 (total: 8 workers)
   Max models: 40
   Max memory: 20 GB
   Background cache: 600 frames per model
   Eviction policy: lfu
   Configured models: 3

📦 Initializing model registry...
✅ Model registry initialized (0 models preloaded)

🖼️  Initializing image registry...
🖼️  Loading backgrounds for model 'sanders'...
✅ Loaded backgrounds for 'sanders' in 1.23s (523 frames, 246.72 MB)
✅ Image registry initialized (1 models loaded)

🎮 GPU Status:
   GPU 0: 0 models, 0 MB used / 24576 MB total

🎵 Initializing audio processing pipeline...
✅ Mel-spectrogram processor initialized
✅ Audio encoder initialized (ONNX)

🌐 Monolithic server listening on port :50053
   Protocol: gRPC with Protobuf
   Features:
      • Inference + Compositing in single process
      • No inter-service communication overhead
      • Multi-GPU inference
      • Real-time audio processing (mel + encoder)
      • Dynamic model loading
      • Automatic eviction (LFU)
      • JPEG-encoded output

✅ Ready to accept connections!
================================================================================
```

## API

### InferBatchComposite

Complete pipeline: audio → inference → compositing → JPEG

**Request:**
```protobuf
message CompositeBatchRequest {
    string model_id = 1;
    bytes visual_frames = 2;
    bytes raw_audio = 3;           // Raw PCM audio (16kHz, float32)
    bytes audio_features = 4;      // DEPRECATED: pre-computed features
    int32 batch_size = 5;
    int32 start_frame_idx = 6;
}
```

**Response:**
```protobuf
message CompositeBatchResponse {
    repeated bytes composited_frames = 1;  // JPEG-encoded frames
    float inference_time_ms = 2;
    float composite_time_ms = 3;
    float total_time_ms = 4;
    float audio_processing_ms = 5;
    bool success = 8;
    int32 gpu_id = 10;
}
```

### Other RPCs

- `Health` - Server health check
- `ListModels` - List all configured models
- `LoadModel` - Explicitly load a model + backgrounds
- `UnloadModel` - Unload a model + backgrounds
- `GetModelStats` - Get usage statistics

## Performance Monitoring

**Request Logging:**
```
🎵 Audio processing: 10240 samples -> 52 frames -> 24 features (7.45ms)
⚡ Inference: model=sanders, batch=24, gpu=0, time=120.28ms
🎨 Compositing: 24 frames, 2.15ms (0.09ms/frame)
```

**Typical Performance (Batch Size 24):**
- Audio Processing: 7-10ms
- Inference (GPU): 120ms
- Compositing: 2-3ms
- JPEG Encoding: (included in compositing)
- **Total: ~130ms** vs ~145ms with separated architecture

## Testing

Use the test client from `go-compositing-server`:

```powershell
cd ..\go-compositing-server

# Edit test_client.go to change server address:
# serverAddr = "localhost:50053"

go build -o test-client.exe test_client.go
.\test-client.exe
```

## Comparison with Other Architectures

| Architecture | Processes | Communication | Overhead | Throughput |
|--------------|-----------|---------------|----------|------------|
| **Monolithic** | 1 | None | ~2-3ms | **Highest** |
| Separated | 2 | gRPC | ~10-15ms | High |
| Python | 1 | None | ~50-100ms | Medium |

## Migration from Separated Architecture

**Minimal Changes Required:**

1. Update client connection:
   ```go
   // OLD: Connect to compositing server
   conn, err := grpc.Dial("localhost:50052", ...)
   
   // NEW: Connect to monolithic server  
   conn, err := grpc.Dial("localhost:50053", ...)
   ```

2. Update proto import (if needed):
   ```go
   // OLD
   pb "go-compositing-server/proto"
   
   // NEW
   pb "go-monolithic-server/proto"
   ```

3. Same API! `InferBatchComposite` works identically

## Troubleshooting

### GPU Memory Issues
```
❌ Failed to load model: insufficient GPU memory
```
**Solution:** Reduce `max_models` in config or increase `max_memory_gb`

### ONNX Runtime Not Found
```
❌ Failed to initialize audio encoder
```
**Solution:** Check `onnx.library_path` in config.yaml points to correct DLL

### Backgrounds Not Loading
```
❌ Failed to load backgrounds: frame 0 not found
```
**Solution:** Verify `background_dir` paths in config.yaml are correct

## File Structure

```
go-monolithic-server/
├── cmd/
│   └── server/
│       └── main.go              # Main server (800+ lines)
├── audio/
│   ├── processor.go             # Mel-spectrogram processor
│   ├── encoder.go               # Audio encoder (ONNX)
│   └── windows.go               # Sliding window extraction
├── lipsyncinfer/
│   └── inferencer.go            # ONNX inference wrapper
├── registry/
│   ├── model_registry.go        # Model/GPU management
│   └── image_registry.go        # Background/crop rect management
├── config/
│   └── config.go                # Configuration loader
├── proto/
│   ├── monolithic.proto         # Proto definition
│   ├── monolithic.pb.go         # Generated proto code
│   └── monolithic_grpc.pb.go    # Generated gRPC code
├── config.yaml                  # Server configuration
├── go.mod                       # Go module definition
└── README.md                    # This file
```

## Development Notes

### Why Monolithic?

The separated architecture was designed for:
- **Horizontal scaling** - Different scaling needs for CPU vs GPU
- **Resource isolation** - Separate memory/CPU limits
- **Independent deployment** - Update compositing without GPU restart

However, for **single-machine deployments**, the monolithic approach is superior because:
- ✅ No network overhead
- ✅ No serialization overhead  
- ✅ No queue delays
- ✅ Simpler deployment
- ✅ Lower latency (70-80% less overhead)

### When to Use Separated Architecture

Use separated servers if you need:
- Multiple compositing servers sharing one inference server
- Kubernetes-style orchestration
- Separate scaling policies (e.g., 10 compositing pods : 1 inference pod)
- Network-level load balancing

## License

Same as parent project.
