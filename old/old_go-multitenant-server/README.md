# Multi-Tenant LipSync gRPC Server

## Overview

A production-grade, multi-tenant lip-sync inference server supporting:
- **Multiple Models**: Run sanders, bob, jane, or any number of models simultaneously
- **Dynamic Loading**: Models load on-demand and unload automatically when not needed
- **Capacity Management**: LRU/LFU eviction when GPU memory is constrained
- **Usage Tracking**: Detailed statistics on model usage, inference time, and memory
- **Full Compositing**: Returns complete 1280x720 PNG frames ready to display

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     gRPC Client                              │
│  (sends model_id + visual/audio data)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Tenant gRPC Server                        │
│                    (port 50053)                              │
├─────────────────────────────────────────────────────────────┤
│  • Receives request with model_id                            │
│  • Checks ModelRegistry                                      │
│  • Loads model if not cached (auto-evicts if capacity full)  │
│  • Runs inference + compositing                              │
│  • Tracks usage statistics                                   │
│  • Returns PNG-encoded frames                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Registry                              │
├─────────────────────────────────────────────────────────────┤
│  models:                                                     │
│    "sanders" ───> ModelInstance                              │
│                    ├─ Inferencer (ONNX)                      │
│                    ├─ Backgrounds (1280x720 PNGs)            │
│                    ├─ Crop Rectangles (JSON)                 │
│                    └─ Stats (usage, last_used, memory)       │
│    "bob" ───> ModelInstance                                  │
│    "jane" ───> ModelInstance                                 │
│                                                              │
│  Capacity Management:                                        │
│    • Max 3 models (configurable)                             │
│    • LRU or LFU eviction                                     │
│    • Idle timeout unloading                                  │
│    • Periodic stats reporting                                │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Multi-Model Support

Each request specifies a `model_id`:

```protobuf
message CompositeBatchRequest {
  string model_id = 1;           // "sanders", "bob", etc.
  repeated float visual_frames = 2;
  repeated float audio_features = 3;
  int32 batch_size = 4;
  int32 start_frame_idx = 5;
}
```

### 2. Dynamic Loading

- Models load **on-demand** when first requested
- Subsequent requests use cached model (fast path)
- No need to preload all models on startup

### 3. Capacity Management

**Configuration:**
```yaml
capacity:
  max_models: 3                    # Max models in memory
  max_memory_gb: 4                 # Approximate GPU limit
  eviction_policy: "lru"           # lru or lfu
  idle_unload_minutes: 30          # Auto-unload after idle time
```

**Eviction Strategies:**
- **LRU (Least Recently Used)**: Evicts model with oldest `last_used` timestamp
- **LFU (Least Frequently Used)**: Evicts model with lowest `usage_count`

When capacity is full and a new model is requested:
1. Server checks if any model can be evicted
2. Evicts least-used model according to policy
3. Loads new model
4. Logs eviction event

### 4. Usage Statistics

Each model tracks:
- `usage_count`: Total inferences performed
- `last_used`: Timestamp of last access
- `total_inference_ms`: Cumulative inference time
- `memory_bytes`: Estimated GPU + RAM usage
- `loaded_at`: When model was loaded

### 5. Full Compositing

Server returns PNG-encoded 1280x720 frames:
- Loads per-frame crop rectangles
- Loads background frames
- Composites 320x320 mouth onto background
- Encodes to PNG
- Client receives ready-to-display images

## Directory Structure

```
go-multitenant-server/
├── cmd/
│   ├── server/
│   │   └── main.go          # Multi-tenant gRPC server
│   └── client/
│       └── main.go          # Test client
├── config/
│   └── config.go            # Configuration loader
├── lipsyncinfer/
│   └── inferencer.go        # ONNX wrapper
├── registry/
│   └── registry.go          # Model lifecycle manager
├── proto/
│   ├── multitenant.proto    # Service definition
│   ├── multitenant.pb.go    # Generated protobuf
│   └── multitenant_grpc.pb.go
├── config.yaml              # Server configuration
├── go.mod
└── README.md               # This file
```

## Configuration

### config.yaml

```yaml
server:
  port: ":50053"
  max_message_size_mb: 200

capacity:
  max_models: 3
  max_memory_gb: 4
  eviction_policy: "lru"       # or "lfu"
  idle_unload_minutes: 30      # 0 = disabled

onnx:
  library_path: "C:\\onnxruntime-1.22.0\\lib\\onnxruntime.dll"
  cuda_enabled: true

models:
  sanders:
    model_path: "d:/Projects/.../sanders/checkpoint/model_best.onnx"
    background_dir: "d:/Projects/.../sanders/frames"
    crop_rects_path: "d:/Projects/.../sanders/cache/crop_rectangles.json"
    num_frames: 100
    preload: true               # Load on startup
    memory_estimate_mb: 500

  bob:
    model_path: "d:/Projects/.../bob/checkpoint/model_best.onnx"
    background_dir: "d:/Projects/.../bob/frames"
    crop_rects_path: "d:/Projects/.../bob/cache/crop_rectangles.json"
    num_frames: 100
    preload: false              # Load on-demand
    memory_estimate_mb: 500

logging:
  level: "info"
  log_model_operations: true
  stats_report_minutes: 5       # Periodic stats (0 = disabled)
```

## Building

```powershell
# Install dependencies
cd go-multitenant-server
go mod download

# Generate protobuf code (if modified)
protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative proto/multitenant.proto

# Build server
cd cmd/server
go build -o multi-tenant-server.exe .

# Build client
cd ../client
go build -o multi-tenant-client.exe .
```

## Running

### Start Server

```powershell
cd go-multitenant-server/cmd/server
.\multi-tenant-server.exe
```

**Output:**
```
================================================================================
🏢 Multi-Tenant LipSync gRPC Server
================================================================================
✅ Configuration loaded from config.yaml
   Max models: 3
   Max memory: 4 GB
   Eviction policy: lru
   Configured models: 2

📦 Initializing model registry...
📦 Preloading model 'sanders'...
   Loading ONNX model: d:/Projects/.../sanders/checkpoint/model_best.onnx
   Loading 100 background frames: d:/Projects/.../sanders/frames
   Loading crop rectangles: d:/Projects/.../sanders/cache/crop_rectangles.json
   Warming up CUDA...
✅ Model 'sanders' loaded in 5.23s (memory: 850.00 MB)
✅ Model registry initialized (1 models preloaded)

🌐 Server listening on port :50053
   Protocol: gRPC with Protobuf
   Features:
      • Multi-model support
      • Dynamic model loading
      • Automatic eviction (LRU/LFU)
      • Usage statistics tracking
      • Compositing with backgrounds

✅ Ready to accept connections!
================================================================================
```

### Run Test Client

```powershell
cd go-multitenant-server/cmd/client
.\multi-tenant-client.exe
```

**Test Flow:**
1. Health check
2. List available models
3. Test inference with sanders (batch sizes: 1, 5, 10)
4. Get model statistics
5. Test model unload/reload

## API Reference

### InferBatchComposite

Main inference endpoint with compositing.

**Request:**
```protobuf
message CompositeBatchRequest {
  string model_id = 1;           // Required: "sanders", "bob", etc.
  repeated float visual_frames = 2;  // [batch_size * 6 * 320 * 320]
  repeated float audio_features = 3; // [32 * 16 * 16] single window
  int32 batch_size = 4;          // 1-25 frames
  int32 start_frame_idx = 5;     // For crop rect/background lookup
}
```

**Response:**
```protobuf
message CompositeBatchResponse {
  repeated bytes composited_frames = 1;  // PNG bytes (1280x720)
  double inference_time_ms = 2;
  double composite_time_ms = 3;
  double total_time_ms = 4;
  bool model_loaded = 5;         // Was model loaded during this request?
  double model_load_time_ms = 6;
  bool success = 7;
  string error = 8;
}
```

### ListModels

List all configured models (loaded and unloaded).

**Request:** Empty

**Response:**
```protobuf
message ListModelsResponse {
  repeated ModelInfo models = 1;
}

message ModelInfo {
  string model_id = 1;
  bool loaded = 2;               // Currently in memory?
  string model_path = 3;
  string background_dir = 4;
  string crop_rects_path = 5;
  ModelStats stats = 6;          // null if not loaded
}
```

### LoadModel

Explicitly load a model (usually not needed - auto-loads on inference).

**Request:**
```protobuf
message LoadModelRequest {
  string model_id = 1;
  bool force_reload = 2;  // Reload even if already loaded
}
```

### UnloadModel

Explicitly unload a model to free memory.

**Request:**
```protobuf
message UnloadModelRequest {
  string model_id = 1;
}
```

### GetModelStats

Get detailed statistics for one or all models.

**Request:**
```protobuf
message GetModelStatsRequest {
  string model_id = 1;  // Empty = all loaded models
}
```

**Response:**
```protobuf
message GetModelStatsResponse {
  repeated ModelInfo models = 1;
  int32 max_models = 2;
  int32 loaded_models = 3;
  int64 total_memory_bytes = 4;
  int64 max_memory_bytes = 5;
}
```

## Adding a New Model

1. **Prepare model files:**
   ```
   models/jane/
   ├── checkpoint/
   │   └── model_best.onnx
   ├── frames/
   │   ├── frame_0000.png
   │   ├── frame_0001.png
   │   └── ...
   └── cache/
       └── crop_rectangles.json
   ```

2. **Add to config.yaml:**
   ```yaml
   models:
     jane:
       model_path: "d:/Projects/.../jane/checkpoint/model_best.onnx"
       background_dir: "d:/Projects/.../jane/frames"
       crop_rects_path: "d:/Projects/.../jane/cache/crop_rectangles.json"
       num_frames: 100
       preload: false
       memory_estimate_mb: 500
   ```

3. **Restart server** (or use LoadModel RPC)

4. **Send inference requests:**
   ```go
   req := &pb.CompositeBatchRequest{
       ModelId: "jane",  // ← New model
       VisualFrames: visualData,
       AudioFeatures: audioData,
       BatchSize: 5,
   }
   ```

## Performance

**Expected throughput per model:**
- Single frame: ~40-50 FPS
- Batch of 5: ~80-120 FPS
- Batch of 10: ~100-150 FPS

**Per-frame timing:**
- Inference: ~21-23ms
- Compositing: ~3-4ms
- PNG encoding: ~10-15ms
- gRPC overhead: ~2-5ms

**Memory usage per model:**
- ONNX model: ~500MB GPU
- Background cache (100 frames): ~350MB RAM
- Total: ~850MB per model

**Capacity example:**
- 4GB GPU → ~4-5 models maximum
- 8GB GPU → ~9-10 models maximum

## Logging

Server logs include:

**Model operations:**
```
📦 Preloading model 'sanders'...
✅ Model 'sanders' loaded in 5.23s (memory: 850.00 MB)
🔄 Model 'bob' loaded on-demand in 4.87s
⚠️  Evicting model 'jane' (policy: lru, usage: 3, last used: 15m)
🗑️  Model 'sanders' unloaded (was active for 2h, used 1247 times)
⏰ Auto-unloading idle model 'bob' (idle for 35m)
```

**Periodic statistics (every 5 minutes):**
```
📊 Model Statistics Report:
   Loaded models: 2/3
   Total memory: 1700.00 MB
   • sanders: usage=1247, last=5s, avg_inference=22.34ms
   • bob: usage=432, last=2m, avg_inference=21.89ms
```

## Monitoring

Use `GetModelStats` RPC to monitor:
- Which models are loaded
- Usage patterns (counts, last access time)
- Memory consumption
- Average inference time per model

## Troubleshooting

**Problem: "capacity full and eviction failed"**
- All models are being actively used
- Solution: Increase `max_models` in config or add more GPU memory

**Problem: Model loading slow**
- CUDA warmup takes ~1.5s on first inference
- Background loading takes ~3-4s for 100 frames
- Solution: Use `preload: true` for frequently-used models

**Problem: Model unloaded unexpectedly**
- Idle timeout reached
- LRU/LFU eviction due to capacity
- Check logs for eviction events

**Problem: High memory usage**
- Backgrounds are cached in RAM
- Solution: Reduce `num_frames` or implement lazy loading

## Comparison: Single-Tenant vs Multi-Tenant

| Feature | Single-Tenant (port 50052) | Multi-Tenant (port 50053) |
|---------|---------------------------|---------------------------|
| **Models** | One hardcoded model | Multiple dynamic models |
| **Loading** | On startup only | On-demand + eviction |
| **Selection** | No choice | model_id in request |
| **Capacity** | N/A | LRU/LFU eviction |
| **Stats** | No tracking | Full usage tracking |
| **Memory** | Fixed | Dynamic management |
| **Use Case** | Production (single user/model) | Production (multi-user/model) |

## License

Same as parent project.

## Credits

Built on top of:
- Go ONNX inference architecture (go-onnx-inference/)
- Single-tenant composite server (port 50052)
- Proven 180 FPS parallel compositing pipeline
