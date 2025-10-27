# Multi-GPU Inference Server

Dedicated GPU-only inference server for lip sync models. Returns raw float32 mouth regions without compositing.

## Features

- **Multi-GPU**: Supports 8× NVIDIA RTX 6000 Blackwell GPUs (768GB total)
- **Multi-tenant**: Load 1,200+ models simultaneously
- **Round-robin GPU assignment**: Automatic load balancing across GPUs
- **Dynamic loading**: Models loaded on-demand
- **LRU/LFU eviction**: Automatic memory management
- **Raw float32 output**: 320×320×3 arrays (1.2MB per frame)
- **No compositing**: Pure inference for maximum GPU utilization

## Quick Start

### 1. Configure

Edit `config.yaml`:

```yaml
server:
  port: ":50051"
  worker_count_per_gpu: 4

gpus:
  count: 8              # Your GPU count
  memory_gb_per_gpu: 96

capacity:
  max_models: 1200
  max_memory_gb: 720

onnx:
  library_path: "C:/onnxruntime-win-x64-gpu-1.21.0/lib/onnxruntime.dll"

models:
  sanders:
    model_path: "path/to/unet_328.onnx"
    preload: false
```

### 2. Build

```bash
# Generate protobuf
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       proto/inference.proto

# Build server
go build -o inference-server.exe .\cmd\server\
```

### 3. Run

```bash
.\inference-server.exe
```

Expected output:
```
🚀 Multi-GPU Inference Server (Inference ONLY)
   GPUs: 8 × 96GB
   Workers per GPU: 4 (total: 32 workers)
   Max models: 1200

🎮 GPU Status:
   GPU 0: 0 models, 0 MB used / 98304 MB total
   ...
   GPU 7: 0 models, 0 MB used / 98304 MB total

✅ Ready to accept connections!
```

## API

### InferBatch - Run GPU Inference

**Request**:
```protobuf
message InferBatchRequest {
    string model_id = 1;           // e.g., "sanders"
    bytes visual_frames = 2;       // 6*320*320 float32 (as bytes)
    bytes audio_features = 3;      // 32*16*16 float32 (as bytes)
    int32 batch_size = 4;          // 1-25 frames
}
```

**Response**:
```protobuf
message InferBatchResponse {
    repeated RawMouthRegion outputs = 1;  // Raw float32 arrays
    float inference_time_ms = 2;
    int32 gpu_id = 6;                     // Which GPU processed this
    bool success = 3;
}

message RawMouthRegion {
    bytes data = 1;  // 3*320*320 float32 = 307,200 floats = 1.2MB
    // Layout: [R_channel (320×320), G_channel (320×320), B_channel (320×320)]
}
```

**Client Example** (Go):
```go
conn, _ := grpc.Dial("localhost:50051", grpc.WithInsecure())
client := pb.NewInferenceServiceClient(conn)

// Prepare data
visualData := make([]float32, 6*320*320)
audioData := make([]float32, 32*16*16)
// ... fill with actual data

// Convert to bytes
visualBytes := float32ToBytes(visualData)
audioBytes := float32ToBytes(audioData)

// Call inference
resp, err := client.InferBatch(context.Background(), &pb.InferBatchRequest{
    ModelId:       "sanders",
    VisualFrames:  visualBytes,
    AudioFeatures: audioBytes,
    BatchSize:     1,
})

// Extract raw output
rawMouth := bytesToFloat32(resp.Outputs[0].Data)
// rawMouth is 320×320×3 float32 array in range [0, 1]
```

### ListModels - Get All Models

```bash
grpcurl -plaintext localhost:50051 inference.InferenceService/ListModels
```

Response shows all configured models, loaded status, GPU assignment, and stats.

### LoadModel - Preload a Model

```bash
grpcurl -plaintext -d '{"model_id": "sanders"}' \
    localhost:50051 inference.InferenceService/LoadModel
```

### GetModelStats - Get Usage Statistics

```bash
grpcurl -plaintext localhost:50051 inference.InferenceService/GetModelStats
```

Returns:
- Per-model: usage count, total inference time, memory usage, last used
- Per-GPU: models loaded, memory used/total
- Global: total models, total memory

### Health - Check Server Status

```bash
grpcurl -plaintext localhost:50051 inference.InferenceService/Health
```

## Performance

### Throughput
- **Single worker**: 150 FPS (23ms per frame)
- **32 workers** (8 GPUs × 4): 4,800 FPS
- **Concurrent users**: 710 conversational users (25 FPS, 27% duty cycle)

### Capacity
- **Max models**: 1,200 models (configurable)
- **GPU memory**: ~500MB per model
- **System RAM**: Minimal (no backgrounds)

### Latency
- **Inference only**: 21-23ms per frame
- **No overhead**: No compositing, encoding, or disk I/O

## Configuration Options

### GPU Assignment Strategies

```yaml
gpus:
  assignment_strategy: "round-robin"  # Options: round-robin, least-loaded
```

- **round-robin**: Distribute models evenly across GPUs
- **least-loaded**: Choose GPU with fewest models

### Eviction Policies

```yaml
capacity:
  eviction_policy: "lfu"  # Options: lru, lfu
```

- **LRU** (Least Recently Used): Evict oldest unused model
- **LFU** (Least Frequently Used): Evict least-used model

### Preferred GPU

```yaml
models:
  important_model:
    preferred_gpu: 0  # Force GPU 0 (-1 = auto)
```

## Integration

This server is designed to be called by a **compositing server** that:
1. Receives client requests
2. Calls this inference server for raw mouth regions
3. Loads backgrounds from disk/cache
4. Composites mouth onto background
5. Encodes and streams to client (PNG or H.264/WebRTC)

See `SEPARATED_ARCHITECTURE.md` for the full architecture.

## Monitoring

### Logs

```
⚡ Inference: model=sanders, batch=16, gpu=2, time=22.34ms, total=23.15ms
🔄 Model 'new_model' loaded on GPU 5 in 1.23s (memory: 500 MB)
⚠️  Evicting model 'old_model' (lfu policy)
```

### Statistics

Call `GetModelStats` to monitor:
- Models per GPU
- Memory usage per GPU
- Inference times
- Usage counts

## Troubleshooting

### GPU Out of Memory

Reduce `max_models` in config.yaml or enable more aggressive eviction:

```yaml
capacity:
  max_models: 800  # Reduce from 1200
  idle_timeout_minutes: 30  # Evict after 30 min idle
```

### Model Load Slow

Check ONNX Runtime library path and GPU drivers:

```yaml
onnx:
  library_path: "C:/onnxruntime-win-x64-gpu-1.21.0/lib/onnxruntime.dll"
```

Ensure CUDA is properly installed and accessible.

### Uneven GPU Load

Switch to `least-loaded` strategy:

```yaml
gpus:
  assignment_strategy: "least-loaded"
```

## Hardware Requirements

**Recommended**:
- NVIDIA GPUs with CUDA support
- 48GB+ GPU memory per GPU
- Fast internal network for compositing server (10 Gbps+)

**Your Setup**:
- 8× NVIDIA RTX 6000 Blackwell (96GB each)
- 2TB system RAM
- **Capacity**: 1,200+ models, 710+ concurrent users

## License

Same as parent project.
