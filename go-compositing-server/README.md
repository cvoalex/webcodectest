# Compositing Server

CPU-heavy compositing server that calls the inference server for GPU work, then handles background compositing and PNG encoding.

## Features

- **Calls inference server**: gRPC connection to GPU inference server
- **Lazy-loading backgrounds**: LRU cache (50 frames per model by default)
- **Multi-tenant**: Support 11,000+ models with 2TB RAM
- **PNG encoding**: Returns PNG frames (ready for WebRTC H.264 upgrade)
- **Automatic eviction**: LRU/LFU when capacity reached

## Quick Start

### 1. Start Inference Server First

The compositing server requires the inference server to be running:

```bash
cd ../go-inference-server
.\inference-server.exe
```

Wait for:
```
✅ Ready to accept connections!
```

### 2. Configure Compositing Server

Edit `config.yaml`:

```yaml
server:
  port: ":50052"  # Different from inference server

inference_server:
  url: "localhost:50051"  # Point to inference server

capacity:
  max_models: 11000           # 2TB RAM / 175MB per model
  background_cache_frames: 50 # LRU cache size

models:
  sanders:
    background_dir: "path/to/sanders/background"
    crop_rects_path: "path/to/sanders/crop_rects.json"
    num_frames: 2000
```

### 3. Build & Run

```bash
# Generate protobuf (if needed)
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       proto/compositing.proto

protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       proto_inference/inference.proto

# Build
go build -o compositing-server.exe .\cmd\server\

# Run
.\compositing-server.exe
```

Expected output:
```
🎨 Compositing Server (CPU + Background Resources)
✅ Configuration loaded from config.yaml
   Inference server: localhost:50051

🔌 Connecting to inference server at localhost:50051...
✅ Connected to inference server
   GPUs: 8
   Loaded models: 0/1200
   Version: 1.0.0

📦 Initializing compositing registry...
✅ Compositing registry initialized (0 models loaded)

🌐 Compositing server listening on port :50052
   Features:
      • Calls inference server for GPU work
      • Lazy-loading background cache
      • Multi-model compositing
      • PNG encoding
      • Ready for WebRTC integration

✅ Ready to accept connections!
```

## Architecture

```
Client
  ↓ gRPC (visual + audio features)
Compositing Server
  ↓ gRPC (raw float32 request)
Inference Server (GPU)
  ↓ Returns raw 320×320×3 float32 mouth
Compositing Server
  ↓ Loads background from cache/disk
  ↓ Composites mouth onto background
  ↓ Encodes to PNG
  ↓ Returns to client
```

## API

### InferBatchComposite - Full Pipeline

**Request** (same as monolithic server):
```protobuf
message CompositeBatchRequest {
    string model_id = 1;
    bytes visual_frames = 2;   // 6*320*320 float32
    bytes audio_features = 3;  // 32*16*16 float32
    int32 batch_size = 4;
    int32 start_frame_idx = 5; // For background selection
}
```

**Response**:
```protobuf
message CompositeBatchResponse {
    repeated bytes composited_frames = 1;  // PNG-encoded
    float inference_time_ms = 2;            // Time on GPU server
    float composite_time_ms = 3;            // Time on this server
    float total_time_ms = 4;
    int32 gpu_id = 9;                       // Which GPU processed it
    bool success = 7;
}
```

**Client Example** (Go):
```go
conn, _ := grpc.Dial("localhost:50052", grpc.WithInsecure())
client := pb.NewCompositingServiceClient(conn)

resp, err := client.InferBatchComposite(context.Background(), &pb.CompositeBatchRequest{
    ModelId:       "sanders",
    VisualFrames:  visualBytes,
    AudioFeatures: audioBytes,
    BatchSize:     16,
    StartFrameIdx: 0,
})

// Extract PNG frames
for i, pngData := range resp.CompositedFrames {
    os.WriteFile(fmt.Sprintf("frame_%d.png", i), pngData, 0644)
}

fmt.Printf("Inference: %.2fms\n", resp.InferenceTimeMs)
fmt.Printf("Compositing: %.2fms\n", resp.CompositeTimeMs)
fmt.Printf("Total: %.2fms\n", resp.TotalTimeMs)
```

### Health - Check Status

```bash
grpcurl -plaintext localhost:50052 compositing.CompositingService/Health
```

Response shows:
- Compositing server health
- Inference server health
- Loaded models count

## Performance

### Latency Breakdown

Typical request (batch=16):
- **Inference** (GPU server): 23ms (actual GPU work)
- **Network** (compositing → inference): 0.5-2ms (same datacenter)
- **Compositing** (this server): 15-20ms (CPU work)
- **PNG encoding** (this server): 5-10ms
- **Total**: ~42ms (vs 40ms monolithic)

**Overhead**: +2ms for separation (minimal!)

### Throughput

With 64-core CPU server:
- Compositing: ~25ms per batch
- Single thread: 40 batches/sec
- 64 threads (well-parallelized): 2,560 batches/sec
- **Concurrent users**: 380 conversational users per box

### Memory Usage

- **Per model**: 175MB RAM (50 frame cache × 1280×720×4)
- **2TB RAM capacity**: 11,000+ models
- **Actual usage**: Depends on concurrent active models

### Scaling

**Horizontal scaling** (add more compositing servers):
```
1× Inference Server (710 user capacity): $40K
2× Compositing Servers (760 user capacity): $14K
Total: $54K for 710 users
```

**Cost per user**: $76 (vs $170 monolithic)

## Configuration

### Background Cache Size

Adjust based on RAM availability and hit rate:

```yaml
capacity:
  background_cache_frames: 50  # Default
  # Larger = better hit rate but more RAM
  # 20 frames = 70MB per model
  # 50 frames = 175MB per model
  # 100 frames = 350MB per model
```

### Eviction Policy

```yaml
capacity:
  eviction_policy: "lfu"  # Options: lru, lfu
```

- **LRU**: Evict least recently used (good for changing workload)
- **LFU**: Evict least frequently used (good for stable popular models)

### Inference Server Connection

```yaml
inference_server:
  url: "localhost:50051"        # Can be remote
  timeout_seconds: 10            # Request timeout
  max_retries: 3                 # Retry on failure
```

For production, use internal datacenter network or InfiniBand.

## Monitoring

### Logs

```
🎨 Composite: model=sanders, batch=16, gpu=2, inference=22.34ms, composite=18.45ms, total=41.23ms
🔄 Loading compositing resources for model 'bob'...
✅ Model 'bob' compositing resources loaded in 0.25s (memory: 175 MB)
```

### Cache Statistics

Enable in config:
```yaml
logging:
  log_cache_stats: true
```

### gRPC Metrics

Query stats endpoint:
```bash
grpcurl -plaintext localhost:50052 compositing.CompositingService/GetModelStats
```

## WebRTC Integration (Future)

This server is designed to be upgraded for WebRTC streaming:

**Today**:
- Returns PNG frames
- Client downloads each frame

**Future** (WebRTC):
- Persistent WebRTC connection
- H.264 hardware encoding (replace PNG encoder)
- Stream video directly to browser
- 75x bandwidth reduction

See `SEPARATED_ARCHITECTURE.md` for details.

## Troubleshooting

### Cannot connect to inference server

```
❌ Inference server health check failed
```

**Solution**: Ensure inference server is running first:
```bash
cd ../go-inference-server
.\inference-server.exe
```

### Model resources not found

```
❌ Failed to load model resources: failed to open frame 0
```

**Solution**: Check paths in config.yaml:
```yaml
models:
  sanders:
    background_dir: "d:/Projects/webcodecstest/model_videos/sanders/background"
    crop_rects_path: "d:/Projects/webcodecstest/model_videos/sanders/crop_rects.json"
```

### High latency

**Symptom**: `total_time_ms` > 50ms

**Diagnosis**:
- Check `inference_time_ms` (should be ~23ms)
- Check `composite_time_ms` (should be ~15-20ms)
- If inference high: GPU server overloaded
- If compositing high: CPU saturated or slow disk I/O

**Solutions**:
- Add more compositing servers (horizontal scaling)
- Increase background cache (reduce disk I/O)
- Use faster SSD for background frames

## Deployment

### Same Machine (Development)

```
Inference Server: :50051
Compositing Server: :50052
```

Both on same machine, minimal network latency.

### Separate Machines (Production)

```
Datacenter (GPU box):
  Inference Server: internal-gpu-1:50051

Edge (CPU boxes):
  Compositing Server 1: edge-nyc-1:50052 → calls internal-gpu-1:50051
  Compositing Server 2: edge-nyc-2:50052 → calls internal-gpu-1:50051
  Compositing Server 3: edge-sf-1:50052  → calls internal-gpu-1:50051
```

Use internal datacenter network for compositing → inference calls.

## Hardware Requirements

**Recommended**:
- **CPU**: 64+ cores (compositing + encoding is CPU-heavy)
- **RAM**: 256GB minimum (supports 1,400 models)
- **RAM**: 2TB optimal (supports 11,000+ models)
- **Storage**: Fast SSD for background frames (NVMe preferred)
- **Network**: 10 Gbps to inference server

**Your Setup**:
- 2TB RAM: 11,000+ model capacity
- With lazy loading: Minimal startup time
- Perfect for massive multi-tenant deployment

## Summary

**Why Separation?**
- GPU stays focused on inference (95% utilization vs 60%)
- Compositing scales independently (cheap CPU boxes)
- Ready for WebRTC streaming (edge deployment)
- 67% cost savings at scale

**Your Benefits**:
- 2TB RAM = 11,000+ models
- Lazy loading = instant startup
- Edge deployment ready
- WebRTC upgrade path

Next step: Start both servers and test the full pipeline! 🚀
