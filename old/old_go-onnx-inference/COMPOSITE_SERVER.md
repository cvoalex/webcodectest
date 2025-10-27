# gRPC Composite Server

## Overview

The **gRPC Composite Server** is a complete lip-sync inference service that:
1. Performs ONNX model inference on lip-sync visual/audio inputs
2. Composites the generated 320x320 mouth regions onto full 1280x720 background frames
3. Returns PNG-encoded full-resolution frames via gRPC/Protobuf

This is a **separate service** from the basic inference-only gRPC server.

## Architecture

### Components

- **Server**: `go-onnx-inference/cmd/grpc-composite-server/main.go`
  - Port: **50052** (different from basic server on 50051)
  - Protocol: gRPC with Protobuf
  - Features:
    - Batch inference (1-25 frames)
    - Per-frame crop rectangle loading
    - Background frame caching
    - Pure Go image compositing
    - PNG encoding of output

- **Client**: `go-onnx-inference/cmd/grpc-composite-client/main.go`
  - Test client for the composite server
  - Tests batch sizes: 1, 5, 10
  - Measures throughput and timing

- **Proto Definition**: `go-onnx-inference/proto/lipsync_composite.proto`
  - Service: `LipSyncComposite`
  - Request: Visual frames + audio window + batch size + start frame index
  - Response: PNG-encoded composited frames + timing info

## Comparison: Composite vs Basic Server

| Feature | Basic Server (port 50051) | Composite Server (port 50052) |
|---------|---------------------------|-------------------------------|
| **Inference** | ✅ Yes | ✅ Yes |
| **Compositing** | ❌ No | ✅ Yes |
| **Output** | 320x320 mouth regions (float32) | 1280x720 full frames (PNG bytes) |
| **Output Size** | ~307KB per frame | ~500KB-1MB per frame (PNG) |
| **Background Loading** | N/A | ✅ Cached in memory |
| **Crop Rectangles** | N/A | ✅ Per-frame from JSON |
| **Use Case** | Client-side compositing | Server-side compositing |

## Data Requirements

The composite server needs these additional resources:

1. **Background Frames**: 
   - Location: `d:/Projects/webcodecstest/minimal_server/models/sanders/frames/`
   - Format: PNG files (1280x720)
   - Naming: `frame_0000.png`, `frame_0001.png`, etc.

2. **Crop Rectangles**:
   - Location: `d:/Projects/webcodecstest/minimal_server/models/sanders/cache/crop_rectangles.json`
   - Format: `{"0": {"rect": [x1, y1, x2, y2]}, "1": {...}, ...}`
   - One rectangle per frame (defines where to composite mouth region)

3. **ONNX Model**:
   - Location: `d:/Projects/webcodecstest/minimal_server/models/sanders/unet_328.onnx`
   - Same model used by basic server

## Building

```powershell
# Build server
cd d:\Projects\webcodecstest\go-onnx-inference\cmd\grpc-composite-server
go build -o grpc-composite-server.exe .

# Build client
cd d:\Projects\webcodecstest\go-onnx-inference\cmd\grpc-composite-client
go build -o grpc-composite-client.exe .
```

## Running

### Start Server

```powershell
cd d:\Projects\webcodecstest\go-onnx-inference\cmd\grpc-composite-server
.\grpc-composite-server.exe
```

Server will:
1. Load ONNX model (~2-3s)
2. Cache background frames (~3-4s for 100 frames)
3. Load crop rectangles from JSON
4. Warm up CUDA (~1.5s first inference)
5. Listen on `localhost:50052`

### Run Test Client

```powershell
cd d:\Projects\webcodecstest\go-onnx-inference\cmd\grpc-composite-client
.\grpc-composite-client.exe
```

Client will:
1. Connect to server
2. Health check
3. Test batch sizes: 1, 5, 10
4. Report throughput and timing

## Expected Performance

After warmup:
- **Batch size 1**: ~40-50 FPS
- **Batch size 5**: ~80-120 FPS
- **Batch size 10**: ~100-150 FPS

Timing breakdown per frame:
- Inference: ~21-23ms
- Compositing: ~3-4ms
- PNG encoding: ~10-15ms
- gRPC overhead: ~2-5ms

## Protocol Details

### Request Format

```protobuf
message BatchRequest {
  repeated float visual_frames = 1;    // [batch_size * 6 * 320 * 320]
  repeated float audio_features = 2;   // [32 * 16 * 16] - one window
  int32 batch_size = 3;                // 1-25 frames
  int32 start_frame_idx = 4;           // For crop rect/background lookup
}
```

### Response Format

```protobuf
message BatchResponse {
  repeated bytes composited_frames = 1;  // PNG bytes for each frame
  double inference_time_ms = 2;
  double composite_time_ms = 3;
  double total_time_ms = 4;
  bool success = 5;
  string error = 6;
}
```

## Technical Details

### Compositing Pipeline

For each frame:
1. **Inference**: Generate 320x320 mouth region (3 channels, BGR)
2. **Convert**: ONNX output [0,1] → Go image.RGBA [0,255]
3. **Resize**: Scale 320x320 to crop rectangle size (e.g., 182x182)
4. **Load Background**: Get cached 1280x720 background for this frame
5. **Composite**: Draw resized mouth onto background at crop position
6. **Encode**: Convert to PNG bytes

### Memory Usage

- **ONNX Session**: ~500MB
- **Background Cache** (100 frames): ~350MB
- **Crop Rectangles**: <1MB
- **Per-request buffers**: ~5-10MB

Total: **~900MB** baseline + per-request overhead

## When to Use Each Server

### Use Basic Server (50051) when:
- Client can do compositing (browser with Canvas/WebCodecs)
- Bandwidth is not a constraint
- Want maximum flexibility on client side
- Need raw float32 outputs

### Use Composite Server (50052) when:
- Client has limited compute (mobile, embedded)
- Want to reduce client complexity
- Need PNG frames ready to display
- Server has GPU for faster compositing
- Bandwidth can handle larger responses

## Next Steps

- [ ] Add WebSocket streaming support
- [ ] Implement batch size auto-tuning based on load
- [ ] Add metrics/monitoring endpoints
- [ ] Docker containerization
- [ ] Load balancing for multiple GPUs
- [ ] Client-side frame caching to reduce latency
