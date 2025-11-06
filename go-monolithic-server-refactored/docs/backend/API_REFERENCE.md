# API Reference

> **Complete gRPC API documentation for the Go Monolithic Lip-Sync Server**

This document provides comprehensive documentation for all gRPC endpoints with request/response formats, examples, and use cases.

---

## Table of Contents

- [Overview](#overview)
- [Connection Details](#connection-details)
- [Endpoints](#endpoints)
  - [InferBatchComposite](#inferbatchcomposite) - Main inference + compositing
  - [Health](#health) - Server health check
  - [ListModels](#listmodels) - Get all configured models
  - [LoadModel](#loadmodel) - Load model into memory
  - [UnloadModel](#unloadmodel) - Unload model from memory
  - [GetModelStats](#getmodelstats) - Get usage statistics
- [Error Handling](#error-handling)
- [Common Use Cases](#common-use-cases)

---

## Overview

The Monolithic Server provides a gRPC API for lip-sync video generation. All endpoints use Protocol Buffers for serialization.

**Service Name:** `monolithic.MonolithicService`  
**Protocol:** gRPC with Protocol Buffers  
**Default Port:** `:50053`

---

## Connection Details

### Server Address

```
localhost:50053  (local development)
your-server.com:50053  (production)
```

### Connection Options

**Max Message Size:** 100 MB (configurable in `config.yaml`)

**Timeouts:**
- Default: 60 seconds
- Inference requests: May take longer for large batches

### Example Connection (Go)

```go
import (
    "google.golang.org/grpc"
    pb "go-monolithic-server/proto"
)

conn, err := grpc.Dial(
    "localhost:50053",
    grpc.WithInsecure(),
    grpc.WithDefaultCallOptions(
        grpc.MaxCallRecvMsgSize(100 * 1024 * 1024), // 100 MB
    ),
)
if err != nil {
    log.Fatal(err)
}
defer conn.Close()

client := pb.NewMonolithicServiceClient(conn)
```

---

## Endpoints

### InferBatchComposite

**Purpose:** Perform batch inference + compositing in a single request

**RPC Definition:**
```protobuf
rpc InferBatchComposite(CompositeBatchRequest) returns (CompositeBatchResponse);
```

---

#### Request: CompositeBatchRequest

**Proto Definition:**
```protobuf
message CompositeBatchRequest {
    string model_id = 1;           // Model to use (e.g., "sanders")
    bytes visual_frames = 2;       // Visual frames data
    bytes raw_audio = 3;           // Raw PCM audio (640ms @ 16kHz)
    bytes audio_features = 4;      // [DEPRECATED] Pre-computed features
    int32 batch_size = 5;          // Number of frames (1-32)
    int32 start_frame_idx = 6;     // Background frame start index
    int32 sample_rate = 7;         // Audio sample rate (default: 16000)
}
```

**Field Details:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | ✅ Yes | ID of the model (must be configured in `config.yaml`) |
| `visual_frames` | bytes | ✅ Yes | Float32 array: `[batch_size][6][320][320]` (BGR format, 640x360 → 320x320) |
| `raw_audio` | bytes | ✅ Yes | Float32 PCM audio (16kHz, mono, 640ms = 10,240 samples) |
| `audio_features` | bytes | ❌ No | Deprecated - use `raw_audio` instead |
| `batch_size` | int32 | ✅ Yes | Number of frames to process (1-32, typically 8 or 25) |
| `start_frame_idx` | int32 | ✅ Yes | Starting index for background frames (0-based) |
| `sample_rate` | int32 | ❌ No | Audio sample rate (default: 16000 Hz) |

**Visual Frames Format:**
```
Shape: [batch_size][6][320][320]
Type: float32
Total bytes: batch_size * 6 * 320 * 320 * 4
Example for batch_size=25: 15,360,000 bytes (14.6 MB)

Layout: BGR channels, 640x360 → resized to 320x320
```

**Audio Format:**
```
Shape: [10,240] (640ms at 16kHz)
Type: float32
Total bytes: 10,240 * 4 = 40,960 bytes (~40 KB)

Format: PCM, mono, normalized float32 [-1.0, 1.0]
```

---

#### Response: CompositeBatchResponse

**Proto Definition:**
```protobuf
message CompositeBatchResponse {
    repeated bytes composited_frames = 1;  // JPEG-encoded frames
    float inference_time_ms = 2;            // GPU inference time
    float composite_time_ms = 3;            // Compositing time
    float total_time_ms = 4;                // Total time
    float audio_processing_ms = 5;          // Audio processing time
    bool model_loaded = 6;                  // Model was loaded on-demand
    float model_load_time_ms = 7;           // Model loading time
    bool success = 8;                       // Request succeeded
    string error = 9;                       // Error message (if failed)
    int32 gpu_id = 10;                      // GPU used for inference
}
```

**Field Details:**

| Field | Type | Description |
|-------|------|-------------|
| `composited_frames` | bytes[] | JPEG-encoded frames (quality: 75), array length = batch_size |
| `inference_time_ms` | float | Time spent in ONNX inference (GPU) |
| `composite_time_ms` | float | Time spent compositing frames |
| `total_time_ms` | float | Total request time |
| `audio_processing_ms` | float | Time spent processing audio |
| `model_loaded` | bool | True if model was loaded during this request |
| `model_load_time_ms` | float | Time to load model (if loaded) |
| `success` | bool | True if request succeeded |
| `error` | string | Error message (empty if success) |
| `gpu_id` | int32 | GPU ID used for inference |

---

#### Example: Go Client

```go
package main

import (
    "context"
    "encoding/binary"
    "io/ioutil"
    "log"
    
    "google.golang.org/grpc"
    pb "go-monolithic-server/proto"
)

func main() {
    // Connect to server
    conn, err := grpc.Dial("localhost:50053", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    
    client := pb.NewMonolithicServiceClient(conn)
    
    // Prepare visual frames (example: random data)
    batchSize := int32(8)
    visualSize := batchSize * 6 * 320 * 320
    visualFrames := make([]byte, visualSize*4) // float32 = 4 bytes
    
    // Prepare audio (example: random data)
    audioSize := 10240 // 640ms at 16kHz
    rawAudio := make([]byte, audioSize*4)
    
    // Send request
    req := &pb.CompositeBatchRequest{
        ModelId:       "sanders",
        VisualFrames:  visualFrames,
        RawAudio:      rawAudio,
        BatchSize:     batchSize,
        StartFrameIdx: 0,
        SampleRate:    16000,
    }
    
    resp, err := client.InferBatchComposite(context.Background(), req)
    if err != nil {
        log.Fatalf("Request failed: %v", err)
    }
    
    if !resp.Success {
        log.Fatalf("Server error: %s", resp.Error)
    }
    
    // Save composited frames
    for i, frameBytes := range resp.CompositedFrames {
        filename := fmt.Sprintf("output_frame_%04d.jpg", i)
        err := ioutil.WriteFile(filename, frameBytes, 0644)
        if err != nil {
            log.Printf("Failed to save frame %d: %v", i, err)
        }
    }
    
    log.Printf("✅ Success! Processed %d frames in %.2f ms", 
        len(resp.CompositedFrames), resp.TotalTimeMs)
    log.Printf("   Inference: %.2f ms", resp.InferenceTimeMs)
    log.Printf("   Compositing: %.2f ms", resp.CompositeTimeMs)
    log.Printf("   Audio: %.2f ms", resp.AudioProcessingMs)
}
```

---

#### Example: grpcurl

**Health Check:**
```bash
grpcurl -plaintext localhost:50053 monolithic.MonolithicService/Health
```

**List Models:**
```bash
grpcurl -plaintext localhost:50053 monolithic.MonolithicService/ListModels
```

**Load Model:**
```bash
grpcurl -plaintext -d '{"model_id": "sanders"}' \
  localhost:50053 monolithic.MonolithicService/LoadModel
```

**Note:** `InferBatchComposite` requires binary data, not easily testable with grpcurl. Use a proper client instead.

---

### Health

**Purpose:** Check if server is healthy and get status

**RPC Definition:**
```protobuf
rpc Health(HealthRequest) returns (HealthResponse);
```

---

#### Request: HealthRequest

**Proto Definition:**
```protobuf
message HealthRequest {
    // Empty
}
```

No parameters required.

---

#### Response: HealthResponse

**Proto Definition:**
```protobuf
message HealthResponse {
    bool healthy = 1;
    int32 loaded_models = 2;
    int32 max_models = 3;
    repeated int32 gpu_ids = 4;
    repeated float gpu_memory_used_gb = 5;
}
```

**Field Details:**

| Field | Type | Description |
|-------|------|-------------|
| `healthy` | bool | True if server is healthy |
| `loaded_models` | int32 | Number of currently loaded models |
| `max_models` | int32 | Maximum models capacity |
| `gpu_ids` | int32[] | List of available GPU IDs |
| `gpu_memory_used_gb` | float[] | Memory used per GPU (GB) |

---

#### Example: grpcurl

```bash
grpcurl -plaintext localhost:50053 monolithic.MonolithicService/Health
```

**Response:**
```json
{
  "healthy": true,
  "loaded_models": 1,
  "max_models": 40,
  "gpu_ids": [0],
  "gpu_memory_used_gb": [2.5]
}
```

---

#### Example: Go Client

```go
resp, err := client.Health(context.Background(), &pb.HealthRequest{})
if err != nil {
    log.Fatal(err)
}

if resp.Healthy {
    log.Printf("✅ Server healthy: %d/%d models loaded", 
        resp.LoadedModels, resp.MaxModels)
} else {
    log.Printf("❌ Server unhealthy")
}
```

---

#### Use Cases

1. **Startup Validation** - Verify server is ready before sending requests
2. **Monitoring** - Periodic health checks in production
3. **Load Balancing** - Check capacity before routing requests
4. **Debugging** - Verify GPU availability and memory usage

---

### ListModels

**Purpose:** Get all configured models and their status

**RPC Definition:**
```protobuf
rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
```

---

#### Request: ListModelsRequest

**Proto Definition:**
```protobuf
message ListModelsRequest {
    // Empty
}
```

No parameters required.

---

#### Response: ListModelsResponse

**Proto Definition:**
```protobuf
message ListModelsResponse {
    repeated ModelInfo models = 1;
}

message ModelInfo {
    string model_id = 1;
    bool loaded = 2;
    int32 gpu_id = 3;
    int64 memory_mb = 4;
    int32 request_count = 5;
    float avg_inference_ms = 6;
}
```

**ModelInfo Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Model identifier (e.g., "sanders") |
| `loaded` | bool | True if model is currently loaded in GPU memory |
| `gpu_id` | int32 | GPU ID where model is loaded (-1 if not loaded) |
| `memory_mb` | int64 | GPU memory used by model (MB) |
| `request_count` | int32 | Total inference requests served |
| `avg_inference_ms` | float | Average inference time (ms) |

---

#### Example: grpcurl

```bash
grpcurl -plaintext localhost:50053 monolithic.MonolithicService/ListModels
```

**Response:**
```json
{
  "models": [
    {
      "model_id": "sanders",
      "loaded": true,
      "gpu_id": 0,
      "memory_mb": 2048,
      "request_count": 157,
      "avg_inference_ms": 125.3
    },
    {
      "model_id": "bob",
      "loaded": false,
      "gpu_id": -1,
      "memory_mb": 0,
      "request_count": 0,
      "avg_inference_ms": 0
    }
  ]
}
```

---

#### Example: Go Client

```go
resp, err := client.ListModels(context.Background(), &pb.ListModelsRequest{})
if err != nil {
    log.Fatal(err)
}

for _, model := range resp.Models {
    status := "Not Loaded"
    if model.Loaded {
        status = fmt.Sprintf("GPU %d, %d MB", model.GpuId, model.MemoryMb)
    }
    
    log.Printf("Model: %s - %s", model.ModelId, status)
    if model.RequestCount > 0 {
        log.Printf("  Requests: %d, Avg time: %.2f ms", 
            model.RequestCount, model.AvgInferenceMs)
    }
}
```

---

#### Use Cases

1. **Discovery** - Find available models
2. **Monitoring** - Check which models are loaded
3. **Capacity Planning** - See memory usage per model
4. **Performance Analysis** - Compare inference times across models

---

### LoadModel

**Purpose:** Explicitly load a model into GPU memory

**RPC Definition:**
```protobuf
rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
```

---

#### Request: LoadModelRequest

**Proto Definition:**
```protobuf
message LoadModelRequest {
    string model_id = 1;
    int32 preferred_gpu = 2;  // -1 for auto
}
```

**Field Details:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | ✅ Yes | Model ID to load (must be in `config.yaml`) |
| `preferred_gpu` | int32 | ❌ No | GPU ID preference (-1 for automatic assignment) |

---

#### Response: LoadModelResponse

**Proto Definition:**
```protobuf
message LoadModelResponse {
    bool success = 1;
    string error = 2;
    int32 gpu_id = 3;
    float load_time_ms = 4;
}
```

**Field Details:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | True if model loaded successfully |
| `error` | string | Error message (empty if success) |
| `gpu_id` | int32 | GPU where model was loaded |
| `load_time_ms` | float | Time taken to load model (ms) |

---

#### Example: grpcurl

```bash
grpcurl -plaintext -d '{"model_id": "sanders", "preferred_gpu": 0}' \
  localhost:50053 monolithic.MonolithicService/LoadModel
```

**Response:**
```json
{
  "success": true,
  "error": "",
  "gpu_id": 0,
  "load_time_ms": 1523.45
}
```

---

#### Example: Go Client

```go
req := &pb.LoadModelRequest{
    ModelId:      "sanders",
    PreferredGpu: 0,  // Use GPU 0, or -1 for auto
}

resp, err := client.LoadModel(context.Background(), req)
if err != nil {
    log.Fatalf("RPC failed: %v", err)
}

if resp.Success {
    log.Printf("✅ Model loaded on GPU %d in %.2f ms", 
        resp.GpuId, resp.LoadTimeMs)
} else {
    log.Printf("❌ Failed to load model: %s", resp.Error)
}
```

---

#### Use Cases

1. **Preloading** - Load frequently-used models at startup
2. **Cache Warming** - Prepare models before high-traffic periods
3. **GPU Assignment** - Control which GPU hosts which model
4. **Testing** - Verify model can be loaded successfully

---

### UnloadModel

**Purpose:** Remove a model from GPU memory

**RPC Definition:**
```protobuf
rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
```

---

#### Request: UnloadModelRequest

**Proto Definition:**
```protobuf
message UnloadModelRequest {
    string model_id = 1;
}
```

**Field Details:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | ✅ Yes | Model ID to unload |

---

#### Response: UnloadModelResponse

**Proto Definition:**
```protobuf
message UnloadModelResponse {
    bool success = 1;
    string error = 2;
}
```

**Field Details:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | True if unloaded successfully |
| `error` | string | Error message (empty if success) |

---

#### Example: grpcurl

```bash
grpcurl -plaintext -d '{"model_id": "sanders"}' \
  localhost:50053 monolithic.MonolithicService/UnloadModel
```

**Response:**
```json
{
  "success": true,
  "error": ""
}
```

---

#### Example: Go Client

```go
req := &pb.UnloadModelRequest{
    ModelId: "sanders",
}

resp, err := client.UnloadModel(context.Background(), req)
if err != nil {
    log.Fatalf("RPC failed: %v", err)
}

if resp.Success {
    log.Printf("✅ Model unloaded successfully")
} else {
    log.Printf("❌ Failed to unload: %s", resp.Error)
}
```

---

#### Use Cases

1. **Memory Management** - Free GPU memory for other models
2. **Maintenance** - Unload before updating model files
3. **Testing** - Test eviction policies
4. **Resource Control** - Manually manage GPU capacity

---

### GetModelStats

**Purpose:** Get detailed usage statistics for all models

**RPC Definition:**
```protobuf
rpc GetModelStats(GetModelStatsRequest) returns (GetModelStatsResponse);
```

---

#### Request: GetModelStatsRequest

**Proto Definition:**
```protobuf
message GetModelStatsRequest {
    // Empty
}
```

No parameters required.

---

#### Response: GetModelStatsResponse

**Proto Definition:**
```protobuf
message GetModelStatsResponse {
    repeated ModelStats models = 1;
}

message ModelStats {
    string model_id = 1;
    bool loaded = 2;
    int32 gpu_id = 3;
    int64 memory_mb = 4;
    int32 request_count = 5;
    int64 total_inference_ms = 6;
    float avg_inference_ms = 7;
    int64 last_used_timestamp = 8;
}
```

**ModelStats Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | Model identifier |
| `loaded` | bool | Currently loaded in memory |
| `gpu_id` | int32 | GPU ID (-1 if not loaded) |
| `memory_mb` | int64 | GPU memory used (MB) |
| `request_count` | int32 | Total requests served |
| `total_inference_ms` | int64 | Total inference time (ms) |
| `avg_inference_ms` | float | Average inference time (ms) |
| `last_used_timestamp` | int64 | Unix timestamp of last use |

---

#### Example: grpcurl

```bash
grpcurl -plaintext localhost:50053 monolithic.MonolithicService/GetModelStats
```

**Response:**
```json
{
  "models": [
    {
      "model_id": "sanders",
      "loaded": true,
      "gpu_id": 0,
      "memory_mb": 2048,
      "request_count": 325,
      "total_inference_ms": 40625,
      "avg_inference_ms": 125.0,
      "last_used_timestamp": 1699305600
    }
  ]
}
```

---

#### Example: Go Client

```go
resp, err := client.GetModelStats(context.Background(), &pb.GetModelStatsRequest{})
if err != nil {
    log.Fatal(err)
}

for _, stats := range resp.Models {
    log.Printf("Model: %s", stats.ModelId)
    log.Printf("  Loaded: %v (GPU %d)", stats.Loaded, stats.GpuId)
    log.Printf("  Requests: %d", stats.RequestCount)
    log.Printf("  Avg Inference: %.2f ms", stats.AvgInferenceMs)
    log.Printf("  Memory: %d MB", stats.MemoryMb)
}
```

---

#### Use Cases

1. **Monitoring** - Track model usage patterns
2. **Performance Analysis** - Identify slow models
3. **Eviction Decisions** - See which models are least-used
4. **Capacity Planning** - Understand memory requirements

---

## Error Handling

### Error Response Format

Errors are returned in two ways:

**1. Application-Level Errors** (returned in response):
```json
{
  "success": false,
  "error": "Model 'unknown' not found in configuration"
}
```

**2. gRPC-Level Errors** (connection/protocol issues):
```
Error: rpc error: code = Unavailable desc = connection refused
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `model_id is required` | Empty model_id field | Provide valid model ID |
| `Model 'X' not found` | Model not in config.yaml | Add model to configuration |
| `batch_size must be 1-32` | Invalid batch size | Use valid range (1-32) |
| `Invalid visual frames size` | Wrong input size | Check array dimensions |
| `ONNX inference failed` | Model execution error | Check model file, GPU memory |
| `Background frame X not found` | Frame not cached | Check background directory |

### Error Handling Best Practices

**1. Check Application Success Flag:**
```go
if !resp.Success {
    return fmt.Errorf("server error: %s", resp.Error)
}
```

**2. Handle gRPC Errors:**
```go
resp, err := client.InferBatchComposite(ctx, req)
if err != nil {
    return fmt.Errorf("grpc error: %w", err)
}
```

**3. Implement Retries:**
```go
var resp *pb.CompositeBatchResponse
var err error

for i := 0; i < 3; i++ {
    resp, err = client.InferBatchComposite(ctx, req)
    if err == nil && resp.Success {
        break
    }
    time.Sleep(time.Second * time.Duration(i+1))
}
```

---

## Common Use Cases

### Use Case 1: Batch Inference Pipeline

**Scenario:** Process video frames in batches of 25

```go
batchSize := 25
totalFrames := 500
numBatches := totalFrames / batchSize

for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
    startFrame := batchIdx * batchSize
    
    req := &pb.CompositeBatchRequest{
        ModelId:       "sanders",
        VisualFrames:  getVisualFrames(batchIdx),
        RawAudio:      getAudioSegment(batchIdx),
        BatchSize:     int32(batchSize),
        StartFrameIdx: int32(startFrame),
    }
    
    resp, err := client.InferBatchComposite(ctx, req)
    if err != nil || !resp.Success {
        log.Printf("Batch %d failed: %v", batchIdx, err)
        continue
    }
    
    saveFrames(resp.CompositedFrames, startFrame)
}
```

---

### Use Case 2: Multi-Model Server

**Scenario:** Manage multiple models with different characteristics

```go
// Load models on startup
models := []string{"sanders", "bob", "alice"}

for _, modelID := range models {
    loadResp, _ := client.LoadModel(ctx, &pb.LoadModelRequest{
        ModelId:      modelID,
        PreferredGpu: -1,  // Auto-assign
    })
    
    if loadResp.Success {
        log.Printf("Loaded %s on GPU %d", modelID, loadResp.GpuId)
    }
}

// Process requests dynamically
for _, userRequest := range userRequests {
    req := &pb.CompositeBatchRequest{
        ModelId:      userRequest.PreferredModel,
        VisualFrames: userRequest.Frames,
        RawAudio:     userRequest.Audio,
        BatchSize:    userRequest.BatchSize,
    }
    
    resp, _ := client.InferBatchComposite(ctx, req)
    // ... handle response
}
```

---

### Use Case 3: Health Monitoring Loop

**Scenario:** Monitor server health continuously

```go
ticker := time.NewTicker(30 * time.Second)
defer ticker.Stop()

for range ticker.C {
    health, err := client.Health(ctx, &pb.HealthRequest{})
    if err != nil {
        log.Printf("❌ Health check failed: %v", err)
        continue
    }
    
    if !health.Healthy {
        log.Printf("⚠️ Server unhealthy!")
        alertOps()
        continue
    }
    
    stats, _ := client.GetModelStats(ctx, &pb.GetModelStatsRequest{})
    for _, model := range stats.Models {
        if model.RequestCount > 1000 && model.AvgInferenceMs > 200 {
            log.Printf("⚠️ Model %s degraded: %.2f ms", 
                model.ModelId, model.AvgInferenceMs)
        }
    }
}
```

---

### Use Case 4: Dynamic Model Loading

**Scenario:** Load models on-demand based on traffic

```go
// Check if model is loaded
models, _ := client.ListModels(ctx, &pb.ListModelsRequest{})
modelLoaded := false

for _, m := range models.Models {
    if m.ModelId == "sanders" && m.Loaded {
        modelLoaded = true
        break
    }
}

// Load if not loaded
if !modelLoaded {
    loadResp, _ := client.LoadModel(ctx, &pb.LoadModelRequest{
        ModelId: "sanders",
    })
    
    if !loadResp.Success {
        return fmt.Errorf("failed to load model: %s", loadResp.Error)
    }
    
    log.Printf("Model loaded in %.2f ms", loadResp.LoadTimeMs)
}

// Now process request
resp, _ := client.InferBatchComposite(ctx, req)
```

---

## Related Documentation

- **[Architecture Overview](ARCHITECTURE.md)** - System design and components
- **[Development Guide](development/DEVELOPMENT_GUIDE.md)** - Setup and getting started
- **[Testing Guide](development/TESTING.md)** - How to test the API
- **[Common Gotchas](development/GOTCHAS.md)** - API pitfalls to avoid

---

**Last Updated:** November 6, 2025  
**API Version:** 1.0.0  
**Protocol:** gRPC with Protocol Buffers
