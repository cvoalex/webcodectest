# Answer: Why Does the Client Load Videos via HTTP?

## 🤔 Your Question

> "So why do we need that path at all? We shouldn't. The client is asking the server for data.  
> The server needs to know what model to use. Shouldn't the server provide some API of models to select?  
> Also we have a gRPC Go proxy server that should be able to do all this."

## ✅ You Were 100% Correct!

**The old design was flawed.** The browser shouldn't load model videos via HTTP - everything should go through the gRPC/WebSocket pipeline.

## 🔧 What I Fixed

### Before (Bad Design):
```
Browser → HTTP Server → Static Files (models/default_model/video.mp4)
Browser → WebSocket → Go Proxy → gRPC Server (lip-sync only)
```

**Problems:**
- Needed separate HTTP static file server
- Two different protocols (HTTP + WebSocket)
- Security issues (exposing file system)
- No load balancing for video delivery
- No server control over video quality/caching

### After (Your Design):
```
Browser → WebSocket → Go Proxy → gRPC Server (models + video + lip-sync)
```

**Benefits:**
- Single WebSocket connection for everything
- No HTTP server needed
- Better security
- Load balancing across all servers
- Server controls video quality/caching

## 📡 New gRPC Endpoints

### 1. ListModels
```protobuf
// Get list of available models
rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
```

### 2. GetModelMetadata
```protobuf
// Get model info (frame count, available videos, bounds)
rpc GetModelMetadata(GetModelMetadataRequest) returns (GetModelMetadataResponse);
```

### 3. GetVideoFrame
```protobuf
// Get specific video frame as JPEG
rpc GetVideoFrame(GetVideoFrameRequest) returns (GetVideoFrameResponse);
```

## 🎯 New Workflow

```javascript
// 1. Connect to WebSocket
ws = new WebSocket('ws://localhost:8086/ws');

// 2. List available models
ws.send({ type: 'list_models' });
// Response: { models: ['default_model', 'sanders'], count: 2 }

// 3. Get model metadata
ws.send({ 
    type: 'get_metadata', 
    model_name: 'default_model' 
});
// Response: { 
//   frame_count: 3318, 
//   available_videos: ['full_body', 'face_regions', 'masked_regions'],
//   bounds: [0.1, 0.2, 0.9, 0.8]
// }

// 4. Load video frames
for (let i = 0; i < frame_count; i++) {
    ws.send({ 
        type: 'get_video_frame',
        model_name: 'default_model',
        frame_id: i,
        video_type: 'full_body'
    });
    // Response: Binary JPEG data
}

// 5. Do real-time lip-sync
ws.send({ 
    type: 'inference',
    model_name: 'default_model',
    frame_id: currentFrame
});
// Response: Lip-synced frame
```

## 📊 What's Implemented

✅ **Server-Side (Python):**
- `get_video_frame()` - Returns JPEG encoded frame
- `get_model_metadata()` - Returns model info
- Added to `optimized_inference_engine.py`
- Added handlers to `optimized_grpc_server.py`

✅ **Protocol (Protobuf):**
- Added `GetVideoFrameRequest/Response`
- Added `GetModelMetadataRequest/Response`
- Regenerated Python protobuf files
- Regenerated Go protobuf files

✅ **Proxy (Go):**
- Updated to new protobuf definitions
- Fixed type compatibility
- Rebuilt `lipsync-proxy.exe`

## 🔄 What's Next

The server is ready! Now we need to update the browser client to use the new API:

1. **Remove HTTP video loading code**
2. **Add WebSocket video frame requests**
3. **Implement progressive loading**
4. **Add caching (IndexedDB)**

See `NEW_VIDEO_API.md` for complete documentation!

## 💡 Summary

You spotted a major architectural flaw. The **Go proxy + gRPC server** should handle everything, including:
- ✅ Model listing
- ✅ Model metadata
- ✅ Video frame delivery
- ✅ Real-time lip-sync inference

No need for a separate HTTP static file server! Everything through one WebSocket connection. 🚀
