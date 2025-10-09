# New Video Frame API - Server-Side Video Delivery

## 🎯 Architecture Change

**Before:** Browser loaded model videos via HTTP from static file server  
**After:** Browser requests video frames through WebSocket → Go Proxy → gRPC Server  

## ✅ Why This Is Better

1. **No Static File Server Required** - Everything goes through the gRPC/WebSocket pipeline
2. **Single Connection** - Audio, video, and lip-sync all through one WebSocket
3. **Better Load Balancing** - Go proxy can route video requests across multiple servers
4. **More Control** - Server controls which models/videos are available
5. **Security** - No need to expose model files via HTTP

## 📡 New gRPC Endpoints

### 1. GetVideoFrame

Request a specific frame from a model's video.

```protobuf
message GetVideoFrameRequest {
    string model_name = 1;      // e.g., "default_model"
    int32 frame_id = 2;          // Frame number (0-based)
    string video_type = 3;       // "full_body", "face_regions", or "masked_regions"
}

message GetVideoFrameResponse {
    bool success = 1;
    bytes frame_data = 2;        // JPEG encoded frame (typically 50-200KB)
    int32 frame_id = 3;
    string video_type = 4;
    optional string error = 5;
}
```

**Example Usage:**
```python
# Request frame 42 from default_model's full_body video
response = await client.GetVideoFrame(
    model_name="default_model",
    frame_id=42,
    video_type="full_body"
)

if response.success:
    # frame_data is JPEG encoded, ready to display
    jpeg_bytes = response.frame_data
```

### 2. GetModelMetadata

Get information about a model including available videos and frame count.

```protobuf
message GetModelMetadataRequest {
    string model_name = 1;
}

message GetModelMetadataResponse {
    bool success = 1;
    string model_name = 2;
    int32 frame_count = 3;                    // Total frames available
    repeated string available_videos = 4;      // ["full_body", "face_regions", "masked_regions"]
    string audio_path = 5;                     // Path to audio file
    repeated float bounds = 6;                 // Default face bounds [x1, y1, x2, y2]
    optional string error = 7;
}
```

**Example Usage:**
```python
# Get metadata for default_model
response = await client.GetModelMetadata(model_name="default_model")

print(f"Model: {response.model_name}")
print(f"Frames: {response.frame_count}")
print(f"Videos: {response.available_videos}")
print(f"Bounds: {response.bounds}")
```

### 3. ListModels (Already Existed)

List all loaded models on the server.

```protobuf
message ListModelsRequest {}

message ListModelsResponse {
    repeated string loaded_models = 1;
    int32 count = 2;
}
```

## 🔄 Updated Client Workflow

### Old Workflow (HTTP Static Files):
```
1. Browser loads HTML from http://localhost:8080/webtest/...
2. Browser requests /models/default_model/video.mp4 via HTTP
3. Static file server serves video file
4. Browser extracts frames client-side
5. Browser connects to WebSocket for lip-sync
```

### New Workflow (gRPC Video API):
```
1. Browser loads HTML (file:// or http://)
2. Browser connects to WebSocket (ws://localhost:8086/ws)
3. Browser requests model list: ListModels
4. Browser requests model metadata: GetModelMetadata("default_model")
5. Browser requests video frames: GetVideoFrame(...) for each frame
6. Browser performs real-time lip-sync inference
```

## 🚀 Performance Considerations

### Video Frame Delivery:
- **JPEG Encoding**: Frames are JPEG encoded (~50-200KB per frame)
- **Parallel Requests**: Browser can request multiple frames in parallel
- **Frame Count**: Typical models have 3,000-4,000 frames
- **Total Transfer**: ~150-800 MB per model video

### Optimization Strategies:

#### 1. Progressive Loading
```javascript
// Load frames in batches
const BATCH_SIZE = 50;
for (let i = 0; i < frameCount; i += BATCH_SIZE) {
    const batch = [];
    for (let j = 0; j < BATCH_SIZE && i + j < frameCount; j++) {
        batch.push(getVideoFrame(i + j));
    }
    await Promise.all(batch);
    updateProgress(i / frameCount * 100);
}
```

#### 2. On-Demand Loading
```javascript
// Only load frames as needed (for playback)
currentFrame = await getVideoFrame(currentFrameIndex);
nextFrame = await getVideoFrame(currentFrameIndex + 1); // Pre-fetch
```

#### 3. Quality Options
```protobuf
// Future enhancement: Add quality parameter
message GetVideoFrameRequest {
    string model_name = 1;
    int32 frame_id = 2;
    string video_type = 3;
    int32 jpeg_quality = 4;  // 1-100 (default: 95)
}
```

## 🛠️ Implementation Status

### ✅ Completed:
- [x] Added `GetVideoFrame` to proto file
- [x] Added `GetModelMetadata` to proto file
- [x] Implemented `get_video_frame()` in `optimized_inference_engine.py`
- [x] Implemented `get_model_metadata()` in `optimized_inference_engine.py`
- [x] Added `GetVideoFrame` handler in `optimized_grpc_server.py`
- [x] Added `GetModelMetadata` handler in `optimized_grpc_server.py`
- [x] Regenerated Python protobuf files
- [x] Regenerated Go protobuf files
- [x] Fixed Go type compatibility issues
- [x] Rebuilt `lipsync-proxy.exe`

### 🔄 TODO:
- [ ] Update browser client to use new API instead of HTTP
- [ ] Add WebSocket message types for video frame requests
- [ ] Implement progressive loading in browser
- [ ] Add caching layer in browser (IndexedDB)
- [ ] Add video frame compression options
- [ ] Test with multiple models
- [ ] Performance benchmarking

## 📝 Example: Updated Browser Client

### Request Model List
```javascript
// Send WebSocket message
ws.send(JSON.stringify({
    type: 'list_models'
}));

// Receive response
{
    type: 'model_list',
    models: ['default_model', 'sanders'],
    count: 2
}
```

### Request Model Metadata
```javascript
ws.send(JSON.stringify({
    type: 'get_metadata',
    model_name: 'default_model'
}));

// Response
{
    type: 'metadata',
    model_name: 'default_model',
    frame_count: 3318,
    available_videos: ['full_body', 'face_regions', 'masked_regions'],
    bounds: [0.1, 0.2, 0.9, 0.8]
}
```

### Request Video Frame
```javascript
ws.send(JSON.stringify({
    type: 'get_video_frame',
    model_name: 'default_model',
    frame_id: 42,
    video_type: 'full_body'
}));

// Response (binary)
// Binary protocol: [type:1][frame_id:4][data_length:4][jpeg_data:N]
```

## 🔧 Testing the New API

### 1. Test GetModelMetadata
```bash
cd D:\Projects\webcodecstest
.\.venv312\Scripts\Activate.ps1
cd minimal_server
python -c "
import asyncio
import optimized_grpc_client as client

async def test():
    c = client.OptimizedGRPCClient('localhost', 50051)
    await c.connect()
    
    # Get metadata
    metadata = await c.get_model_metadata('default_model')
    print(metadata)
    
    await c.close()

asyncio.run(test())
"
```

### 2. Test GetVideoFrame
```bash
python -c "
import asyncio
import optimized_grpc_client as client

async def test():
    c = client.OptimizedGRPCClient('localhost', 50051)
    await c.connect()
    
    # Get frame
    frame = await c.get_video_frame('default_model', 0, 'full_body')
    print(f'Frame size: {len(frame)} bytes')
    
    # Save to file
    with open('frame_0.jpg', 'wb') as f:
        f.write(frame)
    
    await c.close()

asyncio.run(test())
"
```

## 📊 Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| ListModels | <1ms | Instant, reads from memory |
| GetModelMetadata | <1ms | Instant, reads from loaded package |
| GetVideoFrame | 1-3ms | JPEG encoding on server |
| Load Full Video (3K frames) | 10-30s | Depends on network and parallelism |

### Network Bandwidth:
- **Single Frame**: 50-200 KB
- **Full Video (3K frames)**: 150-600 MB
- **Parallel Loading (10 frames/batch)**: 2-6 MB/s transfer
- **Total Time**: 25-100 seconds for full video

## 🎯 Benefits Summary

1. **No HTTP Server** - One less component to manage
2. **Unified Protocol** - Everything through WebSocket/gRPC
3. **Load Balancing** - Go proxy handles routing
4. **Security** - No file system exposure
5. **Flexibility** - Server controls quality, caching, etc.
6. **Scalability** - Can add video caching layer easily

## 🚨 Migration Notes

If you have existing clients using HTTP static files:

1. **Option 1**: Keep HTTP server for backward compatibility
2. **Option 2**: Update clients to use new API
3. **Option 3**: Hybrid - Use HTTP for initial load, gRPC for real-time

**Recommendation**: Update to new API for better architecture!
