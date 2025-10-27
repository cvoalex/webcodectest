# Testing the Complete gRPC Video API System

## 🎯 What We Built

A complete real-time lip-sync system where **ALL data flows through gRPC/WebSocket**:
- ✅ Model listing
- ✅ Model metadata  
- ✅ Video frame delivery
- ✅ Real-time lip-sync inference

**NO HTTP static file server needed!** 🚀

## 🏗️ Architecture

```
Browser (WebSocket)
    ↓
Go Proxy (Load Balancer)
    ↓
gRPC Server (Python)
    ↓
PyTorch Model + Pre-loaded Videos
```

## 📋 Prerequisites

1. **Python Environment**: `.venv312` with PyTorch, gRPC, OpenCV
2. **Go Proxy**: `lipsync-proxy.exe` compiled
3. **Model Package**: `sanders` in `minimal_server/models/sanders/`
4. **Browser**: Chrome/Edge (supports WebCodecs)

## 🚀 Step-by-Step Testing

### Step 1: Start gRPC Server

```powershell
cd D:\Projects\webcodecstest\minimal_server
.\start_grpc_single.bat
```

**Expected Output:**
```
================================================================================
🚀 STARTING ULTRA-OPTIMIZED gRPC SERVER
================================================================================
   ⚡ gRPC with HTTP/2
   ⚡ Protocol Buffers serialization
   ⚡ Pre-loaded videos in RAM
   ⚡ Memory-mapped audio features
   ⚡ Zero I/O overhead
================================================================================

📦 Loading optimized sanders...
🎯 Initializing Optimized Package: sanders
📁 Package directory: models\sanders
📋 Package: sanders v2.0
📊 Frame count: 523
📝 Loading metadata...
  ✅ Crop rectangles: 523 frames cached
🎵 Memory-mapping audio features...
✅ Audio features memory-mapped: (523, 512) (1.02 MB)
🎬 Pre-loading videos into RAM...
✅ full_body_video: 523 frames loaded (1342.56 MB) in 5.23s
✅ crops_328_video: 523 frames loaded (168.92 MB) in 0.87s
✅ masks_video: 523 frames loaded (168.92 MB) in 0.82s
📊 Total video memory: 1680.40 MB
🧠 Loading neural network model...
✅ Model loaded and warmed up on cuda
✅ sanders loaded successfully!
   Frame count: 523
   Initialization time: 12.45s
   Device: cuda
🚀 gRPC server listening on [::]:50051
🎯 Ready for ultra-fast inference!
```

**✅ Success Criteria:**
- Server loads sans model (not default_model)
- All 3 videos loaded into RAM (~1.7GB)
- Model loaded on CUDA
- Server listening on port 50051

### Step 2: Start Go Proxy

Open a **second terminal**:

```powershell
cd D:\Projects\webcodecstest\grpc-websocket-proxy
.\run.bat
```

**Expected Output:**
```
================================================================================
🌉 GRPC-TO-WEBSOCKET PROXY SERVER (Multi-Backend Load Balancer)
================================================================================

📍 Backend configuration:
   [1] localhost:50051

🔌 Connecting to 1 gRPC servers...
   [1/1] Connecting to localhost:50051...
   ✅ localhost:50051 - Status: running, Models: 1
✅ Connected to 1/1 gRPC servers
🚀 WebSocket proxy started on ws://localhost:8086/ws
📁 Static files served from ./static/
🏥 Health check: http://localhost:8086/health
⚖️  Load balancing: Round-robin across 1 backends

Press Ctrl+C to stop
```

**✅ Success Criteria:**
- Connected to gRPC server on 50051
- Health check shows "running" and "Models: 1"
- WebSocket listening on 8086/ws

### Step 3: Open Browser Client

Open in Chrome/Edge:
```
file:///D:/Projects/webcodecstest/webtest/realtime-lipsync-binary.html
```

### Step 4: Connect to Server

1. **Check WebSocket URL**: Should show `ws://localhost:8086/ws`
2. **Click "Connect to Server"**

**Expected in Browser Console:**
```
🔗 Connecting to ws://localhost:8086/ws
✅ WebSocket connected to ws://localhost:8086/ws
📋 Requesting model list...
📋 Available models: sanders
```

**Expected in Go Proxy Terminal:**
```
🌐 Client connected: 127.0.0.1:xxxxx
✅ ListModels: 1 models
```

**✅ Success Criteria:**
- Status shows "Connected" (green)
- Model dropdown populates with "sanders"
- Console shows successful connection

### Step 5: Load Model Video

1. **Select Model**: Should auto-select "sanders"
2. **Click "🎬 Load Video"**

**Expected in Browser Console:**
```
📁 Loading model: sanders
📊 Requesting metadata for sanders...
✅ Metadata loaded: 523 frames, Videos: full_body, face_regions, masked_regions
📥 Loaded 10/523 frames...
📥 Loaded 20/523 frames...
...
📥 Loaded 523/523 frames...
✅ All 523 frames loaded!
```

**Expected in Go Proxy Terminal:**
```
✅ GetModelMetadata: sanders (523 frames)
✅ GetVideoFrame: sanders frame 0 full_body (89234 bytes)
✅ GetVideoFrame: sanders frame 1 full_body (91562 bytes)
...
```

**Expected in gRPC Server Terminal:**
```
(Nothing - video frames served from RAM, no logging)
```

**✅ Success Criteria:**
- Progress bar reaches 100%
- Status shows "523 frames loaded"
- First video frame appears in canvas
- Total transfer: ~50-100 MB (523 frames × ~100KB each)
- Load time: ~10-30 seconds (depends on network)

### Step 6: Test Real-Time Lip-Sync

1. **Click "🎤 Start Audio"**
2. **Allow microphone access**
3. **Speak into microphone**

**Expected in Browser Console:**
```
🎤 Starting audio capture...
✅ Audio capture started
🎵 Audio chunk: 2048 samples
⚡ Sending inference request: frame 0
📨 Binary response received: 85321 bytes
⚡ Binary frame parsed: frame_id=0, processing_time=12ms, image_size=85321
```

**Expected in Go Proxy Terminal:**
```
✅ sanders frame 0 [localhost:50051]: gRPC=12ms total=15ms size=85321 bytes
✅ sanders frame 1 [localhost:50051]: gRPC=11ms total=14ms size=86234 bytes
```

**Expected in gRPC Server Terminal:**
```
(Inference requests are silent by design for performance)
```

**✅ Success Criteria:**
- Lip-sync frames appear in real-time
- Processing time <20ms per frame
- No lag or stuttering
- Metrics show FPS ~30-50

## 🔍 Troubleshooting

### Problem: "No backends available"

**Symptoms:**
- Go proxy shows error connecting to gRPC server
- Browser shows "no backends available"

**Solution:**
1. Check gRPC server is running: `netstat -an | findstr 50051`
2. Restart gRPC server
3. Restart Go proxy

### Problem: "Model video loading fails"

**Symptoms:**
- Progress bar stops
- Console shows timeout errors
- "Failed to load model video"

**Solution:**
1. Check gRPC server loaded model successfully
2. Check Go proxy connected to gRPC server
3. Try smaller batch size (edit `BATCH_SIZE = 10` to `5` in HTML)
4. Check browser console for specific error messages

### Problem: "Frame metadata error"

**Symptoms:**
- gRPC server fails to load model
- Error: "'processed_frames' KeyError"

**Solution:**
- This is fixed in latest code
- Make sure you pulled latest changes
- Check `optimized_inference_engine.py` line 197-200

### Problem: "Video frames not appearing"

**Symptoms:**
- Frames load but don't display
- Canvas is blank

**Solution:**
1. Check browser console for errors
2. Verify JPEG data is valid (check byte size >0)
3. Try reloading the page
4. Check `blobToDataURL()` method works

### Problem: "Lip-sync not working"

**Symptoms:**
- Video loaded but inference doesn't work
- No frames generated

**Solution:**
1. Check microphone permissions
2. Verify audio capture is working (check console for audio chunks)
3. Check gRPC server is responding to inference requests
4. Try restarting all components

## 📊 Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| gRPC Server Start | 10-15s | Loading videos into RAM |
| Go Proxy Start | 1-2s | Connecting to backends |
| Model List | <100ms | From memory |
| Model Metadata | <100ms | From memory |
| Single Video Frame | 10-50ms | JPEG encoding + transfer |
| Full Video Load (523 frames) | 10-30s | Batch loading with progress |
| Inference (per frame) | 10-20ms | GPU inference |
| End-to-End Latency | 50-100ms | Audio → Result |

## 🎯 Success Checklist

- [ ] gRPC server starts and loads sanders model
- [ ] Go proxy connects to gRPC server
- [ ] Browser connects to WebSocket
- [ ] Model list populates ("sanders")
- [ ] Model metadata loads (523 frames)
- [ ] All 523 video frames load successfully
- [ ] First frame displays in canvas
- [ ] Microphone audio capture works
- [ ] Real-time lip-sync inference works
- [ ] Processing time <20ms per frame
- [ ] No lag or stuttering

## 🚀 Next Steps

### If Everything Works:

1. **Test Multi-Model**:
   - Add more models to `minimal_server/models/`
   - Restart gRPC server
   - Switch between models in browser

2. **Test Multi-Backend**:
   - Start multiple gRPC servers on different ports
   - Update Go proxy: `--start-port 50051 --num-servers 4`
   - Verify load balancing works

3. **Performance Optimization**:
   - Implement frame caching in browser (IndexedDB)
   - Add progressive loading (load visible frames first)
   - Add video frame compression options
   - Consider ONNX optimization (15-25% faster)

### If Issues Persist:

1. Check all three terminals for error messages
2. Verify network connectivity (localhost)
3. Check firewall settings
4. Try restarting in sequence: gRPC → Go Proxy → Browser
5. Check browser console (F12) for detailed errors
6. Check `git log` to verify you have latest code

## 📝 Architecture Benefits

**Before (Old Design):**
- ❌ HTTP static file server required
- ❌ Two protocols (HTTP + WebSocket)
- ❌ No load balancing for video
- ❌ Security issues (file system exposure)
- ❌ No server control over quality

**After (New Design):**
- ✅ Single WebSocket connection
- ✅ No HTTP server needed
- ✅ Load balancing for everything
- ✅ Secure (no file exposure)
- ✅ Server controls quality/caching
- ✅ Unified gRPC API
- ✅ Easy to scale (add more servers)

## 💡 Tips

- **Batch Loading**: Adjust `BATCH_SIZE` in browser for optimal loading speed
- **Frame Caching**: Consider caching loaded frames in IndexedDB for faster reloads
- **Progressive Loading**: Load visible frames first, background frames later
- **Quality Control**: Add JPEG quality parameter to reduce transfer size
- **Monitoring**: Watch Go proxy logs to see request patterns
- **Debugging**: Use browser console (F12) to see detailed API calls

**Happy Testing!** 🎉
