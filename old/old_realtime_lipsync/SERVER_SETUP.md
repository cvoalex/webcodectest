# Real-time Lip Sync Server Setup Guide

## 🚀 Complete Server Setup & API Reference

This guide covers all server components, model loading, API endpoints, and troubleshooting for the real-time lip sync system.

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [Server Components](#server-components)
4. [Setup Instructions](#setup-instructions)
5. [Model Management](#model-management)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)
8. [Performance Tuning](#performance-tuning)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser Client                           │
│  Real-time Lip Sync Console + Model Video Manager              │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP/WebSocket
┌─────────────────────▼───────────────────────────────────────────┐
│                    Go Token Server                              │
│  Port: 3000                                                     │
│  • Static file serving                                          │
│  • OpenAI token generation                                      │
│  • Model video serving                                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │ WebSocket
┌─────────────────────▼───────────────────────────────────────────┐
│                Python Frame Generator                           │
│  Port: 8080 (WebSocket)                                         │
│  • Audio processing                                             │
│  • WebSocket bridge                                             │
│  • Buffer management                                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ gRPC
┌─────────────────────▼───────────────────────────────────────────┐
│                  gRPC Lip Sync Service                          │
│  Port: 50051                                                    │
│  • Model inference                                              │
│  • Batch processing                                             │
│  • GPU acceleration                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

### System Requirements
- **OS**: Windows/Linux/macOS
- **RAM**: 16GB+ recommended (for model loading)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 or better)
- **Storage**: 10GB+ free space for models

### Software Dependencies
- **Python**: 3.8+ with pip
- **Go**: 1.21+ (for token server)
- **Node.js**: 18+ (optional, for Node.js server alternative)
- **CUDA**: 11.8+ (for GPU acceleration)

### Required Python Packages
```bash
pip install grpcio grpcio-tools numpy websockets psutil torch torchvision torchaudio
```

### Required Go Modules
```bash
go mod tidy  # Auto-installs dependencies from go.mod
```

---

## 🖥️ Server Components

### 1. **gRPC Lip Sync Service** (Port 50051)
- **Purpose**: High-performance lip sync inference
- **Input**: Audio data + model selection
- **Output**: Mouth regions (320x320) + bounds data
- **Technology**: Python + PyTorch + gRPC

### 2. **Python Frame Generator** (Port 8080)
- **Purpose**: WebSocket bridge between browser and gRPC
- **Features**: Audio buffering, batch processing, model switching
- **Protocol**: WebSocket JSON messages

### 3. **Go Token Server** (Port 3000)
- **Purpose**: OpenAI token generation + static file serving
- **Features**: Model video serving, CORS handling, file streaming
- **Alternative**: Node.js server (same functionality)

---

## 🚀 Setup Instructions

### Step 1: Environment Configuration

Create `.env` file in `realtime_lipsync/` directory:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Audio Settings
AUDIO_SAMPLE_RATE=24000
AUDIO_CHUNK_DURATION_MS=40
AUDIO_BUFFER_SIZE=3000
FRAMES_FOR_INFERENCE=16
INFERENCE_BATCH_SIZE=5
LOOKAHEAD_FRAMES=7

# Server Ports
PORT=3000
GRPC_SERVER_PORT=50051
WEBSOCKET_PORT=8080

# Debug Options (optional)
DEBUG_AUDIO_PROCESSING=false
DEBUG_FRAME_GENERATION=false
DEBUG_WEBSOCKET_MESSAGES=false
```

### Step 2: Start gRPC Lip Sync Service

```bash
# Navigate to your gRPC service directory
cd fast_service

# Start the gRPC server
python grpc_server.py

# Expected output:
# 🚀 gRPC Lip Sync Service starting on port 50051
# 📦 Loading models...
# ✅ Service ready for inference
```

### Step 3: Start Python Frame Generator

```bash
# Navigate to realtime_lipsync directory  
cd realtime_lipsync

# Start the frame generator
python frame_generator.py

# Expected output:
# 🚀 Real-time Lip Sync Frame Generator initialized
# ✅ Connected to gRPC service on localhost:50051
# 🌐 WebSocket server started on ws://localhost:8080
# 📡 Connect your browser to ws://localhost:8080
```

### Step 4: Start Go Token Server

```bash
# Navigate to go-token-server directory
cd go-token-server

# Install dependencies (first time only)
go mod tidy

# Start the server
go run main.go

# Expected output:
# 🚀 Real-time Lip Sync Token Server starting on port 3000
# 📁 Serving static files from: ../realtime_lipsync
# 🔑 Token endpoint: http://localhost:3000/token
# 🌐 Browser access: http://localhost:3000/index.html
```

### Alternative: Node.js Server (instead of Go)

```bash
# Navigate to realtime_lipsync directory
cd realtime_lipsync

# Install Node.js dependencies (first time only)
npm install

# Start the Node.js server
node server.js

# Expected output:
# 🚀 Real-time Lip Sync Server starting on port 3000
# 📁 Serving static files from current directory
# 🔑 Token endpoint: http://localhost:3000/token
```

### Step 5: Access the Console

Open browser and navigate to:
```
http://localhost:3000/index.html
```

---

## 🎯 Model Management

### Model Loading Process

The gRPC service automatically loads models on startup. Models should be placed in:
```
fast_service/models/
├── test_optimized_package_fixed_1/
├── test_optimized_package_fixed_2/
├── test_optimized_package_fixed_3/
├── test_optimized_package_fixed_4/
└── test_optimized_package_fixed_5/
```

### Model Video Setup

For full-body compositing, place model videos in:
```
realtime_lipsync/model_videos/
├── test_optimized_package_fixed_1.mp4
├── test_optimized_package_fixed_2.mp4
├── test_optimized_package_fixed_3.mp4
├── test_optimized_package_fixed_4.mp4
└── test_optimized_package_fixed_5.mp4
```

**Video Requirements:**
- **Format**: MP4 (H.264)
- **Resolution**: Any (auto-resized)
- **Duration**: 10-30 seconds recommended
- **Content**: Clear face visibility, neutral/talking expressions

### Loading New Models

1. **Add model files** to `fast_service/models/new_model_name/`
2. **Add model video** to `realtime_lipsync/model_videos/new_model_name.mp4`
3. **Restart gRPC service** to load new model
4. **Model will appear** in browser dropdown automatically

---

## 📡 API Reference

### Go Token Server APIs

#### `POST /token`
Generate OpenAI ephemeral token for WebRTC connection.

**Request:**
```bash
curl -X POST http://localhost:3000/token \
  -H "Content-Type: application/json"
```

**Response:**
```json
{
  "client_secret": {
    "value": "eph_token_abc123...",
    "expires_at": "2025-09-06T15:30:00Z"
  }
}
```

#### `GET /api/model-video/{model_name}`
Stream model video for client-side compositing.

**Request:**
```bash
curl http://localhost:3000/api/model-video/test_optimized_package_fixed_1
```

**Response:**
- **Content-Type**: `video/mp4`
- **Body**: Video file stream
- **Headers**: `Content-Length`, `Accept-Ranges: bytes`

#### `GET /` (Static Files)
Serves all static files from `realtime_lipsync/` directory.

**Examples:**
- `GET /index.html` → Main console interface
- `GET /realtime_lipsync_app.js` → Application JavaScript
- `GET /model_video_manager.js` → Video management module

### Python Frame Generator WebSocket API

Connect to: `ws://localhost:8080`

#### Message Types

##### `audio_chunk` (Client → Server)
Send audio data for processing.

**Format:**
```json
{
  "type": "audio_chunk",
  "audio_data": "base64_encoded_pcm16_data"
}
```

##### `get_frame` (Client → Server)
Request current synchronized frame.

**Format:**
```json
{
  "type": "get_frame"
}
```

**Response:**
```json
{
  "type": "current_frame",
  "frame_data": "base64_encoded_jpeg",
  "bounds": [xmin, ymin, xmax, ymax],
  "timestamp": 1725631800.123,
  "model_name": "test_optimized_package_fixed_1",
  "prediction_shape": "320x320x3"
}
```

##### `set_model` (Client → Server)
Change active lip sync model.

**Format:**
```json
{
  "type": "set_model",
  "model_name": "test_optimized_package_fixed_2"
}
```

##### `get_stats` (Client → Server)
Request buffer and performance statistics.

**Format:**
```json
{
  "type": "get_stats"
}
```

**Response:**
```json
{
  "type": "stats",
  "audio_buffer_fill": 12,
  "frame_buffer_fill": 8,
  "can_generate": true,
  "current_model": "test_optimized_package_fixed_1"
}
```

##### `frames_generated` (Server → Client)
Notification of new frames available.

**Format:**
```json
{
  "type": "frames_generated",
  "count": 5,
  "buffer_fill": 15
}
```

### gRPC Lip Sync Service API

#### `GenerateBatchInference`
Generate multiple lip sync frames from audio.

**Request:**
```protobuf
message BatchInferenceRequest {
    string model_name = 1;
    repeated int32 frame_ids = 2;
    optional string audio_override = 3;  // base64 audio data
}
```

**Response:**
```protobuf
message BatchInferenceResponse {
    repeated InferenceResponse responses = 1;
    int32 total_processing_time_ms = 2;
}

message InferenceResponse {
    bool success = 1;
    bytes prediction_data = 2;        // JPEG encoded mouth region
    repeated float bounds = 3;        // [xmin, ymin, xmax, ymax]
    int32 processing_time_ms = 4;
    string model_name = 5;
    int32 frame_id = 6;
    string prediction_shape = 8;      // "320x320x3"
    optional string error = 9;
}
```

#### `LoadModel`
Load a new model into the service.

**Request:**
```protobuf
message LoadModelRequest {
    string model_name = 1;
    string package_path = 2;
    optional string audio_override = 3;
}
```

**Response:**
```protobuf
message LoadModelResponse {
    bool success = 1;
    string model_name = 2;
    string message = 3;
    int32 initialization_time_ms = 4;
    optional string error = 5;
}
```

---

## 🔧 Troubleshooting

### Common Startup Issues

#### 1. **gRPC Service Won't Start**
```bash
# Error: Port 50051 already in use
netstat -an | grep 50051
# Kill existing process and restart

# Error: CUDA out of memory
# Reduce number of models or use smaller models
```

#### 2. **Frame Generator Connection Failed**
```bash
# Error: Failed to connect to gRPC service
# Check that gRPC service is running on port 50051
curl localhost:50051  # Should get gRPC response

# Error: WebSocket port 8080 in use
netstat -an | grep 8080
# Change WEBSOCKET_PORT in .env file
```

#### 3. **Token Server Issues**
```bash
# Error: OpenAI API key not configured
echo $OPENAI_API_KEY  # Check environment variable

# Error: Port 3000 in use
netstat -an | grep 3000
# Change PORT in .env file or kill existing process
```

#### 4. **Model Video Not Found**
```bash
# Error: Model video not found for: model_name
# Check file exists:
ls realtime_lipsync/model_videos/model_name.mp4

# Check server logs for searched paths:
# Go server will log all attempted paths
```

### Performance Issues

#### **High Memory Usage**
```bash
# Monitor GPU memory
nvidia-smi

# Monitor system memory
htop

# Reduce cached models in browser:
# ModelVideoManager.maxCachedModels = 3
```

#### **Slow Frame Generation**
```bash
# Check gRPC service performance
tail -f grpc_server.log

# Monitor frame generation timing
tail -f frame_generator.log

# Reduce batch size if needed:
# INFERENCE_BATCH_SIZE=3 (in .env)
```

#### **WebSocket Connection Drops**
```bash
# Check Python frame generator logs
python frame_generator.py  # Watch for connection errors

# Verify WebSocket port not blocked
telnet localhost 8080

# Check browser network tab for WebSocket status
```

### Debugging Commands

#### **Test gRPC Service**
```bash
# Test direct gRPC connection
python -c "
import grpc
channel = grpc.insecure_channel('localhost:50051')
print('gRPC service accessible')
"
```

#### **Test WebSocket**
```bash
# Test WebSocket connection
python -c "
import asyncio
import websockets

async def test():
    async with websockets.connect('ws://localhost:8080') as ws:
        print('WebSocket connected')

asyncio.run(test())
"
```

#### **Test Token Generation**
```bash
# Test OpenAI token endpoint
curl -X POST http://localhost:3000/token \
  -H "Content-Type: application/json" | jq
```

---

## ⚡ Performance Tuning

### GPU Optimization

#### **Multi-GPU Setup**
```python
# In grpc_server.py - distribute models across GPUs
device_map = {
    'model_1': 'cuda:0',
    'model_2': 'cuda:1', 
    'model_3': 'cuda:0',
    'model_4': 'cuda:1'
}
```

#### **Memory Management**
```python
# Reduce precision for more models
torch.backends.cudnn.benchmark = True
model.half()  # Use FP16 instead of FP32
```

### Network Optimization

#### **Audio Buffer Tuning**
```env
# Lower latency (more CPU usage)
AUDIO_CHUNK_DURATION_MS=20
FRAMES_FOR_INFERENCE=8

# Higher throughput (more buffer)  
AUDIO_CHUNK_DURATION_MS=60
FRAMES_FOR_INFERENCE=24
```

#### **WebSocket Settings**
```python
# In frame_generator.py
# Increase buffer sizes for high throughput
websocket_server = await websockets.serve(
    handler, 'localhost', 8080,
    max_size=10*1024*1024,  # 10MB message size
    ping_interval=20,        # 20 second ping
    ping_timeout=10          # 10 second timeout
)
```

### Client-Side Optimization

#### **Model Video Caching**
```javascript
// Adjust cache size based on available RAM
this.maxCachedModels = 10;  // Default: 5

// Preload commonly used models
await modelVideoManager.preloadModels([
    'test_optimized_package_fixed_1',
    'test_optimized_package_fixed_2'
]);
```

#### **Compositing Quality vs Speed**
```javascript
// Higher quality (slower)
const composite = await faceCompositor.compositeWithBlending(
    frame, mouth, bounds, 'multiply'
);

// Faster (basic)
const composite = await faceCompositor.compositeFrame(
    frame, mouth, bounds
);
```

---

## 📊 Monitoring & Logs

### Log Locations
- **gRPC Service**: `fast_service/grpc_server.log`
- **Frame Generator**: `realtime_lipsync/frame_generator.log`  
- **Go Server**: Console output
- **Browser**: Developer Console (F12)

### Performance Metrics
- **Frame Generation**: Target 30+ FPS
- **Audio Latency**: <100ms end-to-end
- **GPU Memory**: Monitor with `nvidia-smi`
- **WebSocket Latency**: Check browser Network tab

### Health Checks
```bash
# Quick system health check
curl http://localhost:3000/token >/dev/null && echo "✅ Token server OK"
curl http://localhost:8080 2>&1 | grep -q "WebSocket" && echo "✅ Frame generator OK"  
nvidia-smi | grep -q "python" && echo "✅ gRPC service OK"
```

---

## 🚀 Production Deployment

### Security Considerations
- **API Keys**: Use environment variables, never commit to code
- **CORS**: Configure proper origins in production
- **HTTPS**: Use TLS for production deployment
- **Rate Limiting**: Implement rate limiting for public APIs

### Scaling Options
- **Load Balancing**: Multiple gRPC service instances
- **GPU Cluster**: Distribute models across multiple GPUs/servers
- **CDN**: Serve model videos from CDN for global distribution
- **Caching**: Redis/Memcached for model metadata

### Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Performance dashboards  
- **ELK Stack**: Centralized logging
- **Health Checks**: Automated service monitoring

---

This comprehensive server guide should help you set up, manage, and troubleshoot the entire real-time lip sync system! 🚀
