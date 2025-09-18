# Unified Server Architecture Guide

## Overview
This guide covers the **UNIFIED 1-SERVER ARCHITECTURE** that consolidates all functionality into a single Go executable. No more multiple servers, Python WebSocket services, or complex deployment!

## Architecture Evolution

### ❌ Before (3 Servers)
1. **gRPC Lip Sync Service** (Python) - Port 50051
2. **Python Frame Generator** (WebSocket) - Port 8080  
3. **Go Web Server** (HTTP/Static) - Port 3000

### ✅ After (1 Server!) 
1. **Go Unified Server** (ALL FUNCTIONALITY) - Port 3000
   - HTTP/Static file serving
   - WebSocket frame generator  
   - OpenAI token generation
   - Model video streaming
   - Embedded gRPC stub client
   - **Works standalone or with optional Python gRPC service**

## Deployment Modes

### Mode 1: Standalone (Recommended for Development)
```bash
# Single command deployment!
start_unified_server.bat

# Or manually:
cd go-token-server
go run main.go
```

**Features:**
- ✅ Complete web application 
- ✅ WebSocket real-time processing
- ✅ OpenAI Realtime API integration
- ✅ Model video serving
- ✅ Stub frame generation (for testing)
- ❌ No AI lip sync (uses dummy frames)

### Mode 2: Full AI (Production)
```bash
# Terminal 1: Start Python gRPC service (optional)
python syncnet.py

# Terminal 2: Start unified server  
cd go-token-server
go run main.go
```

**Features:**
- ✅ Everything from Mode 1
- ✅ Real AI lip sync via gRPC
- ✅ SyncNet model inference
- ✅ Production-ready frame generation

## WebSocket Frame Generator (Integrated)

The Go server now includes a complete WebSocket frame generator that replaces the Python equivalent:

### Core Components
- **Circular Audio Buffer:** 3000-slot ring buffer for audio chunks
- **Circular Frame Buffer:** 3000-slot ring buffer for generated frames
- **gRPC Client:** Connects to Python lip sync service on port 50051
- **Multi-client Support:** Handles multiple WebSocket connections
- **Real-time Processing:** 40ms audio chunks, batch inference

### WebSocket API Endpoints

#### Client → Server Messages
```javascript
// Send audio chunk
{
    "type": "audio_chunk",
    "audio_data": "base64_encoded_audio"
}

// Request current frame
{
    "type": "get_frame"
}

// Change model
{
    "type": "set_model", 
    "model_name": "test_optimized_package_fixed_1"
}

// Get buffer statistics
{
    "type": "get_stats"
}

// Get reference image for compositing
{
    "type": "get_reference_image",
    "model_name": "test_optimized_package_fixed_1",
    "request_id": "unique_id"
}
```

#### Server → Client Messages
```javascript
// Current frame response
{
    "type": "current_frame",
    "frame_data": "base64_encoded_frame",
    "bounds": [x, y, width, height],
    "timestamp": 1234567890,
    "model_name": "test_optimized_package_fixed_1",
    "prediction_shape": "320x320"
}

// Buffer statistics
{
    "type": "stats",
    "audio_buffer_fill": 45,
    "frame_buffer_fill": 12,
    "can_generate": true,
    "current_model": "test_optimized_package_fixed_1"
}

// Frame generation notification
{
    "type": "frames_generated",
    "count": 5,
    "buffer_fill": 17
}
```

## Setup Instructions

### Prerequisites
```bash
# Go dependencies (from go-token-server/)
go mod tidy

# Protobuf compiler (if not installed)
# Windows: Download from https://github.com/protocolbuffers/protobuf/releases
# Add protoc.exe to PATH

# Protocol Buffer Go plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

### Build Process
```bash
cd go-token-server

# Generate protobuf Go files
setup_integrated.bat

# Or manually:
# protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative proto/lipsync.proto
```

### Environment Setup
```bash
# go-token-server/.env (create if needed)
OPENAI_API_KEY=your_openai_api_key_here
PORT=3000
```

## Deployment Sequence

### Start Servers
```bash
# Terminal 1: Start gRPC service (Python)
cd d:\Projects\SyncTalk2D
python syncnet.py

# Terminal 2: Start integrated web server (Go)  
cd go-token-server
go run main_integrated.go
```

### Verify Services
```bash
# Check gRPC service
# Should show model loading messages and "🚀 gRPC server listening on :50051"

# Check web server
# Should show:
# 🚀 Real-time Lip Sync Server (with integrated frame generator) starting on port 3000
# 🔑 Token endpoint: http://localhost:3000/token
# 🌐 WebSocket endpoint: ws://localhost:3000/ws  
# 🎬 Model video endpoint: http://localhost:3000/api/model-video/{model_name}
```

### Test Application
1. **Open Browser:** http://localhost:3000/index.html
2. **Check Connections:** All status indicators should be green
3. **Test Audio:** Start microphone and verify frame generation
4. **Verify Compositing:** Full-body avatar should display instead of mouth-only

## Benefits of Simplified Architecture

### Deployment Advantages
- **Reduced Complexity:** 2 servers instead of 3
- **Fewer Dependencies:** No separate Python WebSocket server
- **Single Web Port:** All HTTP/WebSocket traffic on port 3000
- **Unified Logging:** All client interactions in one place

### Performance Benefits  
- **Native Go Performance:** WebSocket handling in compiled Go
- **Reduced Network Hops:** Direct gRPC client in web server
- **Memory Efficiency:** Go's garbage collector vs Python
- **Concurrent WebSocket Handling:** Goroutines vs Python threading

### Maintenance Benefits
- **Single Web Codebase:** All HTTP/WebSocket logic in Go
- **Consistent Error Handling:** Unified logging and error responses
- **Easier Monitoring:** Fewer processes to track
- **Simplified Scaling:** Scale web tier independently

## Troubleshooting

### Common Issues

#### "Connection refused" on port 50051
```bash
# Ensure gRPC service is running
python syncnet.py
# Look for "🚀 gRPC server listening on :50051"
```

#### WebSocket connection fails
```bash
# Check if Go server is running on port 3000
netstat -an | findstr 3000
# Should show LISTENING on port 3000
```

#### No frames generated
```bash
# Check gRPC connection in Go server logs
# Should see "✅ Connected to gRPC service on localhost:50051"

# Verify model files exist
# Check datasets_test/ directory structure
```

#### Audio processing errors
```bash
# Verify audio format (24kHz, mono, int16)
# Check browser microphone permissions
# Monitor WebSocket messages in browser dev tools
```

### Debug Commands
```bash
# Check server processes
tasklist | findstr go
tasklist | findstr python

# Test gRPC directly (if you have grpcurl)
grpcurl -plaintext localhost:50051 list

# Test WebSocket connection
# Use browser dev tools → Network → WS
```

## Migration from 3-Server Setup

If you have an existing 3-server setup, follow these steps:

1. **Stop Python frame generator:**
   ```bash
   # Stop the Python WebSocket server on port 8080
   # Ctrl+C in the terminal running frame_generator.py
   ```

2. **Update client configuration:**
   - The WebSocket URL is automatically updated to `ws://localhost:3000/ws`
   - No other client changes needed

3. **Start integrated server:**
   ```bash
   cd go-token-server  
   go run main_integrated.go
   ```

4. **Verify functionality:**
   - Test token generation: http://localhost:3000/token
   - Test model videos: http://localhost:3000/api/model-video/test_optimized_package_fixed_1
   - Test WebSocket: Connect via browser and check real-time frames

5. **Remove Python dependencies (optional):**
   - The `frame_generator.py` file is no longer needed
   - Keep `syncnet.py` for the gRPC service

## Production Considerations

### Security
```bash
# Add proper CORS configuration
# Implement rate limiting
# Use HTTPS in production
# Validate WebSocket origins
```

### Performance Monitoring
```bash
# Monitor goroutine counts
# Track WebSocket connection counts  
# Monitor gRPC call latency
# Track audio/frame buffer utilization
```

### Scaling
```bash
# Web server can be horizontally scaled
# gRPC service remains single instance (GPU-bound)
# Consider load balancer for multiple web instances
# Use connection pooling for gRPC clients
```

This simplified architecture provides the same functionality with reduced operational complexity, making it easier to deploy, maintain, and scale the real-time lip sync system.
