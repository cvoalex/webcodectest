# Real-time Lip Sync Console

Revolutionary real-time AI conversation system with perfect lip synchronization, combining OpenAI's Realtime API with high-performance lip sync generation.

## 🎯 Features

- **Real-time Speech-to-Speech**: OpenAI Realtime API with WebRTC
- **Synchronized Lip Sync**: Advanced audio processing with Web Audio API
- **High Performance**: 30+ FPS lip sync generation with your optimized models
- **Intelligent Buffering**: 15-frame target buffer for jitter-free playback
- **Controlled Audio Playback**: Dual audio routing for user experience and lip sync
- **Multi-Model Support**: Switch between multiple optimized lip sync models
- **Real-time Monitoring**: Comprehensive statistics and performance metrics
- **Go/Node.js Server Options**: Choose between Go or Node.js for token generation

## 🏗️ Architecture

```
Browser (WebRTC) ↔ Go/Node.js Server ↔ Python Frame Generator ↔ gRPC Lip Sync Service
     ↓                      ↓                    ↓                       ↓
WebRTC Audio        Token Generation    Audio Processing        Frame Generation
Web Audio API       Static Files        WebSocket Bridge        High-Performance
Event Handling      OpenAI Integration  PCM16 Conversion        84+ FPS Inference
```

## 🚀 Quick Start

### Prerequisites

- Go 1.21+ OR Node.js 18+ 
- Python 3.8+
- OpenAI API Key
- Your gRPC lip sync service running on localhost:50051

### 1. Setup Environment

```bash
# Navigate to the project
cd realtime_lipsync

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key:
OPENAI_API_KEY=your_openai_api_key_here

# Install Python dependencies
pip install grpcio grpcio-tools numpy websockets psutil
```

### 2. Start Services

**Option A: Go Server (Recommended)**
```bash
# Terminal 1: Start your gRPC service
cd ../fast_service
python grpc_server.py

# Terminal 2: Start Python frame generator  
cd ../realtime_lipsync
python frame_generator.py

# Terminal 3: Start Go token server
cd ../go-token-server
go mod tidy
go run main.go
```

**Option B: Node.js Server**
```bash
# Terminal 1: Start your gRPC service
cd ../fast_service  
python grpc_server.py

# Terminal 2: Start Python frame generator
cd ../realtime_lipsync
python frame_generator.py

# Terminal 3: Install Node.js dependencies and start server
npm install
node server.js
```

### 3. Use the Console

1. Open http://localhost:3000/realtime_lipsync.html
2. Click "Start Session" 
3. Grant microphone permissions when prompted
4. Start talking and see synchronized lip sync!

## 🎵 Audio Processing Pipeline

### Web Audio API Integration
- **Real-time Processing**: `createMediaStreamSource()` for direct stream processing
- **PCM16 Conversion**: Float32 samples converted to 16-bit PCM at 24kHz
- **Dual Audio Routing**: 
  - Audio element for user playback (you hear the AI)
  - Audio buffer manager for lip sync generation
- **Zero Latency**: No WebM/Opus encoding/decoding bottlenecks

### Buffer Management
- **Target Buffer**: 15 frames (600ms) for optimal jitter minimization
- **Adaptive Levels**: Critical (3), Min (8), Target (15), Max (25) frames
- **Circular Buffering**: 3000-element arrays for audio and frame storage
- **Real-time Statistics**: Buffer fill levels, FPS, latency monitoring

### Frame Generation Flow
```
OpenAI WebRTC Audio → Web Audio API → PCM16 Conversion → WebSocket
                                                             ↓
Frame Buffer ← gRPC Inference ← Audio Processing ← Python Frame Generator
     ↓
Canvas Display ← Synchronized Playback ← Buffer Management
```
Frame Generation ← gRPC Inference ← 640ms Audio Window ← Buffer Read
        ↓
Frame Buffer → Synchronized Display → Canvas Rendering
```

## 📊 Performance Specifications

- **Audio Latency**: <100ms from OpenAI to buffer
- **Frame Generation**: 5 frames per 640ms audio window
- **Display FPS**: 30 FPS smooth playback
- **Buffer Capacity**: 120 seconds audio + 120 seconds frames
- **Memory Efficiency**: ~470MB per model (40+ models possible)

## �️ Configuration

### OpenAI Realtime API Settings

**Minimal Configuration (Current Implementation)**
```javascript
const sessionConfig = {
    instructions: "You are a helpful assistant...",
    voice: "alloy",  // Must be one of: alloy, echo, fable, onyx, nova, shimmer
    input_audio_format: "pcm16",
    output_audio_format: "pcm16",
    turn_detection: { type: "server_vad" }
};
```

**Available Voices**
- `alloy` - Balanced, natural voice (default)
- `echo` - Clear, articulate voice
- `fable` - Warm, expressive voice  
- `onyx` - Deep, authoritative voice
- `nova` - Bright, energetic voice
- `shimmer` - Soft, gentle voice

### Environment Configuration (.env)
```env
OPENAI_API_KEY=your_openai_api_key_here
AUDIO_SAMPLE_RATE=24000          # OpenAI Realtime API default
AUDIO_CHUNK_DURATION_MS=40       # 40ms per audio chunk
AUDIO_BUFFER_SIZE=3000           # 3000 element circular buffer
FRAMES_FOR_INFERENCE=16          # 16 frames for inference (640ms)
INFERENCE_BATCH_SIZE=5           # 5 frames per batch
LOOKAHEAD_FRAMES=7               # 7 future frames needed
```

### gRPC Service Configuration

**Model Management**
```python
# frame_generator.py - Model switching
MODEL_NAMES = [
    "test_optimized_package_fixed_1", 
    "test_optimized_package_fixed_2", 
    "test_optimized_package_fixed_3",
    "test_optimized_package_fixed_4", 
    "test_optimized_package_fixed_5"
]

# Send model change requests
stub.SetLipSyncModel(set_model_request)
```

**Performance Tuning**
- **Target Buffer**: 15 frames (600ms) for smooth playback
- **Frame Generation**: Batch processing for efficiency  
- **Audio Chunk Size**: 1920 samples (40ms at 48kHz)
- **PCM Format**: 16-bit signed integers at 24kHz
- **Memory Usage**: ~470MB per model (40+ models possible)

## 🌐 API Endpoints

### Web Server (localhost:3000)
- `GET /` - Main console interface
- `GET /token` - Generate OpenAI ephemeral token
- `GET /health` - Service health check
- `GET /config` - System configuration

### Frame Generator WebSocket (localhost:8080)
- `audio_chunk` - Send audio data for processing
- `get_frame` - Request current synchronized frame
- `set_model` - Change active lip sync model
- `get_stats` - Request buffer statistics

### gRPC Service (localhost:50051)
- `GenerateInference` - Single frame generation
- `GenerateBatchInference` - Batch frame generation  
- `GetModelInfo` - Model information
- `LoadModel` - Load additional models

## 📈 Monitoring & Statistics

### Real-time Metrics
- **Audio Buffer Fill**: Current/3000 chunks with visual bar
- **Frame Buffer Fill**: Generated frames ready for display
- **Generation FPS**: Frame generation rate
- **Display FPS**: Canvas rendering rate
- **GPU Memory**: Current model memory usage
- **Inference Time**: Frame generation latency

### Event Log
- Real-time event stream showing:
  - WebRTC session events
  - Audio processing events  
  - Frame generation events
  - OpenAI API responses
  - System status changes

## 🛠️ Development

### File Structure
```
realtime_lipsync/
├── realtime_lipsync.html        # Main console interface  
├── realtime_lipsync_app.js      # Main application controller
├── realtime_lipsync_client.js   # WebRTC client integration
├── audio_buffer_manager.js      # Audio buffering system
├── frame_generator.py           # Python WebSocket server for gRPC integration
├── server.js                    # Node.js Express server (optional)
├── package.json                 # Node.js dependencies
├── setup_and_start.py          # Python setup script
├── start_console.bat            # Windows startup script
├── .env.example                 # Configuration template
└── ../go-token-server/          # Go token server (recommended)
    ├── main.go                  # Go server implementation
    ├── go.mod                   # Go module definition
    └── go.sum                   # Go dependencies
```

### Key Components

**RealtimeLipSyncClient**: WebRTC session management and OpenAI integration with Web Audio API
**AudioBufferManager**: Real-time audio capture, buffering, and controlled playback with PCM16 conversion
**RealtimeLipSyncApp**: Main UI controller and coordination layer with statistics monitoring
**FrameGenerator**: Python WebSocket server for gRPC integration with corrected handler signatures
**Go Token Server**: Lightweight token generation and static file serving (recommended over Node.js)

## � Troubleshooting

### Common Issues & Solutions

**1. "Failed to start session" Error**
```bash
# Check if OpenAI API key is set
echo $OPENAI_API_KEY  # or in .env file

# Verify Go/Node.js server is running
curl http://localhost:3000/token

# Check OpenAI API access
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

**2. Session Configuration Errors**
- **Use minimal config**: Only include required parameters (voice, audio formats, turn_detection)
- **Avoid unsupported params**: Don't include temperature, max_response_output_tokens for session.update
- **Valid voices only**: Must be one of: alloy, echo, fable, onyx, nova, shimmer

**3. Audio Processing Issues**
- **Browser Permissions**: Grant microphone access when prompted
- **HTTPS Required**: Some browsers require HTTPS for microphone (use `localhost` for development)
- **Web Audio API**: Ensure browser support (Chrome/Firefox/Edge recommended)
- **Encoding Errors**: System uses Web Audio API directly, not MediaRecorder

**4. Lip Sync Delays**
```bash
# Check gRPC service performance
python -c "import grpc; print('gRPC available')"

# Monitor frame generation
tail -f frame_generator.log

# Verify buffer levels in browser console
```

**5. WebSocket Connection Failures**
```bash
# Check if frame_generator.py is running
ps aux | grep frame_generator

# Verify port availability
netstat -an | grep 8765

# Test WebSocket manually
python -c "import websockets; print('WebSockets available')"
```

**6. Python WebSocket Handler Errors**
- **Modern websockets library**: Use `async def websocket_handler(websocket)` signature
- **Remove path parameter**: Don't include `path` in handler function for newer websockets versions
- **Audio processing**: Ensure NumPy arrays are properly converted to bytes

### Performance Optimization

**Buffer Tuning**
```javascript
// Adjust buffer levels based on your hardware
const BUFFER_CRITICAL = 3;   // Minimum for smooth playback
const BUFFER_MIN = 8;        // Good performance threshold  
const BUFFER_TARGET = 15;    // Optimal balance
const BUFFER_MAX = 25;       // Maximum before overflow
```

**Model Selection Performance**
- **High Quality**: Use `test_optimized_package_fixed_5` for best visual results
- **Performance**: Use `test_optimized_package_fixed_1` for fastest generation
- **Balance**: Use `test_optimized_package_fixed_3` for good quality + speed

**Network Optimization**
- Ensure stable internet for OpenAI WebRTC
- Use local gRPC service (localhost:50051) for best performance
- Monitor browser DevTools Network tab for bottlenecks

### Debug Mode
Enable detailed logging:
```env
DEBUG_AUDIO_PROCESSING=true
DEBUG_FRAME_GENERATION=true  
DEBUG_WEBSOCKET_MESSAGES=true
```

## 🚀 Performance Optimization

### GPU Memory
- Current usage: ~2.3GB for 5 models
- Theoretical capacity: 40+ models on RTX 4090
- Memory efficiency: <10% utilization

### Frame Generation
- Current performance: 84.2 FPS peak
- Production target: 30 FPS sustained
- Batch processing: 5 frames per inference

### Audio Processing
- Circular buffer prevents memory leaks
- Intelligent read/write pointers
- Zero-copy audio processing where possible

## 🔮 Future Enhancements

- Multi-GPU scaling for higher model capacity
- WebRTC data channel optimizations
- Advanced audio preprocessing (noise reduction, enhancement)
- Real-time quality adaptation based on network conditions
- Support for multiple simultaneous sessions
- Integration with other OpenAI Realtime API features

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📞 Support

For issues and questions:
- Check troubleshooting section
- Review browser console logs
- Check Python frame generator logs
- Verify gRPC service status
