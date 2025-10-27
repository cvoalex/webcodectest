# Real-time Lip Sync Client Architecture

## Overview

This document provides comprehensive technical documentation for the real-time lip sync client implementation. The system performs live audio-driven lip synchronization using a binary WebSocket protocol for high-performance real-time communication with an AI inference server.

## Core Architecture

### System Components

1. **Audio Capture & Processing**
   - Real-time microphone audio capture at 24kHz sample rate
   - 40ms audio chunks (960 samples per chunk)
   - Circular buffer system for 500 audio chunks (20 seconds @ 40ms chunks)
   - RMS audio level calculation for activity detection

2. **Video Frame Management**
   - Model video loading and frame extraction
   - IndexedDB caching for extracted video frames
   - Dual canvas rendering system (720x1280 pixels)
   - Frame-accurate playback synchronization

3. **Binary WebSocket Protocol**
   - High-performance binary message format
   - Compressed audio data transmission
   - ZIP-compressed inference response handling
   - Real-time bidirectional communication

4. **AI Frame Storage & Retrieval**
   - Persistent AI frame storage using Map structure
   - Frame positioning based on audio buffer index
   - Bounding box coordinate extraction from server responses
   - Real-time frame overlay and display

## Technical Specifications

### Audio Processing Pipeline

```
Microphone Input (24kHz) 
    ↓
AudioContext + ScriptProcessorNode (1024 buffer size)
    ↓ 
Sample Accumulation (960 samples = 40ms)
    ↓
RMS Level Calculation & Activity Detection
    ↓
Circular Buffer Storage (500 slots)
    ↓
Binary Protocol Encoding
    ↓
WebSocket Transmission to Server
```

### Binary Protocol Format

**Request Message Structure:**
```
[Header: 16 bytes]
- Model name length (4 bytes, little-endian uint32)
- Frame ID (4 bytes, little-endian uint32) 
- Audio data length (4 bytes, little-endian uint32)
- Reserved (4 bytes)

[Payload: Variable length]
- Model name (UTF-8 string)
- Audio data (Float32Array as bytes)
```

**Response Message Structure:**
```
[Header: 16 bytes]
- Status code (4 bytes, little-endian uint32)
- Frame ID (4 bytes, little-endian uint32)
- ZIP data length (4 bytes, little-endian uint32) 
- Reserved (4 bytes)

[Payload: Variable length]
- ZIP compressed data containing:
  - AI generated image (JPEG)
  - Bounding box coordinates (Float32Array: [x1, y1, x2, y2])
```

### Frame Buffer Management

**Circular Buffer System:**
- **Size:** 500 slots (accommodates 20 seconds of audio at 40ms chunks)
- **Indexing:** `audioBufferIndex % 500` for current position
- **Threading:** Audio capture and AI inference operate independently
- **Synchronization:** Frame ID used for precise audio-visual alignment

**Buffer States:**
- **Empty (Gray):** No audio data captured
- **Audio Only (Green):** Audio captured, inference pending
- **Inference Sent (Yellow):** Request sent to server, awaiting response
- **Complete (Blue):** AI frame received and ready for playback

### Coordinate System & Display

**Canvas Configuration:**
- **Dimensions:** 720x1280 pixels (matches original video)
- **Dual Canvas Setup:**
  - Left Canvas: Original video + AI mouth overlay
  - Right Canvas: AI inference result only
- **No Scaling Required:** Direct coordinate mapping from server

**Bounding Box Format:**
- **Server Response:** `[x1, y1, x2, y2]` as Float32Array
- **Coordinate Space:** Original video dimensions (720x1280)
- **Direct Mapping:** No transformation needed due to matching canvas size

## Real-time Playback Process

### Initialization Sequence

1. **WebSocket Connection**
   ```
   Connect to ws://localhost:8084
   Set binary message handling
   Enable auto-reconnection logic
   ```

2. **Audio Context Setup**
   ```
   Create AudioContext (24kHz sample rate)
   Request microphone permissions
   Initialize ScriptProcessorNode (1024 buffer)
   Setup circular buffer (500 slots)
   ```

3. **Video Loading**
   ```
   Load model video (/models/{model_name}/video.mp4)
   Extract frames at 25fps (0.04s intervals)
   Cache frames in IndexedDB for performance
   Initialize dual canvas system (720x1280)
   ```

### Real-time Processing Loop

**Audio Capture (40ms intervals):**
```javascript
1. Capture 960 samples from microphone
2. Calculate RMS audio level
3. Store in circular buffer at current index
4. Generate binary request with current frame ID
5. Send to server via WebSocket
6. Increment buffer index (index % 500)
```

**Inference Response Handling:**
```javascript
1. Receive binary response from server
2. Parse header (status, frame_id, zip_length)
3. Extract ZIP payload
4. Decompress to get:
   - AI generated mouth image (JPEG)
   - Bounding box coordinates [x1, y1, x2, y2]
5. Store AI frame data using frame_id as key
6. Update buffer visualization
```

**Real-time Playback (25fps):**
```javascript
1. Calculate current playback position
2. Retrieve original video frame for position
3. Check for AI frame at same position
4. If AI frame exists:
   - Display original frame on left canvas
   - Overlay AI mouth using bounding box coordinates
   - Display AI-only result on right canvas
5. Play corresponding audio chunk if available
6. Advance to next frame (40ms later)
```

## Performance Optimizations

### Binary Protocol Benefits
- **Reduced Overhead:** 16-byte headers vs JSON parsing
- **Efficient Encoding:** Direct Float32Array transmission
- **ZIP Compression:** Compressed image and coordinate data
- **Predictable Parsing:** Fixed header structure for fast processing

### Caching Strategy
- **IndexedDB Storage:** Persistent frame caching across sessions
- **Model-based Keys:** Separate cache per model for isolation
- **Instant Loading:** Cached frames eliminate re-extraction time
- **Progressive Loading:** Background frame extraction with progress indicators

### Memory Management
- **Circular Buffers:** Fixed-size allocation prevents memory growth
- **AI Frame Cleanup:** Automatic cleanup of old AI frames
- **Canvas Reuse:** Single canvas contexts for efficient rendering
- **Blob URL Management:** Proper cleanup of temporary object URLs

## Implementation Guidelines for iOS Native Client

### Core Technologies Needed

**Audio Processing:**
- AVAudioEngine for real-time audio capture
- AVAudioPCMBuffer for 24kHz, 40ms chunk processing
- Core Audio for low-latency audio handling
- Real-time audio level calculation using vDSP

**Video Processing:**
- AVAsset for model video loading and frame extraction
- CVPixelBuffer for frame manipulation
- Core Graphics for canvas-like rendering operations
- Metal or Core Animation for high-performance display

**Networking:**
- URLSessionWebSocketTask for WebSocket communication
- Custom binary protocol encoding/decoding
- ZIP decompression using Compression framework
- Background queue management for network operations

### Key iOS-Specific Considerations

**Audio Session Management:**
```
Configure AVAudioSession for:
- .playAndRecord category
- .defaultToSpeaker option for output
- Low-latency audio processing
- Background audio continuation
```

**Real-time Processing:**
```
Use DispatchQueue with QoS:
- .userInteractive for audio capture
- .userInitiated for network requests  
- .default for video frame processing
- .background for caching operations
```

**Memory and Performance:**
```
Implement:
- CVPixelBufferPool for efficient frame reuse
- Metal shaders for fast image composition
- Core Data or SQLite for frame caching
- Automatic memory pressure handling
```

### Protocol Implementation Details

**Binary Message Encoding (Swift):**
```
Required capability to:
1. Encode UTF-8 strings with length prefixes
2. Convert Float arrays to little-endian byte data
3. Construct messages with 16-byte headers
4. Handle WebSocket binary message transmission
```

**ZIP Response Handling:**
```
Must implement:
1. Binary header parsing (16 bytes)
2. ZIP payload extraction and decompression
3. JPEG image decoding to UIImage/CGImage
4. Float32Array coordinate parsing
5. Error handling for malformed responses
```

## Configuration Parameters

### Audio Settings
- **Sample Rate:** 24,000 Hz (required by lip sync model)
- **Chunk Duration:** 40ms (960 samples per chunk)
- **Buffer Size:** 500 chunks (20 seconds total)
- **Processor Buffer:** 1024 samples (nearest power of 2)

### Video Settings
- **Frame Rate:** 25 fps (0.04 second intervals)
- **Canvas Size:** 720x1280 pixels (portrait orientation)
- **Quality:** 0.8 JPEG quality for frame extraction
- **Max Frames:** 250 frames (10 seconds @ 25fps)

### Network Settings
- **WebSocket URL:** ws://localhost:8084 (development)
- **Protocol:** Binary message format
- **Auto-reconnect:** 3-second intervals on disconnect
- **Timeout:** 30 seconds for inference responses

### Performance Targets
- **Audio Latency:** < 50ms capture to transmission
- **Inference Time:** 200-500ms server processing
- **Playback Sync:** ±16ms frame accuracy (1 frame @ 25fps)
- **Memory Usage:** < 100MB for 20-second buffer

## Debugging and Monitoring

### Real-time Metrics
- **Audio Buffer Fill:** Current number of captured chunks
- **Frame Buffer Fill:** Number of AI frames ready
- **Frames Generated:** Total AI frames received
- **Current FPS:** Real-time frame generation rate
- **Latency:** Round-trip time for AI inference
- **Audio Level:** Real-time microphone input level

### Visual Debugging
- **Buffer Visualization:** Color-coded timeline showing capture/inference states
- **Coordinate Display:** Real-time bounding box coordinates with timestamps
- **Frame Information:** Current playback position and frame details
- **Bounding Box Overlay:** Red debug rectangles showing AI mouth placement

### Error Handling
- **WebSocket Reconnection:** Automatic retry with exponential backoff
- **Audio Permission:** Graceful fallback when microphone access denied
- **Model Loading:** Error reporting for missing video files
- **Inference Failures:** Timeout handling and request retry logic

## Future Enhancement Opportunities

### Performance Improvements
- **WebCodecs API:** Hardware-accelerated video encoding/decoding
- **WebAssembly:** CPU-intensive audio processing acceleration
- **WebWorkers:** Background thread processing for heavy operations
- **Progressive Frame Loading:** Streaming video frame extraction

### Quality Enhancements
- **Adaptive Quality:** Dynamic adjustment based on network conditions
- **Multi-model Support:** Seamless switching between different AI models
- **Advanced Synchronization:** Sub-frame timing for perfect lip sync
- **Audio Enhancement:** Noise reduction and echo cancellation

### Platform Extensions
- **Mobile Optimization:** Touch-optimized controls and responsive design
- **Desktop Integration:** Native desktop application wrapper
- **Cloud Deployment:** Scalable server infrastructure support
- **Cross-platform SDK:** Unified API for multiple client platforms

---

This architecture provides a robust foundation for real-time lip synchronization with excellent performance characteristics and clear separation of concerns suitable for native iOS implementation.
