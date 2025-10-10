# Audio Batch Inference API Guide

## Overview

This guide explains how to use the **Audio Batch Inference API** for generating lip-synced video frames with optimal bandwidth efficiency. The API achieves **93.4% bandwidth savings** by eliminating redundant audio data transmission.

## Table of Contents
- [Key Concepts](#key-concepts)
- [Audio Preparation](#audio-preparation)
- [Making Requests](#making-requests)
- [Complete Examples](#complete-examples)
- [Performance Tips](#performance-tips)

---

## Key Concepts

### Audio Chunking
- Each frame requires **16 audio chunks** (8 previous + current + 7 future)
- Each chunk represents **40ms of audio** at **16kHz sample rate**
- Chunk size: **640 samples × 4 bytes (float32) = 2,560 bytes**

### Bandwidth Optimization
**Old Method (Redundant):**
- Send 16 chunks per frame
- For N frames: N × 16 chunks = **93% redundant data**

**New Method (Optimized):**
- Send contiguous array: N + 15 chunks
- Savings: 70% (4 frames) to 93% (240+ frames)

**Example:**
```
240 frames:
  Old: 240 × 16 = 3,840 chunks (9.41 MB)
  New: 240 + 15 = 255 chunks (0.62 MB)
  Savings: 93.4% (8.79 MB saved!)
```

### Audio Padding
For frames at audio boundaries (beginning/end):
- **Repeat first chunk** for frames before audio starts
- **Repeat last chunk** for frames after audio ends
- This allows generating frames for the **entire audio duration**

---

## Audio Preparation

### Step 1: Load Audio File
Load a WAV file (16kHz, mono or stereo):

```go
// Read WAV file
data, err := os.ReadFile("audio.wav")

// Parse WAV header to find data chunk
// Skip RIFF/WAVE headers and metadata
audioData := extractDataChunk(data)
```

### Step 2: Convert to Mono (if stereo)
```go
if channels == 2 {
    // Average left and right channels
    for i := 0; i < len(samples); i += 2 {
        mono[i/2] = (samples[i] + samples[i+1]) / 2
    }
}
```

### Step 3: Resample to 16kHz (if needed)
```go
if sampleRate != 16000 {
    // Use resampling library
    samples = resample(samples, sampleRate, 16000)
}
```

### Step 4: Chunk into 40ms Segments
```go
chunkSize := 640 // samples per 40ms at 16kHz
numChunks := len(samples) / chunkSize

for i := 0; i < numChunks; i++ {
    chunk := make([]byte, 2560) // 640 × 4 bytes
    
    for j := 0; j < 640; j++ {
        // Convert int16 to float32 [-1.0, 1.0]
        floatVal := float32(samples[i*640 + j]) / 32768.0
        
        // Write as little-endian bytes
        binary.LittleEndian.PutUint32(chunk[j*4:], 
            math.Float32bits(floatVal))
    }
    
    audioChunks = append(audioChunks, chunk)
}
```

### Step 5: Add Padding (for complete coverage)
```go
frameCount := len(audioChunks) // Can generate this many frames
audioChunksNeeded := frameCount + 15

paddedChunks := make([][]byte, audioChunksNeeded)

for i := 0; i < audioChunksNeeded; i++ {
    chunkIdx := i - 8 + startFrame
    
    if chunkIdx < 0 {
        // Repeat first chunk
        paddedChunks[i] = audioChunks[0]
    } else if chunkIdx >= len(audioChunks) {
        // Repeat last chunk
        paddedChunks[i] = audioChunks[len(audioChunks)-1]
    } else {
        paddedChunks[i] = audioChunks[chunkIdx]
    }
}
```

---

## Making Requests

### Protocol Buffer Definition

```protobuf
message BatchInferenceWithAudioRequest {
    string model_name = 1;        // e.g., "sanders"
    int32 start_frame_id = 2;     // Starting frame ID
    int32 frame_count = 3;        // Number of frames to generate
    repeated bytes audio_chunks = 4; // Audio data (frame_count + 15 chunks)
}

message BatchInferenceResponse {
    repeated OptimizedInferenceResponse responses = 1;
    int32 total_processing_time_ms = 2;
    double avg_frame_time_ms = 3;
}

message OptimizedInferenceResponse {
    bool success = 1;
    bytes image_data = 2;         // JPEG image
    int32 frame_id = 3;
    int32 processing_time_ms = 4;
    optional string error = 5;
}
```

### gRPC Call Example

```go
// Connect to server
conn, err := grpc.Dial("localhost:50051",
    grpc.WithTransportCredentials(insecure.NewCredentials()),
    grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(50*1024*1024)),
)
defer conn.Close()

client := pb.NewOptimizedLipSyncServiceClient(conn)

// Create request
request := &pb.BatchInferenceWithAudioRequest{
    ModelName:    "sanders",
    StartFrameId: 0,
    FrameCount:   int32(frameCount),
    AudioChunks:  paddedAudioChunks,
}

// Call API
ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
defer cancel()

response, err := client.GenerateBatchWithAudio(ctx, request)
if err != nil {
    log.Fatalf("RPC failed: %v", err)
}

// Process responses
for _, resp := range response.Responses {
    if resp.Success {
        // Save JPEG frame
        os.WriteFile(fmt.Sprintf("frame_%d.jpg", resp.FrameId), 
            resp.ImageData, 0644)
    } else {
        log.Printf("Frame %d failed: %s", resp.FrameId, *resp.Error)
    }
}
```

---

## Complete Examples

### Python Example

```python
import grpc
import numpy as np
from scipy.io import wavfile
import optimized_lipsyncsrv_pb2
import optimized_lipsyncsrv_pb2_grpc

# Load and process audio
sample_rate, audio = wavfile.read('audio.wav')

# Convert to mono if stereo
if len(audio.shape) == 2:
    audio = audio.mean(axis=1).astype(np.int16)

# Resample to 16kHz if needed
if sample_rate != 16000:
    from scipy.signal import resample
    num_samples = int(len(audio) * 16000 / sample_rate)
    audio = resample(audio, num_samples).astype(np.int16)

# Chunk into 40ms segments
chunk_size = 640
num_chunks = len(audio) // chunk_size
chunks = []

for i in range(num_chunks):
    chunk_samples = audio[i * chunk_size:(i + 1) * chunk_size]
    # Convert to float32 [-1.0, 1.0]
    float_chunk = (chunk_samples / 32768.0).astype(np.float32)
    chunks.append(float_chunk.tobytes())

# Add padding
frame_count = len(chunks)
padded_chunks = []

for i in range(frame_count + 15):
    chunk_idx = i - 8
    if chunk_idx < 0:
        padded_chunks.append(chunks[0])
    elif chunk_idx >= len(chunks):
        padded_chunks.append(chunks[-1])
    else:
        padded_chunks.append(chunks[chunk_idx])

# Make gRPC request
channel = grpc.insecure_channel('localhost:50051')
stub = optimized_lipsyncsrv_pb2_grpc.OptimizedLipSyncServiceStub(channel)

request = optimized_lipsyncsrv_pb2.BatchInferenceWithAudioRequest(
    model_name='sanders',
    start_frame_id=0,
    frame_count=frame_count,
    audio_chunks=padded_chunks
)

response = stub.GenerateBatchWithAudio(request)

# Save frames
for resp in response.responses:
    if resp.success:
        with open(f'frame_{resp.frame_id}.jpg', 'wb') as f:
            f.write(resp.image_data)
```

### JavaScript/TypeScript Example

```typescript
import * as grpc from '@grpc/grpc-js';
import { OptimizedLipSyncServiceClient } from './generated/optimized_lipsyncsrv_grpc_pb';
import { BatchInferenceWithAudioRequest } from './generated/optimized_lipsyncsrv_pb';
import * as fs from 'fs';
import WavDecoder from 'wav-decoder';

async function generateLipSync(audioPath: string) {
    // Load WAV file
    const buffer = fs.readFileSync(audioPath);
    const audioData = await WavDecoder.decode(buffer);
    
    // Convert to mono float32 at 16kHz
    const sampleRate = 16000;
    let samples = audioData.channelData[0]; // First channel
    
    // Resample if needed (use resampling library)
    if (audioData.sampleRate !== sampleRate) {
        samples = resample(samples, audioData.sampleRate, sampleRate);
    }
    
    // Chunk into 40ms segments
    const chunkSize = 640;
    const numChunks = Math.floor(samples.length / chunkSize);
    const chunks: Uint8Array[] = [];
    
    for (let i = 0; i < numChunks; i++) {
        const chunk = new Float32Array(chunkSize);
        for (let j = 0; j < chunkSize; j++) {
            chunk[j] = samples[i * chunkSize + j];
        }
        chunks.push(new Uint8Array(chunk.buffer));
    }
    
    // Add padding
    const frameCount = chunks.length;
    const paddedChunks: Uint8Array[] = [];
    
    for (let i = 0; i < frameCount + 15; i++) {
        const chunkIdx = i - 8;
        if (chunkIdx < 0) {
            paddedChunks.push(chunks[0]);
        } else if (chunkIdx >= chunks.length) {
            paddedChunks.push(chunks[chunks.length - 1]);
        } else {
            paddedChunks.push(chunks[chunkIdx]);
        }
    }
    
    // Create gRPC client
    const client = new OptimizedLipSyncServiceClient(
        'localhost:50051',
        grpc.credentials.createInsecure()
    );
    
    // Create request
    const request = new BatchInferenceWithAudioRequest();
    request.setModelName('sanders');
    request.setStartFrameId(0);
    request.setFrameCount(frameCount);
    request.setAudioChunksList(paddedChunks);
    
    // Make call
    return new Promise((resolve, reject) => {
        client.generateBatchWithAudio(request, (error, response) => {
            if (error) {
                reject(error);
            } else {
                // Save frames
                response.getResponsesList().forEach(resp => {
                    if (resp.getSuccess()) {
                        fs.writeFileSync(
                            `frame_${resp.getFrameId()}.jpg`,
                            Buffer.from(resp.getImageData())
                        );
                    }
                });
                resolve(response);
            }
        });
    });
}
```

---

## Performance Tips

### 1. Batch Size Optimization
- **Sweet spot**: 100-250 frames per request
- Larger batches = better GPU utilization
- Smaller batches = lower latency

### 2. Connection Settings
```go
// Increase max message size for large batches
grpc.WithDefaultCallOptions(
    grpc.MaxCallRecvMsgSize(50 * 1024 * 1024), // 50MB
    grpc.MaxCallSendMsgSize(50 * 1024 * 1024)
)
```

### 3. Concurrent Requests
For very long audio:
```go
// Split into chunks of 250 frames each
const batchSize = 250
numBatches := (totalFrames + batchSize - 1) / batchSize

// Process concurrently (but limit to 2-3 concurrent requests)
semaphore := make(chan struct{}, 3)
var wg sync.WaitGroup

for i := 0; i < numBatches; i++ {
    wg.Add(1)
    go func(batchNum int) {
        defer wg.Done()
        semaphore <- struct{}{}
        defer func() { <-semaphore }()
        
        // Process batch...
    }(i)
}
wg.Wait()
```

### 4. Video Assembly
Use FFmpeg to combine frames:
```bash
# Create frame list
for i in {0..255}; do
    echo "file 'frame_${i}.jpg'"
    echo "duration 0.04"
done > frames.txt

# Combine with audio
ffmpeg -f concat -safe 0 -i frames.txt \
       -i audio.wav \
       -c:v libx264 -pix_fmt yuv420p \
       -c:a aac -b:a 128k \
       output.mp4
```

---

## Bandwidth Comparison

| Frames | Old Method | New Method | Savings | Data Saved |
|--------|------------|------------|---------|------------|
| 4      | 64 chunks  | 19 chunks  | 70.3%   | 112 KB     |
| 20     | 320 chunks | 35 chunks  | 89.1%   | 712 KB     |
| 100    | 1,600 chunks | 115 chunks | 92.8% | 3.71 MB    |
| 240    | 3,840 chunks | 255 chunks | 93.4% | 8.79 MB    |
| 500    | 8,000 chunks | 515 chunks | 93.6% | 18.3 MB    |

**Formula**: `savings = (N × 16 - (N + 15)) / (N × 16) × 100%`

---

## Troubleshooting

### Error: "Insufficient audio chunks"
```
Expected: frame_count + 15 chunks
Got: X chunks
```
**Solution**: Ensure you send exactly `frame_count + 15` audio chunks with padding.

### Error: "Invalid audio format"
**Solution**: Each chunk must be:
- Exactly 2,560 bytes (640 float32 values)
- Little-endian byte order
- Float32 values in range [-1.0, 1.0]

### Performance Issues
- Check GPU memory: Should have 4+ GB available
- Reduce batch size if getting OOM errors
- Use compression for network transfer if bandwidth is limited

---

## API Reference

### Server Endpoint
```
localhost:50051
```

### RPC Methods
```protobuf
service OptimizedLipSyncService {
    // Batch inference with optimized audio
    rpc GenerateBatchWithAudio(BatchInferenceWithAudioRequest) 
        returns (BatchInferenceResponse);
}
```

### Response Fields
- `success`: Whether frame generation succeeded
- `image_data`: JPEG-encoded frame (typically 15-18 KB)
- `frame_id`: Frame identifier
- `processing_time_ms`: Server processing time for this frame
- `error`: Error message if success=false

---

## Best Practices

1. ✅ **Always pad audio chunks** for complete coverage
2. ✅ **Use proper audio format** (16kHz, float32, mono)
3. ✅ **Batch requests** for better performance (100-250 frames)
4. ✅ **Check response.success** before using frames
5. ✅ **Set appropriate timeouts** (60s for large batches)
6. ✅ **Clean up temp files** after video assembly

---

## Support

For issues or questions:
- Check server logs for detailed error messages
- Verify audio format matches specification
- Test with small batches first (4-20 frames)
- Review example code in `grpc-test-client/test_real_audio.go`

**Performance Metrics:**
- Throughput: ~50 FPS on RTX 4080
- Latency: ~20ms per frame
- Memory: ~2GB GPU RAM for model + frames
