# 🎵 Audio Batch Optimization

## Problem: Redundant Audio Data in Batch Requests

### Current Approach (Inefficient)

When generating multiple consecutive frames, each frame requires a **16-chunk audio window**:
- 8 chunks BEFORE current frame (320ms history)
- 1 chunk FOR current frame (40ms)
- 7 chunks AFTER current frame (280ms future)

**Example: Generate frames 100-103**

```
Frame 100: [chunk 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107]
Frame 101: [chunk 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
Frame 102: [chunk 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
Frame 103: [chunk 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
```

**Total chunks sent: 64 chunks (4 frames × 16 chunks)**
**Unique chunks needed: 19 chunks (92-110)**
**Redundancy: 70%** ❌

### Optimized Approach (Smart)

Send a **single contiguous audio array** and extract windows on the server:

```
Send once: [chunk 92, 93, 94, ..., 110]  (19 chunks total)

Server generates:
- Frame 100: extract chunks[0:16]   → [92-107]
- Frame 101: extract chunks[1:17]   → [93-108]
- Frame 102: extract chunks[2:18]   → [94-109]
- Frame 103: extract chunks[3:19]   → [95-110]
```

**Total chunks sent: 19 chunks**
**Data reduction: 70% less!** ✅

## Formula

For generating `N` consecutive frames starting at frame `F`:

### Required Audio Chunks
```
start_chunk = F - 8
end_chunk = F + N + 7
total_chunks = 8 + N + 7 = N + 15
```

### Examples

| Frames | Start | Count | Chunks Needed | Old Method | Savings |
|--------|-------|-------|---------------|------------|---------|
| 100    | 100   | 1     | 16            | 16         | 0%      |
| 100-101| 100   | 2     | 17            | 32         | 47%     |
| 100-103| 100   | 4     | 19            | 64         | 70%     |
| 100-107| 100   | 8     | 23            | 128        | 82%     |
| 100-119| 100   | 20    | 35            | 320        | 89%     |

**The more frames you batch, the more efficient it becomes!** 🚀

## Implementation

### Protocol Buffers Definition

```protobuf
message BatchInferenceWithAudioRequest {
    string model_name = 1;
    int32 start_frame_id = 2;           // First frame to generate
    int32 frame_count = 3;              // Number of consecutive frames
    repeated bytes audio_chunks = 4;    // Contiguous audio chunks (40ms each)
                                        // Must provide: start_frame_id-8 through start_frame_id+frame_count+6
}
```

### Client Usage (Pseudo-code)

```javascript
// User wants frames 100-103 with real-time audio
const startFrame = 100;
const frameCount = 4;

// Calculate audio range
const audioStart = startFrame - 8;     // 92
const audioEnd = startFrame + frameCount + 7;  // 111
const audioChunks = extractAudioChunks(audioStart, audioEnd);  // 19 chunks

// Send optimized batch request
const request = {
    model_name: "sanders",
    start_frame_id: startFrame,
    frame_count: frameCount,
    audio_chunks: audioChunks  // Only 19 chunks vs 64!
};

const response = await client.GenerateBatchWithAudio(request);
```

### Server Processing

```python
def generate_batch_with_audio(request):
    start_frame = request.start_frame_id
    frame_count = request.frame_count
    audio_chunks = request.audio_chunks
    
    # Verify we have enough audio
    required_chunks = frame_count + 15
    if len(audio_chunks) < required_chunks:
        raise ValueError(f"Need {required_chunks} chunks, got {len(audio_chunks)}")
    
    results = []
    for i in range(frame_count):
        frame_id = start_frame + i
        
        # Extract 16-chunk window for this frame
        window_start = i        # Offset in audio array
        window_end = i + 16     # 16 chunks per frame
        frame_audio = audio_chunks[window_start:window_end]
        
        # Generate frame with extracted audio
        result = generate_frame(frame_id, frame_audio)
        results.append(result)
    
    return results
```

## Bandwidth Savings

### Comparison for 4-Frame Batch

**Audio chunk size: ~16KB** (40ms of audio, typical)

| Method | Chunks | Data Size | Network Time (10 Mbps) |
|--------|--------|-----------|------------------------|
| Old (redundant) | 64 | 1.0 MB | 800ms |
| New (optimized) | 19 | 0.3 MB | 240ms |
| **Savings** | **-45 chunks** | **-0.7 MB** | **-560ms** |

### Real-World Impact

**Scenario: 4 users at 25 FPS, batching 4 frames each**

Old method:
- 4 users × 6.25 batches/sec × 64 chunks = 1,600 chunks/sec
- 1,600 × 16KB = 25.6 MB/sec

New method:
- 4 users × 6.25 batches/sec × 19 chunks = 475 chunks/sec
- 475 × 16KB = 7.6 MB/sec

**Network bandwidth saved: 18 MB/sec (70%)** 🎉

## API Compatibility

### Maintains Backward Compatibility

1. **Old API** (`GenerateBatchInference`): Still works, uses pre-extracted features
2. **New API** (`GenerateBatchWithAudio`): Optimized for real-time audio

### Migration Path

```javascript
// Old way (still supported)
for (let frame = 100; frame < 104; frame++) {
    const audio = getAudioWindow(frame - 8, frame + 7);  // 16 chunks
    await client.GenerateInference({
        model_name: "sanders",
        frame_id: frame,
        audio: audio
    });
}
// Sends 64 audio chunks total

// New way (optimized)
const audioChunks = getAudioChunks(92, 110);  // 19 chunks once
await client.GenerateBatchWithAudio({
    model_name: "sanders",
    start_frame_id: 100,
    frame_count: 4,
    audio_chunks: audioChunks
});
// Sends 19 audio chunks total (70% reduction)
```

## Edge Cases

### Non-Consecutive Frames

If frames are **not consecutive** (e.g., [100, 105, 200]), use original `GenerateBatchInference` API.

**This optimization only works for consecutive frame sequences.**

### Frame Range Calculation

```python
def calculate_audio_range(start_frame, frame_count):
    """
    Calculate required audio chunk range
    
    Args:
        start_frame: First frame to generate
        frame_count: Number of consecutive frames
    
    Returns:
        (audio_start_chunk, audio_end_chunk)
    """
    audio_start = start_frame - 8
    audio_end = start_frame + frame_count + 7
    
    return audio_start, audio_end

# Example
audio_start, audio_end = calculate_audio_range(100, 4)
# Returns: (92, 111) - need chunks 92 through 110 (19 total)
```

### Validation

```python
def validate_audio_chunks(request):
    """Validate correct number of audio chunks provided"""
    required = request.frame_count + 15
    provided = len(request.audio_chunks)
    
    if provided < required:
        raise ValueError(
            f"Insufficient audio chunks: need {required}, got {provided}"
        )
    
    if provided > required + 10:
        # Warning: excessive audio chunks
        logger.warning(f"Extra audio chunks provided: {provided} vs {required}")
```

## Performance Impact

### CPU/Memory

- **Less network I/O**: 70% reduction in audio data transfer
- **Same GPU usage**: Inference workload unchanged
- **Minimal server overhead**: Simple array slicing

### Latency Improvement

For 4-frame batch over 10 Mbps connection:
- **Network time reduced**: 800ms → 240ms (**-560ms**)
- **Processing time**: Unchanged (~50-100ms)
- **Total latency**: 70% faster end-to-end

## Use Cases

### Best For

✅ **Real-time streaming**: Continuous audio input, consecutive frames
✅ **Video generation**: Processing full videos with audio
✅ **Multi-user scenarios**: Many users batching frames simultaneously
✅ **Mobile clients**: Limited bandwidth connections

### Not Ideal For

❌ **Random frame access**: Non-consecutive frame IDs
❌ **Single frame requests**: No batching benefit
❌ **Pre-extracted features**: Already using optimized package

## Future Enhancements

1. **Audio Compression**: Compress audio chunks before transfer (additional 50-70% reduction)
2. **Streaming Audio**: Stream audio chunks as they arrive
3. **Adaptive Batching**: Auto-calculate optimal batch size based on audio buffer
4. **WebRTC Integration**: Direct audio stream to batch API

## Summary

### Benefits

✅ **70% less audio data** for 4-frame batches
✅ **Up to 89% savings** for larger batches (20+ frames)
✅ **Faster network transfer** and lower bandwidth usage
✅ **Same inference quality** and performance
✅ **Backward compatible** with existing API

### Trade-offs

⚠️ **Only works for consecutive frames** (limitation)
⚠️ **Client must calculate audio range** (slight complexity)
⚠️ **Larger individual requests** (but fewer requests overall)

### Recommendation

**Use `GenerateBatchWithAudio` for all consecutive frame generation with real-time audio.** The bandwidth savings and latency improvements are substantial, especially for multi-user scenarios.

---

**Last Updated**: October 10, 2025
**Status**: Specification Complete - Implementation Pending
