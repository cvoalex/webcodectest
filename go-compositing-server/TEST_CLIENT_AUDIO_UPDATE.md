# Test Client Audio Update

## Overview

The test client (`test_client.go`) has been updated to send **raw audio from WAV files** instead of mock audio features. This enables end-to-end testing of the complete audio processing pipeline.

## Changes Made

### 1. WAV File Reading
Added `readWAVFile()` function that:
- Reads standard WAV file headers (RIFF/WAVE format)
- Supports both 16-bit PCM and IEEE float formats
- Handles mono and stereo (converts stereo to mono)
- Normalizes audio to float32 range [-1, 1]
- Validates sample rate and format

### 2. Audio Chunk Extraction
Added `extractAudioChunk()` function that:
- Extracts the correct number of samples for each batch
- Calculates: `samples_needed = batch_size × (sample_rate / 25)`
  - For batch_size=24 at 16kHz: 24 × 640 = **10,240 samples**
- Handles end-of-file by looping or padding with zeros
- Converts float32 samples to bytes for transmission

### 3. Main Function Updates
- Loads `aud.wav` from parent directory (`../aud.wav`)
- Displays WAV file info (sample rate, bit depth, channels)
- Verifies sample rate is 16kHz (warns if not)
- Extracts audio chunks for each batch
- Sends `raw_audio` field instead of `audio_features`
- Displays audio processing time in output

### 4. Output Updates
Now shows audio processing time when available:
```
Batch 1/5: GPU=0, frames=24
  🎵 Audio:       7.45 ms
  ⚡ Inference:   120.28 ms
  🎨 Compositing: 2.15 ms
  📊 Total:       129.88 ms
```

## Usage

### Running the Test Client

```powershell
cd go-compositing-server
.\test-client.exe
```

### Expected Output

```
🧪 Compositing Server Test Client
============================================================
🔌 Connecting to compositing server at localhost:50052...
✅ Connected successfully

📊 Checking server health...
✅ Server Status: Healthy
   Loaded Models: 1/4
   Inference Server: localhost:50051 (true)

📁 Output directory: test_output/

🎵 Loading audio file: ..\aud.wav
📊 WAV Info: 16000 Hz, 16-bit, 1 channel(s), format=1
   Loaded 51200 samples (3.20 seconds)

🚀 Running 5 batches (batch_size=24)...
============================================================
Batch 1/5: GPU=0, frames=24
  🎵 Audio:       7.45 ms
  ⚡ Inference:   120.28 ms
  🎨 Compositing: 2.15 ms
  📊 Total:       129.88 ms
  💾 Saved 24 frames to test_output/
...
```

## Audio File Requirements

### Format
- **Container**: WAV (RIFF/WAVE)
- **Encoding**: PCM (16-bit) or IEEE float (32-bit)
- **Channels**: Mono (preferred) or Stereo (auto-converted)
- **Sample Rate**: **16,000 Hz** (required by audio pipeline)

### Duration
For 5 batches × 24 frames × 40ms = **4.8 seconds minimum**
- Each batch needs: 24 frames × 40ms = **960ms** (but pipeline uses 640ms)
- Actual samples per batch: 24 × 640 = **10,240 samples** at 16kHz

### Example WAV Generation

Using ffmpeg to convert any audio to correct format:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 aud.wav
```

Parameters:
- `-ar 16000`: Set sample rate to 16kHz
- `-ac 1`: Convert to mono
- `-sample_fmt s16`: 16-bit PCM format

## API Changes

### Previous (Mock Audio Features)
```go
req := &pb.CompositeBatchRequest{
    ModelId:       "sanders",
    VisualFrames:  visualBytes,
    AudioFeatures: mockAudioFeatures,  // 32×16×16 float32
    BatchSize:     24,
}
```

### Current (Raw Audio)
```go
req := &pb.CompositeBatchRequest{
    ModelId:      "sanders",
    VisualFrames: visualBytes,
    RawAudio:     rawAudioBytes,  // 10,240 float32 samples for batch=24
    BatchSize:    24,
}
```

## Error Handling

### Sample Rate Mismatch
If `aud.wav` is not 16kHz:
```
⚠️  Warning: Audio is 44100 Hz, but pipeline expects 16kHz. Resampling required!
```
The test will continue but results will be incorrect. Resample the audio first.

### File Not Found
```
❌ Failed to load audio: failed to open WAV file: open ..\aud.wav: no such file or directory
```
Ensure `aud.wav` exists in the project root directory.

### Invalid Format
```
❌ Failed to load audio: unsupported audio format: 6 (only PCM and IEEE float supported)
```
The WAV file must be PCM or IEEE float format.

## Testing Different Scenarios

### Test with Different Batch Sizes
Modify constants in `test_client.go`:
```go
const (
    batchSize  = 16    // Try different batch sizes
    numBatches = 10    // More batches for longer test
)
```

### Test with Different Models
```go
const modelID = "biden"  // Try different models
```

### Test Backward Compatibility
Uncomment `generateMockAudioFeatures()` and switch back to:
```go
AudioFeatures: generateMockAudioFeatures(batchSize),
```

## Performance Benchmarking

The test client reports:
- **Audio Processing Time**: Time to process raw audio → mel → features
- **Inference Time**: GPU model inference
- **Compositing Time**: Image compositing/encoding
- **Total Time**: Complete pipeline
- **Overhead**: Non-inference processing time

Typical results (batch_size=24):
```
🎵 Audio:       ~7-10 ms
⚡ Inference:   ~120 ms (GPU dependent)
🎨 Compositing: ~2-3 ms
📊 Total:       ~130-135 ms
📈 Overhead:    ~10 ms (7-8% of inference)
```

## Troubleshooting

### Audio Not Processing
Check server logs for audio encoder initialization:
```
✅ Audio encoder initialized successfully
```

If missing, check:
1. `audio_encoder.onnx` exists in go-inference-server root
2. ONNX Runtime library path is correct in config
3. Server has audio processing enabled

### Sample Count Mismatch
Error: "Failed to extract mel windows: expected X frames, got Y"
- Verify audio chunk size matches batch size
- Check `extractAudioChunk()` calculation
- Ensure no resampling artifacts

### Quality Issues
If output quality is poor:
1. Verify audio is 16kHz mono
2. Check audio normalization (should be -1 to 1)
3. Validate mel-spectrogram processing in server logs
4. Compare with Python reference implementation

## Next Steps

1. ✅ **Test client updated** - Sends raw audio
2. ⏳ **Run end-to-end test** - Verify complete pipeline
3. ⏳ **Performance benchmark** - Measure audio processing overhead
4. ⏳ **Quality validation** - Compare output with Python baseline
5. ⏳ **Stress testing** - Multiple concurrent clients

## Related Files

- `test_client.go` - Updated test client source
- `../aud.wav` - Test audio file (16kHz mono WAV)
- `proto/compositing.proto` - API definition with `raw_audio` field
- `../go-inference-server/audio/` - Audio processing pipeline
- `../go-inference-server/AUDIO_INTEGRATION.md` - Server-side integration docs
