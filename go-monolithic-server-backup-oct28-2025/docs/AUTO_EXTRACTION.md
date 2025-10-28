# Auto-Extraction Feature

## Overview

The monolithic server now **automatically extracts background frames** from source video files if the frames directory is empty. This eliminates the need for manual frame extraction as a setup step.

## How It Works

### Configuration

Add a `source_video` path to your model configuration in `config.yaml`:

```yaml
models:
  sanders:
    model_path: "sanders/checkpoint/model_best.onnx"
    background_dir: "sanders/frames"
    source_video: "sanders/full_body_video.mp4"  # ← NEW: Auto-extract source
    crop_rects_path: "sanders/crop_rects.json"
    num_frames: 523
```

### Automatic Behavior

When you first load a model (on first inference request):

1. **Check if frames exist:** Server checks if `background_dir` contains any image files
2. **Extract if missing:** If directory is empty and `source_video` is configured:
   - Creates the `background_dir` directory
   - Extracts `num_frames` frames from the source video
   - Saves as `frame_000000.jpg`, `frame_000001.jpg`, etc.
   - Uses JPEG quality 95 for good quality/size balance
3. **Load into memory:** All frames loaded and cached in RAM (same as before)
4. **Subsequent loads:** Uses cached frames from disk (no re-extraction)

## Extraction Methods

The server tries multiple extraction methods in order of preference:

### 1. FFmpeg (Fastest - Preferred)

**Speed:** ~10-15 seconds for 523 frames  
**Requirements:** FFmpeg in system PATH

```bash
ffmpeg -i video.mp4 -vf "select='lt(n,523)'" -vsync 0 -q:v 2 -start_number 0 frame_%06d.jpg
```

**Advantages:**
- ✅ Fastest extraction method
- ✅ High quality output
- ✅ Efficient memory usage
- ✅ Battle-tested video processing

### 2. Python/OpenCV (Fallback)

**Speed:** ~15-20 seconds for 523 frames  
**Requirements:** Python with OpenCV (`cv2`)

```python
import cv2
cap = cv2.VideoCapture("video.mp4")
for i in range(num_frames):
    ret, frame = cap.read()
    cv2.imwrite(f"frame_{i:06d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
```

**Advantages:**
- ✅ Works if FFmpeg not available
- ✅ Same quality as FFmpeg
- ✅ Python usually already installed for ML projects

## Performance Impact

### First-Time Setup (No Frames Cached)

| Phase | Time | Description |
|-------|------|-------------|
| Frame extraction (FFmpeg) | 10-15s | Extract 523 frames from video |
| Frame extraction (Python) | 15-20s | Fallback if FFmpeg unavailable |
| Frame loading | 25-30s | Load 523 JPEG → RGBA in memory |
| **Total cold start** | **35-50s** | One-time cost on first request |

### Subsequent Requests

| Phase | Time | Description |
|-------|------|-------------|
| Frame loading | 25-30s | Load from disk (first request only) |
| Inference | 697ms | Batch of 25 frames (warm cache) |
| **Total warm** | **697ms** | Typical performance after cache |

### Disk Space

- Source video: ~50-100 MB (full_body_video.mp4)
- Extracted frames: ~50-60 MB (523 × ~100KB JPEG each)
- **Total:** ~100-160 MB per model

## Configuration Examples

### Basic Setup (Auto-extract enabled)

```yaml
models:
  my_model:
    background_dir: "my_model/frames"
    source_video: "my_model/full_body_video.mp4"
    num_frames: 523
```

**Behavior:**
- First load: Auto-extracts 523 frames, then loads into memory
- Subsequent loads: Uses cached frames from disk

### Pre-extracted Frames (Skip extraction)

```yaml
models:
  my_model:
    background_dir: "my_model/frames"  # Already has 523 frame_*.jpg files
    # source_video: (omitted - not needed)
    num_frames: 523
```

**Behavior:**
- First load: Loads existing frames from disk
- Skips extraction entirely

### Shared Frames (Multiple models, one video)

```yaml
models:
  sanders:
    background_dir: "sanders/frames"
    source_video: "sanders/full_body_video.mp4"
    num_frames: 523
    
  sanders1:
    background_dir: "sanders/frames"  # Same directory
    source_video: "sanders/full_body_video.mp4"
    num_frames: 2000  # Can use more frames from same video
    
  sanders2:
    background_dir: "sanders/frames"
    source_video: "sanders/full_body_video.mp4"
    num_frames: 2000
```

**Behavior:**
- First model to load extracts frames
- Other models reuse existing frames
- Can specify different `num_frames` per model

## Error Handling

### Missing Source Video

```
ERROR: background directory 'sanders/frames' is empty and no source_video configured
```

**Solution:** Add `source_video` to config or manually extract frames

### Invalid Video Path

```
ERROR: source video not found: sanders/full_body_video.mp4
```

**Solution:** Check video path is correct (relative to `models_root`)

### Extraction Failed

```
ERROR: failed to extract frames: ffmpeg failed: ...
ERROR: python extraction failed: ...
```

**Solution:** Install FFmpeg or Python with OpenCV (`pip install opencv-python`)

## Manual Extraction (Alternative)

If you prefer manual control or need custom extraction settings:

### Using Python Script

```bash
cd go-monolithic-server
python extract_backgrounds.py
```

**Script:** `extract_backgrounds.py` (included in repo)

### Using FFmpeg Directly

```bash
ffmpeg -i full_body_video.mp4 -vf "select='lt(n,523)'" -q:v 2 frame_%06d.jpg
```

### Using Python/OpenCV

```python
import cv2
import os

video_path = "full_body_video.mp4"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
for i in range(523):
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_dir}/frame_{i:06d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
cap.release()
```

## Benefits

### Zero Manual Setup ✅
- No need to run extraction scripts before starting server
- Server handles everything automatically
- Just provide source video and go

### Idempotent Behavior ✅
- Safe to delete frames directory - server will re-extract
- Safe to restart server - uses existing frames if present
- No duplicate extraction if frames already exist

### Developer-Friendly ✅
- Clear error messages if misconfigured
- Logs extraction progress
- Falls back gracefully if FFmpeg unavailable

### Production-Ready ✅
- One-time extraction cost (amortized over many requests)
- Frames cached on disk permanently
- Memory cache for instant access during inference

## Troubleshooting

### Slow Extraction

**Problem:** Extraction takes >30 seconds  
**Solutions:**
- Install FFmpeg for faster extraction (10-15s vs 15-20s)
- Pre-extract frames manually during deployment
- Use SSD for faster disk I/O

### Memory Issues

**Problem:** Server runs out of memory during load  
**Solutions:**
- Reduce `num_frames` to extract fewer frames
- Use smaller resolution source video
- Increase server RAM or use memory limits

### Disk Space Issues

**Problem:** Not enough disk space for frames  
**Solutions:**
- Reduce JPEG quality (modify `IMWRITE_JPEG_QUALITY` in code)
- Extract fewer frames (`num_frames`)
- Clean up old/unused model frames

## Implementation Details

### Code Location

- **Config:** `config/config.go` - Added `SourceVideo` field to `ModelConfig`
- **Extraction:** `registry/video_extractor.go` - FFmpeg and Python extraction
- **Loading:** `registry/image_registry.go` - Auto-extraction trigger in `loadModel()`

### Key Functions

```go
// Check if extraction needed
if isDirectoryEmpty(modelCfg.BackgroundDir) {
    if modelCfg.SourceVideo == "" {
        return error // No source video configured
    }
    extractFramesFromVideo(sourceVideo, outputDir, numFrames)
}

// Extract using FFmpeg (preferred)
func extractWithFFmpeg(sourceVideo, outputDir string, numFrames int) error

// Extract using Python/OpenCV (fallback)
func extractWithPython(sourceVideo, outputDir string, numFrames int) error

// Check if directory is empty or has no image files
func isDirectoryEmpty(dir string) bool
```

## See Also

- [EXTRACTION_VS_CACHING.md](EXTRACTION_VS_CACHING.md) - Full performance analysis
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - Implementation guide
- [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md) - Benchmark results
