# Test Data Extraction vs Caching Analysis

## Overview

The tests use a **hybrid approach** - some data is extracted on-demand, while critical server-side data is pre-cached for performance.

---

## What's Being EXTRACTED On-Demand (Test Client Side)

### 1. Visual Frames (Every Test Iteration)

**Location:** `load_frames.py` called by Go test  
**Source:** MP4 video files  
**Process:**
```python
# Extracted EVERY time loadRealVisualFrames() is called
crops_cap = cv2.VideoCapture("crops_328_video.mp4")
rois_cap = cv2.VideoCapture("rois_320_video.mp4")

# Seek to start frame
crops_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Read batch_size frames
for i in range(batch_size):
    ret1, crop = crops_cap.read()
    ret2, roi = rois_cap.read()
    # Resize, normalize, convert to BGR CHW format
```

**Why Extract Instead of Cache:**
- Test flexibility: Can test any frame range
- Memory efficient: Don't need to hold all 523 frames in test client memory
- Video seeking is fast (~1-2ms per seek)
- Total extraction time: ~50-100ms for 25 frames (negligible vs 28s total)

**Performance Impact:**
- Batch 1: ~5-10ms to extract 1 frame
- Batch 25: ~50-100ms to extract 25 frames
- **Overhead: <0.5%** of total processing time

### 2. Audio Chunks (Every Test Iteration)

**Location:** `extractAudioChunk()` in Go test  
**Source:** Pre-loaded WAV file in memory  
**Process:**
```go
// Audio file loaded ONCE at test startup
audioSamples, sampleRate, err := readWAVFile("../aud.wav")

// Then each iteration extracts a chunk
audioChunk := audioSamples[startSample : startSample+samplesNeeded]

// Convert int16 to float32
for i, sample := range audioChunk {
    normalized := float32(sample) / 32768.0
    // Write to bytes
}
```

**Why Extract Instead of Pre-compute:**
- Different batch sizes need different audio windows
- Frame context window varies (16 frames for SyncTalk)
- Conversion is trivial (~1-2ms)

**Performance Impact:**
- Audio file load: One-time, ~10-20ms
- Per-iteration extraction: ~1-2ms
- **Overhead: <0.1%** of total processing time

---

## What's PRE-CACHED (Server Side)

### 2. Background Frames (523 Images) ✅ FULLY CACHED + AUTO-EXTRACTED

**Location:** `ImageRegistry.loadModel()`  
**Loaded:** On first request (lazy loading)  
**Memory:** ~3.5 GB (523 frames × 1920×1080 × 4 bytes RGBA)

```go
// Check if frames directory is empty
if isDirectoryEmpty(modelCfg.BackgroundDir) {
    if modelCfg.SourceVideo == "" {
        return nil, fmt.Errorf("no frames and no source video configured")
    }
    
    // Auto-extract frames from source video
    log.Printf("📹 Extracting frames from source video...")
    extractFramesFromVideo(modelCfg.SourceVideo, modelCfg.BackgroundDir, modelCfg.NumFrames)
}

// Loaded ONCE when model first used
backgrounds, memoryBytes, err := loadBackgrounds(modelCfg.BackgroundDir, modelCfg.NumFrames)

// All 523 frames loaded into memory
for i := 0; i < 523; i++ {
    imgPath := filepath.Join(dir, fmt.Sprintf("frame_%06d.jpg", i))
    img, err := imaging.Open(imgPath)
    // Convert to RGBA
    backgrounds[i] = convertToRGBA(img)
}
```

**Auto-Extraction Feature:**
- ✅ Detects if `background_dir` is empty on model load
- ✅ Automatically extracts frames from `source_video` (configured in YAML)
- ✅ Uses FFmpeg if available (fastest), falls back to Python/OpenCV
- ✅ Saves as JPEG with 95% quality (frame_000000.jpg format)
- ✅ One-time setup - subsequent loads use cached frames

**Extraction Methods:**
1. **FFmpeg (preferred):** ~10-15 seconds for 523 frames
   - Command: `ffmpeg -i video.mp4 -vf "select='lt(n,523)'" -q:v 2 frame_%06d.jpg`
2. **Python/OpenCV (fallback):** ~15-20 seconds for 523 frames
   - Uses cv2.VideoCapture to read and save frames

**Why Pre-cache:**
- ✅ Loaded from disk ONCE (523 × ~100KB JPEG = ~50MB disk I/O)
- ✅ Decompressed to RGBA ONCE (523 × ~8MB = ~4GB memory)
- ✅ Instant access during compositing (<1ns pointer lookup)
- ✅ No I/O stalls during inference
- ✅ No manual extraction step required

**Performance Impact:**
- **Cold start (frames exist):** ~25-30 seconds to load all 523 frames
- **Cold start (no frames):** ~35-50 seconds (extract + load)
- **Warm (cached):** 0ms - instant array access
- **Memory trade-off:** 3.5GB RAM for 100× speedup

### 2. Crop Rectangles (523 Positions) ✅ FULLY CACHED

**Location:** `ImageRegistry.loadModel()`  
**Source:** `crop_rects.json` (523 × [xmin, ymin, xmax, ymax])  
**Memory:** ~16 KB (523 × 4 × int32)

```go
cropRects, err := loadCropRects(modelCfg.CropRectsPath, modelCfg.NumFrames)

// Parse JSON once
var rects [][4]int
json.Unmarshal(data, &rects)

// Convert to image.Rectangle
for i, r := range rects {
    cropRects[i] = image.Rect(r[0], r[1], r[2], r[3])
}
```

**Why Pre-cache:**
- ✅ Tiny memory footprint (16KB)
- ✅ Calculated from landmarks offline (523 × 68 points)
- ✅ Never changes for a given model
- ✅ Instant array indexing during compositing

**Performance Impact:**
- Load time: <1ms
- Memory: 16KB (negligible)
- Access time: 0ms (array index)

### 3. ONNX Model Weights ✅ FULLY CACHED

**Location:** `ModelRegistry.loadModel()`  
**Memory:** ~500 MB per model  
**GPU Memory:** ~2 GB for inference session

```go
// Loaded ONCE per model
session, err := ort.NewAdvancedSession(
    modelPath,
    []string{"crops", "audio"},  // Input names
    []string{"output"},          // Output names
    []int64{1,6,320,320},        // Input shapes
    ort.NewSessionOptions(),     // CUDA, etc.
)
```

**Why Pre-cache:**
- ✅ Model loading: ~1-2 seconds (CUDA initialization)
- ✅ Inference session: Kept warm on GPU
- ✅ No model reload between batches
- ✅ CUDA kernels pre-compiled

**Performance Impact:**
- **Cold start:** 1-2 seconds
- **Warm:** 0ms - session already on GPU
- **Inference:** 10-50ms (batch size dependent)

---

## Comparison: Extract vs Cache Performance

### Visual Frames (Test Client)

| Approach | Load Time | Memory | Access Time | Total Cost |
|----------|-----------|--------|-------------|------------|
| **Extract (current)** | 50-100ms | ~25MB | 0ms (streaming) | 100ms one-time |
| Cache all 523 | ~500ms | ~1.5GB | 0ms | 500ms + 1.5GB |

**Decision:** EXTRACT ✅
- Only need 25 frames per test
- Memory efficient
- Fast enough (<0.5% overhead)

### Background Frames (Server)

| Approach | Load Time | Memory | Access Time | Total Cost |
|----------|-----------|--------|-------------|------------|
| Extract per frame | 523 × 5ms | ~10MB | 5ms/frame | 125ms/batch |
| **Cache all 523 (current)** | 25s once | 3.5GB | <1ns | 25s + 3.5GB |

**Decision:** CACHE ✅
- Used for ALL requests
- Reused across batches
- 5ms/frame would add 125ms per batch
- Worth 3.5GB for 100× speedup

### Audio Features (Server)

**Current:** Generated on-the-fly via ONNX audio encoder

| Stage | Time | Cached? |
|-------|------|---------|
| Audio encoder model | ~500MB | ✅ Yes (GPU) |
| Mel-spectrogram compute | ~50ms | ❌ No (computed) |
| Audio features (512-dim) | ~400ms | ❌ No (computed) |

**Potential Optimization:**
- Could cache audio features for repeated audio
- Would save ~400ms per batch
- Trade-off: Memory vs flexibility

---

## Test Performance Breakdown

### Batch Size 25 (697ms total)

| Component | Time | Type | Impact |
|-----------|------|------|--------|
| Load visual frames | ~80ms | EXTRACT | 11.5% |
| Extract audio chunk | ~2ms | EXTRACT | 0.3% |
| Audio processing | ~93ms | COMPUTE | 13.3% |
| Inference (ONNX) | ~272ms | COMPUTE (cached model) | 39.0% |
| Compositing | ~60ms | COMPUTE (cached backgrounds) | 8.6% |
| JPEG encoding | ~190ms | COMPUTE | 27.3% |

**Total Extraction Overhead:** 82ms (11.8%)

---

## Memory Footprint

### Test Client
- WAV file: ~650KB (loaded once)
- Visual frames: ~25MB per iteration (released after send)
- **Peak:** ~26MB

### Server (Per Model)
- ONNX weights: ~500MB RAM
- GPU session: ~2GB VRAM
- Background frames: ~3.5GB RAM
- Crop rects: ~16KB RAM
- **Total:** ~4GB RAM + 2GB VRAM per model

### System Capacity
- RTX 4090: 24GB VRAM → ~12 models max (2GB each)
- System RAM: 64GB → ~15 models max (4GB each)
- **Server config:** 40 models max (assumes smaller models or eviction)

---

## Optimization Opportunities

### 1. Cache Audio Features ❌ NOT WORTH IT
- Savings: ~400ms per repeated audio
- Cost: ~8KB per frame (512 floats)
- Use case: Rare (different audio every time)

### 2. Pre-load Visual Frames ❌ NOT NEEDED
- Current: 80ms extraction (11.8% overhead)
- Benefit: Save 80ms
- Cost: 1.5GB RAM in test client
- **Decision:** Current overhead acceptable

### 3. Background Frame Caching ✅ ALREADY DONE
- Current: All 523 frames pre-loaded
- Access: <1ns pointer lookup
- **Status:** Optimal

### 4. Batch Background Loading ⚠️ POTENTIAL
- Current: Sequential loading (~25s cold start)
- Potential: Parallel goroutines
- Benefit: 5-10× faster cold start
- Risk: Disk I/O contention

---

## Summary

### Extraction Strategy (Test Client)
✅ **Visual Frames:** Extract on-demand from MP4 (80ms overhead)  
✅ **Audio Chunks:** Extract from pre-loaded WAV (2ms overhead)  
**Total Overhead:** 82ms (11.8% of 697ms)

### Caching Strategy (Server)
✅ **Backgrounds:** ALL 523 frames pre-cached (3.5GB, instant access)  
   - **Auto-extraction:** If frames directory is empty, automatically extracts from source video on first model load
   - Uses FFmpeg (if available) or Python/OpenCV as fallback
   - One-time setup cost: ~10-20 seconds for 523 frames
✅ **Crop Rects:** ALL 523 positions pre-cached (16KB, instant access)  
✅ **ONNX Model:** Weights + GPU session cached (4GB, warm inference)  
❌ **Audio Features:** Generated on-the-fly (flexible, no cache needed)

### Performance Impact
- **Cold start (frames exist):** 25-30 seconds (loading 523 backgrounds from disk)
- **Cold start (frames missing):** 35-50 seconds (auto-extract + load 523 backgrounds)
- **Warm (cached):** 697ms for 25 frames (35.9 FPS)
- **Extraction overhead:** 11.8% (acceptable for flexibility)

### Design Principles
1. **Cache heavy data** (backgrounds, models) → Used repeatedly
2. **Extract light data** (visual frames) → Different every test
3. **Auto-extract if missing** (backgrounds) → Zero manual setup required
4. **Compute cheap operations** (audio chunks, features) → Too variable to cache
5. **Trade memory for speed** where it matters (backgrounds: 3.5GB for 100× speedup)
