# Additional Memory Inefficiencies Found

## Executive Summary
While gaming, I analyzed the codebase and found **CRITICAL** memory allocation issues beyond what we already fixed. These are causing significant GC pressure in the audio processing pipeline.

## 🔴 CRITICAL Issues (High Impact)

### 1. **Mel Window Allocations** (Lines 319-322 in main.go)
**Location**: `cmd/server/main.go` - Audio processing loop

**Problem**:
```go
for frameIdx := 0; frameIdx < numVideoFrames; frameIdx++ {
    window := make([][]float32, 80)  // ❌ Allocates 80 slices
    for i := 0; i < 80; i++ {
        window[i] = make([]float32, 16)  // ❌ 80 more allocations
    }
    // ... 
    allMelWindows = append(allMelWindows, window)  // Keeps all in memory
}
```

**Impact per Request**:
- For batch 25: ~41 video frames to encode
- Each frame: 80 + 1 = 81 allocations
- Total: 41 × 81 = **3,321 allocations**
- Memory: 41 × 80 × 16 × 4 bytes = **210KB wasted**

**Solution**: Pool the mel window buffers with sync.Pool

---

### 2. **Audio Processor STFT Allocations** (audio/processor.go)
**Location**: Lines 267, 273, 280 in `audio/processor.go`

**Problem**:
```go
func (p *Processor) stft(samples []float32) ([][]float32, error) {
    for frameIdx := 0; frameIdx < numFrames; frameIdx++ {
        windowed := make([]float64, p.config.WindowSize)    // ❌ Per frame
        fftInput := make([]float64, p.config.NumFFT)        // ❌ Per frame  
        magnitudes := make([]float32, numFreqBins)          // ❌ Per frame
        // ...
    }
}
```

**Impact per Request**:
- Called once per batch with full audio
- ~600-800 mel frames for 20 second audio
- 3 allocations × 800 frames = **2,400 allocations**
- Window: 640 × 8 bytes = 5KB
- FFT Input: 1024 × 8 bytes = 8KB
- Magnitudes: 513 × 4 bytes = 2KB
- Total per frame: **15KB**
- **Total: 800 × 15KB = 12MB**

**Solution**: Reuse buffers across loop iterations

---

### 3. **Mel Filterbank Application** (audio/processor.go line 313)
**Location**: `applyMelFilterbank()` function

**Problem**:
```go
for frameIdx := 0; frameIdx < numFrames; frameIdx++ {
    melFrame := make([]float32, p.config.NumMelBins)  // ❌ 80 floats per frame
    // ...
}
```

**Impact**: 
- 800 frames × 80 floats × 4 bytes = **256KB**
- 800 separate allocations

**Solution**: Pre-allocate result matrix, fill in-place

---

### 4. **Power-to-DB Conversions** (audio/processor.go lines 340, 373)
**Location**: `powerToDb()` and `normalizeDb()` functions

**Problem**:
```go
func (p *Processor) powerToDb(melSpec [][]float32) [][]float32 {
    result := make([][]float32, len(melSpec))
    for i := 0; i < len(melSpec); i++ {
        frame := make([]float32, len(melSpec[i]))  // ❌ Per frame allocation
        // ...
    }
}
```

**Impact**:
- Each function allocates 800 frames
- 800 × 80 × 4 bytes = **256KB** per function
- Total for both: **512KB**

**Solution**: Operate in-place on existing slice

---

### 5. **JPEG Data Copy** (cmd/server/main.go line 743-744)
**Location**: `compositeFrame()` - After JPEG encoding

**Problem**:
```go
jpegData := make([]byte, buf.Len())  // ❌ Allocates ~50KB per frame
copy(jpegData, buf.Bytes())
```

**Why it exists**: Buffer is returned to pool, so we copy out

**Impact**:
- Batch 25: 25 × 50KB = **1.25MB**
- Batch 8: 8 × 50KB = **400KB**

**Better Solution**: 
- Return `buf.Bytes()` directly (DON'T return buffer to pool yet)
- Let gRPC serialize it
- Return buffer to pool in a defer AFTER gRPC sends response
- **OR** Use a longer-lived buffer pool that survives until response is sent

---

## 🟡 MEDIUM Issues

### 6. **Audio Features Padding** (main.go line 370)
```go
paddedFeatures := make([][]float32, len(allFrameFeatures)+2)
```
**Impact**: One allocation of ~43 × 512 × 4 bytes = **88KB**
**Solution**: Could pre-allocate with capacity

### 7. **Pre-computation Allocations** (audio/processor.go)
Various pre-computation functions allocate temporary buffers:
- `preEmphasis()` - line 233
- `applyPadding()` - line 104
- `hanningWindow()` - line 444

**Impact**: Low (called once during initialization or once per request)

---

## 📊 Total Impact Summary

### Per Batch 25 Request (Before Fixes):
| Component | Allocations | Memory |
|-----------|-------------|--------|
| Mel windows (main.go) | 3,321 | 210 KB |
| STFT (processor.go) | 2,400 | 12 MB |
| Mel filterbank | 800 | 256 KB |
| Power-to-DB | 800 | 256 KB |
| DB normalization | 800 | 256 KB |
| JPEG copies | 25 | 1.25 MB |
| **TOTAL** | **8,146** | **~14 MB** |

### Combined with Previous Findings:
- **Original compositing**: 241 MB (FIXED ✅)
- **Audio processing**: 14 MB (NOT FIXED ❌)
- **NEW TOTAL**: 14 MB per request

---

## 🛠️ Recommended Fixes (Priority Order)

### Priority 1: STFT Buffer Pooling
**File**: `audio/processor.go`
**Impact**: Eliminates 12MB + 2,400 allocations

```go
var stftBufferPool = sync.Pool{
    New: func() interface{} {
        return &stftBuffers{
            windowed: make([]float64, 640),
            fftInput: make([]float64, 1024),
            magnitudes: make([]float32, 513),
        }
    },
}

type stftBuffers struct {
    windowed   []float64
    fftInput   []float64
    magnitudes []float32
}
```

### Priority 2: Mel Window Pooling
**File**: `cmd/server/main.go` (and `audio/windows.go`)
**Impact**: Eliminates 3,321 allocations + 210KB

```go
var melWindowPool = sync.Pool{
    New: func() interface{} {
        window := make([][]float32, 80)
        for i := 0; i < 80; i++ {
            window[i] = make([]float32, 16)
        }
        return window
    },
}
```

### Priority 3: In-Place Operations
**Files**: `audio/processor.go` - powerToDb, normalizeDb, applyMelFilterbank
**Impact**: Eliminates 2,400 allocations + 768KB

Modify functions to operate in-place:
```go
func (p *Processor) powerToDbInPlace(melSpec [][]float32) {
    for i := 0; i < len(melSpec); i++ {
        for j := 0; j < len(melSpec[i]); j++ {
            melSpec[i][j] = 10.0 * float32(math.Log10(math.Max(float64(melSpec[i][j]), 1e-10)))
        }
    }
}
```

### Priority 4: JPEG Buffer Management
**File**: `cmd/server/main.go`
**Impact**: Eliminates 1.25MB per batch 25

Use `buf.Bytes()` directly, defer buffer return until after response sent.

---

## 🎯 Expected Results After All Fixes

### Memory Allocations:
- **Before**: 241 MB (compositing) + 14 MB (audio) = **255 MB**
- **After all fixes**: <1 MB
- **Reduction**: 99.6%

### Allocation Count:
- **Before**: 8,146 + previous = ~8,200 per request
- **After**: <100 per request
- **Reduction**: 98.8%

### Performance Impact:
- Reduced GC pause frequency
- More consistent frame times
- Better sustained throughput
- Lower CPU overhead

---

## 🎮 Implementation Status

- ✅ Compositing pooling (DONE - 241MB saved)
- ⏳ Audio processing pooling (PENDING - 14MB to save)

---

## 🔍 How I Found These

Tools used:
1. `grep` for `make(`, `append(`, `copy(`
2. Manual code review of hot paths
3. Understanding data flow through audio pipeline
4. Calculating allocation sizes and frequencies

---

## 💡 Next Steps

1. Implement STFT buffer pooling (highest impact)
2. Implement mel window pooling
3. Convert audio functions to in-place operations
4. Optimize JPEG buffer handling
5. Run pprof to verify improvements
6. Benchmark before/after

---

**Enjoy your gaming! This analysis will be ready when you return.** 🎮
