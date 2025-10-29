# Code Review: Parallelization & Optimization Opportunities

**Date:** October 28, 2025  
**Reviewed:** go-monolithic-server-refactored  
**Current Performance:** 43.9 FPS peak (batch 25), 23.1 FPS (batch 8)  
**Target:** Further optimization to approach theoretical limits

---

## Executive Summary

✅ **Already Well-Optimized Areas:**
- Compositing is fully parallelized (goroutines per frame)
- Audio encoder pool uses worker pattern with 4 parallel instances
- Memory pools eliminate allocations (5 pools active)
- JPEG encoding uses pooled buffers

⚠️ **Optimization Opportunities Found:**
1. **outputToImage()** - Pixel loop can be parallelized (320×320 = 102,400 iterations)
2. **resizeImagePooled()** - Bilinear interpolation loop can be parallelized
3. **Mel window extraction** - Sequential loop can use goroutines
4. **Audio feature zero-padding** - Sequential loops for padding

---

## 🔥 HIGH-IMPACT OPTIMIZATIONS

### 1. **Parallelize `outputToImage()` Pixel Conversion** ⭐⭐⭐⭐⭐

**Current Code:** `helpers.go:67-85`
```go
// Sequential pixel-by-pixel conversion (102,400 iterations)
for y := 0; y < 320; y++ {
    for x := 0; x < 320; x++ {
        b := outputData[0*320*320+y*320+x]
        g := outputData[1*320*320+y*320+x]
        r := outputData[2*320*320+y*320+x]
        
        rByte := uint8(clampFloat(r * 255.0))
        gByte := uint8(clampFloat(g * 255.0))
        bByte := uint8(clampFloat(b * 255.0))
        
        img.SetRGBA(x, y, color.RGBA{R: rByte, G: gByte, B: bByte, A: 255})
    }
}
```

**Impact:** Called 25 times per batch, 102,400 pixels per call = **2.56M pixels/batch**

**Optimization:**
```go
// Parallel row processing with goroutines
func outputToImage(outputData []float32) *image.RGBA {
    img := rgbaPool320.Get().(*image.RGBA)
    
    const numWorkers = 8 // Match worker count
    rowsPerWorker := 320 / numWorkers
    var wg sync.WaitGroup
    
    for worker := 0; worker < numWorkers; worker++ {
        wg.Add(1)
        startY := worker * rowsPerWorker
        endY := startY + rowsPerWorker
        if worker == numWorkers-1 {
            endY = 320 // Last worker handles remainder
        }
        
        go func(yStart, yEnd int) {
            defer wg.Done()
            
            for y := yStart; y < yEnd; y++ {
                for x := 0; x < 320; x++ {
                    // BGR order from ONNX model
                    b := outputData[0*320*320+y*320+x]
                    g := outputData[1*320*320+y*320+x]
                    r := outputData[2*320*320+y*320+x]
                    
                    rByte := uint8(clampFloat(r * 255.0))
                    gByte := uint8(clampFloat(g * 255.0))
                    bByte := uint8(clampFloat(b * 255.0))
                    
                    img.SetRGBA(x, y, color.RGBA{R: rByte, G: gByte, B: bByte, A: 255})
                }
            }
        }(startY, endY)
    }
    
    wg.Wait()
    return img
}
```

**Expected Speedup:** 3-5x faster (from ~1ms to ~0.2-0.3ms per call)  
**Total Batch Impact:** Saves ~17-20ms per batch of 25 frames  
**FPS Improvement:** +1.5-2 FPS

---

### 2. **Parallelize `resizeImagePooled()` Bilinear Interpolation** ⭐⭐⭐⭐

**Current Code:** `helpers.go:153-190`
```go
// Sequential bilinear interpolation
for dstY := 0; dstY < targetHeight; dstY++ {
    for dstX := 0; dstX < targetWidth; dstX++ {
        // Complex bilinear calculation per pixel
        srcX := float32(dstX) * xRatio
        srcY := float32(dstY) * yRatio
        // ... interpolation logic ...
    }
}
```

**Impact:** Called 25 times per batch, variable dimensions (typically ~300×200)

**Optimization:**
```go
func resizeImagePooled(src *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
    srcBounds := src.Bounds()
    srcWidth := srcBounds.Dx()
    srcHeight := srcBounds.Dy()
    
    dst := rgbaPoolResize.Get().(*image.RGBA)
    
    if targetWidth > 400 || targetHeight > 400 {
        return resizeImage(src, targetWidth, targetHeight)
    }
    
    xRatio := float32(srcWidth) / float32(targetWidth)
    yRatio := float32(srcHeight) / float32(targetHeight)
    
    // Parallel row processing
    const numWorkers = 8
    rowsPerWorker := targetHeight / numWorkers
    var wg sync.WaitGroup
    
    for worker := 0; worker < numWorkers; worker++ {
        wg.Add(1)
        startY := worker * rowsPerWorker
        endY := startY + rowsPerWorker
        if worker == numWorkers-1 {
            endY = targetHeight
        }
        
        go func(yStart, yEnd int) {
            defer wg.Done()
            
            for dstY := yStart; dstY < yEnd; dstY++ {
                for dstX := 0; dstX < targetWidth; dstX++ {
                    srcX := float32(dstX) * xRatio
                    srcY := float32(dstY) * yRatio
                    
                    x0 := int(srcX)
                    y0 := int(srcY)
                    x1 := x0 + 1
                    y1 := y0 + 1
                    
                    if x1 >= srcWidth {
                        x1 = srcWidth - 1
                    }
                    if y1 >= srcHeight {
                        y1 = srcHeight - 1
                    }
                    
                    xWeight := srcX - float32(x0)
                    yWeight := srcY - float32(y0)
                    
                    c00 := src.RGBAAt(x0, y0)
                    c10 := src.RGBAAt(x1, y0)
                    c01 := src.RGBAAt(x0, y1)
                    c11 := src.RGBAAt(x1, y1)
                    
                    r := bilinearInterp(c00.R, c10.R, c01.R, c11.R, xWeight, yWeight)
                    g := bilinearInterp(c00.G, c10.G, c01.G, c11.G, xWeight, yWeight)
                    b := bilinearInterp(c00.B, c10.B, c01.B, c11.B, xWeight, yWeight)
                    
                    dst.SetRGBA(dstX, dstY, color.RGBA{R: r, G: g, B: b, A: 255})
                }
            }
        }(startY, endY)
    }
    
    wg.Wait()
    return dst
}
```

**Expected Speedup:** 4-6x faster  
**Total Batch Impact:** Saves ~10-15ms per batch of 25 frames  
**FPS Improvement:** +1-1.5 FPS

---

### 3. **Parallelize Mel Window Extraction** ⭐⭐⭐

**Current Code:** `inference.go:109-169`
```go
// Sequential extraction of mel windows
for frameIdx := 0; frameIdx < numVideoFrames; frameIdx++ {
    startIdx := int(float64(80*frameIdx) / 25.0)
    endIdx := startIdx + 16
    
    window := melWindowPool.Get().([][]float32)
    
    // Extract and transpose window
    for step := 0; step < 16; step++ {
        srcIdx := startIdx + step
        if srcIdx >= numMelFrames {
            srcIdx = numMelFrames - 1
        }
        for m := 0; m < 80; m++ {
            window[m][step] = melSpec[srcIdx][m]
        }
    }
    
    allMelWindows = append(allMelWindows, window)
}
```

**Impact:** For batch 25: ~40 windows to extract (each 16×80 = 1,280 values)

**Optimization:**
```go
// Parallel window extraction
allMelWindows := make([][][]float32, numVideoFrames)
var wg sync.WaitGroup

for frameIdx := 0; frameIdx < numVideoFrames; frameIdx++ {
    wg.Add(1)
    go func(idx int) {
        defer wg.Done()
        
        startIdx := int(float64(80*idx) / 25.0)
        endIdx := startIdx + 16
        
        if endIdx > numMelFrames {
            endIdx = numMelFrames
            startIdx = endIdx - 16
        }
        if startIdx < 0 {
            startIdx = 0
        }
        
        // Allocate window (can't use pool safely with goroutines)
        window := make([][]float32, 80)
        for m := 0; m < 80; m++ {
            window[m] = make([]float32, 16)
        }
        
        for step := 0; step < 16; step++ {
            srcIdx := startIdx + step
            if srcIdx >= numMelFrames {
                srcIdx = numMelFrames - 1
            }
            for m := 0; m < 80; m++ {
                window[m][step] = melSpec[srcIdx][m]
            }
        }
        
        allMelWindows[idx] = window
    }(frameIdx)
}

wg.Wait()
```

**Expected Speedup:** 6-8x faster (trivially parallelizable)  
**Total Batch Impact:** Saves ~5-10ms per batch  
**FPS Improvement:** +0.5-1 FPS

---

## 🔧 MEDIUM-IMPACT OPTIMIZATIONS

### 4. **Optimize Audio Feature Zero-Padding** ⭐⭐⭐

**Current Code:** `inference.go:225-263`
```go
// Three separate loops for zero-padding
for i := 0; i < padLeft; i++ {
    destOffset := outputOffset + frameCounter*512
    for j := 0; j < 512; j++ {
        audioData[destOffset+j] = 0.0
    }
    frameCounter++
}
```

**Optimization:**
```go
// Use Go's built-in zero initialization
// Option 1: Use copy with pre-zeroed slice
var zeroBlock [512]float32 // Static zero block

for i := 0; i < padLeft; i++ {
    destOffset := outputOffset + frameCounter*512
    copy(audioData[destOffset:destOffset+512], zeroBlock[:])
    frameCounter++
}

// Option 2: Bulk zero with single operation
if padLeft > 0 {
    destStart := outputOffset + frameCounter*512
    destEnd := destStart + padLeft*512
    for i := destStart; i < destEnd; i++ {
        audioData[i] = 0.0
    }
    frameCounter += padLeft
}
```

**Expected Speedup:** 2-3x faster for padding operations  
**Total Batch Impact:** Saves ~2-5ms per batch  
**FPS Improvement:** +0.3-0.5 FPS

---

### 5. **SIMD-Optimized Pixel Format Conversion** ⭐⭐

**Approach:** Use Go's `golang.org/x/image/draw` package with SIMD intrinsics

**Current:** Manual pixel-by-pixel conversion  
**Alternative:** Use optimized draw operations

```go
import "golang.org/x/image/draw"

func outputToImageSIMD(outputData []float32) *image.RGBA {
    img := rgbaPool320.Get().(*image.RGBA)
    
    // Create temporary float image (if such type existed)
    // Then use draw.Draw with optimized scaler
    // This requires custom implementation or third-party library
    
    // For now, parallel approach in #1 is best native Go solution
    return img
}
```

**Note:** Go doesn't have native SIMD float→uint8 conversion in stdlib.  
Would require CGO or assembly for true SIMD gains.  
**Recommendation:** Stick with parallel approach (#1) for pure Go.

---

## 📊 PROFILING RECOMMENDATIONS

### 6. **Add Detailed Timing Breakdowns**

Add timing instrumentation to identify bottlenecks:

```go
// In compositeFrame()
t0 := time.Now()
mouthImg := outputToImage(mouthRegion)
t1 := time.Now()
resized := resizeImagePooled(mouthImg, w, h)
t2 := time.Now()
// ... rest of compositing ...
t3 := time.Now()

if s.cfg.Logging.LogCompositingTimes {
    log.Printf("🎨 Frame %d compositing breakdown: "+
        "outputToImage=%.2fms, resize=%.2fms, jpeg=%.2fms",
        frameIdx,
        t1.Sub(t0).Seconds()*1000,
        t2.Sub(t1).Seconds()*1000,
        t3.Sub(t2).Seconds()*1000)
}
```

---

## 💡 LOW-IMPACT / FUTURE OPTIMIZATIONS

### 7. **Consider Batch JPEG Encoding**

**Current:** Sequential JPEG encoding in goroutines  
**Alternative:** Group multiple frames and encode in parallel batches

**Complexity:** High  
**Gain:** Minimal (JPEG encoding already parallel via goroutines)

---

### 8. **GPU Acceleration for Image Operations**

**Idea:** Offload pixel conversion and resize to GPU  
**Approach:** Use CUDA/OpenGL compute shaders  

**Pros:**
- Massive parallelism (thousands of threads)
- GPU already active for inference

**Cons:**
- CPU→GPU→CPU transfer overhead
- Complexity of maintaining CUDA code
- Current CPU operations already fast enough

**Recommendation:** Only if targeting 100+ FPS

---

## 🚀 IMPLEMENTATION PRIORITY

### Phase 1: Quick Wins (1-2 hours work, ~3-5 FPS gain)
1. ✅ Parallelize `outputToImage()` - **Expected: +2 FPS**
2. ✅ Parallelize `resizeImagePooled()` - **Expected: +1.5 FPS**
3. ✅ Optimize zero-padding - **Expected: +0.5 FPS**

**Total Expected: +4 FPS → 47-48 FPS (batch 25)**

### Phase 2: Medium Effort (3-4 hours, +1-2 FPS gain)
4. ⏳ Parallelize mel window extraction - **Expected: +1 FPS**
5. ⏳ Add detailed profiling - **Identifies further bottlenecks**

**Total Expected: +5 FPS → 48-49 FPS (batch 25)**

### Phase 3: Advanced (1-2 days, diminishing returns)
6. ⏳ SIMD intrinsics (CGO/assembly)
7. ⏳ GPU acceleration for compositing

---

## ⚠️ IMPORTANT NOTES

### Thread Safety Considerations

**Memory Pools:** Current pools are thread-safe via `sync.Pool`  
**ONNX Sessions:** Already parallelized via AudioEncoderPool  
**Goroutine Overhead:** Minimal for image operations (amortized)

### Goroutine vs Serial Tradeoff

**Rule of Thumb:**
- **Parallelize if:** Loop iterations > 1,000 AND work per iteration > 100ns
- **Stay serial if:** Loop iterations < 100 OR setup overhead > work time

**Our Cases:**
- ✅ `outputToImage()`: 102,400 iterations × ~50ns = 5ms → **Worth parallelizing**
- ✅ `resizeImagePooled()`: 60,000+ iterations × ~100ns = 6ms → **Worth parallelizing**
- ✅ Mel extraction: 40 iterations × 250μs = 10ms → **Worth parallelizing**
- ⚠️ Zero-padding: 16 iterations × 50μs = 0.8ms → **Marginal benefit**

---

## 📈 EXPECTED PERFORMANCE AFTER OPTIMIZATIONS

| Metric | Current | Phase 1 | Phase 2 |
|--------|---------|---------|---------|
| Batch 8 FPS | 23.1 | 26-27 | 27-28 |
| Batch 25 FPS | 43.9 | 47-48 | 48-49 |
| Audio Proc | 106ms | 106ms | 100ms |
| Inference | 244ms | 244ms | 244ms |
| Compositing | 48ms | 30ms | 28ms |
| **Total** | **398ms** | **380ms** | **372ms** |

**Phase 1 Target:** 47-48 FPS (batch 25) - **Achievable in 2 hours**  
**Phase 2 Target:** 48-49 FPS (batch 25) - **Stretch goal**

---

## 🎯 NEXT STEPS

1. **Implement Phase 1 optimizations** (outputToImage + resizeImagePooled)
2. **Benchmark before/after** using existing test suite
3. **Validate correctness** (visual output quality unchanged)
4. **Profile with pprof** to identify any remaining bottlenecks
5. **Consider Phase 2** if further gains needed

---

## 📝 CODE QUALITY NOTES

**Strengths:**
- ✅ Clean separation of concerns (helpers.go vs inference.go)
- ✅ Excellent use of memory pools (zero allocations in hot path)
- ✅ Already parallelized at batch level (compositing goroutines)
- ✅ Well-commented, maintainable code

**Opportunities:**
- ⚠️ Some tight loops still sequential (pixel operations)
- ⚠️ Could benefit from more granular timing metrics
- ⚠️ No current CPU profiling instrumentation

**Overall Assessment:** 🌟🌟🌟🌟 (4/5 stars)

Code is production-quality and well-optimized. The suggested optimizations are "icing on the cake" rather than critical fixes. Current performance (43.9 FPS) is already excellent for this workload.

---

**Reviewed by:** GitHub Copilot  
**Last Updated:** October 28, 2025
