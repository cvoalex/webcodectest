# Memory Allocation Optimizations

## Summary
Implemented comprehensive memory pooling to eliminate per-frame allocations that were causing GC pressure and performance variance.

## Problems Identified

### Critical Issues
1. **Background Clone (207MB per batch 25)** - Line 704 in main.go
   - Each compositeFrame() allocated a full background image (1920×1080 RGBA = 8.3MB)
   - For batch 25: 25 × 8.3MB = **207MB per request**
   - For batch 8: 8 × 8.3MB = **66MB per request**

2. **Resize Image (4MB per batch 25)** - Line 775 in main.go
   - Each resize allocated new image.RGBA (~200×200 = 160KB typical)
   - For batch 25: 25 × 160KB = **4MB per request**

3. **Inference Output Buffers (30MB per batch 25)** - Line 145 in inferencer.go
   - Each Infer() call allocated new outputData (320×320×3 float32 = 1.2MB)
   - For batch 25: 25 × 1.2MB = **30MB per request**

### Total Allocation Per Request
- **Batch 25**: 207MB + 4MB + 30MB = **~241MB**
- **Batch 8**: 66MB + 1.3MB + 9.6MB = **~77MB**

These allocations happened in parallel goroutines, causing massive GC pressure.

## Solutions Implemented

### 1. Background Image Pooling
**File**: `cmd/server/main.go`

Added `rgbaPoolFullHD` sync.Pool for 1920×1080 images:
```go
var rgbaPoolFullHD = sync.Pool{
    New: func() interface{} {
        return image.NewRGBA(image.Rect(0, 0, 1920, 1080))
    },
}
```

Helper functions:
- `getPooledImageForSize()` - Gets appropriate pooled image
- `returnPooledImageForSize()` - Returns image to pool

**Impact**: Eliminates 207MB (batch 25) or 66MB (batch 8) allocations per request

### 2. Resize Image Pooling
**File**: `cmd/server/main.go`

Added `rgbaPoolResize` sync.Pool for resize operations:
```go
var rgbaPoolResize = sync.Pool{
    New: func() interface{} {
        return image.NewRGBA(image.Rect(0, 0, 400, 400))
    },
}
```

New function `resizeImagePooled()` uses pooled destination image.

**Impact**: Eliminates 4MB (batch 25) or 1.3MB (batch 8) allocations per request

### 3. Inference Output Buffer Pooling
**File**: `lipsyncinfer/inferencer.go`

Added `outputBufferPool` sync.Pool:
```go
var outputBufferPool = sync.Pool{
    New: func() interface{} {
        return make([]float32, 3*320*320)
    },
}
```

Modified:
- `Infer()` - Uses pooled buffer instead of `make([]float32, ...)`
- `InferBatch()` - Returns buffers to pool after copying

**Impact**: Eliminates 30MB (batch 25) or 9.6MB (batch 8) allocations per request

### 4. Existing Optimizations (Already Present)
- ✅ JPEG encoding buffer pool (`bufferPool`)
- ✅ 320×320 RGBA image pool (`rgbaPool320`)

## Expected Performance Improvements

### Memory Allocation Reduction
- **Before**: ~241MB allocated per batch 25 request
- **After**: ~0MB (all from pools, reused across requests)
- **Reduction**: 99.6% fewer allocations

### GC Pressure Reduction
- Eliminates 25+ major allocations per request
- Reduces GC pause frequency and duration
- More predictable latency

### Performance Gains
- Reduced memory allocation overhead
- Less CPU time spent in GC
- More consistent frame times
- Better throughput under sustained load

## Testing Plan

1. **Restart server** to clear any cached state
2. **Run batch 25 test** - Should see more consistent FPS (28-35 FPS stable)
3. **Run batch 8 test** - Should see improved FPS (closer to batch 25 efficiency)
4. **Monitor GC stats** - Check with `GODEBUG=gctrace=1` for reduced GC activity

## Code Changes

### Files Modified
1. `cmd/server/main.go`
   - Added `rgbaPoolFullHD` and `rgbaPoolResize` pools
   - Added `resizeImagePooled()`, `getPooledImageForSize()`, `returnPooledImageForSize()`
   - Updated `compositeFrame()` to use pooled images with deferred cleanup

2. `lipsyncinfer/inferencer.go`
   - Added `outputBufferPool` with sync.Pool
   - Updated `Infer()` to use pooled output buffer
   - Updated `InferBatch()` to return buffers to pool after copying

### Backward Compatibility
✅ All changes are internal optimizations
✅ No API changes
✅ No breaking changes to existing code

## Monitoring

To verify optimizations are working:

```powershell
# Run with GC trace to see reduced allocations
$env:GODEBUG="gctrace=1"
.\monolithic-server.exe

# In another terminal, run test
.\testing\test_batch_25_full.exe
```

Look for:
- Reduced "gc" lines in server output
- Smaller "heap" values
- Fewer "scvg" (scavenge) operations

## Future Optimizations

1. **Add more size-specific pools** for common background dimensions (1280×720, 2560×1440)
2. **Pool audio window slices** to reduce mel processing allocations
3. **Consider true GPU batching** by re-exporting ONNX model with dynamic batch size
4. **Profile with pprof** to identify any remaining allocation hotspots

## Benchmark Results

### Before Optimizations
- Batch 25: 28-35 FPS (variable due to GC pauses)
- Batch 8: 11-16 FPS (poor efficiency)
- Memory: ~241MB allocated per request

### After Optimizations
- *To be measured after deployment*
- Expected: More consistent FPS, reduced variance
- Expected: Batch 8 closer to batch 25 efficiency
- Expected: <1MB allocations per request
