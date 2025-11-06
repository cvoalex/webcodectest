# Common Gotchas

> **Pitfalls, traps, and subtle issues to avoid when working with the Go Monolithic Lip-Sync Server**

This document catalogs common mistakes, debugging tips, and non-obvious behaviors you should know about.

---

## Table of Contents

- [Parallel Processing](#parallel-processing)
- [Memory Management](#memory-management)
- [ONNX Runtime](#onnx-runtime)
- [Audio Processing](#audio-processing)
- [Image Processing](#image-processing)
- [Configuration](#configuration)
- [gRPC](#grpc)
- [Performance](#performance)
- [Race Conditions](#race-conditions)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Parallel Processing

### ⚠️ Goroutine Leaks from Missing WaitGroup

**Problem:**
```go
// ❌ WRONG: Goroutines may not complete
for i := 0; i < batchSize; i++ {
    go processFrame(i)
}
// Continues without waiting!
```

**Solution:**
```go
// ✅ CORRECT: Wait for all goroutines
var wg sync.WaitGroup
for i := 0; i < batchSize; i++ {
    wg.Add(1)
    go func(idx int) {
        defer wg.Done()
        processFrame(idx)
    }(i)
}
wg.Wait()  // Wait for completion
```

**Detection:**
```powershell
# Run with race detector
go test -race ./...
```

---

### ⚠️ Closure Variable Capture in Goroutines

**Problem:**
```go
// ❌ WRONG: All goroutines see final value of 'i'
for i := 0; i < 10; i++ {
    go func() {
        processFrame(i)  // i == 10 for all goroutines!
    }()
}
```

**Why it fails:**
- `i` is captured by reference, not value
- By the time goroutines run, loop has finished
- All goroutines see `i == 10`

**Solution:**
```go
// ✅ CORRECT: Pass variable as parameter
for i := 0; i < 10; i++ {
    go func(idx int) {
        processFrame(idx)  // idx is a copy
    }(i)  // Pass i by value
}
```

---

### ⚠️ Incorrect Row Distribution

**Problem:**
```go
// ❌ WRONG: Integer division loses remainder
rowsPerWorker := totalRows / numWorkers
for w := 0; w < numWorkers; w++ {
    startRow := w * rowsPerWorker
    endRow := (w + 1) * rowsPerWorker
    // Last worker misses remaining rows!
}
```

**Example:**
```
320 rows ÷ 8 workers = 40 rows/worker
Workers process rows 0-319? NO!
Workers process rows 0-319? NO!
Last worker: 280-320 (should be 280-319)
Rows 280-319 processed, but 320 missed!
```

**Solution:**
```go
// ✅ CORRECT: Last worker gets remaining rows
for w := 0; w < numWorkers; w++ {
    startRow := w * rowsPerWorker
    endRow := (w + 1) * rowsPerWorker
    if w == numWorkers-1 {
        endRow = totalRows  // Ensure all rows covered
    }
    // ...
}
```

---

### ⚠️ Deadlock from Unbuffered Channels

**Problem:**
```go
// ❌ WRONG: Deadlock if no receiver ready
results := make(chan Result)  // Unbuffered
for i := 0; i < 100; i++ {
    go func(idx int) {
        results <- processFrame(idx)  // Blocks if channel full!
    }(i)
}
// Deadlock: all goroutines waiting to send
```

**Solution:**
```go
// ✅ CORRECT: Use buffered channel
results := make(chan Result, 100)  // Buffer size = num goroutines
for i := 0; i < 100; i++ {
    go func(idx int) {
        results <- processFrame(idx)
    }(i)
}

// Collect results
for i := 0; i < 100; i++ {
    result := <-results
}
```

---

## Memory Management

### ⚠️ Not Using Memory Pools

**Problem:**
```go
// ❌ WRONG: Allocates new buffer every frame
func processFrame() {
    buffer := make([]float32, 320*320*3)  // 1000+ allocs/sec
    // process...
}
```

**Performance Impact:**
```
Allocations: 10,000 per second
GC pressure: High
Latency:     150ms → 300ms (2x slower)
```

**Solution:**
```go
// ✅ CORRECT: Use sync.Pool
var bufferPool = sync.Pool{
    New: func() interface{} {
        return make([]float32, 320*320*3)
    },
}

func processFrame() {
    buffer := bufferPool.Get().([]float32)
    defer bufferPool.Put(buffer)  // Return to pool
    // process...
}
```

**Performance After:**
```
Allocations: 10 (99.9% reduction)
GC pressure: Minimal
Latency:     50ms ✅
```

---

### ⚠️ Pool Buffer Reuse Without Clearing

**Problem:**
```go
// ❌ WRONG: Reused buffer has stale data
buffer := bufferPool.Get().([]float32)
defer bufferPool.Put(buffer)

// If previous use set values, they persist!
// buffer[0:100] might have old data
```

**Solution:**
```go
// ✅ CORRECT: Clear buffer before use
buffer := bufferPool.Get().([]float32)
defer bufferPool.Put(buffer)

// Option 1: Clear manually
for i := range buffer {
    buffer[i] = 0
}

// Option 2: Use slice zeroing (faster)
for i := 0; i < len(buffer); i++ {
    buffer[i] = 0
}

// Option 3: Only clear needed section
copy(buffer[:neededSize], make([]float32, neededSize))
```

---

### ⚠️ Memory Leaks from Unclosed Resources

**Problem:**
```go
// ❌ WRONG: ONNX session never freed
session := ort.CreateSession(modelPath)
// If function returns early (error), session leaks!
```

**Solution:**
```go
// ✅ CORRECT: Always defer cleanup
session, err := ort.CreateSession(modelPath)
if err != nil {
    return err
}
defer session.Destroy()  // Freed even if panic/error
```

---

### ⚠️ Large Slice Copying Performance

**Problem:**
```go
// ❌ WRONG: Slow manual copy
src := []float32{...}  // 1M elements
dst := make([]float32, len(src))
for i := 0; i < len(src); i++ {
    dst[i] = src[i]  // Slow!
}
```

**Solution:**
```go
// ✅ CORRECT: Use built-in copy (10x faster)
dst := make([]float32, len(src))
copy(dst, src)  // Optimized assembly
```

---

## ONNX Runtime

### ⚠️ Model Not Found Error

**Error:**
```
Error loading model: model file not found: models/sanders_unet_328.onnx
```

**Causes:**
1. **Wrong working directory:**
```powershell
# ❌ WRONG: Running from wrong directory
cd go-monolithic-server-refactored/cmd/server
go run main.go
# Looks for: cmd/server/models/... (doesn't exist!)
```

2. **Incorrect models_root in config.yaml:**
```yaml
# ❌ WRONG: Absolute path not set
models_root: "models"  # Relative path fragile
```

**Solution:**
```powershell
# ✅ CORRECT: Run from project root
cd go-monolithic-server-refactored
go run cmd/server/main.go
```

```yaml
# ✅ CORRECT: Use absolute path
models_root: "D:/Projects/webcodecstest/model"
```

---

### ⚠️ ONNX Tensor Shape Mismatch

**Error:**
```
ONNX inference failed: shape mismatch
Expected: [1,6,320,320]
Got:      [1,320,320,6]
```

**Problem:**
```go
// ❌ WRONG: Wrong dimension order
tensor := ort.NewTensor(
    []int64{1, 320, 320, 6},  // WRONG ORDER!
    data,
)
```

**Solution:**
```go
// ✅ CORRECT: Match model's expected shape
// Most models expect: [batch, channels, height, width]
tensor := ort.NewTensor(
    []int64{1, 6, 320, 320},  // CORRECT ORDER
    data,
)
```

**How to Check Expected Shape:**
```python
# Use Netron to visualize model
# https://netron.app/
# Or:
import onnx
model = onnx.load("model.onnx")
for input in model.graph.input:
    print(f"{input.name}: {input.type.tensor_type.shape}")
```

---

### ⚠️ GPU Out of Memory

**Error:**
```
CUDA error: out of memory
```

**Causes:**
1. **Too many models loaded:**
```go
// ❌ WRONG: Loading all models at startup
for _, modelID := range allModels {
    registry.LoadModel(modelID)  // 40 models × 2GB = 80GB!
}
```

2. **Large batch size:**
```go
// ❌ WRONG: Batch too large for GPU
batchSize := 100  // Requires 50GB GPU memory!
```

**Solution:**
```go
// ✅ CORRECT: Load on-demand
// Only load models when needed
// Implement LRU eviction policy

// ✅ CORRECT: Use reasonable batch size
batchSize := 25  // ~5GB GPU memory
if gpuMemoryGB < 8 {
    batchSize = 8  // Reduce for smaller GPUs
}
```

---

### ⚠️ ONNX Session Not Thread-Safe

**Problem:**
```go
// ❌ WRONG: Multiple goroutines using same session
var session *ort.Session

func processFrames() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(idx int) {
            defer wg.Done()
            session.Run(inputs)  // RACE CONDITION!
        }(i)
    }
    wg.Wait()
}
```

**Solution:**
```go
// ✅ CORRECT: Use mutex to serialize access
var sessionMutex sync.Mutex

func runInference(session *ort.Session, inputs map[string]interface{}) {
    sessionMutex.Lock()
    defer sessionMutex.Unlock()
    
    outputs := session.Run(inputs)
    return outputs
}
```

**Better Solution:**
```go
// ✅ BETTER: Create separate sessions per goroutine
// (if memory allows)
func processFrames() {
    sessions := make([]*ort.Session, numWorkers)
    for i := 0; i < numWorkers; i++ {
        sessions[i] = ort.CreateSession(modelPath)
        defer sessions[i].Destroy()
    }
    
    // Each worker uses its own session (no mutex needed)
}
```

---

## Audio Processing

### ⚠️ Audio Length Mismatch

**Error:**
```
Expected 10,240 audio samples (640ms @ 16kHz)
Got: 8,000 samples
```

**Causes:**
1. **Wrong sample rate:**
```go
// ❌ WRONG: Using 44.1kHz audio
sampleRate := 44100  // Should be 16000!
```

2. **Wrong duration:**
```go
// ❌ WRONG: Sending 500ms instead of 640ms
duration := 500 * time.Millisecond  // Should be 640ms
```

**Solution:**
```go
// ✅ CORRECT: Verify sample rate and duration
const (
    sampleRate = 16000
    windowDuration = 640 * time.Millisecond
    expectedSamples = 10240  // 16000 * 0.64
)

if len(audioSamples) != expectedSamples {
    log.Printf("Zero-padding audio: %d → %d samples", 
        len(audioSamples), expectedSamples)
    audioSamples = zeroPad(audioSamples, expectedSamples)
}
```

---

### ⚠️ Mel Spectrogram Dimension Errors

**Error:**
```
Expected mel shape: [16][80]
Got: [80][16]
```

**Problem:**
```go
// ❌ WRONG: Dimensions transposed
melSpec := extractMelSpectrogram(audio)  // Returns [80][16]
```

**Solution:**
```go
// ✅ CORRECT: Verify shape matches expected
// [num_frames][num_mel_bands] = [16][80]
melSpec := extractMelSpectrogram(audio)
if len(melSpec) != 16 || len(melSpec[0]) != 80 {
    return fmt.Errorf("invalid mel shape: [%d][%d]", 
        len(melSpec), len(melSpec[0]))
}
```

---

### ⚠️ Window Extraction Out-of-Bounds

**Error:**
```
panic: runtime error: index out of range [100] with length 100
```

**Problem:**
```go
// ❌ WRONG: Window extends past end
frameIdx := 90
windowSize := 16
window := melSpec[frameIdx : frameIdx+windowSize]  // Panic if melSpec has 100 frames!
```

**Solution:**
```go
// ✅ CORRECT: Validate bounds
if frameIdx+windowSize > len(melSpec) {
    return fmt.Errorf("window out of bounds: frame %d, mel length %d", 
        frameIdx, len(melSpec))
}
window := melSpec[frameIdx : frameIdx+windowSize]
```

---

## Image Processing

### ⚠️ BGR vs RGB Confusion

**Problem:**
```go
// ❌ WRONG: Assuming RGB order
r := pixel[0]  // Actually Blue!
g := pixel[1]  // Green (correct)
b := pixel[2]  // Actually Red!
```

**Why:**
- OpenCV uses BGR order (not RGB)
- Visual frames come from Python (OpenCV)
- Must convert BGR → RGBA for correct colors

**Solution:**
```go
// ✅ CORRECT: BGR input order
func convertBGRToRGBA(bgr []byte, width, height int) []byte {
    rgba := make([]byte, width*height*4)
    
    for i := 0; i < width*height; i++ {
        rgba[i*4+0] = bgr[i*3+2]  // R = BGR[2]
        rgba[i*4+1] = bgr[i*3+1]  // G = BGR[1]
        rgba[i*4+2] = bgr[i*3+0]  // B = BGR[0]
        rgba[i*4+3] = 255          // A = 255
    }
    
    return rgba
}
```

---

### ⚠️ Integer Overflow in Color Calculations

**Problem:**
```go
// ❌ WRONG: Can overflow uint8
r := byte((int(r1) + int(r2)) / 2)  // Overflow if r1+r2 > 255
```

**Example:**
```
r1 = 200
r2 = 200
r1 + r2 = 400  // Overflow! wraps to 144
result = 144 / 2 = 72  (WRONG! should be 200)
```

**Solution:**
```go
// ✅ CORRECT: Use int for intermediate calculations
r := int(r1) + int(r2)
if r > 255 {
    r = 255  // Clamp
}
result := byte(r / 2)
```

---

### ⚠️ Bilinear Interpolation Bounds Errors

**Problem:**
```go
// ❌ WRONG: No bounds checking
x := int(srcX)
y := int(srcY)
pixel := img[y*width + x]  // Panic if x or y out of bounds!
```

**Solution:**
```go
// ✅ CORRECT: Clamp coordinates
func clamp(val, min, max int) int {
    if val < min {
        return min
    }
    if val > max {
        return max
    }
    return val
}

x := clamp(int(srcX), 0, width-1)
y := clamp(int(srcY), 0, height-1)
```

---

## Configuration

### ⚠️ YAML Indentation Errors

**Problem:**
```yaml
# ❌ WRONG: Inconsistent indentation
server:
  port: 50053
   log_level: "info"  # Mixed spaces/tabs!
models:
- model_id: "sanders"
  model_path: "sanders_unet_328.onnx"
    background_dir: "backgrounds"  # Misaligned!
```

**Error:**
```
yaml: line 4: mapping values are not allowed in this context
```

**Solution:**
```yaml
# ✅ CORRECT: Consistent 2-space indentation
server:
  port: 50053
  log_level: "info"

models:
  - model_id: "sanders"
    model_path: "sanders_unet_328.onnx"
    background_dir: "backgrounds"
```

---

### ⚠️ Relative Paths Break When Running from Different Directories

**Problem:**
```yaml
# ❌ WRONG: Relative paths fragile
models_root: "models"
background_root: "backgrounds"
```

```powershell
# Works from project root
cd go-monolithic-server-refactored
go run cmd/server/main.go  ✅

# Breaks from subdirectory
cd cmd/server
go run main.go  ❌ (models not found)
```

**Solution:**
```yaml
# ✅ CORRECT: Absolute paths
models_root: "D:/Projects/webcodecstest/model"
background_root: "D:/Projects/webcodecstest/backgrounds"
```

**Or use environment variable:**
```yaml
models_root: "${PROJECT_ROOT}/model"
```

---

### ⚠️ Missing Model Configuration

**Error:**
```
Model 'bob' not found in configuration
```

**Problem:**
```yaml
# ❌ WRONG: Model used but not configured
models:
  - model_id: "sanders"
    model_path: "sanders_unet_328.onnx"
# 'bob' model missing!
```

**Solution:**
```yaml
# ✅ CORRECT: Add all models
models:
  - model_id: "sanders"
    model_path: "sanders_unet_328.onnx"
    background_dir: "sanders_backgrounds"
    
  - model_id: "bob"
    model_path: "bob_unet_328.onnx"
    background_dir: "bob_backgrounds"
```

---

## gRPC

### ⚠️ Message Size Limit Exceeded

**Error:**
```
rpc error: code = ResourceExhausted desc = grpc: received message larger than max (15728640 vs 4194304)
```

**Problem:**
- Default gRPC max message size: 4 MB
- Batch of 25 frames: ~15 MB
- Request rejected!

**Solution:**
```go
// ✅ CORRECT: Increase message size limit (server)
opts := []grpc.ServerOption{
    grpc.MaxRecvMsgSize(100 * 1024 * 1024),  // 100 MB
    grpc.MaxSendMsgSize(100 * 1024 * 1024),  // 100 MB
}
server := grpc.NewServer(opts...)
```

```go
// ✅ CORRECT: Increase message size limit (client)
conn, err := grpc.Dial(
    "localhost:50053",
    grpc.WithInsecure(),
    grpc.WithDefaultCallOptions(
        grpc.MaxCallRecvMsgSize(100 * 1024 * 1024),
        grpc.MaxCallSendMsgSize(100 * 1024 * 1024),
    ),
)
```

---

### ⚠️ Context Timeout

**Error:**
```
rpc error: code = DeadlineExceeded desc = context deadline exceeded
```

**Problem:**
```go
// ❌ WRONG: Default timeout too short
ctx := context.Background()
resp, err := client.InferBatchComposite(ctx, req)  // Times out after default (varies)
```

**Solution:**
```go
// ✅ CORRECT: Set reasonable timeout
ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
defer cancel()

resp, err := client.InferBatchComposite(ctx, req)
```

---

### ⚠️ Connection Not Closed

**Problem:**
```go
// ❌ WRONG: Connection leaks
func sendRequest() {
    conn, _ := grpc.Dial("localhost:50053", grpc.WithInsecure())
    client := pb.NewMonolithicServiceClient(conn)
    client.InferBatchComposite(ctx, req)
    // Connection never closed!
}
```

**Solution:**
```go
// ✅ CORRECT: Always defer Close()
func sendRequest() {
    conn, err := grpc.Dial("localhost:50053", grpc.WithInsecure())
    if err != nil {
        return err
    }
    defer conn.Close()  // Cleanup guaranteed
    
    client := pb.NewMonolithicServiceClient(conn)
    client.InferBatchComposite(ctx, req)
}
```

---

## Performance

### ⚠️ Not Measuring Actual Performance

**Problem:**
```go
// ❌ WRONG: Guessing if optimization worked
// "I parallelized it, so it must be faster!"
```

**Solution:**
```go
// ✅ CORRECT: Always measure before/after
func measurePerformance() {
    // Baseline
    start := time.Now()
    processSequential()
    baseline := time.Since(start)
    
    // Optimized
    start = time.Now()
    processParallel()
    optimized := time.Since(start)
    
    speedup := float64(baseline) / float64(optimized)
    log.Printf("Speedup: %.2fx (%s → %s)", speedup, baseline, optimized)
}
```

---

### ⚠️ Too Many Goroutines

**Problem:**
```go
// ❌ WRONG: Creating goroutine per pixel!
for y := 0; y < 320; y++ {
    for x := 0; x < 320; x++ {
        go processPixel(x, y)  // 102,400 goroutines!
    }
}
```

**Performance Impact:**
```
Goroutines: 102,400
Context switches: Massive overhead
Performance: 10x SLOWER than sequential
```

**Solution:**
```go
// ✅ CORRECT: Reasonable number of workers
numWorkers := 8  // Match CPU cores
rowsPerWorker := 320 / numWorkers

var wg sync.WaitGroup
for w := 0; w < numWorkers; w++ {
    wg.Add(1)
    go func(workerID int) {
        defer wg.Done()
        startRow := workerID * rowsPerWorker
        endRow := (workerID + 1) * rowsPerWorker
        
        for y := startRow; y < endRow; y++ {
            for x := 0; x < 320; x++ {
                processPixel(x, y)
            }
        }
    }(w)
}
wg.Wait()
```

---

### ⚠️ False Sharing in Parallel Code

**Problem:**
```go
// ❌ WRONG: Workers writing to adjacent memory (cache line contention)
type WorkerStats struct {
    count int64  // Each worker updates its own counter
}

stats := make([]WorkerStats, numWorkers)

// Workers on different cores fight for same cache line!
```

**Solution:**
```go
// ✅ CORRECT: Pad structs to avoid false sharing
type WorkerStats struct {
    count int64
    _     [56]byte  // Padding to 64 bytes (cache line size)
}

stats := make([]WorkerStats, numWorkers)
```

---

## Race Conditions

### ⚠️ Data Race on Shared Map

**Problem:**
```go
// ❌ WRONG: Concurrent map access without synchronization
var cache = make(map[string][]byte)

func getOrLoad(key string) []byte {
    if val, ok := cache[key]; ok {  // RACE!
        return val
    }
    
    val := loadData(key)
    cache[key] = val  // RACE!
    return val
}
```

**Solution:**
```go
// ✅ CORRECT: Use sync.Map or mutex
var cacheMutex sync.RWMutex
var cache = make(map[string][]byte)

func getOrLoad(key string) []byte {
    // Read lock for checking
    cacheMutex.RLock()
    if val, ok := cache[key]; ok {
        cacheMutex.RUnlock()
        return val
    }
    cacheMutex.RUnlock()
    
    // Write lock for updating
    cacheMutex.Lock()
    defer cacheMutex.Unlock()
    
    // Double-check after acquiring write lock
    if val, ok := cache[key]; ok {
        return val
    }
    
    val := loadData(key)
    cache[key] = val
    return val
}
```

---

### ⚠️ Unprotected Counter Increment

**Problem:**
```go
// ❌ WRONG: Race condition on counter
var requestCount int

func handleRequest() {
    requestCount++  // RACE! Not atomic!
}
```

**Solution:**
```go
// ✅ CORRECT: Use atomic operations
var requestCount int64

func handleRequest() {
    atomic.AddInt64(&requestCount, 1)
}
```

---

## Testing

### ⚠️ Tests Depend on Execution Order

**Problem:**
```go
// ❌ WRONG: Test2 depends on Test1's side effects
var globalState int

func TestSetup(t *testing.T) {
    globalState = 42  // Test1 sets state
}

func TestUseState(t *testing.T) {
    if globalState != 42 {  // Test2 assumes Test1 ran first
        t.Fatal("State not set!")
    }
}
```

**Problem:**
- Tests run in random order (Go 1.7+)
- Test2 fails if run in isolation

**Solution:**
```go
// ✅ CORRECT: Each test self-contained
func TestUseState(t *testing.T) {
    // Setup state within test
    state := 42
    
    if state != 42 {
        t.Fatal("Test failed")
    }
}
```

---

### ⚠️ Not Using Table-Driven Tests

**Problem:**
```go
// ❌ WRONG: Repetitive test code
func TestResize1x1(t *testing.T) {
    input := createImage(1, 1)
    output := resize(input, 320, 320)
    if output.Width != 320 { t.Fatal() }
}

func TestResize100x100(t *testing.T) {
    input := createImage(100, 100)
    output := resize(input, 320, 320)
    if output.Width != 320 { t.Fatal() }
}
// ... 10 more similar tests
```

**Solution:**
```go
// ✅ CORRECT: Table-driven tests
func TestResize(t *testing.T) {
    tests := []struct {
        name string
        inW, inH int
        outW, outH int
    }{
        {"1x1", 1, 1, 320, 320},
        {"100x100", 100, 100, 320, 320},
        {"640x360", 640, 360, 320, 320},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            input := createImage(tt.inW, tt.inH)
            output := resize(input, tt.outW, tt.outH)
            if output.Width != tt.outW {
                t.Errorf("Width: got %d, want %d", output.Width, tt.outW)
            }
        })
    }
}
```

---

## Deployment

### ⚠️ Missing CUDA/cuDNN Dependencies

**Error:**
```
ONNX Runtime error: CUDA runtime library not found
```

**Problem:**
- CUDA 11.8+ required for ONNX GPU inference
- cuDNN library missing
- Incorrect CUDA version

**Solution:**
```powershell
# Verify CUDA installation
nvidia-smi

# Should show CUDA 11.8 or later
# If not, install from: https://developer.nvidia.com/cuda-downloads
```

---

### ⚠️ Port Already in Use

**Error:**
```
listen tcp :50053: bind: address already in use
```

**Problem:**
- Another process using port 50053
- Previous server instance still running

**Solution:**
```powershell
# Find process using port
netstat -ano | findstr :50053

# Kill process
taskkill /PID <PID> /F

# Or change port in config.yaml
server:
  port: 50054  # Use different port
```

---

### ⚠️ File Descriptors Exhausted

**Error:**
```
too many open files
```

**Problem:**
- Background frames not closed after loading
- Model files kept open
- Image files leaked

**Solution:**
```go
// ✅ CORRECT: Always close files
file, err := os.Open(path)
if err != nil {
    return err
}
defer file.Close()  // Guaranteed cleanup
```

---

## Quick Reference

### Common Errors Checklist

Before asking for help, check:

- [ ] Running from correct directory (project root)
- [ ] config.yaml paths are absolute
- [ ] All models exist in models_root
- [ ] Background frames loaded for model
- [ ] GPU has enough memory (check nvidia-smi)
- [ ] CUDA 11.8+ installed
- [ ] Port not already in use
- [ ] gRPC message size limits increased
- [ ] Using sync.Pool for buffers
- [ ] defer WaitGroup.Wait() in parallel code
- [ ] Passing loop variables to goroutines correctly
- [ ] Run tests with -race flag

---

## Debugging Tips

### Enable Verbose Logging

```yaml
# config.yaml
server:
  log_level: "debug"  # Shows detailed execution trace
```

### Profile Performance

```powershell
# CPU profiling
go test -cpuprofile=cpu.prof -bench=.

# Memory profiling
go test -memprofile=mem.prof -bench=.

# View profile
go tool pprof cpu.prof
```

### Detect Race Conditions

```powershell
# Build with race detector
go build -race cmd/server/main.go

# Run tests with race detector
go test -race ./...
```

---

## Related Documentation

- **[Architecture](../ARCHITECTURE.md)** - System design details
- **[Testing Guide](TESTING.md)** - How to write and run tests
- **[API Reference](../API_REFERENCE.md)** - gRPC endpoint documentation
- **[Development Guide](DEVELOPMENT_GUIDE.md)** - Setup and getting started

---

**Last Updated:** November 6, 2025  
**Covers:** 40+ common pitfalls and gotchas
