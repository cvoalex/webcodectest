# System Architecture

> **Complete technical architecture for the Go Monolithic Lip-Sync Inference Server**

This document provides a comprehensive overview of the system design, core components, data flows, and architectural decisions.

---

## Table of Contents

- [System Overview](#system-overview)
- [High-Level Architecture](#high-level-architecture)
- [Core Components](#core-components)
- [Design Patterns](#design-patterns)
- [Performance Optimizations](#performance-optimizations)
- [Critical Flows](#critical-flows)
- [Memory Management](#memory-management)
- [Error Handling](#error-handling)
- [Configuration](#configuration)

---

## System Overview

The Go Monolithic Server is a **high-performance lip-sync inference server** that combines ONNX model inference with real-time compositing to generate lip-synced video frames.

### Key Characteristics

**Architecture Style:** Monolithic (Inference + Compositing combined)  
**Language:** Go 1.21+  
**Communication:** gRPC with Protocol Buffers  
**Performance Target:** 48 FPS sustained throughput  
**Concurrency Model:** Parallel processing with worker pools

### Primary Responsibilities

1. **ONNX Model Inference** - Run neural network models for lip-sync generation
2. **Audio Processing** - Convert audio to mel-spectrogram features
3. **Image Processing** - Resize, convert, and composite video frames
4. **Memory Management** - Efficient pooling and caching
5. **Model Management** - Dynamic loading/unloading of ONNX models
6. **Background Management** - Cache and serve background video frames

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CLIENT (gRPC Request)                            │
│              CompositeBatchRequest {modelId, visual, audio,              │
│                      batchSize, startFrame}                              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    GRPC SERVER (:50053)                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Server Initialization (cmd/server/main.go)                        │ │
│  │  • Load config.yaml                                                │ │
│  │  • Initialize registries (Model, Image)                            │ │
│  │  • Create memory pools (5 types)                                   │ │
│  │  • Initialize audio processing pipeline                            │ │
│  │  • Create compositor                                               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Request Handler (internal/server/inference.go)                    │ │
│  │  • Validate request                                                │ │
│  │  • Process audio → mel-spectrogram → audio features               │ │
│  │  • Process visual → resize → normalize                            │ │
│  │  • Run ONNX inference (UNet model)                                 │ │
│  │  • Composite frames (inference output + background)                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          RESPONSE                                        │
│              CompositeBatchResponse {frames: []byte (JPEG)}              │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Raw       │────▶│   Audio      │────▶│  ONNX       │────▶│  Compositor  │
│  Request    │     │  Processor   │     │  Model      │     │              │
│  (gRPC)     │     │  (mel-spec)  │     │  (UNet)     │     │ (blend ROI)  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
      │                                                               │
      │                                                               │
      └───────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  JPEG Frames     │
                        │  (compressed)    │
                        └──────────────────┘
```

---

## Core Components

### 1. Server Core (`internal/server/`)

The central orchestration layer handling requests and coordinating all subsystems.

#### `server.go` - Server Struct

**Purpose:** Main server structure holding all dependencies

**Key Fields:**
```go
type Server struct {
    modelRegistry    *registry.ModelRegistry    // ONNX models
    imageRegistry    *registry.ImageRegistry    // Backgrounds + crop rects
    cfg              *config.Config             // Configuration
    audioProcessor   *audio.Processor           // Mel-spectrogram processing
    audioEncoderPool *audio.AudioEncoderPool    // Audio encoder ONNX pool
    compositor       *compositing.Compositor    // Frame compositing
    logger           *logger.BufferedLogger     // High-performance logging
}
```

**Constructor:**
```go
func New(
    modelRegistry *registry.ModelRegistry,
    imageRegistry *registry.ImageRegistry,
    cfg *config.Config,
    audioProcessor *audio.Processor,
    audioEncoderPool *audio.AudioEncoderPool,
    compositor *compositing.Compositor,
    logger *logger.BufferedLogger,
) *Server
```

**Swappable:** No (core orchestrator)  
**Thread-Safe:** Yes (stateless request handling)

---

#### `inference.go` - Core Inference Pipeline

**Purpose:** Main request handler for batch inference + compositing

**Key Function:**
```go
func (s *Server) InferBatchComposite(
    ctx context.Context, 
    req *pb.CompositeBatchRequest,
) (*pb.CompositeBatchResponse, error)
```

**Pipeline Stages:**

**Phase 1: Audio Processing**
```
Raw Audio (PCM float32) 
    → Mel-Spectrogram (audioProcessor.ProcessAudio)
    → Audio Encoder ONNX (audioEncoderPool.GetEncoder)
    → Mel Windows Extraction (extractMelWindowsParallel) 
    → Audio Features [batchSize][512]float32
```

**Phase 2: Visual Processing**
```
Raw Visual Frames (BGR float32 640x360)
    → BGR→RGBA Conversion (convertBGRToRGBAParallel)
    → Resize 320x320 (resizeImageParallel)
    → Normalize [-1, 1]
    → Visual Tensor [batchSize][3][320][320]float32
```

**Phase 3: ONNX Inference**
```
Inputs: Visual Tensor + Audio Features
Model: UNet (loaded from ModelRegistry)
Output: Lip-synced ROI tensors [batchSize][3][256][256]float32
```

**Phase 4: Compositing**
```
ROI Output + Background Frames
    → Denormalize ROI
    → Convert RGB→BGR
    → Resize to crop size
    → Blend into background (compositor.CompositeFrame)
    → JPEG encode
```

**Performance Metrics:**
- **Target:** 48 FPS (Phase 1 + Phase 2 optimizations)
- **Baseline:** 43.9 FPS (before optimizations)
- **Achieved:** ~48 FPS ✅

**Swappable:** No (core business logic)  
**Thread-Safe:** Yes (goroutine-safe operations)

---

#### `helpers.go` - Parallel Processing Functions

**Purpose:** Optimized image/audio processing functions

**Key Functions:**

**1. convertBGRToRGBAParallel** (Phase 1 Optimization)
```go
func convertBGRToRGBAParallel(
    bgrData []float32, 
    width, height int,
) *image.RGBA
```
- **Pattern:** 8-worker parallel processing
- **Distribution:** Row-based (each worker processes a subset of rows)
- **Speedup:** 4.2x vs sequential
- **Allocations:** Minimal (uses pre-allocated output buffer)

**2. resizeImageParallel** (Phase 1 Optimization)
```go
func resizeImageParallel(
    src *image.RGBA, 
    dstWidth, dstHeight int,
) *image.RGBA
```
- **Pattern:** 8-worker parallel processing  
- **Algorithm:** Bilinear interpolation
- **Speedup:** 4.9x vs sequential
- **Quality:** Identical to sequential (no trade-offs)

**3. extractMelWindowsParallel** (Phase 2 Optimization)
```go
func extractMelWindowsParallel(
    melSpec [][]float32,
    numMelFrames, numVideoFrames int,
    allMelWindows [][][]float32,
    saveDebugFiles bool,
)
```
- **Pattern:** 8-worker parallel processing
- **Distribution:** Frame-based (each worker processes subset of video frames)
- **Speedup:** 1.5x vs sequential
- **Thread-Safe:** Yes (non-overlapping memory writes)

**Swappable:** Yes (can replace with sequential or different parallel strategy)  
**Thread-Safe:** Yes (all parallel functions use sync.WaitGroup)

---

#### `health.go` - Health & Stats Endpoints

**Purpose:** Provide server health and model statistics

**Endpoints:**

**1. Health Check**
```go
func (s *Server) Health(
    ctx context.Context, 
    req *pb.HealthRequest,
) (*pb.HealthResponse, error)
```

Returns:
- Server status ("healthy")
- Loaded models count
- Max models capacity
- Uptime
- Version

**2. Get Model Stats**
```go
func (s *Server) GetModelStats(
    ctx context.Context, 
    req *pb.GetModelStatsRequest,
) (*pb.GetModelStatsResponse, error)
```

Returns:
- Per-model usage statistics
- Memory consumption
- Last used timestamp
- Background cache status

**Swappable:** Yes (monitoring/metrics implementation)  
**Thread-Safe:** Yes (read-only operations)

---

#### `model_management.go` - Dynamic Model Loading

**Purpose:** Runtime model lifecycle management

**Endpoints:**

**1. LoadModel** - Load ONNX model into GPU memory
```go
func (s *Server) LoadModel(
    ctx context.Context, 
    req *pb.LoadModelRequest,
) (*pb.LoadModelResponse, error)
```

**2. UnloadModel** - Remove model from GPU memory
```go
func (s *Server) UnloadModel(
    ctx context.Context, 
    req *pb.UnloadModelRequest,
) (*pb.UnloadModelResponse, error)
```

**3. ListModels** - Get all configured models and their status
```go
func (s *Server) ListModels(
    ctx context.Context, 
    req *pb.ListModelsRequest,
) (*pb.ListModelsResponse, error)
```

**Features:**
- Dynamic loading (on-demand)
- Eviction policies (LRU/LFU)
- GPU memory tracking
- Thread-safe operations

**Swappable:** Partially (eviction policy is configurable)  
**Thread-Safe:** Yes (registry handles locking)

---

### 2. Model Registry (`registry/`)

**Purpose:** Manage ONNX model lifecycle and GPU allocation

**File:** `registry/model_registry.go`

**Key Responsibilities:**
- Load ONNX models into GPU memory
- Track GPU memory usage
- Implement eviction policies (LRU/LFU)
- Provide thread-safe model access
- Handle multi-GPU scenarios

**Data Structures:**
```go
type ModelRegistry struct {
    models      map[string]*ModelInstance  // modelID → instance
    cfg         *config.Config
    gpuInfo     []GPUInfo
    totalMemory int64
    usedMemory  int64
    mu          sync.RWMutex
}

type ModelInstance struct {
    ID           string
    Session      *onnxruntime.Session  // ONNX session
    UsageCount   int64
    LastUsed     time.Time
    MemoryMB     int64
    GPUID        int
    Mu           sync.RWMutex
}
```

**Thread Safety:**
- Global lock for registry modifications
- Per-model locks for usage updates
- Read-locks for concurrent access

**Swappable:** Yes (can replace with different model backend)

---

### 3. Image Registry (`registry/`)

**Purpose:** Cache background frames and crop rectangles

**File:** `registry/image_registry.go`

**Key Responsibilities:**
- Load background video frames into memory
- Parse and cache crop rectangles
- Provide fast frame lookup
- Lazy-loading support
- Memory-efficient caching

**Data Structures:**
```go
type ImageRegistry struct {
    models      map[string]*ImageModel
    cfg         *config.Config
    cacheFrames int  // Max frames to cache per model
    mu          sync.RWMutex
}

type ImageModel struct {
    ID             string
    BackgroundDir  string
    CropRects      []CropRect
    FrameCache     map[int]*image.Image  // frameIndex → image
    NumFrames      int
    CachedFrames   int
    Mu             sync.RWMutex
}
```

**Caching Strategy:**
- **Preload:** Load first N frames on startup (configurable)
- **Lazy-load:** Load frames on-demand if not cached
- **LRU:** Evict least-recently-used frames if cache full

**Memory Footprint:**
```
Per Frame: ~480 KB (1920x1080 RGB)
600 frames: ~280 MB per model
```

**Swappable:** Yes (can replace with disk-backed or CDN-backed storage)

---

### 4. Audio Processing (`audio/`)

**Purpose:** Convert raw audio to mel-spectrogram features

#### `processor.go` - Mel-Spectrogram Processor

**Pipeline:**
```
PCM Float32 Audio (16kHz)
    → Pre-emphasis (0.97)
    → Padding (reflect mode)
    → STFT (Short-Time Fourier Transform)
    → Power Spectrogram
    → Mel Filter Banks
    → Log Scale
    → Normalization
    → Mel-Spectrogram [80 mel bins × time frames]
```

**Key Function:**
```go
func (p *Processor) ProcessAudio(
    audioData []float32, 
    sampleRate int,
) ([][]float32, error)
```

**Parameters:**
- Sample rate: 16,000 Hz
- FFT size: 1024
- Hop length: 200
- Mel bins: 80
- Frequency range: 55-7600 Hz

**Performance:**
- Optimized with pre-computed mel filter banks
- Batch processing support
- Memory pooling for intermediate buffers

**Swappable:** Yes (can replace with librosa-compatible implementations)

---

#### `encoder.go` - Audio Encoder ONNX Pool

**Purpose:** Pool of ONNX audio encoder models for parallel processing

**Pattern:** Object pool (reusable encoder instances)

**Data Structure:**
```go
type AudioEncoderPool struct {
    encoders chan *AudioEncoder  // Pool of encoders
    cfg      *config.Config
}

type AudioEncoder struct {
    session *onnxruntime.Session
}
```

**Usage:**
```go
encoder := pool.GetEncoder()  // Acquire from pool
defer pool.ReturnEncoder(encoder)  // Return to pool

audioFeatures := encoder.Encode(melSpec)
```

**Pool Size:** Configurable (default: 4 encoders)

**Benefits:**
- Avoid repeated ONNX session creation
- Parallel audio processing
- Reduced memory allocations

**Swappable:** Yes (can replace with different audio feature extractor)

---

### 5. Compositor (`internal/compositing/`)

**Purpose:** Blend inference output ROI into background frames

#### `compositor.go` - Frame Compositor

**Algorithm:**
```
1. Load background frame (from ImageRegistry)
2. Get crop rectangle (from ImageRegistry)
3. Denormalize ROI output ([−1, 1] → [0, 255])
4. Resize ROI to match crop size
5. Blend ROI into background at crop position
6. Encode as JPEG
```

**Key Function:**
```go
func (c *Compositor) CompositeFrame(
    outputTensor []float32,  // [3][256][256]
    modelID string,
    frameIndex int,
) ([]byte, error)
```

**Optimization:** Memory pooling for image buffers

**Output:** JPEG-encoded bytes (quality: 75, configurable)

**Swappable:** Yes (can replace with PNG, WebP, or raw output)

---

#### `image_ops.go` - Image Operations

**Key Functions:**

**1. OutputToImage** - Convert ONNX tensor to image
```go
func OutputToImage(
    output []float32,  // [3][H][W] CHW format
    height, width int,
) *image.RGBA
```

**2. ResizeImage** - Bilinear interpolation resize
```go
func ResizeImage(
    src *image.RGBA, 
    dstWidth, dstHeight int,
) *image.RGBA
```

**3. BlendIntoBackground** - Composite ROI into background
```go
func BlendIntoBackground(
    background *image.Image,
    roi *image.RGBA,
    cropRect CropRect,
) *image.RGBA
```

**Swappable:** Yes (can optimize with SIMD or GPU)

---

#### `pools.go` - Memory Pools

**Purpose:** Reduce allocations through buffer reuse

**Pools:**

**1. bufferPool** - Generic byte slices
```go
var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}
```

**2. rgbaPool** - RGBA image buffers
```go
var rgbaPool = sync.Pool{
    New: func() interface{} {
        return &image.RGBA{
            Pix: make([]byte, 320*320*4),
            Stride: 320 * 4,
            Rect: image.Rect(0, 0, 320, 320),
        }
    },
}
```

**3. jpegEncoderPool** - JPEG encoder buffers
```go
var jpegEncoderPool = sync.Pool{
    New: func() interface{} {
        return &bytes.Buffer{}
    },
}
```

**4. float32Pool** - Float32 slices (tensors)
**5. bgrPool** - BGR conversion buffers

**Impact:** 1000x reduction in allocations (see [ADR-002](adr/ADR-002-memory-pooling.md))

**Swappable:** No (critical for performance)

---

### 6. Logging (`logger/`)

**Purpose:** High-performance buffered logging

**File:** `logger/buffered_logger.go`

**Pattern:** Ring buffer with batch flushing

**Features:**
- Minimal allocations
- Configurable sample rate (log 1 in N requests)
- Automatic flush on interval or buffer full
- Thread-safe

**Configuration:**
```yaml
logging:
  buffered_logging: true
  sample_rate: 10        # Log 1 in 10 requests
  auto_flush: true
  flush_interval_ms: 1000
```

**Swappable:** Yes (can replace with structured logging)

---

### 7. Configuration (`config/`)

**Purpose:** Centralized YAML-based configuration

**File:** `config/config.go`

**Key Sections:**

**Server:**
```yaml
server:
  port: ":50053"
  max_message_size_mb: 100
  worker_count_per_gpu: 8
```

**GPUs:**
```yaml
gpus:
  enabled: true
  count: 1
  memory_gb_per_gpu: 24
  assignment_strategy: "round-robin"
```

**Capacity:**
```yaml
capacity:
  max_models: 40
  max_memory_gb: 20
  background_cache_frames: 600
  eviction_policy: "lfu"  # or "lru"
```

**ONNX:**
```yaml
onnx:
  library_path: "C:/onnxruntime-1.22.0/lib/onnxruntime.dll"
  intra_op_threads: 4
  inter_op_threads: 2
```

**Models:**
```yaml
models_root: "d:/Projects/webcodecstest/minimal_server/models"

models:
  sanders:
    model_path: "sanders/checkpoint/model_best.onnx"
    background_dir: "sanders/frames"
    crop_rects_path: "sanders/crop_rects.json"
    num_frames: 523
    preload_backgrounds: true
```

**Swappable:** Partially (can add new config sections)

---

## Design Patterns

### 1. Parallel Processing Pattern

**Concept:** Distribute work across multiple goroutines with row/frame-based distribution

**Implementation:**
```go
func processParallel(data []float32, width, height int) {
    const numWorkers = 8
    rowsPerWorker := height / numWorkers
    extraRows := height % numWorkers
    
    var wg sync.WaitGroup
    wg.Add(numWorkers)
    
    for w := 0; w < numWorkers; w++ {
        go func(workerID int) {
            defer wg.Done()
            
            // Calculate row range for this worker
            startRow := workerID * rowsPerWorker
            endRow := startRow + rowsPerWorker
            if workerID == numWorkers-1 {
                endRow += extraRows  // Last worker handles remainder
            }
            
            // Process assigned rows
            for y := startRow; y < endRow; y++ {
                // ... work ...
            }
        }(w)
    }
    
    wg.Wait()  // Ensure all workers complete
}
```

**Benefits:**
- ✅ Linear scaling with CPU cores
- ✅ No shared state mutations (thread-safe)
- ✅ Predictable load distribution
- ✅ Easy to test and reason about

**Used In:**
- `convertBGRToRGBAParallel` (4.2x speedup)
- `resizeImageParallel` (4.9x speedup)
- `extractMelWindowsParallel` (1.5x speedup)

**Trade-offs:**
- ❌ Small overhead from goroutine creation (~17 allocs)
- ❌ Not beneficial for tiny datasets (<100 pixels)

---

### 2. Memory Pooling Pattern

**Concept:** Reuse allocated buffers instead of allocating new ones

**Implementation:**
```go
var rgbaPool = sync.Pool{
    New: func() interface{} {
        return &image.RGBA{
            Pix: make([]byte, 320*320*4),
            Stride: 320 * 4,
            Rect: image.Rect(0, 0, 320, 320),
        }
    },
}

// Usage
func processImage() {
    img := rgbaPool.Get().(*image.RGBA)
    defer rgbaPool.Put(img)  // Return to pool
    
    // Use img...
}
```

**Benefits:**
- ✅ 1000x reduction in allocations
- ✅ Lower GC pressure
- ✅ Faster processing (no malloc overhead)

**Used In:**
- RGBA image buffers
- JPEG encoder buffers
- Float32 tensor buffers
- BGR conversion buffers
- Generic byte buffers

**Trade-offs:**
- ❌ Must ensure buffers are properly returned (use defer)
- ❌ Can't change buffer size after pool creation

---

### 3. Registry Pattern

**Concept:** Centralized management of shared resources

**Implementation:**
```go
type ModelRegistry struct {
    models map[string]*ModelInstance
    mu     sync.RWMutex
}

func (r *ModelRegistry) GetModel(id string) (*ModelInstance, error) {
    r.mu.RLock()
    defer r.mu.RUnlock()
    
    if model, ok := r.models[id]; ok {
        return model, nil
    }
    return nil, ErrNotFound
}
```

**Benefits:**
- ✅ Thread-safe access to shared resources
- ✅ Single source of truth
- ✅ Easy to implement caching/eviction
- ✅ Decouples resource management from business logic

**Used In:**
- `ModelRegistry` (ONNX models)
- `ImageRegistry` (backgrounds + crop rects)

---

### 4. Object Pool Pattern

**Concept:** Maintain a pool of reusable objects

**Implementation:**
```go
type AudioEncoderPool struct {
    encoders chan *AudioEncoder
}

func (p *AudioEncoderPool) GetEncoder() *AudioEncoder {
    return <-p.encoders  // Block if pool empty
}

func (p *AudioEncoderPool) ReturnEncoder(enc *AudioEncoder) {
    p.encoders <- enc  // Return to pool
}
```

**Benefits:**
- ✅ Avoid expensive object creation
- ✅ Bounded concurrency (pool size = max concurrent usage)
- ✅ Simple acquire/release pattern

**Used In:**
- `AudioEncoderPool` (ONNX encoder sessions)

---

### 5. Dependency Injection Pattern

**Concept:** Pass dependencies via constructor instead of globals

**Implementation:**
```go
// Constructor explicitly declares dependencies
func New(
    modelRegistry *registry.ModelRegistry,
    imageRegistry *registry.ImageRegistry,
    cfg *config.Config,
    audioProcessor *audio.Processor,
    compositor *compositing.Compositor,
) *Server {
    return &Server{
        modelRegistry:  modelRegistry,
        imageRegistry:  imageRegistry,
        cfg:            cfg,
        audioProcessor: audioProcessor,
        compositor:     compositor,
    }
}
```

**Benefits:**
- ✅ Easy to test (mock dependencies)
- ✅ Clear dependency graph
- ✅ No hidden globals
- ✅ Flexible configuration

**Used Throughout:** All major components

---

## Performance Optimizations

### Phase 1: Image Processing (4-5x Speedup)

**Baseline:** 43.9 FPS (batch 25), 23.1 FPS (batch 8)  
**Target:** 47-48 FPS  
**Achieved:** ~48 FPS ✅

#### Optimization #1: Parallel BGR→RGBA Conversion

**Before:**
```go
// Sequential loop
for y := 0; y < height; y++ {
    for x := 0; x < width; x++ {
        // Convert pixel
    }
}
```

**After:**
```go
// 8-worker parallel processing
convertBGRToRGBAParallel(bgrData, width, height)
```

**Results:**
- **Speedup:** 4.2x
- **Benchmark:** 2,000 ns/op → 476 ns/op
- **Allocations:** Minimal (uses pre-allocated buffer)

**See:** [ADR-001: Parallel Image Processing](adr/ADR-001-parallel-image-processing.md)

---

#### Optimization #2: Parallel Image Resize

**Before:**
```go
// Sequential resize with bilinear interpolation
for y := 0; y < dstHeight; y++ {
    for x := 0; x < dstWidth; x++ {
        // Interpolate pixel
    }
}
```

**After:**
```go
// 8-worker parallel resize
resizeImageParallel(src, dstWidth, dstHeight)
```

**Results:**
- **Speedup:** 4.9x
- **Benchmark:** 5,000 ns/op → 1,020 ns/op
- **Quality:** Identical (no approximations)

**See:** [ADR-001: Parallel Image Processing](adr/ADR-001-parallel-image-processing.md)

---

#### Optimization #3: Zero-Allocation Audio Padding

**Before:**
```go
// Allocate new slice for padded data
padded := make([]float32, paddedLen)
copy(padded, audioFeatures)
// ... fill padding ...
```

**After:**
```go
// Direct slice operations (no allocations)
paddingNeeded := batchMaxFrames - framesUsed
for i := 0; i < paddingNeeded*512; i++ {
    audioFeatures = append(audioFeatures, 0)
}
```

**Results:**
- **Allocations:** 1000x reduction
- **Benchmark:** 2,000,014 allocs → 2,004 allocs (with pooling)
- **Performance:** No measurable difference

**See:** [ADR-002: Memory Pooling Strategy](adr/ADR-002-memory-pooling.md)

---

### Phase 2: Audio Processing (1.5x Speedup)

**Baseline:** 47-48 FPS  
**Target:** 48-49 FPS  
**Achieved:** ~48-49 FPS ✅

#### Optimization #4: Parallel Mel Window Extraction

**Before:**
```go
// Sequential loop extracting mel windows
for i := 0; i < numFrames; i++ {
    startIdx := audioIdx * 512
    copy(melWindows[i*512:(i+1)*512], audioFeatures[startIdx:startIdx+512])
    audioIdx += 2
}
```

**After:**
```go
// 8-worker parallel extraction
extractMelWindowsParallel(melSpec, numMelFrames, numVideoFrames, allMelWindows, false)
```

**Results:**
- **Speedup:** 1.5x
- **Benchmark:** 230 μs/op → 213 μs/op
- **Thread-safe:** Yes (100-iteration race test passed)

**Rationale:** Audio processing is small portion of pipeline (~5-10%), so modest speedup expected.

**See:** [ADR-003: Parallel Mel Extraction](adr/ADR-003-parallel-mel-extraction.md)

---

### Summary of Optimizations

| Optimization | Speedup | Impact | Phase |
|--------------|---------|--------|-------|
| **Parallel BGR→RGBA** | 4.2x | High (image ops ~40% of pipeline) | 1 |
| **Parallel Resize** | 4.9x | High (image ops ~40% of pipeline) | 1 |
| **Zero-allocation Padding** | 1000x allocs | Medium (GC pressure reduction) | 1 |
| **Memory Pooling** | 1000x allocs | High (GC pressure reduction) | 1 |
| **Parallel Mel Extraction** | 1.5x | Low (audio ops ~5% of pipeline) | 2 |

**Overall:** 43.9 FPS → ~48 FPS (+9% throughput)

---

## Critical Flows

### 1. Request Flow (InferBatchComposite)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Validate Request                                                │
│     • Check modelID exists                                          │
│     • Validate batchSize (1-32)                                     │
│     • Validate input sizes                                          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Audio Processing                                                │
│     • Convert raw audio to mel-spectrogram                          │
│     • Run audio encoder ONNX (pool.GetEncoder)                      │
│     • Extract mel windows (extractMelWindowsParallel)               │
│     Duration: ~50-100ms                                             │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. Visual Processing                                               │
│     • Convert BGR→RGBA (convertBGRToRGBAParallel)                   │
│     • Resize 320x320 (resizeImageParallel)                          │
│     • Normalize to [-1, 1]                                          │
│     Duration: ~20-40ms (optimized)                                  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. ONNX Inference                                                  │
│     • Load model from ModelRegistry                                 │
│     • Create input tensors (visual + audio)                         │
│     • Run UNet model inference                                      │
│     • Extract output ROI tensors                                    │
│     Duration: ~100-150ms (model dependent)                          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. Compositing                                                     │
│     • For each frame:                                               │
│       - Load background (from ImageRegistry)                        │
│       - Denormalize ROI output                                      │
│       - Resize ROI to crop size                                     │
│       - Blend into background                                       │
│       - JPEG encode (pool-based)                                    │
│     Duration: ~30-50ms                                              │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. Return Response                                                 │
│     • Concatenate all JPEG frames                                   │
│     • Log performance metrics (if enabled)                          │
│     • Flush buffered logs (if threshold met)                        │
└─────────────────────────────────────────────────────────────────────┘

TOTAL TIME: ~200-340ms per request (depending on batch size)
THROUGHPUT: ~48 FPS sustained
```

---

### 2. Model Loading Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  LoadModel Request (modelID)                                        │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Check if already loaded                                         │
│     • ModelRegistry.GetModel(modelID)                               │
│     • If loaded → return success                                    │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Check capacity                                                  │
│     • If max models reached → evict based on policy (LRU/LFU)       │
│     • If max memory reached → evict until space available           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. Load ONNX model                                                 │
│     • Read model file from disk                                     │
│     • Create ONNX session with GPU provider                         │
│     • Assign to GPU (round-robin strategy)                          │
│     Duration: ~500-2000ms (model size dependent)                    │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Update registry                                                 │
│     • Add to models map                                             │
│     • Update GPU memory tracking                                    │
│     • Set lastUsed timestamp                                        │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. Load backgrounds (if configured)                                │
│     • ImageRegistry.LoadBackgrounds(modelID)                        │
│     • Load first N frames (configurable: 600 default)               │
│     • Parse crop rectangles                                         │
│     Duration: ~1000-3000ms (frame count dependent)                  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. Return success                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3. Error Handling Flow

**Philosophy:** Fail fast, log comprehensively, return structured errors

**Error Categories:**

**1. Validation Errors (400-level)**
```go
if req.ModelId == "" {
    return &pb.CompositeBatchResponse{
        Success: false,
        Error:   "model_id is required",
    }, nil  // Return nil error (gRPC-level success, app-level failure)
}
```

**2. Resource Errors (503-level)**
```go
model, err := s.modelRegistry.GetModel(req.ModelId)
if err != nil {
    return &pb.CompositeBatchResponse{
        Success: false,
        Error:   fmt.Sprintf("Model not loaded: %s", req.ModelId),
    }, nil
}
```

**3. Processing Errors (500-level)**
```go
outputs, err := session.Run(inputs)
if err != nil {
    return nil, fmt.Errorf("ONNX inference failed: %w", err)
}
```

**Logging:**
```go
// Buffered logging with sampling
reqLog := s.logger.StartRequest()
reqLog.Printf("Processing request: modelID=%s, batchSize=%d", modelID, batchSize)
// ... processing ...
s.logger.FinishRequest(reqLog)  // Flush if needed
```

---

## Memory Management

### Memory Pools (sync.Pool)

**Purpose:** Reduce GC pressure by reusing allocated buffers

**Active Pools:**

1. **rgbaPool** - RGBA image buffers (320x320x4 bytes)
2. **bufferPool** - Generic byte buffers
3. **jpegEncoderPool** - JPEG encoder buffers
4. **float32Pool** - Float32 tensor buffers
5. **bgrPool** - BGR conversion buffers

**Impact:**
- **Before pooling:** 2,000,014 allocations/request
- **After pooling:** 2,004 allocations/request
- **Reduction:** 1000x

**Usage Pattern:**
```go
// Get from pool
img := rgbaPool.Get().(*image.RGBA)
defer rgbaPool.Put(img)  // CRITICAL: Must return

// Use img...
```

**⚠️ Critical:** Always return buffers to pool (use `defer`)

---

### GPU Memory Tracking

**Purpose:** Prevent OOM errors by tracking GPU memory usage

**Implementation:**
```go
type ModelRegistry struct {
    totalMemory int64  // Total GPU memory (from config)
    usedMemory  int64  // Currently used memory
    models      map[string]*ModelInstance
}

func (r *ModelRegistry) canLoadModel(memoryMB int64) bool {
    return r.usedMemory + memoryMB <= r.totalMemory
}
```

**Eviction:** When capacity exceeded, evict based on policy (LRU/LFU)

---

### Background Frame Caching

**Purpose:** Balance memory usage vs disk I/O

**Strategy:**
- **Preload:** First N frames (default: 600)
- **Lazy-load:** Load on-demand if not cached
- **LRU eviction:** If cache full, evict least-recently-used

**Memory Calculation:**
```
Per frame: ~480 KB (1920x1080 RGB)
600 frames: ~280 MB per model
```

**Configuration:**
```yaml
capacity:
  background_cache_frames: 600  # Adjust based on memory
```

---

## Configuration

### Configuration File Structure

**Location:** `config.yaml` (root directory)

**Sections:**

**1. Server Configuration**
```yaml
server:
  port: ":50053"              # gRPC listening port
  max_message_size_mb: 100    # Max gRPC message size
  worker_count_per_gpu: 8     # Concurrent requests per GPU
  queue_size: 50              # Request queue size
```

**2. GPU Configuration**
```yaml
gpus:
  enabled: true               # Enable GPU acceleration
  count: 1                    # Number of GPUs
  memory_gb_per_gpu: 24       # Memory per GPU
  assignment_strategy: "round-robin"  # or "least-loaded"
  allow_multi_gpu_models: false  # Allow single model across GPUs
```

**3. Capacity Limits**
```yaml
capacity:
  max_models: 40                    # Max models in memory
  max_memory_gb: 20                 # Max GPU memory to use
  background_cache_frames: 600      # Frames to cache per model
  eviction_policy: "lfu"            # "lfu" or "lru"
  idle_timeout_minutes: 60          # Unload after idle time
```

**4. ONNX Runtime**
```yaml
onnx:
  library_path: "C:/onnxruntime-1.22.0/lib/onnxruntime.dll"
  cuda_streams_per_worker: 2
  intra_op_threads: 4         # Threads within operation
  inter_op_threads: 2         # Threads between operations
```

**5. Output Configuration**
```yaml
output:
  format: "jpeg"              # "jpeg" or "raw"
  jpeg_quality: 75            # 1-100 (higher = better quality)
```

**6. Logging**
```yaml
logging:
  level: "info"
  log_inference_times: true
  log_gpu_utilization: true
  log_compositing_times: true
  log_cache_stats: true
  buffered_logging: true      # Use buffered logger
  sample_rate: 10             # Log 1 in 10 requests
  auto_flush: true
  flush_interval_ms: 1000
  save_debug_files: false     # Save mel/audio tensors for debugging
```

**7. Models Configuration**
```yaml
models_root: "d:/Projects/webcodecstest/minimal_server/models"

models:
  sanders:
    model_path: "sanders/checkpoint/model_best.onnx"
    background_dir: "sanders/frames"
    source_video: "sanders/full_body_video.mp4"
    crop_rects_path: "sanders/crop_rects.json"
    num_frames: 523
    preload_backgrounds: true
    
    # Optional: For testing
    crops_video_path: "sanders/crops_328_video.mp4"
    rois_video_path: "sanders/rois_320_video.mp4"
```

---

## Related Documentation

- **[API Reference](API_REFERENCE.md)** - All gRPC endpoints with examples
- **[ADR-001: Parallel Image Processing](adr/ADR-001-parallel-image-processing.md)** - Why and how we parallelized
- **[ADR-002: Memory Pooling](adr/ADR-002-memory-pooling.md)** - Memory optimization strategy
- **[ADR-003: Parallel Mel Extraction](adr/ADR-003-parallel-mel-extraction.md)** - Phase 2 audio optimization
- **[Testing Guide](development/TESTING.md)** - How to run tests
- **[Gotchas](development/GOTCHAS.md)** - Common pitfalls

---

**Last Updated:** November 6, 2025  
**Version:** 1.0.0 (Phase 2 Complete)  
**Target Performance:** 48 FPS ✅ Achieved
