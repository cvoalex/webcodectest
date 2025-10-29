# Monolithic Server - Request Flow Diagram

**Visual guide for understanding, maintaining, and debugging the inference pipeline**

---

## 📋 Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Request Flow Overview](#request-flow-overview)
3. [Audio Processing Pipeline](#audio-processing-pipeline)
4. [Visual Processing Pipeline](#visual-processing-pipeline)
5. [Inference Pipeline](#inference-pipeline)
6. [Compositing Pipeline](#compositing-pipeline)
7. [Memory Management Flow](#memory-management-flow)
8. [Error Handling Flow](#error-handling-flow)
9. [Debugging Guide](#debugging-guide)

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT (gRPC Request)                           │
│                 CompositeBatchRequest {modelId, visual,                 │
│                       audio, batchSize, startFrame}                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    GRPC SERVER (main.go)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Server Initialization                                           │  │
│  │  • Load config (port, models path, GPU IDs)                      │  │
│  │  • Initialize memory pools (5 types)                             │  │
│  │  • Create registries (model, image)                              │  │
│  │  • Initialize audio processor (8 STFT + 8 Mel workers)           │  │
│  │  • Create encoder pool (4 instances)                             │  │
│  │  • Start gRPC server with keepalive                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 INFERENCE SERVICE (inference.go)                        │
│                    InferBatchComposite()                                │
│  ┌──────────────┬──────────────┬──────────────┬──────────────────┐     │
│  │   Validate   │   Process    │   Inference  │   Composite      │     │
│  │   Input      │   Audio+     │   (GPU)      │   Frames         │     │
│  │              │   Visual     │              │   (Parallel)     │     │
│  └──────────────┴──────────────┴──────────────┴──────────────────┘     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  RESPONSE (CompositeBatchResponse)                      │
│         {success, compositedFrames[], timing metrics, error}            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request Flow Overview

### End-to-End Timeline

```
Client Request
     │
     ├─► [0ms] Validate Input
     │        ├─ Check modelId exists
     │        ├─ Verify visualFrames size (6 * 320 * 320 * 4 * batchSize)
     │        └─ Verify rawAudio size (16000 Hz * frames/25 * 4 bytes)
     │
     ├─► [0-25ms] Audio Processing ⚡ PARALLEL
     │        ├─ Convert bytes → float32 (zero-copy)
     │        ├─ STFT (8 workers, parallel windows)
     │        ├─ Mel-spectrogram (8 workers, parallel)
     │        ├─ Create 25fps windows (16 frames each, 50% overlap)
     │        ├─ Zero-pad to match batch size
     │        └─ Encode features (pooled encoder)
     │             Result: [batchSize, 32, 16, 16] audio features
     │
     ├─► [25-30ms] Visual Processing
     │        ├─ Parse visualFrames bytes
     │        ├─ Split: first half = crops, second half = ROIs
     │        ├─ Convert bytes → float32 (zero-copy)
     │        └─ Prepare for inference
     │             Result: [batchSize, 6, 320, 320] visual frames
     │
     ├─► [30-190ms] GPU Inference 🚀
     │        ├─ Load model from registry
     │        ├─ Prepare inputs (visual + audio features)
     │        ├─ Run inference on GPU
     │        └─ Get outputs: [batchSize, 3, 320, 320]
     │
     ├─► [190-195ms] Parallel Compositing ⚡ GOROUTINES
     │        ├─ Launch goroutine per frame
     │        ├─ Convert output → image (BGR→RGB)
     │        ├─ Resize 320x320 → 1920x1080 (pooled buffers)
     │        ├─ Encode JPEG (quality 95, pooled buffers)
     │        └─ Collect results
     │             Result: [batchSize] JPEG byte arrays
     │
     └─► [195ms] Return Response
              └─ CompositeBatchResponse with JPEGs + timing
```

---

## 🎵 Audio Processing Pipeline

### Detailed Flow

```
Raw Audio Bytes (client input)
     │
     │ Input: []byte of float32 values (little-endian)
     │ Size: batchSize * (16000/25) * 4 bytes
     │       = batchSize * 640 samples * 4 bytes
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Bytes → Float32 Conversion (helpers.go)               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  bytesToFloat32(audioBytes []byte) []float32              │ │
│  │  • Uses unsafe.Slice for ZERO-COPY conversion            │ │
│  │  • Reinterprets byte slice as float32 slice              │ │
│  │  • No memory allocation, instant                         │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: []float32 (16kHz audio samples)
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: STFT (Short-Time Fourier Transform)                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  audioProcessor.ComputeSTFT(samples, 16000)               │ │
│  │  • Window size: 400 samples (25ms at 16kHz)               │ │
│  │  • Hop length: 160 samples (10ms at 16kHz)                │ │
│  │  • Hanning window applied                                 │ │
│  │  • FFT per window → frequency bins                        │ │
│  │  • PARALLEL: 8 worker goroutines                          │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: [timeSteps, freqBins] complex spectrogram
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Mel-Spectrogram Conversion                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  audioProcessor.STFTToMel(stft, 16000, 80)                │ │
│  │  • Apply mel filter bank (80 mel bins)                    │ │
│  │  • Magnitude: sqrt(real² + imag²)                         │ │
│  │  • Power spectrum: magnitude²                             │ │
│  │  • Log scale: log(mel + 1e-5)                             │ │
│  │  • PARALLEL: 8 worker goroutines                          │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: [timeSteps, 80] mel-spectrogram
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Extract 25fps Windows (inference.go)                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  For each frame at 25fps:                                 │ │
│  │    frameTime = frameIdx / 25.0  (seconds)                 │ │
│  │    centerCol = frameTime * 100  (mel-spec columns)        │ │
│  │    Extract window: [centerCol-8 : centerCol+8]            │ │
│  │                    = 16 columns centered on frame         │ │
│  │                                                            │ │
│  │  Window structure (16 frames, 50% overlap):               │ │
│  │    Frame 0: columns [0:16]   ─┐                           │ │
│  │    Frame 1: columns [1:17]    ├─ 50% overlap (8 columns)  │ │
│  │    Frame 2: columns [2:18]   ─┘                           │ │
│  │    ...                                                     │ │
│  │                                                            │ │
│  │  Use POOLED mel window buffer (avoid allocation)          │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: [numWindows, 80, 16] mel windows
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Zero-Padding (if needed)                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  If numWindows < batchSize:                               │ │
│  │    padLeft = (batchSize - numWindows) / 2                 │ │
│  │    padRight = batchSize - numWindows - padLeft            │ │
│  │                                                            │ │
│  │    Result = [padLeft zeros] + [mel windows] + [padRight]  │ │
│  │                                                            │ │
│  │  Example (3 windows → batch 8):                           │ │
│  │    padLeft = 2, padRight = 3                              │ │
│  │    [0, 0, W1, W2, W3, 0, 0, 0]                            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: [batchSize, 80, 16] padded mel windows
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Audio Encoding (pooled encoder)                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  encoderPool.Get() → audio encoder instance               │ │
│  │  encoder.Run(melWindows) → audio features                 │ │
│  │  encoderPool.Put(encoder) → return to pool                │ │
│  │                                                            │ │
│  │  Pool: 4 encoder instances (avoid loading overhead)       │ │
│  │  Output shape: [batchSize, 32, 16, 16]                    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: [batchSize, 32, 16, 16] audio features
     ▼
Ready for Inference (combined with visual frames)
```

### Audio Processing Key Points

| Aspect | Details |
|--------|---------|
| **Input** | Raw audio bytes (16kHz, float32) |
| **STFT** | 8 parallel workers, 400-sample windows, 160-sample hop |
| **Mel** | 80 mel bins, 8 parallel workers, log scale |
| **Windows** | 16 frames per window, 50% overlap, 25fps timing |
| **Padding** | Center alignment with zeros if needed |
| **Encoder** | Pooled (4 instances), outputs [32, 16, 16] |
| **Time** | ~23ms total (parallel processing) |

---

## 👁️ Visual Processing Pipeline

### Detailed Flow

```
Visual Frames Bytes (client input)
     │
     │ Input: Flattened byte array
     │ Layout: [all crops] + [all ROIs]
     │ Size: batchSize * 2 * (3 * 320 * 320 * 4) bytes
     │       = batchSize * 6 channels * 320 * 320 * 4 bytes
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Parse Visual Frames (inference.go)                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Expected size = batchSize * 6 * 320 * 320 * 4            │ │
│  │               = batchSize * 1,228,800 bytes/frame          │ │
│  │                                                            │ │
│  │  Layout:                                                   │ │
│  │    Bytes [0 : halfSize]           = All crop faces        │ │
│  │    Bytes [halfSize : fullSize]    = All ROIs              │ │
│  │                                                            │ │
│  │  Where:                                                    │ │
│  │    cropSize = 3 * 320 * 320 * 4 = 1,228,800 bytes        │ │
│  │    halfSize = batchSize * cropSize                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │
     ├─► Crops:  visualFrames[0 : halfSize]
     │           ├─ Frame 0 crop: [0:1228800]
     │           ├─ Frame 1 crop: [1228800:2457600]
     │           └─ ... (all batch frames)
     │
     └─► ROIs:   visualFrames[halfSize : end]
                 ├─ Frame 0 ROI: [halfSize:halfSize+1228800]
                 ├─ Frame 1 ROI: [halfSize+1228800:...]
                 └─ ... (all batch frames)
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Combine Crops + ROIs (inference.go)                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  For each frame:                                           │ │
│  │    visualFrame = [crop (3 ch), ROI (3 ch)]                │ │
│  │                = 6 channels total                          │ │
│  │                                                            │ │
│  │  Structure per frame:                                      │ │
│  │    Channel 0: Crop R (320x320)                            │ │
│  │    Channel 1: Crop G (320x320)                            │ │
│  │    Channel 2: Crop B (320x320)                            │ │
│  │    Channel 3: ROI R (320x320)                             │ │
│  │    Channel 4: ROI G (320x320)                             │ │
│  │    Channel 5: ROI B (320x320)                             │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: [batchSize, 6, 320, 320] float32 tensor
     ▼
Ready for Inference (combined with audio features)
```

### Visual Frame Structure

```
┌──────────────────────────────────────────────────────────────┐
│  Single Visual Frame (6 channels, 320x320)                   │
│                                                              │
│  ┌────────────────────┐  ┌────────────────────┐             │
│  │  CROP FACE (3 ch)  │  │  ROI (3 ch)        │             │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │             │
│  │  │     R        │  │  │  │     R        │  │             │
│  │  │   320x320    │  │  │  │   320x320    │  │             │
│  │  └──────────────┘  │  │  └──────────────┘  │             │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │             │
│  │  │     G        │  │  │  │     G        │  │             │
│  │  │   320x320    │  │  │  │   320x320    │  │             │
│  │  └──────────────┘  │  │  └──────────────┘  │             │
│  │  ┌──────────────┐  │  │  ┌──────────────┐  │             │
│  │  │     B        │  │  │  │     B        │  │             │
│  │  │   320x320    │  │  │  │   320x320    │  │             │
│  │  └──────────────┘  │  │  └──────────────┘  │             │
│  └────────────────────┘  └────────────────────┘             │
│                                                              │
│  Purpose:                                                    │
│  • Crop: Detected face region (tight crop)                  │
│  • ROI: Region of Interest (context around face)            │
│  • Model uses both for better lip-sync synthesis            │
└──────────────────────────────────────────────────────────────┘
```

---

## 🧠 Inference Pipeline

### Model Execution Flow

```
Inputs Ready
     │
     ├─► Visual: [batchSize, 6, 320, 320] float32
     └─► Audio:  [batchSize, 32, 16, 16] float32
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Model (model_management.go)                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  modelRegistry.GetModel(modelId)                          │ │
│  │  • Check if model loaded in registry                      │ │
│  │  • If not found, return error                             │ │
│  │  • Model already on GPU (loaded at startup)               │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Model handle retrieved
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Prepare Input Tensors (inference.go)                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Input tensor map:                                         │ │
│  │    "visual_frames" → [batchSize, 6, 320, 320]             │ │
│  │    "audio_features" → [batchSize, 32, 16, 16]             │ │
│  │                                                            │ │
│  │  Tensors already in correct format (float32)              │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Inputs prepared
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: GPU Inference (ONNX Runtime)                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  model.Run(inputs)                                         │ │
│  │  • Execute on GPU (CUDA)                                   │ │
│  │  • Neural network forward pass                            │ │
│  │  • SyncTalk model architecture:                           │ │
│  │    - Visual encoder                                        │ │
│  │    - Audio encoder (already encoded)                      │ │
│  │    - Cross-modal fusion                                   │ │
│  │    - Decoder (generates lip-synced frames)                │ │
│  │                                                            │ │
│  │  Time: ~165ms for batch 8                                 │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Outputs generated
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Extract Outputs (inference.go)                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Output tensor: [batchSize, 3, 320, 320]                  │ │
│  │  • 3 channels: RGB                                         │ │
│  │  • 320x320: Output resolution                             │ │
│  │  • float32 values [0, 255] range                          │ │
│  │  • Each frame is lip-synced result                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: [batchSize, 3, 320, 320] float32
     ▼
Ready for Compositing (convert to JPEGs)
```

### Inference Key Points

| Aspect | Details |
|--------|---------|
| **Model Type** | ONNX (SyncTalk 2D lip-sync model) |
| **Runtime** | ONNX Runtime with CUDA provider |
| **GPU** | RTX 4090 (dev), RTX 6000 Blackwell Pro (prod) |
| **Batch Size** | 8 frames (default), up to 25 frames |
| **Input 1** | Visual frames [batch, 6, 320, 320] |
| **Input 2** | Audio features [batch, 32, 16, 16] |
| **Output** | RGB frames [batch, 3, 320, 320] |
| **Time** | ~165ms for batch 8 |

---

## 🎨 Compositing Pipeline

### Parallel Frame Processing

```
Inference Outputs: [batchSize, 3, 320, 320]
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Launch Parallel Goroutines (1 per frame)                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  for i := 0; i < batchSize; i++ {                         │ │
│  │      go compositeFrame(outputTensor[i], i, resultChan)    │ │
│  │  }                                                         │ │
│  │                                                            │ │
│  │  Each goroutine processes independently:                  │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │
     ├─► Goroutine 1: Frame 0 ──┐
     ├─► Goroutine 2: Frame 1   ├─► All run in PARALLEL
     ├─► Goroutine 3: Frame 2   │
     └─► ... (all frames)      ──┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Convert Output to Image (helpers.go)                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  outputToImage(tensor [3, 320, 320]) → *image.RGBA        │ │
│  │                                                            │ │
│  │  For each pixel (x, y):                                    │ │
│  │    R = clamp(tensor[0][y][x], 0, 255)                     │ │
│  │    G = clamp(tensor[1][y][x], 0, 255)                     │ │
│  │    B = clamp(tensor[2][y][x], 0, 255)                     │ │
│  │    A = 255 (fully opaque)                                  │ │
│  │                                                            │ │
│  │  NOTE: Tensor is BGR order, swap to RGB:                  │ │
│  │    img[R] = tensor[2]  (B→R swap)                         │ │
│  │    img[G] = tensor[1]  (G→G)                              │ │
│  │    img[B] = tensor[0]  (R→B swap)                         │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: *image.RGBA (320x320)
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Resize to Full HD (helpers.go)                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  resizeImagePooled(img, 1920, 1080) → *image.RGBA         │ │
│  │                                                            │ │
│  │  • Get pooled RGBA image (1920x1080) - ZERO ALLOCATION   │ │
│  │  • Bilinear interpolation for smooth scaling              │ │
│  │  • For each output pixel:                                 │ │
│  │      - Calculate source position (float)                  │ │
│  │      - Get 4 neighboring pixels                           │ │
│  │      - Interpolate color values                           │ │
│  │  • Return pooled buffer after use                         │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: *image.RGBA (1920x1080)
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Encode to JPEG (helpers.go)                           │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Get pooled buffer from bufferPool                        │ │
│  │  jpeg.Encode(buffer, img, quality: 95)                    │ │
│  │  • Quality 95: High quality, reasonable size              │ │
│  │  • Progressive encoding: Off                              │ │
│  │  • Typical size: ~50-100KB per frame                      │ │
│  │  Return pooled buffer after copying bytes                 │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: []byte (JPEG compressed)
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Send to Result Channel                                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  resultChan <- FrameResult{                                │ │
│  │      Index:      frameIdx,                                 │ │
│  │      JPEGBytes:  jpegData,                                 │ │
│  │      Error:      nil,                                      │ │
│  │  }                                                         │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Collect All Results (inference.go)                             │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  for i := 0; i < batchSize; i++ {                         │ │
│  │      result := <-resultChan                                │ │
│  │      compositedFrames[result.Index] = result.JPEGBytes    │ │
│  │  }                                                         │ │
│  │                                                            │ │
│  │  Wait for ALL goroutines to complete                      │ │
│  │  Results collected in correct order (by index)            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
     │ Output: [][]byte (JPEG frames in order)
     ▼
Return to Client
```

### Compositing Flow Diagram

```
    Inference Output
         │
         ├──────┬──────┬──────┬──────┬──────┬──────┬──────┐
         │      │      │      │      │      │      │      │
        F0     F1     F2     F3     F4     F5     F6     F7
         │      │      │      │      │      │      │      │
         ├─────→├─────→├─────→├─────→├─────→├─────→├─────→├──→ PARALLEL
         │      │      │      │      │      │      │      │
    ┌────▼───┐ │      │      │      │      │      │      │
    │BGR→RGB │ │      │      │      │      │      │      │
    │Converter│ │      │      │      │      │      │      │
    └────┬───┘ │      │      │      │      │      │      │
         │  ┌──▼───┐  │      │      │      │      │      │
         │  │BGR→RGB│ │      │      │      │      │      │
         │  └──┬───┘  │      │      │      │      │      │
         │     │  ┌───▼───┐  │      │      │      │      │
    ┌────▼────┐│  │BGR→RGB│ │      │      │      │      │
    │ Resize  ││  └───┬───┘  │      │      │      │      │
    │320→1920 ││      │  ┌───▼───┐  │      │      │      │
    │320→1080 ││ ┌────▼────┐│BR→RGB│ │      │      │      │
    └────┬────┘│ │ Resize  │└───┬───┘ │      │      │      │
         │     │ │320→1920 │    │ ┌───▼───┐ │      │      │
    ┌────▼───┐ │ │320→1080 │┌───▼────┐R→RGB│ │      │      │
    │  JPEG  │ │ └────┬────┘│ Resize │┬───┘  │      │      │
    │Encoder │ │      │     │320→1920││  ┌───▼───┐  │      │
    │Quality │ │ ┌────▼───┐ │320→1080││  │BR→RGB │  │      │
    │  95    │ │ │  JPEG  │ └────┬───┘│  └───┬───┘  │      │
    └────┬───┘ │ │Encoder │      │ ┌──▼────┐ │  ┌───▼───┐  │
         │     │ │Quality │ ┌────▼───┐esize │ │  │BR→RGB │  │
    ┌────▼───┐ │ │  95    │ │  JPEG  │20→1920│  └───┬───┘  │
    │ Result │ │ └────┬───┘ │Encoder │20→1080│  ┌───▼────┐  │
    │Channel │ │      │     │Quality │───┬───┘  │ Resize │  │
    └────────┘ │ ┌────▼───┐ │  95    │   │  ┌───▼────────┐  │
               │ │ Result │ └────┬───┘   │  │20→1920     │  │
               │ │Channel │      │   ┌───▼──▼─20→1080    │  │
               │ └────────┘ ┌────▼───┐ Result │───┬───┐   │  │
               │            │ Result │Channel │PEG │   │   │  │
               │            │Channel │────────┘coder   │   │  │
               │            └────────┘       Quality   │   │  │
               │                               95 ─────┘   │  │
               │                          ┌────┬───┐       │  │
               │                          │sult    │  ┌────▼──▼─┐
               │                          │annel   │  │ JPEG    │
               │                          └────────┘  │Encoder  │
               │                                      │Quality  │
               │                                      │  95     │
               │                                      └────┬────┘
               │                                           │
               │                                      ┌────▼────┐
               │                                      │ Result  │
               │                                      │Channel  │
               │                                      └─────────┘
               │
               ▼
    Collect All Results (ordered by index)
```

### Compositing Key Points

| Aspect | Details |
|--------|---------|
| **Parallelism** | 1 goroutine per frame (all parallel) |
| **Color Conv** | BGR→RGB swap during conversion |
| **Resize** | 320x320 → 1920x1080 (6x scale, bilinear) |
| **JPEG Quality** | 95 (high quality, ~50-100KB per frame) |
| **Memory Pools** | Pooled RGBA images, pooled buffers |
| **Time** | ~4ms total (parallel execution) |
| **Output** | Array of JPEG byte slices (ordered) |

---

## 💾 Memory Management Flow

### Pool Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MEMORY POOLS (constants.go)                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. bufferPool (bytes.Buffer pool)                      │   │
│  │     • Purpose: JPEG encoding buffers                    │   │
│  │     • Reuse: Get() → Encode → Put()                     │   │
│  │     • Avoids: Repeated allocations for encoding         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  2. rgbaPool320 (image.RGBA pool, 320x320)             │   │
│  │     • Purpose: Store inference outputs                  │   │
│  │     • Reuse: Get() → Fill → Convert → Put()            │   │
│  │     • Avoids: Allocating new 320x320 images            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3. rgbaPoolFullHD (image.RGBA pool, 1920x1080)        │   │
│  │     • Purpose: Resized output frames                    │   │
│  │     • Reuse: Get() → Resize into → Encode → Put()      │   │
│  │     • Avoids: Allocating large 1920x1080 images        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  4. rgbaPoolResize (image.RGBA pool, variable sizes)   │   │
│  │     • Purpose: Temporary resize operations              │   │
│  │     • Reuse: Get(width, height) → Use → Put()          │   │
│  │     • Avoids: Allocations for intermediate resizes     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  5. melWindowPool (float32 slice pool, 80x16)          │   │
│  │     • Purpose: Mel-spectrogram window extraction        │   │
│  │     • Reuse: Get() → Fill window → Encode → Put()      │   │
│  │     • Avoids: Allocating mel windows per frame         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Pool Lifecycle Example: Compositing a Frame

```
START
  │
  ├─► Get pooled 320x320 RGBA image
  │   rgbaPool320.Get() → img320
  │   └─► Fill with inference output (BGR→RGB)
  │
  ├─► Get pooled 1920x1080 RGBA image
  │   rgbaPoolFullHD.Get() → imgFullHD
  │   └─► Resize img320 into imgFullHD (bilinear)
  │
  ├─► Get pooled buffer
  │   bufferPool.Get() → buffer
  │   └─► Encode imgFullHD to JPEG into buffer
  │
  ├─► Copy JPEG bytes out
  │   jpegBytes := buffer.Bytes()
  │
  ├─► Return all to pools
  │   rgbaPool320.Put(img320)
  │   rgbaPoolFullHD.Put(imgFullHD)
  │   bufferPool.Put(buffer)
  │
END
  └─► Memory reused for next frame (ZERO GC pressure)
```

### Memory Savings

| Operation | Without Pools | With Pools | Savings |
|-----------|--------------|------------|---------|
| **320x320 RGBA** | 409,600 bytes × batch | Reused | ~3.2MB saved (batch 8) |
| **1920x1080 RGBA** | 8,294,400 bytes × batch | Reused | ~66MB saved (batch 8) |
| **JPEG buffers** | ~100KB × batch | Reused | ~800KB saved (batch 8) |
| **Mel windows** | 5,120 bytes × batch | Reused | ~40KB saved (batch 8) |
| **Total saved** | - | - | **~70MB per request** |

---

## ⚠️ Error Handling Flow

### Validation & Error Paths

```
Request Received
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Input Validation (inference.go)                           │
│                                                             │
│  ┌─ Model ID Check ─────────────────────────────────────┐  │
│  │  if modelId == "" {                                   │  │
│  │    return Error: "modelId cannot be empty"            │  │
│  │  }                                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ Model Exists Check ──────────────────────────────────┐  │
│  │  model := registry.GetModel(modelId)                  │  │
│  │  if model == nil {                                     │  │
│  │    return Error: "model not found: {modelId}"         │  │
│  │  }                                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ Visual Frames Size Check ────────────────────────────┐  │
│  │  expectedSize = batchSize * 6 * 320 * 320 * 4         │  │
│  │  if len(visualFrames) != expectedSize {                │  │
│  │    return Error: "visual frames size mismatch"        │  │
│  │    Details: got {len}, expected {expectedSize}        │  │
│  │  }                                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ Audio Size Check ────────────────────────────────────┐  │
│  │  if len(rawAudio) == 0 {                               │  │
│  │    return Error: "audio cannot be empty"              │  │
│  │  }                                                     │  │
│  │  expectedSamples = batchSize * (16000 / 25)           │  │
│  │  if len(rawAudio)/4 != expectedSamples {               │  │
│  │    return Error: "audio size mismatch"                │  │
│  │  }                                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
     │
     │ ✅ Validation passed
     ▼
Processing...
     │
     ├─► Audio Processing Error?
     │   ├─ STFT failed → return Error + log details
     │   └─ Mel conversion failed → return Error + log details
     │
     ├─► Inference Error?
     │   ├─ Model.Run() failed → return Error + stack trace
     │   └─ Output shape wrong → return Error + details
     │
     └─► Compositing Error?
         ├─ JPEG encoding failed → return Error for that frame
         └─ Channel timeout → return Error + partial results
     │
     ▼
Error Response
     │
     └─► CompositeBatchResponse {
           Success: false,
           Error: "detailed error message",
           CompositedFrames: [] (empty),
         }
```

### Error Categories

| Category | Example | Response |
|----------|---------|----------|
| **Validation** | Invalid input sizes | HTTP 400, clear error message |
| **Model** | Model not found | HTTP 404, model ID in error |
| **Processing** | Audio processing failed | HTTP 500, processing stage in error |
| **Inference** | GPU out of memory | HTTP 500, CUDA error details |
| **Compositing** | JPEG encoding failed | HTTP 500, frame index in error |

---

## 🐛 Debugging Guide

### Debug File Locations

When debug file saving is enabled (check code for debug flags):

```
test_output/
├── mel_spec_{timestamp}.npy           ← Mel-spectrogram (before encoding)
├── audio_features_{timestamp}.npy     ← Audio features (after encoding)
├── visual_input_{timestamp}.npy       ← Visual frames input
├── inference_output_{timestamp}.npy   ← Raw inference output
└── frame_{idx}_{timestamp}.jpg        ← Individual composited frames
```

### Performance Profiling Points

```go
// Key timing points to measure:

1. Audio Processing:
   start := time.Now()
   // STFT + Mel + Windowing + Encoding
   audioTime := time.Since(start)
   
2. Inference:
   start := time.Now()
   outputs := model.Run(inputs)
   inferTime := time.Since(start)
   
3. Compositing:
   start := time.Now()
   // Parallel compositing of all frames
   compositeTime := time.Since(start)
   
4. Total:
   totalTime := audioTime + inferTime + compositeTime
```

### Common Issues & Solutions

| Issue | Symptoms | Debug Steps | Solution |
|-------|----------|-------------|----------|
| **Slow audio processing** | >30ms audio time | Check worker count, profile STFT/Mel | Increase workers (default: 8 each) |
| **Slow inference** | >200ms infer time | Check GPU utilization, batch size | Optimize batch size, check GPU memory |
| **Slow compositing** | >10ms composite time | Check goroutine count, JPEG quality | Reduce quality or increase pool sizes |
| **High memory** | >100MB per request | Check pool returns, look for leaks | Ensure all `Put()` calls happen |
| **Wrong output** | Visual artifacts | Save debug files, check input shapes | Verify BGR→RGB, check normalization |
| **Audio desync** | Lips don't match | Check 25fps windowing, padding | Verify mel window extraction logic |

### Logging Best Practices

```go
// Current logging pattern (buffered logger):

logger.Info("Starting inference",
    "modelId", modelId,
    "batchSize", batchSize,
    "startFrame", startFrameIdx)

// Add timing logs:
logger.Debug("Audio processing complete",
    "duration_ms", audioTime.Milliseconds(),
    "mel_shape", melShape)

logger.Debug("Inference complete",
    "duration_ms", inferTime.Milliseconds(),
    "output_shape", outputShape)

logger.Debug("Compositing complete",
    "duration_ms", compositeTime.Milliseconds(),
    "frames", len(compositedFrames))
```

### Profiling Commands

```bash
# CPU profiling
go test -cpuprofile=cpu.prof -bench=.
go tool pprof cpu.prof

# Memory profiling
go test -memprofile=mem.prof -bench=.
go tool pprof mem.prof

# Trace execution
go test -trace=trace.out -bench=.
go tool trace trace.out

# Run with race detector
go run -race cmd/server/main.go
```

---

## 📊 Summary Cheat Sheet

### Request Processing Pipeline

| Stage | Time | Parallelism | Key Files |
|-------|------|-------------|-----------|
| **Validation** | <1ms | Serial | `inference.go` |
| **Audio** | ~23ms | 8+8 workers | `inference.go` + audio processor |
| **Visual** | ~2ms | Serial | `inference.go` |
| **Inference** | ~165ms | GPU batch | `inference.go` + model |
| **Compositing** | ~4ms | Per-frame | `helpers.go` + goroutines |
| **TOTAL** | ~195ms | Mixed | All files |

### Memory Pools Quick Reference

| Pool | Size | Purpose | File |
|------|------|---------|------|
| `bufferPool` | Variable | JPEG buffers | `constants.go` |
| `rgbaPool320` | 320×320 | Inference outputs | `constants.go` |
| `rgbaPoolFullHD` | 1920×1080 | Resized frames | `constants.go` |
| `rgbaPoolResize` | Variable | Temp resizes | `constants.go` |
| `melWindowPool` | 80×16 | Mel windows | `constants.go` |

### Key Constants

| Constant | Value | Purpose | File |
|----------|-------|---------|------|
| `visualFrameSize` | 6×320×320 | 6-channel visual input | `constants.go` |
| `audioFrameSize` | 32×16×16 | Audio feature dims | `constants.go` |
| `outputFrameSize` | 3×320×320 | RGB output dims | `constants.go` |
| Mel bins | 80 | Frequency resolution | Audio processor |
| Mel window | 16 frames | Temporal context | `inference.go` |
| Sample rate | 16000 Hz | Audio frequency | Audio processor |
| FPS | 25 | Video frame rate | `inference.go` |

---

**Last Updated:** October 28, 2025  
**Purpose:** Maintenance, debugging, and onboarding reference  
**Status:** Complete and validated
