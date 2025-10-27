# UNet-328 Model Input/Output Specification

**DEFINITIVE TECHNICAL REFERENCE - Last Updated: October 23, 2025**

This document specifies the exact input and output formats for the UNet-328 lip-sync model. This is the **single source of truth** for all implementations (Python, Go, JavaScript, etc.).

---

## Table of Contents
1. [Model Architecture Overview](#model-architecture-overview)
2. [Visual Input Tensor](#visual-input-tensor)
3. [Audio Input Tensor](#audio-input-tensor)
4. [Output Tensor](#output-tensor)
5. [Data Pipeline](#data-pipeline)
6. [Common Mistakes](#common-mistakes)
7. [Validation Checklist](#validation-checklist)

---

## Model Architecture Overview

**Model**: UNet-328 (Audio-Visual Encoder mode)  
**File**: `unet_328.py`  
**Purpose**: Generate photorealistic lip-sync video frames from face regions and audio features

### ONNX Model Inputs/Outputs
```python
# Input names (as seen in ONNX model):
- "input"  : Visual tensor
- "audio"  : Audio tensor

# Output name:
- "output" : Generated face region
```

⚠️ **CRITICAL**: Different models may use different input names:
- Sanders model: `"input"`, `"audio"` 
- Other models may use: `"visual_input"`, `"audio_input"`

**Always check your specific model with:**
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
for inp in session.get_inputs():
    print(f"{inp.name}: {inp.shape}")
```

---

## Visual Input Tensor

### Shape: `[batch_size, 6, 320, 320]`

### Data Type: `float32`

### Value Range: `[0.0, 1.0]`

### Channel Organization:
```
Channels 0-2: Face ROI (Region of Interest)
  - Channel 0: BLUE channel of face region
  - Channel 1: GREEN channel of face region  
  - Channel 2: RED channel of face region

Channels 3-5: Masked Face Region
  - Channel 3: BLUE channel of masked face
  - Channel 4: GREEN channel of masked face
  - Channel 5: RED channel of masked face
```

### ⚠️ CRITICAL: Color Format is **BGR** (NOT RGB!)

OpenCV uses BGR ordering by default. The model was trained with `cv2.imread()` which reads images in BGR format. **DO NOT convert to RGB!**

### Generation Process:

```python
import cv2
import numpy as np

# Step 1: Read face ROI (320x320) from video
cap = cv2.VideoCapture("rois_320_video.mp4")
ret, roi_frame = cap.read()  # Shape: [320, 320, 3] in BGR format

# Step 2: Read masked face region (320x320) from video
cap_mask = cv2.VideoCapture("model_inputs_video.mp4")
ret, masked_frame = cap_mask.read()  # Shape: [320, 320, 3] in BGR format

# Step 3: Normalize to [0, 1] range
roi_norm = roi_frame.astype(np.float32) / 255.0      # [320, 320, 3] BGR
masked_norm = masked_frame.astype(np.float32) / 255.0  # [320, 320, 3] BGR

# Step 4: Transpose to channels-first format
roi_chw = np.transpose(roi_norm, (2, 0, 1))      # [3, 320, 320] BGR
masked_chw = np.transpose(masked_norm, (2, 0, 1))  # [3, 320, 320] BGR

# Step 5: Concatenate along channel dimension
visual_input = np.concatenate([roi_chw, masked_chw], axis=0)  # [6, 320, 320]

# Step 6: Add batch dimension
visual_input = np.expand_dims(visual_input, axis=0)  # [1, 6, 320, 320]
```

### Memory Layout:
```
Total elements: 1 × 6 × 320 × 320 = 614,400 float32 values
Memory size: 614,400 × 4 bytes = 2,457,600 bytes = 2.4 MB per frame
```

### Common Sources:
- **Sanders dataset**: `minimal_server/models/sanders/`
  - Face ROI: `rois_320_video.mp4` (already 320x320)
  - Masked face: `model_inputs_video.mp4` (already 320x320)

---

## Audio Input Tensor

### Shape: `[batch_size, 32, 16, 16]`

### Data Type: `float32`

### Value Range: Varies (pre-extracted AVE features, typically normalized)

### ⚠️ CRITICAL: Audio is a **16-FRAME WINDOW** (NOT single frame!)

The model requires audio context from multiple frames to generate natural lip movements.

### Window Composition:
```
Frame window: [current - 8] to [current + 7]
- 8 frames BEFORE current frame
- 1 CURRENT frame
- 7 frames AFTER current frame
Total: 16 frames
```

### Generation Process:

```python
import numpy as np

# Step 1: Load pre-extracted AVE (Audio-Visual Encoder) features
# These are extracted using a separate audio encoding model
audio_features = np.load("aud_ave.npy")  # Shape: [num_frames, 512]
# Each frame has 512-dimensional audio feature vector

# Step 2: For target frame_id, extract 16-frame window
frame_id = 42  # Example: generating frame 42

# Calculate window boundaries with padding
left = max(0, frame_id - 8)
right = min(len(audio_features), frame_id + 8)

# Extract window
audio_window = audio_features[left:right]  # Shape: [up_to_16, 512]

# Step 3: Pad if necessary (for frames near start/end of video)
if len(audio_window) < 16:
    # Replicate first/last frame to fill window
    if frame_id < 8:
        # Pad at beginning
        pad_left = 8 - frame_id
        audio_window = np.concatenate([
            np.tile(audio_features[0], (pad_left, 1)),
            audio_window
        ], axis=0)
    if frame_id + 8 > len(audio_features):
        # Pad at end
        pad_right = (frame_id + 8) - len(audio_features)
        audio_window = np.concatenate([
            audio_window,
            np.tile(audio_features[-1], (pad_right, 1))
        ], axis=0)

# Step 4: Flatten and reshape to [32, 16, 16]
audio_flat = audio_window.flatten()  # [16, 512] → [8192]
audio_input = audio_flat.reshape(32, 16, 16)  # [8192] → [32, 16, 16]

# Step 5: Add batch dimension
audio_input = np.expand_dims(audio_input, axis=0)  # [1, 32, 16, 16]
```

### Why This Shape?
```
16 frames × 512 features = 8,192 total values
8,192 = 32 × 16 × 16

The 2D reshape [32, 16, 16] allows the model to use 2D convolutions 
to process temporal audio patterns efficiently.
```

### Memory Layout:
```
Total elements: 1 × 32 × 16 × 16 = 8,192 float32 values
Memory size: 8,192 × 4 bytes = 32,768 bytes = 32 KB per frame
```

### Audio Feature Extraction:
The 512-dimensional AVE features are pre-computed using a separate audio encoding model:
```python
# This is typically done in preprocessing (not during inference)
from data_utils.ave import AVEEncoder

encoder = AVEEncoder()
audio_waveform = load_audio("audio.wav")
ave_features = encoder.extract(audio_waveform)  # [num_frames, 512]
np.save("aud_ave.npy", ave_features)
```

---

## Output Tensor

### Shape: `[batch_size, 3, 320, 320]`

### Data Type: `float32`

### Value Range: `[0.0, 1.0]` (model uses `F.sigmoid()` activation)

### Channel Organization:
```
Channel 0: BLUE channel of generated face
Channel 1: GREEN channel of generated face
Channel 2: RED channel of generated face
```

### ⚠️ CRITICAL: Output is in **BGR format** (same as input!)

The model preserves the BGR color ordering from the input.

### Post-Processing:

```python
import cv2
import numpy as np

# Step 1: Run inference
output = model.infer(visual_input, audio_input)  # [1, 3, 320, 320]

# Step 2: Remove batch dimension
output = output[0]  # [3, 320, 320]

# Step 3: Transpose to height-width-channels format
output_hwc = np.transpose(output, (1, 2, 0))  # [320, 320, 3] BGR

# Step 4: Denormalize from [0, 1] to [0, 255]
output_uint8 = (output_hwc * 255.0).astype(np.uint8)  # [320, 320, 3] BGR uint8

# Step 5: Resize to target ROI size (328x328 for compositing)
output_resized = cv2.resize(output_uint8, (328, 328), interpolation=cv2.INTER_LINEAR)

# Step 6: Composite onto full frame (if needed)
# The output is just the face region - must be composited back onto full body frame
full_frame = cv2.imread("full_body_frame.jpg")  # BGR format
x1, y1, x2, y2 = roi_bounds  # Coordinates of face in full frame

# Insert prediction at center of 328x328 ROI (skipping 4-pixel border)
roi_328 = np.zeros((328, 328, 3), dtype=np.uint8)
roi_328[4:324, 4:324] = output_uint8

# Resize to original ROI dimensions
roi_resized = cv2.resize(roi_328, (x2-x1, y2-y1))

# Composite back onto full frame
full_frame[y1:y2, x1:x2] = roi_resized

# Step 7: Save or display
cv2.imwrite("output_frame.jpg", full_frame)  # Already in BGR for cv2.imwrite
```

### What the Output Represents:

The output is a **320×320 pixel region containing ONLY the face area** with:
- ✅ Lip movements synchronized to audio
- ✅ Natural facial expressions
- ✅ Preserved lighting and skin tones
- ✅ Seamless blending at face boundaries

The output does NOT include:
- ❌ Full body
- ❌ Background
- ❌ Shoulders/neck
- ❌ Hair (outside face region)

These must be taken from the original full-resolution video frame.

### Memory Layout:
```
Total elements: 1 × 3 × 320 × 320 = 307,200 float32 values
Memory size: 307,200 × 4 bytes = 1,228,800 bytes = 1.2 MB per frame
```

---

## Data Pipeline

### Complete End-to-End Flow:

```
1. Original Video (e.g., 1920×1080) + Audio WAV
   ↓
2. Face Detection & Landmark Detection
   - Detect face bounding box
   - Extract 68 facial landmarks
   ↓
3. Extract Face ROI (320×320)
   - Crop face region
   - Resize to 320×320
   - Save to rois_320_video.mp4 (BGR)
   ↓
4. Create Masked Face Region (320×320)
   - Apply mask around mouth area
   - Save to model_inputs_video.mp4 (BGR)
   ↓
5. Extract Audio Features
   - Load audio waveform
   - Extract AVE features (512-dim per frame)
   - Save to aud_ave.npy [num_frames, 512]
   ↓
6. Inference (for each frame)
   - Load frame ROIs (BGR, keep as BGR!)
   - Normalize to [0, 1]
   - Transpose to [6, 320, 320]
   - Load 16-frame audio window
   - Reshape to [32, 16, 16]
   - Run model inference
   ↓
7. Output (320×320 BGR)
   - Denormalize to [0, 255]
   - Resize to target ROI size
   - Composite onto full frame
   ↓
8. Final Video (1920×1080) with lip-sync
```

### File Structure (Sanders Dataset Example):
```
minimal_server/models/sanders/
├── full_body_video.mp4           # Original video (1920×1080 or similar)
├── rois_320_video.mp4             # Face ROIs (320×320, BGR, 25 FPS)
├── model_inputs_video.mp4         # Masked faces (320×320, BGR, 25 FPS)
├── aud_ave.npy                    # Audio features [num_frames, 512]
├── roi_bounds.npy                 # Face coordinates [num_frames, 4] (x1,y1,x2,y2)
└── checkpoint/
    └── model_best.onnx            # Trained model
```

---

## Common Mistakes

### ❌ MISTAKE #1: Converting BGR to RGB
```python
# WRONG:
frame = cv2.imread("face.jpg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ❌ DON'T DO THIS!
```

**Why it's wrong**: Model was trained with BGR format. Converting to RGB swaps red/blue channels, causing blue faces.

**Correct**:
```python
# CORRECT:
frame = cv2.imread("face.jpg")  # Keep as BGR
# Don't convert! Just normalize and use directly.
```

### ❌ MISTAKE #2: Using Single Audio Frame
```python
# WRONG:
audio_feat = audio_features[frame_id]  # ❌ Only 512 values!
audio_input = audio_feat.reshape(32, 16, 16)  # Shape mismatch!
```

**Why it's wrong**: Model expects 16-frame window (8192 values), not single frame (512 values).

**Correct**:
```python
# CORRECT:
left = max(0, frame_id - 8)
right = min(len(audio_features), frame_id + 8)
audio_window = audio_features[left:right]  # [16, 512] = 8192 values
audio_input = audio_window.flatten().reshape(32, 16, 16)
```

### ❌ MISTAKE #3: Wrong Normalization Range
```python
# WRONG:
frame_norm = (frame / 255.0 - 0.5) * 2.0  # ❌ Produces [-1, 1] range
```

**Why it's wrong**: Model expects [0, 1] range. The output uses sigmoid activation which naturally produces [0, 1].

**Correct**:
```python
# CORRECT:
frame_norm = frame.astype(np.float32) / 255.0  # [0, 1] range
```

### ❌ MISTAKE #4: Wrong Input Names
```python
# WRONG (for Sanders model):
session.run(["output"], {
    "visual_input": visual,  # ❌ Sanders model doesn't have this name
    "audio_input": audio
})
```

**Why it's wrong**: Different models use different input names.

**Correct**:
```python
# CORRECT (for Sanders model):
# Always check model input names first!
session.run(["output"], {
    "input": visual,   # ✅ Sanders model uses "input"
    "audio": audio     # ✅ Sanders model uses "audio"
})
```

### ❌ MISTAKE #5: Wrong Axis for Transpose
```python
# WRONG:
frame_chw = np.transpose(frame, (0, 1, 2))  # ❌ No change - still HWC
```

**Why it's wrong**: Need to move channels from last axis to first axis.

**Correct**:
```python
# CORRECT:
frame_chw = np.transpose(frame, (2, 0, 1))  # ✅ HWC → CHW
```

---

## Validation Checklist

Use this checklist to verify your implementation is correct:

### ✅ Visual Input Checklist:
- [ ] Shape is exactly `[1, 6, 320, 320]`
- [ ] Data type is `float32`
- [ ] Values are in range `[0.0, 1.0]`
- [ ] Color format is **BGR** (not RGB)
- [ ] First 3 channels are face ROI (BGR)
- [ ] Last 3 channels are masked face (BGR)
- [ ] No BGR→RGB conversion was applied
- [ ] Transpose was applied: HWC → CHW

### ✅ Audio Input Checklist:
- [ ] Shape is exactly `[1, 32, 16, 16]`
- [ ] Data type is `float32`
- [ ] Uses 16-frame window (not single frame)
- [ ] Window includes 8 frames before + current + 7 frames after
- [ ] Padding applied for frames near start/end
- [ ] Flattened to 8,192 values before reshape
- [ ] Reshape to [32, 16, 16] is correct

### ✅ Output Checklist:
- [ ] Shape is exactly `[1, 3, 320, 320]`
- [ ] Data type is `float32`
- [ ] Values are in range `[0.0, 1.0]`
- [ ] Output is in **BGR** format (same as input)
- [ ] Denormalized by multiplying by 255.0
- [ ] Converted to `uint8` before saving
- [ ] Composited back onto full frame (not used standalone)

### ✅ Quality Verification:
- [ ] Output faces look **photorealistic** (not blue/noisy)
- [ ] Lip movements are **synchronized** with audio
- [ ] Facial expressions look **natural**
- [ ] Skin tones are **correct** (not blue/purple)
- [ ] No visible **artifacts** or **discontinuities**

### ✅ Performance Verification:
- [ ] Inference completes in **< 50ms** per frame (GPU)
- [ ] Memory usage is **reasonable** (< 4GB VRAM)
- [ ] No memory leaks over multiple frames
- [ ] CUDA execution provider is **enabled**

---

## Quick Reference

### Model I/O Summary:
```
INPUT:
  Visual: [1, 6, 320, 320] float32 [0,1] BGR (face + masked face)
  Audio:  [1, 32, 16, 16]  float32      (16-frame window of AVE features)

OUTPUT:
  Image:  [1, 3, 320, 320] float32 [0,1] BGR (generated face region)
```

### Critical Rules:
1. ⚠️ **ALWAYS** keep BGR format (don't convert to RGB)
2. ⚠️ **ALWAYS** use 16-frame audio window (not single frame)
3. ⚠️ **ALWAYS** normalize to [0, 1] range (not [-1, 1])
4. ⚠️ **ALWAYS** check model input names (they vary by model)
5. ⚠️ **ALWAYS** composite output onto full frame (don't use 320×320 alone)

### Debugging:
If output looks wrong, check in this order:
1. **Blue faces?** → You're converting BGR→RGB (don't!)
2. **Noisy/random?** → Wrong audio window size or normalization
3. **All black?** → Wrong input names or shapes
4. **Inference error?** → Check input names match your model
5. **Memory error?** → Check tensor shapes are exactly correct

---

**Document Version**: 1.0  
**Last Updated**: October 23, 2025  
**Model**: UNet-328 (Sanders checkpoint)  
**Validated On**: Python 3.12 + ONNX Runtime 1.22.0 + Go 1.24.6

---

## References

- Model architecture: `unet_328.py`
- Python implementation: `fast_service/batch_video_processor_onnx.py`
- Go implementation: `go-onnx-inference/cmd/benchmark-sanders/main.go`
- Benchmark results: `PYTHON_VS_GO_BENCHMARK.md`
- Working dataset: `minimal_server/models/sanders/`
