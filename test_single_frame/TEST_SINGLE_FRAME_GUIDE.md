# Single Frame ONNX Test Guide

## Purpose
This test script processes exactly **one frame (default: frame 8)** with the required 640ms of audio input and logs all tensor values in detail. Use this to verify that your iOS ONNX inference produces identical results to the Python reference implementation.

## Files Created
- `test_single_frame_onnx.py` - Python script with detailed tensor logging
- `test_single_frame_onnx.bat` - Batch file launcher

## Usage

### Basic Usage
```batch
.\test_single_frame_onnx.bat <name> <audio_path> [asr_mode] [frame_index]
```

### Example (from your command)
```batch
.\test_single_frame_onnx.bat sanders dataset\aoc\aud.wav ave
```

This will:
1. Process frame 8 (default) using the sanders dataset
2. Use audio from `dataset\aoc\aud.wav`
3. Use ASR mode "ave"
4. Extract exactly 640ms of audio (indices 0-16 for frame 8)

### Optional Parameters
- `asr_mode` - Default: "ave", Options: "ave", "hubert", "wenet"
- `frame_index` - Default: 8, any valid frame number

## Output Files

All outputs are saved to `dataset\<name>\test_output\`:

### 1. JSON Statistics (`frame_8_tensors.json`)
Contains:
- Tensor shapes for all inputs/outputs
- Statistics (min, max, mean, std) for verification
- First 10 and last 10 values of each tensor (flattened)
- Quick checksums for iOS comparison

### 2. NumPy Arrays (`tensors_npy/` folder)
Exact tensor values saved as `.npy` files:
- `frame_8_audio_window.npy` - Audio window (16, 512)
- `frame_8_audio_input.npy` - Reshaped audio input (1, 32, 16, 16)
- `frame_8_image_input.npy` - 6-channel image input (1, 6, 320, 320)
- `frame_8_output.npy` - Generator output (3, 320, 320)

### 3. Visual Verification Images
- `frame_8_crop.jpg` - Original cropped face region
- `frame_8_roi_input.jpg` - ROI before masking
- `frame_8_masked.jpg` - Masked input visualization
- `frame_8_pred.jpg` - Generator prediction output
- `frame_8_final.jpg` - Final composite result

## How to Verify iOS Inference

### Step 1: Compare Tensor Shapes
Check that your iOS tensors match these shapes:
```
Audio Input:      (1, 32, 16, 16)   for "ave" mode
Image Input:      (1, 6, 320, 320)
Generator Output: (3, 320, 320)
```

### Step 2: Verify Statistics
Compare min/max/mean/std from the JSON file:
```json
"audio_window_stats": {
  "min": 0.0,
  "max": 7.261715,
  "mean": 0.388071,
  "std": 0.705689
}
```

### Step 3: Check Exact Values
Compare first 10 and last 10 values (from JSON):
```json
"audio_window_first_10": [0.0, 0.678675, 0.182425, ...],
"audio_window_last_10": [0.0, 0.0, 0.167092, ...]
```

### Step 4: Pixel-Perfect Comparison (Advanced)
Load the `.npy` files in Python and compare against your iOS output:
```python
import numpy as np

# Load reference
ref_output = np.load('frame_8_output.npy')

# Load your iOS output (convert to numpy)
ios_output = your_ios_tensor_as_numpy

# Compare
diff = np.abs(ref_output - ios_output)
print(f"Max difference: {diff.max()}")
print(f"Mean difference: {diff.mean()}")
```

## Console Output Explanation

The script logs detailed information for each processing step:

### Step 1: Audio Processing
- Loads audio and extracts mel spectrograms
- Runs through audio encoder ONNX
- Logs first batch of audio encoder input/output

### Step 2: Audio Window Extraction
- Extracts 16 frames of audio features (640ms)
- Covers indices 0-16 for frame 8
- Reshapes to (1, 32, 16, 16) for "ave" mode

### Step 3: Image Preparation
- Loads frame 8 image and landmarks
- Crops based on landmarks
- Creates 6-channel input (3 real + 3 masked)

### Step 4: Generator Inference
- Runs ONNX generator model
- Logs input/output shapes
- Shows provider information (CUDA/CPU)

### Step 5: Post-Processing
- Converts output to BGR image
- Saves intermediate visualizations

### Step 6: Data Export
- Saves JSON with statistics
- Exports .npy files for exact verification

## Technical Details

### Audio Window Calculation
For frame 8:
- Left boundary: 8 - 8 = 0
- Right boundary: 8 + 8 = 16
- Total: 16 frames of audio features
- Duration: 640ms at 25fps (ave mode)

### Tensor Transformations
1. Audio: (16, 512) → reshape → (32, 16, 16) → unsqueeze → (1, 32, 16, 16)
2. Image: (320, 320, 3) → transpose → (3, 320, 320) → concat → (6, 320, 320) → unsqueeze → (1, 6, 320, 320)
3. Output: (1, 3, 320, 320) → squeeze → (3, 320, 320)

### Why Frame 8?
Frame 8 is ideal for testing because:
- It has sufficient preceding audio context (frames 0-7)
- The audio window (0-16) doesn't require zero-padding at the start
- It's early enough to avoid end-of-sequence issues
- It represents a typical mid-sequence frame

## Troubleshooting

### Missing ONNX Models
If you get errors about missing models:
- Generator: Should be in `dataset\<name>\checkpoint\model_best.onnx`
- Audio Encoder: Should be in `model\checkpoints\audio_encoder.onnx`

### Wrong Dataset Structure
The script auto-detects flattened distributions (models in `dataset\<name>\models\`)

### Comparing Different Frames
To test a different frame:
```batch
.\test_single_frame_onnx.bat sanders dataset\aoc\aud.wav ave 15
```

Note: Early frames (< 8) or late frames may have zero-padding in audio window.

## Integration with iOS

1. **Export tensors from iOS** using the same frame and audio
2. **Compare shapes** first - they must match exactly
3. **Compare statistics** - min/max/mean/std should be very close (<0.001 difference)
4. **Compare sample values** - first/last 10 values should match
5. **Visual check** - Compare generated images side-by-side

If you see differences:
- Check input preprocessing (image normalization, audio extraction)
- Verify ONNX model versions match
- Confirm tensor reshaping operations
- Check for different ONNX runtime providers

## Advanced: Batch Testing Multiple Frames
To test multiple frames, create a loop:
```batch
for /L %%i in (5,1,15) do (
    .\test_single_frame_onnx.bat sanders dataset\aoc\aud.wav ave %%i
)
```

This tests frames 5 through 15 and creates separate output files for each.
