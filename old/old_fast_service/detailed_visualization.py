#!/usr/bin/env python3
"""
Create a detailed visualization of:
1. Input face
2. Input masked face
3. PyTorch output
4. ONNX output
"""

import cv2
import numpy as np

output_dir = "d:/Projects/webcodecstest/output_comparison"
debug_dir = "d:/Projects/webcodecstest/debug_inputs"

# Load all images
input_face = cv2.imread(f"{debug_dir}/face.png")
input_masked = cv2.imread(f"{debug_dir}/masked.png")
pytorch_out = cv2.imread(f"{output_dir}/2_pytorch_output.png")
onnx_out = cv2.imread(f"{output_dir}/3_onnx_output.png")

# Create comparison grid
h, w = 320, 320
grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)

# Add labels
def add_label(img, text):
    img_copy = img.copy()
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return img_copy

grid[0:h, 0:w] = add_label(input_face, "Input: Face")
grid[0:h, w:w*2] = add_label(input_masked, "Input: Masked")
grid[h:h*2, 0:w] = add_label(pytorch_out, "Output: PyTorch")
grid[h:h*2, w:w*2] = add_label(onnx_out, "Output: ONNX")

cv2.imwrite(f"{output_dir}/comparison_grid.png", grid)

print("=" * 80)
print("VISUAL COMPARISON")
print("=" * 80)
print()
print(f"Created comparison grid: {output_dir}/comparison_grid.png")
print()
print("Grid layout:")
print("  Top-left: Input face (full face)")
print("  Top-right: Input masked (mouth area masked out)")
print("  Bottom-left: PyTorch generated face")
print("  Bottom-right: ONNX generated face")
print()

# Analyze outputs
print("Output Analysis:")
print(f"  Input face mean: {input_face.mean():.2f}")
print(f"  Input masked mean: {input_masked.mean():.2f}")
print(f"  PyTorch output mean: {pytorch_out.mean():.2f}")
print(f"  ONNX output mean: {onnx_out.mean():.2f}")
print()

# Check if outputs look reasonable
if pytorch_out.mean() < 50:
    print("⚠️ WARNING: PyTorch output is very dark (mean < 50)")
elif pytorch_out.mean() > 200:
    print("⚠️ WARNING: PyTorch output is very bright (mean > 200)")
else:
    print("✅ PyTorch output brightness seems reasonable")

# Check color distribution
pytorch_bgr_mean = pytorch_out.mean(axis=(0,1))
print(f"\nPyTorch output color distribution (BGR):")
print(f"  B: {pytorch_bgr_mean[0]:.2f}")
print(f"  G: {pytorch_bgr_mean[1]:.2f}")
print(f"  R: {pytorch_bgr_mean[2]:.2f}")

if pytorch_bgr_mean[0] > pytorch_bgr_mean[1] * 1.5 and pytorch_bgr_mean[0] > pytorch_bgr_mean[2] * 1.5:
    print("  ❌ Output is very BLUE - this is WRONG!")
elif abs(pytorch_bgr_mean[0] - pytorch_bgr_mean[1]) < 30 and abs(pytorch_bgr_mean[1] - pytorch_bgr_mean[2]) < 30:
    print("  ✅ Color balance looks reasonable")
else:
    print("  ⚠️ Color distribution might be unusual")
