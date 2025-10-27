#!/usr/bin/env python3
"""
Check for specific issues: blue tint, noise, etc.
"""

import cv2
import numpy as np

output_dir = "d:/Projects/webcodecstest/output_comparison"

pytorch_out = cv2.imread(f"{output_dir}/2_pytorch_output.png")
onnx_out = cv2.imread(f"{output_dir}/3_onnx_output.png")

print("=" * 80)
print("CHECKING FOR SPECIFIC ISSUES")
print("=" * 80)
print()

# Check for blue tint
print("1. BLUE TINT CHECK:")
bgr_mean = pytorch_out.mean(axis=(0,1))
print(f"   Channel means (BGR): B={bgr_mean[0]:.1f}, G={bgr_mean[1]:.1f}, R={bgr_mean[2]:.1f}")

if bgr_mean[0] > bgr_mean[1] + 20 and bgr_mean[0] > bgr_mean[2] + 20:
    print("   ❌ SIGNIFICANT BLUE TINT DETECTED")
elif bgr_mean[0] > bgr_mean[1] + 10:
    print("   ⚠️ Slight blue tint")
else:
    print("   ✅ No significant blue tint")
print()

# Check for noise
print("2. NOISE CHECK:")
# Compute local variance
gray = cv2.cvtColor(pytorch_out, cv2.COLOR_BGR2GRAY).astype(np.float32)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
noise = np.abs(gray - blur)
noise_level = noise.mean()

print(f"   Noise level (mean abs diff from blur): {noise_level:.2f}")

if noise_level > 10:
    print("   ❌ HIGH NOISE DETECTED")
elif noise_level > 5:
    print("   ⚠️ Moderate noise")
else:
    print("   ✅ Low noise")
print()

# Check for reasonable face structure
print("3. FACE STRUCTURE CHECK:")
# Check if there's contrast (not all same color)
std_dev = pytorch_out.std()
print(f"   Standard deviation: {std_dev:.2f}")

if std_dev < 10:
    print("   ❌ Very low contrast - might be mostly flat color")
elif std_dev < 30:
    print("   ⚠️ Low contrast")
else:
    print("   ✅ Good contrast - has structure")
print()

# Check pixel value distribution
print("4. PIXEL VALUE DISTRIBUTION:")
hist = cv2.calcHist([cv2.cvtColor(pytorch_out, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256])
hist = hist.flatten() / hist.sum()

# Check if concentrated in one range
max_bin_percent = hist.max() * 100
print(f"   Maximum single bin percentage: {max_bin_percent:.1f}%")

if max_bin_percent > 50:
    print("   ❌ Very concentrated - might be mostly one color")
elif max_bin_percent > 20:
    print("   ⚠️ Somewhat concentrated")
else:
    print("   ✅ Good distribution")
print()

# Sample specific regions
print("5. REGION SAMPLING:")
h, w = pytorch_out.shape[:2]
regions = {
    "Top-left corner": pytorch_out[10:30, 10:30].mean(axis=(0,1)),
    "Center (face)": pytorch_out[h//2-20:h//2+20, w//2-20:w//2+20].mean(axis=(0,1)),
    "Bottom-right": pytorch_out[h-30:h-10, w-30:w-10].mean(axis=(0,1)),
}

for region_name, bgr in regions.items():
    print(f"   {region_name}: B={bgr[0]:.1f}, G={bgr[1]:.1f}, R={bgr[2]:.1f}")
print()

# Create histogram visualization
hist_img = np.zeros((256, 256, 3), dtype=np.uint8)
for channel, color in enumerate([(255,0,0), (0,255,0), (0,0,255)]):
    hist_channel = cv2.calcHist([pytorch_out], [channel], None, [256], [0,256])
    hist_channel = hist_channel.flatten()
    hist_channel = (hist_channel / hist_channel.max() * 255).astype(np.int32)
    
    for i in range(256):
        cv2.line(hist_img, (i, 255), (i, 255 - hist_channel[i]), color, 1)

cv2.imwrite(f"{output_dir}/histogram.png", hist_img)
print(f"✅ Histogram saved to {output_dir}/histogram.png")
print("   (B=blue line, G=green line, R=red line)")
