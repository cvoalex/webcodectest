#!/usr/bin/env python3
"""
Check what preprocessing the PyTorch model actually expects
by comparing with the reference code
"""

import cv2
import numpy as np

# From the reference test_single_frame_pth.py:
# img_real_ex = roi_img.transpose(2, 0, 1).astype(np.float32) / 255.0
# img_masked = masked_img.transpose(2, 0, 1).astype(np.float32) / 255.0

print("PyTorch Reference Preprocessing:")
print("1. Load image (uint8, 0-255)")
print("2. transpose(2, 0, 1) -> [C, H, W]")
print("3. astype(np.float32) / 255.0 -> range [0, 1]")
print("4. NO mean subtraction or normalization to [-1, 1]")
print()

print("Our Current Preprocessing:")
print("1. Load image (uint8, 0-255)")
print("2. / 255.0 -> [0, 1]")
print("3. - 0.5 -> [-0.5, 0.5]")
print("4. * 2.0 -> [-1, 1]")
print()

print("PROBLEM: We're normalizing to [-1, 1] but model expects [0, 1]!")
print()

# Test with a sample value
sample_pixel = 128  # mid-gray

# Our way (WRONG):
our_way = (sample_pixel / 255.0 - 0.5) * 2.0
print(f"Sample pixel value 128:")
print(f"  Our way: {our_way:.4f} (range [-1, 1])")

# Correct way:
correct_way = sample_pixel / 255.0
print(f"  Correct way: {correct_way:.4f} (range [0, 1])")
