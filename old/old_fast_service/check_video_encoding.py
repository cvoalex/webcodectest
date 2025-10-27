#!/usr/bin/env python3
"""
Check if the video files might have encoding issues
"""

import cv2
import numpy as np

package_dir = "d:/Projects/webcodecstest/fast_service/models/default_model"

print("=" * 80)
print("VIDEO FILE ENCODING CHECK")
print("=" * 80)
print()

# Open video files
face_cap = cv2.VideoCapture(f"{package_dir}/face_regions_320.mp4")
masked_cap = cv2.VideoCapture(f"{package_dir}/masked_regions_320.mp4")

# Get properties
print("Face regions video:")
print(f"  Codec: {int(face_cap.get(cv2.CAP_PROP_FOURCC))}")
print(f"  FPS: {face_cap.get(cv2.CAP_PROP_FPS)}")
print(f"  Total frames: {int(face_cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"  Width: {int(face_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
print(f"  Height: {int(face_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print()

print("Masked regions video:")
print(f"  Codec: {int(masked_cap.get(cv2.CAP_PROP_FOURCC))}")
print(f"  FPS: {masked_cap.get(cv2.CAP_PROP_FPS)}")
print(f"  Total frames: {int(masked_cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"  Width: {int(masked_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
print(f"  Height: {int(masked_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print()

# Read frame 50
for _ in range(50):
    face_cap.read()
    masked_cap.read()

ret1, face_frame = face_cap.read()
ret2, masked_frame = masked_cap.read()

face_cap.release()
masked_cap.release()

print(f"Frame 50 loaded:")
print(f"  Face shape: {face_frame.shape}, dtype: {face_frame.dtype}")
print(f"  Masked shape: {masked_frame.shape}, dtype: {masked_frame.dtype}")
print()

# Check pixel values
print("Raw pixel values (BGR from cv2.VideoCapture):")
print(f"  Face center pixel (BGR): {face_frame[160, 160]}")
print(f"  Masked center pixel (BGR): {masked_frame[160, 160]}")
print()

# Check corners and edges for compression artifacts
print("Checking for compression artifacts:")
corners = [(0, 0), (0, 319), (319, 0), (319, 319)]
for y, x in corners:
    print(f"  Corner ({y},{x}) face: {face_frame[y, x]}, masked: {masked_frame[y, x]}")
print()

# Check if the masked region is TOO masked (all black)
black_pixels = np.sum(masked_frame.sum(axis=2) == 0)
total_pixels = 320 * 320
black_percent = black_pixels / total_pixels * 100

print(f"Masked image analysis:")
print(f"  Black pixels: {black_pixels} / {total_pixels} ({black_percent:.1f}%)")
print(f"  Mean value: {masked_frame.mean():.2f}")
print()

if black_percent > 90:
    print("  ⚠️ WARNING: Masked image is >90% black!")
    print("  This might be TOO masked - model might not have enough context")
print()

# Save raw frames for inspection
cv2.imwrite("debug_inputs/face_frame_50_raw_bgr.png", face_frame)
cv2.imwrite("debug_inputs/masked_frame_50_raw_bgr.png", masked_frame)

print("✅ Raw BGR frames saved to debug_inputs/")
print()

# Now check RGB conversion
face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
masked_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

print("After BGR->RGB conversion:")
print(f"  Face center pixel (RGB): {face_rgb[160, 160]}")
print(f"  Masked center pixel (RGB): {masked_rgb[160, 160]}")
