#!/usr/bin/env python3
"""
Check if the INPUT face colors are correct
"""

import cv2
import numpy as np

debug_dir = "d:/Projects/webcodecstest/debug_inputs"

# Load input face
input_face = cv2.imread(f"{debug_dir}/face.png")

print("=" * 80)
print("INPUT FACE COLOR ANALYSIS")
print("=" * 80)
print()

# Analyze BGR
bgr_mean = input_face.mean(axis=(0,1))
print(f"Input face BGR means: B={bgr_mean[0]:.1f}, G={bgr_mean[1]:.1f}, R={bgr_mean[2]:.1f}")
print()

if bgr_mean[0] > bgr_mean[2]:
    print("❌ INPUT is BLUE-dominant (B > R)")
    print("   This is WRONG for a normal skin tone!")
    print("   Normal skin should have R > G > B")
elif bgr_mean[2] > bgr_mean[1] > bgr_mean[0]:
    print("✅ INPUT has normal skin tone (R > G > B)")
else:
    print("⚠️ INPUT has unusual color distribution")
print()

# Sample the face region (should be skin)
h, w = input_face.shape[:2]
face_region = input_face[h//2-20:h//2+20, w//2-20:w//2+20]
face_bgr = face_region.mean(axis=(0,1))

print(f"Face region (center) BGR: B={face_bgr[0]:.1f}, G={face_bgr[1]:.1f}, R={face_bgr[2]:.1f}")
print()

# Expected skin tone check
if face_bgr[2] > face_bgr[0]:
    print("✅ Face region has skin-like colors (R > B)")
else:
    print("❌ Face region is blue-ish (B >= R)")
    print("   The INPUT VIDEO might already be corrupted!")
print()

# Check the raw video frame
print("=" * 80)
print("CHECKING RAW VIDEO FRAME")
print("=" * 80)
print()

import cv2
package_dir = "d:/Projects/webcodecstest/fast_service/models/default_model"

face_cap = cv2.VideoCapture(f"{package_dir}/face_regions_320.mp4")

# Go to frame 50
for _ in range(50):
    face_cap.read()

ret, raw_frame = face_cap.read()
face_cap.release()

if ret:
    raw_bgr = raw_frame.mean(axis=(0,1))
    print(f"Raw video frame BGR: B={raw_bgr[0]:.1f}, G={raw_bgr[1]:.1f}, R={raw_bgr[2]:.1f}")
    
    # Check center face region
    raw_face_region = raw_frame[160-20:160+20, 160-20:160+20]
    raw_face_bgr = raw_face_region.mean(axis=(0,1))
    print(f"Raw face region BGR: B={raw_face_bgr[0]:.1f}, G={raw_face_bgr[1]:.1f}, R={raw_face_bgr[2]:.1f}")
    print()
    
    if raw_face_bgr[2] > raw_face_bgr[0]:
        print("✅ Raw video has correct colors")
    else:
        print("❌ Raw video is ALREADY BLUE!")
        print("   The preprocessed video files are corrupted!")
    
    # Save for inspection
    cv2.imwrite(f"{debug_dir}/raw_video_frame_50.png", raw_frame)
    print(f"\nSaved raw frame to: {debug_dir}/raw_video_frame_50.png")
