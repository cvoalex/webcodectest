"""
Find the actual crop bounds by comparing crops_328 with full frame
The crops_328_video.mp4 is likely the actual face region that was extracted
"""

import numpy as np
import cv2
import os

sanders_dir = "minimal_server/models/sanders"

# The crops_328 is 328x328, and the landmarks are centered in it
# We need to figure out where in the full frame the 328x328 crop should go

# For frame 0, we know:
# - Landmark bounds: (531, 155) to (715, 343) = 184x188
# - Crops size: 328x328
# - This means the crop extends beyond the landmarks

frame_id = 0

# Load landmarks
landmark_file = os.path.join(sanders_dir, "landmarks", f"{frame_id}.lms")
landmarks = []
with open(landmark_file, 'r') as f:
    for line in f:
        x, y = line.strip().split()
        landmarks.append([float(x), float(y)])
landmarks = np.array(landmarks)

xs = landmarks[:, 0]
ys = landmarks[:, 1]

# Center of landmarks
cx = (xs.min() + xs.max()) / 2
cy = (ys.min() + ys.max()) / 2

print(f"Landmark center: ({cx:.1f}, {cy:.1f})")
print(f"Landmark bounds: ({xs.min():.0f}, {ys.min():.0f}) to ({xs.max():.0f}, {ys.max():.0f})")
print(f"Landmark size: {xs.max() - xs.min():.0f} x {ys.max() - ys.min():.0f}")

# The 328x328 crop should be centered on the landmark center
crop_size = 328
x1_crop = int(cx - crop_size / 2)
y1_crop = int(cy - crop_size / 2)
x2_crop = x1_crop + crop_size
y2_crop = y1_crop + crop_size

print(f"\n328x328 crop bounds (centered on landmarks):")
print(f"  Position: ({x1_crop}, {y1_crop}) to ({x2_crop}, {y2_crop})")

# Verify with full frame
full_cap = cv2.VideoCapture(os.path.join(sanders_dir, "full_body_video.mp4"))
ret, full_frame = full_cap.read()
full_cap.release()

print(f"\nFull frame size: {full_frame.shape[1]}x{full_frame.shape[0]}")

# Check if bounds are valid
if x1_crop >= 0 and y1_crop >= 0 and x2_crop <= full_frame.shape[1] and y2_crop <= full_frame.shape[0]:
    print("✅ Crop bounds are within frame")
else:
    print("⚠️  Crop bounds may need adjustment")
    print(f"   x1={x1_crop}, y1={y1_crop}, x2={x2_crop}, y2={y2_crop}")
    print(f"   Frame: 0 to {full_frame.shape[1]}x{full_frame.shape[0]}")

# Extract that region and compare with crops_328
extracted_crop = full_frame[y1_crop:y2_crop, x1_crop:x2_crop]
crops_cap = cv2.VideoCapture(os.path.join(sanders_dir, "crops_328_video.mp4"))
ret, crop_frame = crops_cap.read()
crops_cap.release()

print(f"\nExtracted crop from full frame: {extracted_crop.shape}")
print(f"Crops_328 video frame: {crop_frame.shape}")

# Save for visual comparison
os.makedirs("debug_crops", exist_ok=True)
cv2.imwrite("debug_crops/extracted_from_full.jpg", extracted_crop)
cv2.imwrite("debug_crops/crops_328_video.jpg", crop_frame)
print(f"\n💾 Saved comparison images to debug_crops/")
print(f"   If they match, we've found the correct bounds!")
