"""
Verify that compositing is working correctly by comparing frame sizes
"""

import cv2
import os

print("\n🔍 Verifying Compositing...")

# Check a few frames
for frame_id in [0, 25, 50, 75, 99]:
    frame_path = f"output_batch_onnx/frame_{frame_id:04d}.jpg"
    if os.path.exists(frame_path):
        img = cv2.imread(frame_path)
        print(f"   Frame {frame_id}: {img.shape[1]}x{img.shape[0]} pixels")
    else:
        print(f"   Frame {frame_id}: NOT FOUND")

# Compare with full body video
sanders_dir = "minimal_server/models/sanders"
full_cap = cv2.VideoCapture(os.path.join(sanders_dir, "full_body_video.mp4"))
ret, full_frame = full_cap.read()
if ret:
    print(f"\n📺 Full body video: {full_frame.shape[1]}x{full_frame.shape[0]} pixels")
    print(f"   Expected: Output frames should match this size")
else:
    print("\n❌ Could not read full body video")
full_cap.release()

# Check ROI video for comparison
roi_cap = cv2.VideoCapture(os.path.join(sanders_dir, "rois_320_video.mp4"))
ret, roi_frame = roi_cap.read()
if ret:
    print(f"\n👤 ROI video: {roi_frame.shape[1]}x{roi_frame.shape[0]} pixels")
    print(f"   (This is just the face crop, not composited)")
roi_cap.release()
