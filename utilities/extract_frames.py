#!/usr/bin/env python3
"""
Extract frames from Sanders full_body_video.mp4 to individual PNG files
NO PYTHON FOR INFERENCE - This is just a ONE-TIME setup tool!
"""
import cv2
import os

video_path = "minimal_server/models/sanders/full_body_video.mp4"
output_dir = "minimal_server/models/sanders/frames"

print("================================================================================")
print("🎬 EXTRACTING FRAMES FROM VIDEO (One-time setup)")
print("================================================================================")
print(f"Video: {video_path}")
print(f"Output: {output_dir}")
print()

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Error: Cannot open video file: {video_path}")
    exit(1)

# Get video info
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"📊 Video Info:")
print(f"   Resolution: {width}x{height}")
print(f"   FPS: {fps}")
print(f"   Total frames: {total_frames}")
print()

print("💾 Extracting frames...")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame
    frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_path, frame)
    
    frame_count += 1
    if frame_count % 10 == 0 or frame_count == 1:
        print(f"   Extracted {frame_count}/{total_frames} frames...")

cap.release()

print()
print(f"✅ Extraction complete!")
print(f"   Total frames extracted: {frame_count}")
print(f"   Output directory: {output_dir}")
print(f"   File size: ~{frame_count * width * height * 3 / (1024*1024):.1f} MB")
print()
print("🚀 Now you can run the Go benchmark with compositing!")
print("================================================================================")
