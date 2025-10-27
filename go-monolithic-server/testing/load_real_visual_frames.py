import cv2
import numpy as np

# Load actual face crops from the sanders model
video_path = "../old/old_minimal_server/models/sanders/crops_328_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit(1)

# Read first 6 frames for visual reference
visual_frames = []
for i in range(6):
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Failed to read frame {i}")
        exit(1)
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"Frame {i} shape: {frame_rgb.shape}, dtype: {frame_rgb.dtype}, range: [{frame_rgb.min()}, {frame_rgb.max()}]")
    visual_frames.append(frame_rgb)

cap.release()

# Stack frames: [6, H, W, 3]
visual_stack = np.stack(visual_frames, axis=0)
print(f"\nStacked visual frames: {visual_stack.shape}")

# The server expects [batch_size, 6, 320, 320] float32
# We need to convert [6, H, W, 3] -> [6, 320, 320] (grayscale or 6-channel format)

# Check what the model actually expects
print(f"\nExpected format: [batch_size, 6, 320, 320] float32")
print(f"Current format: {visual_stack.shape} {visual_stack.dtype}")

# Save for inspection
np.save("visual_frames_6.npy", visual_stack)
print(f"\n✅ Saved visual_frames_6.npy")
