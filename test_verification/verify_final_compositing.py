"""
Verify the final compositing is correct by comparing with crops_328 video
"""

import cv2
import numpy as np
import os

sanders_dir = "minimal_server/models/sanders"

# Check a middle frame
frame_id = 50

# Load our output
output_frame = cv2.imread(f"output_batch_onnx/frame_{frame_id:04d}.jpg")

# Load original crops_328 for comparison
crops_cap = cv2.VideoCapture(os.path.join(sanders_dir, "crops_328_video.mp4"))
for _ in range(frame_id):
    crops_cap.read()
ret, crops_frame = crops_cap.read()
crops_cap.release()

# Load full body frame
full_cap = cv2.VideoCapture(os.path.join(sanders_dir, "full_body_video.mp4"))
for _ in range(frame_id):
    full_cap.read()
ret, full_frame = full_cap.read()
full_cap.release()

# Load landmarks to show where the crop should be
landmark_file = os.path.join(sanders_dir, "landmarks", f"{frame_id}.lms")
landmarks = []
with open(landmark_file, 'r') as f:
    for line in f:
        x, y = line.strip().split()
        landmarks.append([float(x), float(y)])
landmarks = np.array(landmarks)

xs = landmarks[:, 0]
ys = landmarks[:, 1]
cx = int((xs.min() + xs.max()) / 2)
cy = int((ys.min() + ys.max()) / 2)

# Calculate 328x328 bounds
x1 = int(cx - 164)
y1 = int(cy - 164)
x2 = x1 + 328
y2 = y1 + 328

print(f"\n📊 Frame {frame_id} Analysis:")
print(f"   Output frame: {output_frame.shape[1]}x{output_frame.shape[0]}")
print(f"   Full body: {full_frame.shape[1]}x{full_frame.shape[0]}")
print(f"   Crops_328: {crops_frame.shape[1]}x{crops_frame.shape[0]}")
print(f"   Landmark center: ({cx}, {cy})")
print(f"   328x328 bounds: ({x1}, {y1}) to ({x2}, {y2})")

# Extract the region from our output where we composited
extracted = output_frame[y1:y2, x1:x2]

print(f"   Extracted region: {extracted.shape}")

# Create visual comparison
# Draw rectangle on full frame copy to show where face should be
full_with_box = full_frame.copy()
cv2.rectangle(full_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(full_with_box, f"328x328 crop area", (x1, y1-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Create comparison grid
comparison = np.hstack([
    cv2.resize(full_with_box, (640, 360)),
    cv2.resize(output_frame, (640, 360))
])

os.makedirs("verification", exist_ok=True)
cv2.imwrite("verification/full_with_bounds.jpg", full_with_box)
cv2.imwrite("verification/our_output.jpg", output_frame)
cv2.imwrite("verification/original_crops_328.jpg", crops_frame)
cv2.imwrite("verification/extracted_from_output.jpg", extracted)
cv2.imwrite("verification/comparison.jpg", comparison)

print(f"\n✅ Verification images saved to verification/")
print(f"   full_with_bounds.jpg - Original with green box showing 328x328 area")
print(f"   our_output.jpg - Our generated output")
print(f"   original_crops_328.jpg - Original crops_328 for reference")
print(f"   extracted_from_output.jpg - The 328x328 region we composited")
print(f"   comparison.jpg - Side by side")

# Check if sizes match
if extracted.shape == crops_frame.shape:
    print(f"\n✅ Size matches! ({extracted.shape})")
else:
    print(f"\n❌ Size mismatch: extracted={extracted.shape}, crops={crops_frame.shape}")
