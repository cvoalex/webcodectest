"""
Verify that face size matches the original crop rectangles
"""

import cv2
import numpy as np
import json

# Load crop rectangles
with open('minimal_server/models/sanders/cache/crop_rectangles.json', 'r') as f:
    crop_rects = json.load(f)

# Check frame 0
frame_id = 0
rect = crop_rects[str(frame_id)]["rect"]
x1, y1, x2, y2 = rect
orig_width = x2 - x1
orig_height = y2 - y1

print(f"\n📐 Frame {frame_id} Crop Rectangle:")
print(f"   Position: ({x1}, {y1}) to ({x2}, {y2})")
print(f"   Size: {orig_width}x{orig_height}")

# Load original crops_328
cap = cv2.VideoCapture('minimal_server/models/sanders/crops_328_video.mp4')
ret, crop_328 = cap.read()
cap.release()

if ret:
    print(f"\n📦 crops_328 frame: {crop_328.shape[1]}x{crop_328.shape[0]}")
    print(f"   This should be resized to {orig_width}x{orig_height} and placed at position ({x1}, {y1})")

# Load our output
output_frame = cv2.imread(f'output_batch_onnx/frame_{frame_id:04d}.jpg')
if output_frame is not None:
    print(f"\n✅ Output frame: {output_frame.shape[1]}x{output_frame.shape[0]}")
    
    # Load full body for comparison
    full_cap = cv2.VideoCapture('minimal_server/models/sanders/full_body_video.mp4')
    ret, full_frame = full_cap.read()
    full_cap.release()
    
    if ret:
        # Draw rectangle on both frames to compare
        output_with_rect = output_frame.copy()
        full_with_rect = full_frame.copy()
        
        cv2.rectangle(output_with_rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(full_with_rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save comparison
        comparison = np.hstack([full_with_rect, output_with_rect])
        cv2.imwrite('face_size_comparison.jpg', comparison)
        
        print(f"\n📊 Saved face_size_comparison.jpg")
        print(f"   Left: Original full frame with crop rectangle")
        print(f"   Right: Our output with crop rectangle")
        print(f"   The face should match the size and position of the green rectangle")
