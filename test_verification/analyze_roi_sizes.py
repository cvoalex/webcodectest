"""
Analyze the relationship between ROI video frames and landmark bounds
"""

import numpy as np
import cv2
import os

sanders_dir = "minimal_server/models/sanders"

# Load a few frames and their landmarks
for frame_id in [0, 25, 50]:
    print(f"\n{'='*60}")
    print(f"Frame {frame_id}:")
    
    # Load ROI from video (this is what we're inferencing on)
    roi_cap = cv2.VideoCapture(os.path.join(sanders_dir, "rois_320_video.mp4"))
    for _ in range(frame_id):
        roi_cap.read()
    ret, roi_frame = roi_cap.read()
    roi_cap.release()
    
    print(f"  ROI video frame size: {roi_frame.shape[1]}x{roi_frame.shape[0]}")
    
    # Load landmarks
    landmark_file = os.path.join(sanders_dir, "landmarks", f"{frame_id}.lms")
    if os.path.exists(landmark_file):
        landmarks = []
        with open(landmark_file, 'r') as f:
            for line in f:
                x, y = line.strip().split()
                landmarks.append([float(x), float(y)])
        landmarks = np.array(landmarks)
        
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        width, height = x2 - x1, y2 - y1
        
        print(f"  Landmark bounds: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Landmark size: {width}x{height}")
        print(f"  Ratio: ROI is {320/width:.2f}x larger than landmark bounds")
    
    # Check if there's a crops_328 video (might have different info)
    crops_cap = cv2.VideoCapture(os.path.join(sanders_dir, "crops_328_video.mp4"))
    for _ in range(frame_id):
        crops_cap.read()
    ret, crops_frame = crops_cap.read()
    crops_cap.release()
    
    if ret:
        print(f"  Crops 328 video frame size: {crops_frame.shape[1]}x{crops_frame.shape[0]}")

print(f"\n{'='*60}")
print("\n💡 INSIGHT:")
print("The ROI video (320x320) is a CROPPED and RESIZED version of the face.")
print("The landmarks show the ORIGINAL position in the full frame.")
print("We need to find the ORIGINAL face region size, not just landmark bounds!")
