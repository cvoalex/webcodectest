import cv2
import numpy as np

# Check crops_328 video
cap = cv2.VideoCapture('minimal_server/models/sanders/crops_328_video.mp4')
ret, frame = cap.read()
if ret:
    print(f"crops_328 video: {frame.shape}")
else:
    print("ERROR: Could not read crops_328 video")
cap.release()

# Check ROI video  
cap2 = cv2.VideoCapture('minimal_server/models/sanders/rois_320_video.mp4')
ret2, frame2 = cap2.read()
if ret2:
    print(f"rois_320 video: {frame2.shape}")
else:
    print("ERROR: Could not read rois_320 video")
cap2.release()

# Load landmark to calculate original bounds
with open('minimal_server/models/sanders/landmarks/0.lms', 'r') as f:
    landmarks = []
    for line in f:
        x, y = line.strip().split()
        landmarks.append([float(x), float(y)])
landmarks = np.array(landmarks)

xs = landmarks[:, 0]
ys = landmarks[:, 1]
x1, y1 = int(xs.min()), int(ys.min())
x2, y2 = int(xs.max()), int(ys.max())

print(f"\nLandmark bounds: ({x1}, {y1}) to ({x2}, {y2})")
print(f"Landmark size: {x2-x1} x {y2-y1}")

# The crops_328 must be resized back to the ORIGINAL bounds size
# before being cropped to 320x320 and before being resized to 328x328
print(f"\nKey insight: crops_328 (328x328) needs to be resized to the ORIGINAL face region size")
print(f"then composited at the ORIGINAL position on the full frame")
