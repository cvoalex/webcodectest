#!/usr/bin/env python3
"""Helper script to load visual frames from video files for Go test."""
import sys
import cv2
import numpy as np

def load_frames(start_frame, batch_size):
    """Load real frames from sanders videos."""
    crops_path = "../old/old_minimal_server/models/sanders/crops_328_video.mp4"
    rois_path = "../old/old_minimal_server/models/sanders/rois_320_video.mp4"
    
    # Open videos
    crops_cap = cv2.VideoCapture(crops_path)
    rois_cap = cv2.VideoCapture(rois_path)
    
    if not crops_cap.isOpened() or not rois_cap.isOpened():
        print("ERROR: Failed to open videos", file=sys.stderr)
        sys.exit(1)
    
    # Set position
    crops_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rois_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    crops = []
    rois = []
    
    for i in range(batch_size):
        # Read frames
        ret1, crop = crops_cap.read()
        ret2, roi = rois_cap.read()
        
        if not ret1 or not ret2:
            print(f"ERROR: Failed to read frame {start_frame + i}", file=sys.stderr)
            sys.exit(1)
        
        # Resize BOTH to 320x320 (to match server expectations)
        if crop.shape[:2] != (320, 320):
            crop = cv2.resize(crop, (320, 320))
        if roi.shape[:2] != (320, 320):
            roi = cv2.resize(roi, (320, 320))
        
        # Keep as BGR (model expects BGR, not RGB!) and normalize to float32 [0,1]
        crop_bgr = crop.astype(np.float32) / 255.0
        roi_bgr = roi.astype(np.float32) / 255.0
        
        # Convert to CHW format (C, H, W)
        crop_chw = np.transpose(crop_bgr, (2, 0, 1))  # (3, 320, 320) BGR
        roi_chw = np.transpose(roi_bgr, (2, 0, 1))    # (3, 320, 320) BGR
        
        crops.append(crop_chw)
        rois.append(roi_chw)
    
    crops_cap.release()
    rois_cap.release()
    
    # Stack and write to stdout as binary
    crops_array = np.stack(crops, axis=0)  # (batch, 3, 320, 320)
    rois_array = np.stack(rois, axis=0)    # (batch, 3, 320, 320)
    
    # Write crops then rois
    sys.stdout.buffer.write(crops_array.astype(np.float32).tobytes())
    sys.stdout.buffer.write(rois_array.astype(np.float32).tobytes())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: load_frames.py <start_frame> <batch_size>", file=sys.stderr)
        sys.exit(1)
    
    start_frame = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    
    load_frames(start_frame, batch_size)
