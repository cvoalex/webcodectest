#!/usr/bin/env python3
"""Helper script to load visual frames from pre-extracted frame files or video for Go test."""
import sys
import cv2
import numpy as np
import os

def load_frames_from_disk(crops_dir, rois_dir, start_frame, batch_size):
    """Load frames from pre-extracted JPEG files (fastest method)."""
    crops = []
    rois = []
    
    for i in range(batch_size):
        frame_idx = start_frame + i
        
        # Load crop frame
        crop_path = os.path.join(crops_dir, f"frame_{frame_idx:06d}.jpg")
        if not os.path.exists(crop_path):
            return None  # Frame doesn't exist, fall back to video
        crop = cv2.imread(crop_path)
        if crop is None:
            return None
        
        # Load ROI frame
        roi_path = os.path.join(rois_dir, f"frame_{frame_idx:06d}.jpg")
        if not os.path.exists(roi_path):
            return None
        roi = cv2.imread(roi_path)
        if roi is None:
            return None
        
        # Resize BOTH to 320x320 (should already be this size from extraction)
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
    
    # Stack into arrays
    crops_array = np.stack(crops, axis=0)  # (batch, 3, 320, 320)
    rois_array = np.stack(rois, axis=0)    # (batch, 3, 320, 320)
    
    return crops_array, rois_array

def load_frames_from_video(crops_path, rois_path, start_frame, batch_size):
    """Load frames from video files (slower fallback method)."""
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
        
        # Resize BOTH to 320x320
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
    
    # Stack into arrays
    crops_array = np.stack(crops, axis=0)  # (batch, 3, 320, 320)
    rois_array = np.stack(rois, axis=0)    # (batch, 3, 320, 320)
    
    return crops_array, rois_array

def load_frames(start_frame, batch_size):
    """Load real frames - try pre-extracted files first, fall back to video."""
    # Try to load from pre-extracted frames first (faster)
    crops_dir = "../old/old_minimal_server/models/sanders/crops_frames"
    rois_dir = "../old/old_minimal_server/models/sanders/rois_frames"
    
    if os.path.isdir(crops_dir) and os.path.isdir(rois_dir):
        result = load_frames_from_disk(crops_dir, rois_dir, start_frame, batch_size)
        if result is not None:
            return result  # Success! Frames loaded from disk
    
    # Fall back to extracting from video files (slower)
    crops_path = "../old/old_minimal_server/models/sanders/crops_328_video.mp4"
    rois_path = "../old/old_minimal_server/models/sanders/rois_320_video.mp4"
    
    return load_frames_from_video(crops_path, rois_path, start_frame, batch_size)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: load_frames.py <start_frame> <batch_size>", file=sys.stderr)
        sys.exit(1)
    
    start_frame = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    
    crops_array, rois_array = load_frames(start_frame, batch_size)
    
    # Write crops then rois to stdout as binary
    sys.stdout.buffer.write(crops_array.astype(np.float32).tobytes())
    sys.stdout.buffer.write(rois_array.astype(np.float32).tobytes())
