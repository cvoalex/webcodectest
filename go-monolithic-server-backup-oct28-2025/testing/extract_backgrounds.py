#!/usr/bin/env python3
"""Extract background frames from full body video for compositing."""
import cv2
import os
import sys

def extract_frames(video_path, output_dir, max_frames=None):
    """Extract frames from video."""
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"Extracting first {total_frames} frames")
    
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Extracted {frame_count}/{total_frames} frames...")
    
    cap.release()
    print(f"✅ Extracted {frame_count} frames to {output_dir}")
    return True

if __name__ == "__main__":
    video_path = "../old/old_minimal_server/models/sanders/full_body_video.mp4"
    output_dir = "../old/old_minimal_server/models/sanders/frames"
    
    # Extract up to 523 frames (as configured in config.yaml)
    extract_frames(video_path, output_dir, max_frames=523)
