#!/usr/bin/env python3
"""Convert landmark files to crop rectangles JSON."""
import os
import json
import numpy as np

def load_landmarks(lms_path):
    """Load landmarks from .lms file."""
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    return np.array(lms_list, dtype=np.int32)

def calculate_crop_rect(lms):
    """Calculate crop rectangle from landmarks (same logic as inference_328.py)."""
    xmin = int(lms[1][0])
    ymin = int(lms[52][1])
    xmax = int(lms[31][0])
    width = xmax - xmin
    ymax = ymin + width
    return [xmin, ymin, xmax, ymax]

def main():
    landmarks_dir = "../old/old_minimal_server/models/sanders/landmarks"
    output_file = "../old/old_minimal_server/models/sanders/crop_rects.json"
    
    # Get all landmark files
    lms_files = sorted([f for f in os.listdir(landmarks_dir) if f.endswith('.lms')],
                      key=lambda x: int(x.split('.')[0]))
    
    print(f"Found {len(lms_files)} landmark files")
    
    crop_rects = []
    for lms_file in lms_files:
        lms_path = os.path.join(landmarks_dir, lms_file)
        lms = load_landmarks(lms_path)
        crop_rect = calculate_crop_rect(lms)
        crop_rects.append(crop_rect)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(crop_rects, f)
    
    print(f"✅ Saved {len(crop_rects)} crop rectangles to {output_file}")
    print(f"   Example crop_rect[0]: {crop_rects[0]}")
    print(f"   Example crop_rect[10]: {crop_rects[10]}")
    print(f"   Example crop_rect[100]: {crop_rects[100]}")

if __name__ == "__main__":
    main()
