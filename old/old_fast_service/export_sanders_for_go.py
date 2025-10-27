#!/usr/bin/env python3
"""
Export sanders data for Go testing (with correct BGR format)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import json

def export_frames_for_go(sanders_dir, num_frames, output_dir):
    """Export preprocessed frames for Go"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📦 Exporting {num_frames} frames from sanders dataset...")
    
    # Load audio features
    audio_features = np.load(os.path.join(sanders_dir, "aud_ave.npy"))
    
    # Open videos
    roi_cap = cv2.VideoCapture(os.path.join(sanders_dir, "rois_320_video.mp4"))
    model_input_cap = cv2.VideoCapture(os.path.join(sanders_dir, "model_inputs_video.mp4"))
    
    visual_inputs = []
    audio_inputs = []
    
    for frame_id in range(num_frames):
        # Read frames
        ret1, roi_frame = roi_cap.read()
        ret2, model_input_frame = model_input_cap.read()
        
        if not ret1 or not ret2:
            break
        
        # CRITICAL: Keep BGR format (don't convert to RGB!)
        roi_norm = roi_frame.astype(np.float32) / 255.0
        model_input_norm = model_input_frame.astype(np.float32) / 255.0
        
        roi_tensor = np.transpose(roi_norm, (2, 0, 1))
        model_input_tensor = np.transpose(model_input_norm, (2, 0, 1))
        
        visual_input = np.concatenate([roi_tensor, model_input_tensor], axis=0)
        visual_inputs.append(visual_input)
        
        # Get 16-frame audio window
        left = frame_id - 8
        right = frame_id + 8
        
        pad_left = max(0, -left)
        pad_right = max(0, right - len(audio_features))
        left = max(0, left)
        right = min(len(audio_features), right)
        
        audio_window = audio_features[left:right]
        
        if pad_left > 0:
            audio_window = np.concatenate([
                np.tile(audio_features[0:1], (pad_left, 1)),
                audio_window
            ], axis=0)
        if pad_right > 0:
            audio_window = np.concatenate([
                audio_window,
                np.tile(audio_features[-1:], (pad_right, 1))
            ], axis=0)
        
        audio_flat = audio_window.flatten()
        audio_reshaped = audio_flat.reshape(32, 16, 16)
        audio_inputs.append(audio_reshaped)
        
        if (frame_id + 1) % 10 == 0:
            print(f"   Exported {frame_id + 1}/{num_frames} frames")
    
    roi_cap.release()
    model_input_cap.release()
    
    # Save as binary
    visual_array = np.array(visual_inputs, dtype=np.float32)
    audio_array = np.array(audio_inputs, dtype=np.float32)
    
    print(f"\n💾 Saving data...")
    print(f"   Visual: {visual_array.shape}")
    print(f"   Audio: {audio_array.shape}")
    
    visual_array.tofile(os.path.join(output_dir, "visual_input.bin"))
    audio_array.tofile(os.path.join(output_dir, "audio_input.bin"))
    
    # Save metadata
    metadata = {
        "num_frames": len(visual_inputs),
        "start_frame": 0,
        "visual_shape": list(visual_array.shape),
        "audio_shape": list(audio_array.shape),
        "dtype": "float32",
        "format": "BGR",  # IMPORTANT!
        "note": "Visual input is in BGR format (not RGB) to match cv2.imread behavior"
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Data exported to {output_dir}/")
    print(f"   visual_input.bin - {visual_array.nbytes / 1024 / 1024:.2f} MB")
    print(f"   audio_input.bin - {audio_array.nbytes / 1024 / 1024:.2f} MB")
    print(f"   metadata.json")

if __name__ == "__main__":
    sanders_dir = "d:/Projects/webcodecstest/minimal_server/models/sanders"
    output_dir = "test_data_sanders_for_go"
    num_frames = 100
    
    export_frames_for_go(sanders_dir, num_frames, output_dir)
