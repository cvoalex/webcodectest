"""
Export Sanders dataset with landmark data for Go implementation
"""

import numpy as np
import cv2
import json
import os

def export_for_go(num_frames=100, output_dir="../test_data_sanders_for_go"):
    """Export preprocessed sanders data for Go with landmarks"""
    
    sanders_dir = "d:/Projects/webcodecstest/minimal_server/models/sanders"
    
    print(f"\n📦 Exporting {num_frames} frames from sanders dataset with landmarks...")
    
    # Open video files
    roi_cap = cv2.VideoCapture(os.path.join(sanders_dir, "rois_320_video.mp4"))
    model_input_cap = cv2.VideoCapture(os.path.join(sanders_dir, "model_inputs_video.mp4"))
    
    # Load audio features
    audio_features = np.load(os.path.join(sanders_dir, "aud_ave.npy"))
    
    visual_inputs = []
    audio_inputs = []
    landmarks_list = []
    
    for frame_id in range(num_frames):
        # Load video frames
        ret1, roi_frame = roi_cap.read()
        ret2, model_input_frame = model_input_cap.read()
        
        if not ret1 or not ret2:
            break
        
        # KEEP BGR FORMAT (don't convert to RGB!)
        roi_norm = roi_frame.astype(np.float32) / 255.0
        model_input_norm = model_input_frame.astype(np.float32) / 255.0
        
        roi_tensor = np.transpose(roi_norm, (2, 0, 1))
        model_input_tensor = np.transpose(model_input_norm, (2, 0, 1))
        
        visual_input = np.concatenate([roi_tensor, model_input_tensor], axis=0)
        
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
        
        visual_inputs.append(visual_input)
        audio_inputs.append(audio_reshaped)
        
        # Load landmarks
        landmark_file = os.path.join(sanders_dir, "landmarks", f"{frame_id}.lms")
        if os.path.exists(landmark_file):
            landmarks = []
            with open(landmark_file, 'r') as f:
                for line in f:
                    x, y = line.strip().split()
                    landmarks.append([float(x), float(y)])
            landmarks = np.array(landmarks)
            
            # Calculate ROI bounds
            xs = landmarks[:, 0]
            ys = landmarks[:, 1]
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max())
            y2 = int(ys.max())
            
            landmarks_list.append({
                "frame_id": frame_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1
            })
        else:
            landmarks_list.append(None)
        
        if (frame_id + 1) % 10 == 0:
            print(f"   Exported {frame_id + 1}/{num_frames} frames")
    
    roi_cap.release()
    model_input_cap.release()
    
    # Stack into arrays
    visual_data = np.stack(visual_inputs, axis=0).astype(np.float32)
    audio_data = np.stack(audio_inputs, axis=0).astype(np.float32)
    
    print(f"\n💾 Saving data...")
    print(f"   Visual: {visual_data.shape}")
    print(f"   Audio: {audio_data.shape}")
    print(f"   Landmarks: {len(landmarks_list)} frames")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary data
    visual_data.tofile(os.path.join(output_dir, "visual_input.bin"))
    audio_data.tofile(os.path.join(output_dir, "audio_input.bin"))
    
    # Save landmarks as JSON
    with open(os.path.join(output_dir, "landmarks.json"), 'w') as f:
        json.dump(landmarks_list, f, indent=2)
    
    # Save metadata
    metadata = {
        "num_frames": len(visual_inputs),
        "start_frame": 0,
        "visual_shape": list(visual_data.shape),
        "audio_shape": list(audio_data.shape),
        "dtype": "float32",
        "format": "BGR",
        "note": "CRITICAL: Data is in BGR format (NOT RGB). Keep as-is for inference."
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print file sizes
    visual_size = os.path.getsize(os.path.join(output_dir, "visual_input.bin")) / (1024 * 1024)
    audio_size = os.path.getsize(os.path.join(output_dir, "audio_input.bin")) / (1024 * 1024)
    
    print(f"\n✅ Data exported to {output_dir}/")
    print(f"   visual_input.bin - {visual_size:.2f} MB")
    print(f"   audio_input.bin - {audio_size:.2f} MB")
    print(f"   landmarks.json - ROI bounds for compositing")
    print(f"   metadata.json")

if __name__ == "__main__":
    export_for_go(num_frames=100)
