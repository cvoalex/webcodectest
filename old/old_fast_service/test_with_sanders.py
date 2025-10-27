#!/usr/bin/env python3
"""
Test with the ACTUAL sanders dataset and model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import onnxruntime as ort
from unet_328 import Model

def load_sanders_frame(frame_id, sanders_dir="d:/Projects/webcodecstest/minimal_server/models/sanders"):
    """Load frame from sanders dataset videos"""
    
    # Load from the actual video files
    roi_cap = cv2.VideoCapture(os.path.join(sanders_dir, "rois_320_video.mp4"))
    model_input_cap = cv2.VideoCapture(os.path.join(sanders_dir, "model_inputs_video.mp4"))
    
    # Skip to frame
    for _ in range(frame_id):
        roi_cap.read()
        model_input_cap.read()
    
    ret1, roi_frame = roi_cap.read()
    ret2, model_input_frame = model_input_cap.read()
    
    roi_cap.release()
    model_input_cap.release()
    
    if not ret1 or not ret2:
        raise ValueError(f"Could not read frame {frame_id}")
    
    # DON'T convert BGR to RGB - keep as BGR for model!
    # The reference code loads with cv2.imread which gives BGR
    # roi_img = cv2.imread(roi_path)  # This is BGR!
    
    # Normalize and transpose (keep BGR order)
    roi_norm = roi_frame.astype(np.float32) / 255.0
    model_input_norm = model_input_frame.astype(np.float32) / 255.0
    
    roi_tensor = np.transpose(roi_norm, (2, 0, 1))  # [3, 320, 320] BGR
    model_input_tensor = np.transpose(model_input_norm, (2, 0, 1))  # [3, 320, 320] BGR
    
    # Concatenate
    visual_input = np.concatenate([roi_tensor, model_input_tensor], axis=0)
    visual_input = np.expand_dims(visual_input, axis=0)
    
    # Load audio
    audio_features = np.load(os.path.join(sanders_dir, "aud_ave.npy"))
    
    # Get 16-frame window
    left = frame_id - 8
    right = frame_id + 8
    
    if left < 0:
        pad_left = -left
        left = 0
    else:
        pad_left = 0
    
    if right > len(audio_features):
        pad_right = right - len(audio_features)
        right = len(audio_features)
    else:
        pad_right = 0
    
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
    
    # Flatten and reshape
    audio_flat = audio_window.flatten()
    audio_reshaped = audio_flat.reshape(32, 16, 16)
    audio_input = np.expand_dims(audio_reshaped, axis=0).astype(np.float32)
    
    return visual_input.astype(np.float32), audio_input, roi_frame

def test_onnx(visual_input, audio_input, model_path):
    """Test with ONNX model"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    outputs = session.run(
        output_names,
        {
            input_names[0]: visual_input,
            input_names[1]: audio_input
        }
    )
    
    return outputs[0][0]

def main():
    frame_id = 50
    sanders_dir = "d:/Projects/webcodecstest/minimal_server/models/sanders"
    onnx_model = os.path.join(sanders_dir, "checkpoint/model_best.onnx")
    
    print("\n" + "="*80)
    print("🎯 TESTING WITH ACTUAL SANDERS DATA")
    print("="*80)
    
    # Load data
    print(f"\n📊 Loading frame {frame_id} from sanders dataset...")
    visual_input, audio_input, roi_frame_bgr = load_sanders_frame(frame_id, sanders_dir)
    
    print(f"   Visual input: {visual_input.shape}, range [{visual_input.min():.3f}, {visual_input.max():.3f}]")
    print(f"   Audio input: {audio_input.shape}, range [{audio_input.min():.3f}, {audio_input.max():.3f}]")
    print(f"   ROI frame: {roi_frame_bgr.shape}, BGR format")
    
    # Test ONNX
    print(f"\n⚡ Running ONNX inference...")
    output = test_onnx(visual_input, audio_input, onnx_model)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Output mean: {output.mean():.6f}")
    
    # Convert to image (keep BGR format to match input)
    output_img = np.transpose(output, (1, 2, 0))
    output_img = (output_img * 255).clip(0, 255).astype(np.uint8)
    
    # Save outputs
    output_dir = "output_sanders_test"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f"{output_dir}/1_input_roi_bgr.png", roi_frame_bgr)
    cv2.imwrite(f"{output_dir}/2_output_bgr.png", output_img)
    
    # Create comparison
    comparison = np.hstack([roi_frame_bgr, output_img])
    cv2.imwrite(f"{output_dir}/3_comparison.png", comparison)
    
    print(f"\n✅ Results saved to {output_dir}/")
    print(f"   1_input_roi_bgr.png - Input ROI (BGR)")
    print(f"   2_output_bgr.png - Generated output (BGR)")
    print(f"   3_comparison.png - Side-by-side")
    print()
    print("🎯 Check if the colors look correct now!")

if __name__ == "__main__":
    main()
