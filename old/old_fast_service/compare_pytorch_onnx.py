#!/usr/bin/env python3
"""
Compare PyTorch .pth vs ONNX inference using the same preprocessed data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import onnxruntime as ort
from unet_328 import Model

def load_frame_data(frame_id, package_dir="models/default_model"):
    """Load preprocessed data for one frame"""
    
    # Load face regions
    face_cap = cv2.VideoCapture(os.path.join(package_dir, "face_regions_320.mp4"))
    masked_cap = cv2.VideoCapture(os.path.join(package_dir, "masked_regions_320.mp4"))
    
    for _ in range(frame_id):
        face_cap.read()
        masked_cap.read()
    
    ret1, face_frame = face_cap.read()
    ret2, masked_frame = masked_cap.read()
    
    face_cap.release()
    masked_cap.release()
    
    if not ret1 or not ret2:
        raise ValueError(f"Could not read frame {frame_id}")
    
    # Convert BGR to RGB
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and transpose
    face_norm = face_frame.astype(np.float32) / 255.0
    masked_norm = masked_frame.astype(np.float32) / 255.0
    
    face_tensor = np.transpose(face_norm, (2, 0, 1))
    masked_tensor = np.transpose(masked_norm, (2, 0, 1))
    
    # Concatenate to 6 channels
    visual_input = np.concatenate([face_tensor, masked_tensor], axis=0)
    visual_input = np.expand_dims(visual_input, axis=0)
    
    # Load audio features
    audio_features = np.load(os.path.join(package_dir, "aud_ave.npy"))
    
    # CRITICAL FIX: Use 16-frame window (8 before + current + 7 after)
    left = frame_id - 8
    right = frame_id + 8
    
    # Handle boundaries with padding
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
    
    # Get the window of 16 frames
    audio_window = audio_features[left:right]  # [num_frames, 512]
    
    # Pad if needed
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
    
    # Flatten and reshape: [16, 512] -> [8192] -> [32, 16, 16]
    audio_flat = audio_window.flatten()  # [8192]
    audio_reshaped = audio_flat.reshape(32, 16, 16)  # [32, 16, 16]
    audio_input = np.expand_dims(audio_reshaped, axis=0).astype(np.float32)
    
    return visual_input.astype(np.float32), audio_input, face_frame

def test_pytorch(visual_input, audio_input, model_path):
    """Test with PyTorch .pth model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = Model(6, 'ave').to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    # Convert to tensors
    visual_tensor = torch.from_numpy(visual_input).to(device)
    audio_tensor = torch.from_numpy(audio_input).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(visual_tensor, audio_tensor)
    
    # Convert to numpy
    result = output[0].cpu().numpy()  # [3, 320, 320]
    
    return result

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
    
    result = outputs[0][0]  # [3, 320, 320]
    
    return result

def main():
    frame_id = 50  # Use frame with proper audio context (not 0 or near end)
    package_dir = "d:/Projects/webcodecstest/fast_service/models/default_model"
    pth_model = "d:/Projects/webcodecstest/fast_service/models/default_model/models/99.pth"
    onnx_model = "d:/Projects/webcodecstest/fast_service/models/default_model/models/99.onnx"
    
    print("\n" + "="*80)
    print("🔬 PYTORCH vs ONNX COMPARISON")
    print("="*80)
    
    # Load data
    print(f"\n📊 Loading frame {frame_id}...")
    visual_input, audio_input, face_img = load_frame_data(frame_id, package_dir)
    
    print(f"   Visual input: {visual_input.shape}, range [{visual_input.min():.3f}, {visual_input.max():.3f}]")
    print(f"   Audio input: {audio_input.shape}, range [{audio_input.min():.3f}, {audio_input.max():.3f}]")
    
    # Test PyTorch
    print(f"\n🔥 Testing PyTorch .pth model...")
    pytorch_output = test_pytorch(visual_input, audio_input, pth_model)
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output range: [{pytorch_output.min():.3f}, {pytorch_output.max():.3f}]")
    print(f"   Output mean: {pytorch_output.mean():.6f}")
    
    # Test ONNX
    print(f"\n⚡ Testing ONNX model...")
    onnx_output = test_onnx(visual_input, audio_input, onnx_model)
    print(f"   Output shape: {onnx_output.shape}")
    print(f"   Output range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")
    print(f"   Output mean: {onnx_output.mean():.6f}")
    
    # Compare
    print(f"\n📊 Comparison:")
    diff = np.abs(pytorch_output - onnx_output)
    print(f"   Mean absolute difference: {diff.mean():.6f}")
    print(f"   Max absolute difference: {diff.max():.6f}")
    print(f"   Percentage of pixels with diff > 0.01: {(diff > 0.01).sum() / diff.size * 100:.2f}%")
    
    # Save outputs
    output_dir = "output_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to images
    pytorch_img = np.transpose(pytorch_output, (1, 2, 0))
    pytorch_img = (pytorch_img * 255).clip(0, 255).astype(np.uint8)
    
    onnx_img = np.transpose(onnx_output, (1, 2, 0))
    onnx_img = (onnx_img * 255).clip(0, 255).astype(np.uint8)
    
    diff_img = np.transpose(diff, (1, 2, 0))
    diff_img = (diff_img * 255 * 10).clip(0, 255).astype(np.uint8)  # Amplify difference
    
    cv2.imwrite(f"{output_dir}/1_input_face.png", cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/2_pytorch_output.png", cv2.cvtColor(pytorch_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/3_onnx_output.png", cv2.cvtColor(onnx_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{output_dir}/4_difference_x10.png", cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))
    
    print(f"\n✅ Results saved to {output_dir}/")
    print(f"   1_input_face.png - Input face")
    print(f"   2_pytorch_output.png - PyTorch .pth output")
    print(f"   3_onnx_output.png - ONNX output")
    print(f"   4_difference_x10.png - Difference (amplified 10x)")

if __name__ == "__main__":
    main()
