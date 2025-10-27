#!/usr/bin/env python3
"""
Test PyTorch model on single frame to compare with ONNX output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2

from unet_328 import Model

def test_pytorch_single_frame(
    model_path="models/default_model/models/99.pth",
    package_dir="models/default_model",
    frame_idx=100,
    output_dir="test_pytorch_single_frame"
):
    """Test PyTorch model on single frame"""
    
    print("\n" + "="*80)
    print("🔥 PYTORCH SINGLE FRAME TEST")
    print("="*80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    model = Model(n_channels=6, mode='ave')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {model_path}")
    
    # Load test data
    face_video_path = os.path.join(package_dir, "face_regions_320.mp4")
    masked_video_path = os.path.join(package_dir, "masked_regions_320.mp4")
    audio_features_path = os.path.join(package_dir, "aud_ave.npy")
    
    print(f"\n📦 Loading frame {frame_idx} from test data...")
    
    face_cap = cv2.VideoCapture(face_video_path)
    masked_cap = cv2.VideoCapture(masked_video_path)
    
    # Skip to frame
    for _ in range(frame_idx):
        face_cap.read()
        masked_cap.read()
    
    # Read frame
    ret1, face_frame = face_cap.read()
    ret2, masked_frame = masked_cap.read()
    
    if not ret1 or not ret2:
        print("❌ Failed to read frame")
        return
    
    face_cap.release()
    masked_cap.release()
    
    # Convert BGR to RGB
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1]
    face_tensor = torch.from_numpy(face_frame).float() / 255.0
    face_tensor = (face_tensor - 0.5) * 2.0
    
    masked_tensor = torch.from_numpy(masked_frame).float() / 255.0
    masked_tensor = (masked_tensor - 0.5) * 2.0
    
    # Transpose to [C, H, W]
    face_tensor = face_tensor.permute(2, 0, 1)
    masked_tensor = masked_tensor.permute(2, 0, 1)
    
    # Concatenate and add batch dimension [1, 6, 320, 320]
    visual_input = torch.cat([face_tensor, masked_tensor], dim=0).unsqueeze(0).to(device)
    
    # Load audio
    audio_features_full = np.load(audio_features_path)
    audio_frame = audio_features_full[frame_idx]
    
    # Reshape audio
    audio_reshaped = audio_frame[:512].reshape(32, 16)
    audio_tiled = np.tile(audio_reshaped[:, :, np.newaxis], (1, 1, 16))
    audio_input = torch.from_numpy(audio_tiled).float().unsqueeze(0).to(device)
    
    print(f"   Visual input: {visual_input.shape}")
    print(f"   Audio input: {audio_input.shape}")
    
    # Run inference
    print("\n🚀 Running PyTorch inference...")
    with torch.no_grad():
        output = model(visual_input, audio_input)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Output mean: {output.mean():.6f}")
    print(f"   Output std: {output.std():.6f}")
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to image
    output_np = output[0].cpu().numpy()  # [3, 320, 320]
    output_img = np.transpose(output_np, (1, 2, 0))  # [320, 320, 3]
    output_img = ((output_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    # Save PyTorch output
    cv2.imwrite(f"{output_dir}/pytorch_output.png", output_img_bgr)
    
    # Save input face
    input_img = ((face_frame / 255.0 * 2.0 - 1.0 + 1) * 127.5).clip(0, 255).astype(np.uint8)
    input_img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/input_face.png", input_img_bgr)
    
    # Create comparison
    comparison = np.hstack([input_img_bgr, output_img_bgr])
    cv2.imwrite(f"{output_dir}/comparison.png", comparison)
    
    print(f"\n✅ Output saved to: {output_dir}/")
    print(f"   - pytorch_output.png: Model output")
    print(f"   - input_face.png: Original input")
    print(f"   - comparison.png: Side-by-side")
    
    return output_np

if __name__ == "__main__":
    test_pytorch_single_frame()
