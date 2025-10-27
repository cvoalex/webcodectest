#!/usr/bin/env python3
"""
Detailed comparison of PyTorch vs ONNX output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import onnxruntime as ort
from unet_328 import Model

def compare_single_frame(frame_id=100):
    """Compare PyTorch vs ONNX on a single frame with detailed analysis"""
    
    print("\n" + "="*80)
    print(f"🔬 DETAILED COMPARISON: Frame {frame_id}")
    print("="*80)
    
    # Load model
    model_path = "models/default_model/models/99.pth"
    onnx_path = "models/default_model/models/99.onnx"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(n_channels=6, mode='ave')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load ONNX
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Load data
    face_cap = cv2.VideoCapture("models/default_model/face_regions_320.mp4")
    masked_cap = cv2.VideoCapture("models/default_model/masked_regions_320.mp4")
    audio_features = np.load("models/default_model/aud_ave.npy")
    
    # Skip to frame
    for _ in range(frame_id):
        face_cap.read()
        masked_cap.read()
    
    # Read frame
    ret1, face_frame = face_cap.read()
    ret2, masked_frame = masked_cap.read()
    
    face_cap.release()
    masked_cap.release()
    
    # Convert and normalize
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    
    face_norm = (face_frame.astype(np.float32) / 255.0 - 0.5) * 2.0
    masked_norm = (masked_frame.astype(np.float32) / 255.0 - 0.5) * 2.0
    
    # Transpose to [C, H, W]
    face_tensor = np.transpose(face_norm, (2, 0, 1))
    masked_tensor = np.transpose(masked_norm, (2, 0, 1))
    
    # Concatenate
    visual_input = np.concatenate([face_tensor, masked_tensor], axis=0)
    visual_input = np.expand_dims(visual_input, axis=0)  # [1, 6, 320, 320]
    
    # Prepare audio
    audio_feat = audio_features[frame_id]
    audio_reshaped = audio_feat[:512].reshape(32, 16)
    audio_tiled = np.tile(audio_reshaped[:, :, np.newaxis], (1, 1, 16))
    audio_input = np.expand_dims(audio_tiled, axis=0).astype(np.float32)  # [1, 32, 16, 16]
    
    print(f"\n📊 INPUT ANALYSIS")
    print(f"{'='*80}")
    print(f"Visual input shape: {visual_input.shape}")
    print(f"Visual input range: [{visual_input.min():.3f}, {visual_input.max():.3f}]")
    print(f"Visual input mean: {visual_input.mean():.6f}")
    print(f"Audio input shape: {audio_input.shape}")
    print(f"Audio input range: [{audio_input.min():.3f}, {audio_input.max():.3f}]")
    print(f"Audio input mean: {audio_input.mean():.6f}")
    
    # PyTorch inference
    print(f"\n🔥 PYTORCH INFERENCE")
    print(f"{'='*80}")
    
    visual_torch = torch.from_numpy(visual_input).to(device)
    audio_torch = torch.from_numpy(audio_input).to(device)
    
    with torch.no_grad():
        pytorch_output = model(visual_torch, audio_torch)
    
    pytorch_np = pytorch_output.cpu().numpy()[0]
    
    print(f"PyTorch output shape: {pytorch_np.shape}")
    print(f"PyTorch output range: [{pytorch_np.min():.3f}, {pytorch_np.max():.3f}]")
    print(f"PyTorch output mean: {pytorch_np.mean():.6f}")
    print(f"PyTorch output std: {pytorch_np.std():.6f}")
    
    # Count values in different ranges
    near_zero = np.sum((pytorch_np >= -0.1) & (pytorch_np <= 0.1)) / pytorch_np.size * 100
    near_one = np.sum((pytorch_np >= 0.9) & (pytorch_np <= 1.0)) / pytorch_np.size * 100
    neg_one = np.sum((pytorch_np >= -1.0) & (pytorch_np <= -0.9)) / pytorch_np.size * 100
    
    print(f"Values near 0 [-0.1, 0.1]: {near_zero:.2f}%")
    print(f"Values near 1 [0.9, 1.0]: {near_one:.2f}%")
    print(f"Values near -1 [-1.0, -0.9]: {neg_one:.2f}%")
    
    # ONNX inference
    print(f"\n🟦 ONNX INFERENCE")
    print(f"{'='*80}")
    
    onnx_output = session.run(['output'], {
        'visual_input': visual_input,
        'audio_input': audio_input
    })[0][0]
    
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"ONNX output range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")
    print(f"ONNX output mean: {onnx_output.mean():.6f}")
    print(f"ONNX output std: {onnx_output.std():.6f}")
    
    near_zero = np.sum((onnx_output >= -0.1) & (onnx_output <= 0.1)) / onnx_output.size * 100
    near_one = np.sum((onnx_output >= 0.9) & (onnx_output <= 1.0)) / onnx_output.size * 100
    neg_one = np.sum((onnx_output >= -1.0) & (onnx_output <= -0.9)) / onnx_output.size * 100
    
    print(f"Values near 0 [-0.1, 0.1]: {near_zero:.2f}%")
    print(f"Values near 1 [0.9, 1.0]: {near_one:.2f}%")
    print(f"Values near -1 [-1.0, -0.9]: {neg_one:.2f}%")
    
    # Comparison
    print(f"\n📊 PYTORCH vs ONNX COMPARISON")
    print(f"{'='*80}")
    
    diff = np.abs(pytorch_np - onnx_output)
    print(f"Absolute difference mean: {diff.mean():.6f}")
    print(f"Absolute difference max: {diff.max():.6f}")
    print(f"Absolute difference std: {diff.std():.6f}")
    
    match_pct = np.sum(diff < 0.01) / diff.size * 100
    print(f"Values matching within 0.01: {match_pct:.2f}%")
    
    # Save comparison images
    os.makedirs("detailed_comparison", exist_ok=True)
    
    # Original input
    orig_display = (face_frame).astype(np.uint8)
    cv2.imwrite("detailed_comparison/01_input.png", cv2.cvtColor(orig_display, cv2.COLOR_RGB2BGR))
    
    # PyTorch output
    pytorch_display = np.transpose(pytorch_np, (1, 2, 0))
    pytorch_display = ((pytorch_display + 1) * 127.5).clip(0, 255).astype(np.uint8)
    cv2.imwrite("detailed_comparison/02_pytorch_output.png", cv2.cvtColor(pytorch_display, cv2.COLOR_RGB2BGR))
    
    # ONNX output  
    onnx_display = np.transpose(onnx_output, (1, 2, 0))
    onnx_display = ((onnx_display + 1) * 127.5).clip(0, 255).astype(np.uint8)
    cv2.imwrite("detailed_comparison/03_onnx_output.png", cv2.cvtColor(onnx_display, cv2.COLOR_RGB2BGR))
    
    # Difference heatmap
    diff_display = (diff * 255).clip(0, 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(np.transpose(diff_display[0], (1, 2, 0)), cv2.COLORMAP_JET)
    cv2.imwrite("detailed_comparison/04_difference_heatmap.png", diff_heatmap)
    
    # Side by side
    comparison = np.hstack([
        cv2.cvtColor(orig_display, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(pytorch_display, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(onnx_display, cv2.COLOR_RGB2BGR)
    ])
    cv2.imwrite("detailed_comparison/05_comparison.png", comparison)
    
    print(f"\n✅ Saved comparison images to detailed_comparison/")
    print(f"   01_input.png - Original input face")
    print(f"   02_pytorch_output.png - PyTorch output")
    print(f"   03_onnx_output.png - ONNX output")
    print(f"   04_difference_heatmap.png - Difference visualization")
    print(f"   05_comparison.png - Side-by-side (Input | PyTorch | ONNX)")
    
    return pytorch_np, onnx_output

if __name__ == "__main__":
    pytorch_out, onnx_out = compare_single_frame(frame_id=100)
    print("\n✅ Analysis complete!")
