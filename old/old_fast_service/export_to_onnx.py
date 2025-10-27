#!/usr/bin/env python3
"""
Export PyTorch lip sync model to ONNX format for optimized inference
"""

import torch
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet_328 import Model

def export_model_to_onnx(
    model_path="models/default_model/models/99.pth",
    output_path="models/default_model/models/99.onnx",
    opset_version=17
):
    """
    Export the PyTorch model to ONNX format
    
    Args:
        model_path: Path to the PyTorch .pth model file
        output_path: Path where the ONNX model will be saved
        opset_version: ONNX opset version (17 is good for modern CUDA/TensorRT)
    """
    
    print("🔄 Loading PyTorch model...")
    
    # Initialize model architecture
    model = Model(
        n_channels=6,
        mode='ave'
    )
    
    # Load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy inputs for export
    # Visual input: (batch_size=1, channels=6, height=320, width=320)
    dummy_visual = torch.randn(1, 6, 320, 320, device=device)
    # Audio input: (batch_size=1, channels=32, height=16, width=16)
    dummy_audio = torch.randn(1, 32, 16, 16, device=device)
    
    print("🔄 Exporting to ONNX...")
    print(f"   Visual input shape: {dummy_visual.shape}")
    print(f"   Audio input shape: {dummy_audio.shape}")
    print(f"   Opset version: {opset_version}")
    
    # Export with optimization
    torch.onnx.export(
        model,
        (dummy_visual, dummy_audio),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimize constant folding
        input_names=['visual_input', 'audio_input'],
        output_names=['output'],
        dynamic_axes={
            'visual_input': {0: 'batch_size'},  # Variable batch size
            'audio_input': {0: 'batch_size'},   # Variable batch size
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported to: {output_path}")
    
    # Get file sizes for comparison
    pth_size = os.path.getsize(model_path) / (1024 * 1024)
    onnx_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"📦 PyTorch model size: {pth_size:.2f} MB")
    print(f"📦 ONNX model size: {onnx_size:.2f} MB")
    
    return output_path

if __name__ == '__main__':
    export_model_to_onnx()
