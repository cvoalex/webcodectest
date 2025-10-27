#!/usr/bin/env python3
"""
Validate that ONNX inference produces correct outputs by comparing with PyTorch
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import onnxruntime as ort
from unet_328 import Model
import cv2

def validate_onnx_inference(
    pytorch_path="models/default_model/models/99.pth",
    onnx_path="models/default_model/models/99.onnx",
    save_comparison=True
):
    """
    Compare PyTorch and ONNX inference outputs to validate correctness
    """
    
    print("\n" + "="*80)
    print("🔍 VALIDATING ONNX INFERENCE")
    print("="*80)
    
    # Load PyTorch model
    print("\n📦 Loading PyTorch model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_model = Model(n_channels=6, mode='ave')
    pytorch_model.load_state_dict(torch.load(pytorch_path, map_location=device))
    pytorch_model.to(device)
    pytorch_model.eval()
    print(f"✅ PyTorch model loaded on {device}")
    
    # Load ONNX model
    print("\n📦 Loading ONNX model...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    onnx_session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"✅ ONNX model loaded with provider: {onnx_session.get_providers()[0]}")
    
    # Create test inputs (NOT random - use realistic values)
    print("\n🎲 Creating test inputs...")
    batch_size = 1
    
    # Visual input: simulate normalized image data [-1, 1]
    visual_input_torch = torch.randn(batch_size, 6, 320, 320, device=device) * 0.5
    
    # Audio input: simulate mel spectrogram features
    audio_input_torch = torch.randn(batch_size, 32, 16, 16, device=device) * 0.3
    
    print(f"   Visual input range: [{visual_input_torch.min():.3f}, {visual_input_torch.max():.3f}]")
    print(f"   Audio input range: [{audio_input_torch.min():.3f}, {audio_input_torch.max():.3f}]")
    
    # Run PyTorch inference
    print("\n🔥 Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = pytorch_model(visual_input_torch, audio_input_torch)
    
    print(f"   PyTorch output shape: {pytorch_output.shape}")
    print(f"   PyTorch output range: [{pytorch_output.min():.3f}, {pytorch_output.max():.3f}]")
    print(f"   PyTorch output mean: {pytorch_output.mean():.6f}")
    print(f"   PyTorch output std: {pytorch_output.std():.6f}")
    
    # Run ONNX inference
    print("\n🚀 Running ONNX inference...")
    visual_input_np = visual_input_torch.cpu().numpy()
    audio_input_np = audio_input_torch.cpu().numpy()
    
    input_names = [inp.name for inp in onnx_session.get_inputs()]
    output_names = [out.name for out in onnx_session.get_outputs()]
    
    onnx_outputs = onnx_session.run(
        output_names,
        {
            input_names[0]: visual_input_np,
            input_names[1]: audio_input_np
        }
    )
    onnx_output = onnx_outputs[0]
    
    print(f"   ONNX output shape: {onnx_output.shape}")
    print(f"   ONNX output range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")
    print(f"   ONNX output mean: {onnx_output.mean():.6f}")
    print(f"   ONNX output std: {onnx_output.std():.6f}")
    
    # Compare outputs
    print("\n📊 COMPARING OUTPUTS...")
    pytorch_output_np = pytorch_output.cpu().numpy()
    
    # Calculate differences
    abs_diff = np.abs(pytorch_output_np - onnx_output)
    rel_diff = abs_diff / (np.abs(pytorch_output_np) + 1e-8)
    
    print(f"   Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"   Max absolute difference: {abs_diff.max():.6f}")
    print(f"   Mean relative difference: {rel_diff.mean():.6f}")
    print(f"   Max relative difference: {rel_diff.max():.6f}")
    
    # Check if outputs are close enough
    atol = 1e-3  # Absolute tolerance
    rtol = 1e-3  # Relative tolerance
    
    are_close = np.allclose(pytorch_output_np, onnx_output, atol=atol, rtol=rtol)
    
    if are_close:
        print(f"\n✅ SUCCESS! Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n⚠️ WARNING! Outputs differ more than tolerance (atol={atol}, rtol={rtol})")
        
        # Find where differences are largest
        max_diff_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        print(f"   Largest difference at index {max_diff_idx}:")
        print(f"     PyTorch: {pytorch_output_np[max_diff_idx]:.6f}")
        print(f"     ONNX: {onnx_output[max_diff_idx]:.6f}")
        print(f"     Diff: {abs_diff[max_diff_idx]:.6f}")
    
    # Save comparison images
    if save_comparison:
        print("\n💾 Saving comparison images...")
        os.makedirs("validation_output", exist_ok=True)
        
        # PyTorch output
        pytorch_img = pytorch_output_np[0]  # [3, 320, 320]
        pytorch_img = np.transpose(pytorch_img, (1, 2, 0))  # [320, 320, 3]
        
        # Check if output is in [-1, 1] or [0, 1] range
        if pytorch_img.min() < -0.5:
            # Assume [-1, 1] range, convert to [0, 255]
            pytorch_img = ((pytorch_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        else:
            # Assume [0, 1] range, convert to [0, 255]
            pytorch_img = (pytorch_img * 255).clip(0, 255).astype(np.uint8)
        
        pytorch_img_bgr = cv2.cvtColor(pytorch_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("validation_output/pytorch_output.png", pytorch_img_bgr)
        print("   Saved: validation_output/pytorch_output.png")
        
        # ONNX output
        onnx_img = onnx_output[0]  # [3, 320, 320]
        onnx_img = np.transpose(onnx_img, (1, 2, 0))  # [320, 320, 3]
        
        if onnx_img.min() < -0.5:
            onnx_img = ((onnx_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        else:
            onnx_img = (onnx_img * 255).clip(0, 255).astype(np.uint8)
        
        onnx_img_bgr = cv2.cvtColor(onnx_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("validation_output/onnx_output.png", onnx_img_bgr)
        print("   Saved: validation_output/onnx_output.png")
        
        # Difference map
        diff_img = np.abs(pytorch_img.astype(float) - onnx_img.astype(float))
        diff_img = (diff_img / diff_img.max() * 255).astype(np.uint8)
        cv2.imwrite("validation_output/difference_map.png", diff_img)
        print("   Saved: validation_output/difference_map.png")
        
        print("\n   Check these images to see if they look reasonable!")
        print("   They should show a face/lip-sync frame, not random noise.")
    
    # Test with another random input to verify consistency
    print("\n🔄 Testing with second input...")
    visual_input_torch2 = torch.randn(batch_size, 6, 320, 320, device=device) * 0.5
    audio_input_torch2 = torch.randn(batch_size, 32, 16, 16, device=device) * 0.3
    
    with torch.no_grad():
        pytorch_output2 = pytorch_model(visual_input_torch2, audio_input_torch2)
    
    visual_input_np2 = visual_input_torch2.cpu().numpy()
    audio_input_np2 = audio_input_torch2.cpu().numpy()
    
    onnx_outputs2 = onnx_session.run(
        output_names,
        {
            input_names[0]: visual_input_np2,
            input_names[1]: audio_input_np2
        }
    )
    onnx_output2 = onnx_outputs2[0]
    
    pytorch_output_np2 = pytorch_output2.cpu().numpy()
    abs_diff2 = np.abs(pytorch_output_np2 - onnx_output2)
    
    print(f"   Second test - Mean abs diff: {abs_diff2.mean():.6f}, Max abs diff: {abs_diff2.max():.6f}")
    
    are_close2 = np.allclose(pytorch_output_np2, onnx_output2, atol=atol, rtol=rtol)
    if are_close2:
        print(f"   ✅ Second test also matches!")
    else:
        print(f"   ⚠️ Second test also shows differences")
    
    print("\n" + "="*80)
    if are_close and are_close2:
        print("✅ VALIDATION PASSED: ONNX inference is correct!")
    else:
        print("⚠️ VALIDATION ISSUES: ONNX inference may have problems")
    print("="*80)
    
    return are_close and are_close2

if __name__ == "__main__":
    validate_onnx_inference()
