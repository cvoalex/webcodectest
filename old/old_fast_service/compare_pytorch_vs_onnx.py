#!/usr/bin/env python3
"""
Compare PyTorch vs ONNX output side by side
"""

import cv2
import numpy as np

# Load images
pytorch_out = cv2.imread("test_pytorch_single_frame/pytorch_output.png")
onnx_out = cv2.imread("test_single_frame/output_frame_0000.png")
input_face = cv2.imread("test_single_frame/input_face_0000.png")

print("PyTorch output shape:", pytorch_out.shape if pytorch_out is not None else "None")
print("ONNX output shape:", onnx_out.shape if onnx_out is not None else "None")
print("Input face shape:", input_face.shape if input_face is not None else "None")

if pytorch_out is not None and onnx_out is not None and input_face is not None:
    # Calculate difference
    diff = cv2.absdiff(pytorch_out, onnx_out)
    diff_enhanced = cv2.multiply(diff, 10)  # Enhance difference for visibility
    
    # Create 4-way comparison
    top_row = np.hstack([input_face, pytorch_out])
    bottom_row = np.hstack([onnx_out, diff_enhanced])
    comparison = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Input Face", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "PyTorch Output", (330, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "ONNX Output", (10, 350), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Diff x10", (330, 350), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite("comparison_pytorch_vs_onnx.png", comparison)
    print("\n✅ Comparison saved to: comparison_pytorch_vs_onnx.png")
    
    # Calculate numerical difference
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\n📊 Difference Statistics:")
    print(f"   Max pixel difference: {max_diff}")
    print(f"   Mean pixel difference: {mean_diff:.2f}")
    print(f"   Percentage difference: {mean_diff/255*100:.2f}%")
    
    # Check if images are visually similar
    if mean_diff < 5:
        print("\n✅ Outputs are very similar (mean diff < 5)")
    elif mean_diff < 20:
        print("\n⚠️  Outputs have noticeable differences (mean diff 5-20)")
    else:
        print("\n❌ Outputs are significantly different (mean diff > 20)")
else:
    print("\n❌ Failed to load all images")
    print("Make sure you ran:")
    print("  1. test_pytorch_single_frame.py")
    print("  2. test_onnx_with_real_data.py --frames 1")
