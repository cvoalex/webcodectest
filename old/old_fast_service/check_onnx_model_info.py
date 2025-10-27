#!/usr/bin/env python3
"""
Check what the actual ONNX model expects
"""

import onnxruntime as ort
import numpy as np

model_path = "d:/Projects/webcodecstest/fast_service/models/default_model/models/99.onnx"

print("=" * 80)
print("ONNX MODEL INSPECTION")
print("=" * 80)
print()

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

print("Model Inputs:")
print("-" * 80)
for inp in session.get_inputs():
    print(f"  Name: {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type: {inp.type}")
    print()

print("Model Outputs:")
print("-" * 80)
for out in session.get_outputs():
    print(f"  Name: {out.name}")
    print(f"  Shape: {out.shape}")
    print(f"  Type: {out.type}")
    print()

print("=" * 80)
print("EXPECTED INPUT FORMATS:")
print("=" * 80)
print()

print("Based on model signature:")
visual_input = session.get_inputs()[0]
audio_input = session.get_inputs()[1]

print(f"Visual Input: {visual_input.name}")
print(f"  Expected shape: {visual_input.shape}")
print(f"  Interpretation: [batch, channels, height, width]")
print(f"  Channels: 6 (3 for face + 3 for masked face)")
print()

print(f"Audio Input: {audio_input.name}")
print(f"  Expected shape: {audio_input.shape}")
print(f"  Interpretation: [batch, channels, height, width]")
print()

# Check the comment in the code
print("=" * 80)
print("NOTES FROM unet_328.py:")
print("=" * 80)
print()
print("Test code at bottom shows:")
print("  img = torch.zeros([1, 6, 160, 160])    # Different size!")
print("  audio = torch.zeros([1, 16, 32, 32])   # Different shape!")
print()
print("But actual model we have expects:")
print(f"  visual: {visual_input.shape}")
print(f"  audio: {audio_input.shape}")
print()

if visual_input.shape[2] == 320 and visual_input.shape[3] == 320:
    print("✅ Model expects 320x320 images (what we're using)")
else:
    print(f"❌ Model expects {visual_input.shape[2]}x{visual_input.shape[3]} images!")

if audio_input.shape[1] == 32 and audio_input.shape[2] == 16:
    print("✅ Model expects [32, 16, 16] audio (AVE mode - what we're using)")
elif audio_input.shape[1] == 16 and audio_input.shape[2] == 32:
    print("❌ Model expects [16, 32, 32] audio (Hubert/Wenet mode?)")
