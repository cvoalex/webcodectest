"""
Create reference audio tensor to compare with Go implementation
"""
import numpy as np
import torch

# Load reference features from aud_ave.npy
aud_ave = np.load("old/old_minimal_server/models/sanders/aud_ave.npy")
print(f"Loaded aud_ave.npy: {aud_ave.shape}")

# Take first frame's features
frame_0_features = aud_ave[0]  # [512]
print(f"\nFrame 0 features: {frame_0_features.shape}")
print(f"First 10 values: {frame_0_features[:10]}")

# Apply the EXACT Python transformation from inference_engine.py
audio_slice = torch.from_numpy(frame_0_features).unsqueeze(0)  # [1, 512]
print(f"\nAfter unsqueeze(0): {audio_slice.shape}")

audio_tensor = audio_slice.unsqueeze(0)  # [1, 1, 512]
print(f"After unsqueeze(0) again: {audio_tensor.shape}")

audio_reshaped = audio_tensor.view(1, 32, 16).unsqueeze(-1).repeat(1, 1, 1, 16)  # [1, 32, 16, 16]
print(f"After view+unsqueeze+repeat: {audio_reshaped.shape}")

# Flatten and save first 100 values for comparison
flat = audio_reshaped.flatten().numpy()
print(f"\nFlattened shape: {flat.shape}")
print(f"First 20 values: {flat[:20]}")
print(f"Values 16-20 (should be feature[1] repeated): {flat[16:20]}")
print(f"Values 256-260 (should be feature[16] repeated): {flat[256:260]}")

# Save for Go comparison
np.save("audio_reference_flat.npy", flat)
print(f"\n✅ Saved reference to audio_reference_flat.npy")
print(f"   First value (feature[0]): {frame_0_features[0]}")
print(f"   Should appear 16 times at indices 0-15")
