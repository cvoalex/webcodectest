#!/usr/bin/env python3
"""
Check the structure of saved audio features
"""

import numpy as np

# Load the saved audio features
audio_features = np.load("models/default_model/aud_ave.npy")

print(f"Audio features shape: {audio_features.shape}")
print(f"Audio features dtype: {audio_features.dtype}")
print(f"Audio features min: {audio_features.min()}")
print(f"Audio features max: {audio_features.max()}")
print(f"Audio features mean: {audio_features.mean()}")
print(f"Audio features std: {audio_features.std()}")

print(f"\nFirst frame audio feature:")
print(f"  Shape: {audio_features[0].shape}")
print(f"  First 10 values: {audio_features[0][:10]}")

print(f"\nFrame 100 audio feature:")
print(f"  Shape: {audio_features[100].shape}")
print(f"  First 10 values: {audio_features[100][:10]}")
