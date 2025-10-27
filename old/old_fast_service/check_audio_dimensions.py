#!/usr/bin/env python3
"""
Check what audio dimensions we should actually be using
"""

import numpy as np

# From the reference code, get_audio_window returns 16 frames of 512-dim features
# a_win shape should be [16, 512]

print("Audio Window Analysis:")
print("=" * 60)
print()

print("1. get_audio_window returns: [16, 512]")
print("   - 16 frames (8 previous + current + 7 future)")
print("   - 512 features per frame (from AVE encoder)")
print()

print("2. Reshaping for 'ave' mode:")
print("   a_win_t.view(32, 16, 16)")
print()

# Calculate what this means
total_elements = 16 * 512
print(f"   Total elements: 16 × 512 = {total_elements}")
print(f"   Reshaped to: 32 × 16 × 16 = {32 * 16 * 16}")
print()

if total_elements == 32 * 16 * 16:
    print("   ✅ Dimensions match!")
else:
    print("   ❌ Dimensions DON'T match!")
print()

print("3. So the final audio input shape should be:")
print("   [batch, 32, 16, 16] = [1, 32, 16, 16]")
print()

print("4. But the comment in unet_328.py says:")
print("   # 输入维度[32,32,16,16]")
print("   This looks WRONG - probably a typo in the comment")
print("   The actual input is [batch=1, channels=32, H=16, W=16]")
print()

# Now let's verify what we're actually doing
print("=" * 60)
print("OUR CURRENT IMPLEMENTATION:")
print("=" * 60)
print()

# Load sample audio features
audio_features = np.load("models/default_model/aud_ave.npy")
print(f"Audio features shape: {audio_features.shape}")
print(f"Per-frame feature dimension: {audio_features.shape[1]}")
print()

# Check what we're doing
frame_id = 100
audio_feat = audio_features[frame_id]  # Shape: [512]

print(f"Single frame audio: {audio_feat.shape}")
print()

# Our reshaping
audio_reshaped = audio_feat[:512].reshape(32, 16)
audio_tiled = np.tile(audio_reshaped[:, :, np.newaxis], (1, 1, 16))

print(f"After reshape(32, 16): {audio_reshaped.shape}")
print(f"After tile to [32, 16, 16]: {audio_tiled.shape}")
print()

print("❌ PROBLEM FOUND!")
print("We're only using ONE frame's audio features!")
print("We should be using a WINDOW of 16 frames!")
print()

print("CORRECT approach:")
print("1. Get 16 frames of audio (8 before + current + 7 after)")
print("2. Flatten to [16 × 512] = [8192] elements")
print("3. Reshape to [32, 16, 16] (or view as [32, 16, 16])")
print("4. Add batch dimension: [1, 32, 16, 16]")
