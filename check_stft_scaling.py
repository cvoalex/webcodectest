#!/usr/bin/env python3
"""
Check if librosa STFT has any scaling
"""

import numpy as np
import librosa

# Simple test signal
constant = np.ones(10240, dtype=np.float32)

# STFT
D = librosa.stft(constant, n_fft=800, hop_length=200, win_length=800, window='hann', center=True)

S = np.abs(D)

print(f"Constant signal STFT:")
print(f"  Shape: {S.shape}")
print(f"  Frame 2 DC: {S[0, 2]}")  # Frame 2 should be the first centered on actual audio
print(f"  Expected (window sum): ~400")

# Check window sum
window = np.hanning(800)
print(f"\nWindow sum: {window.sum()}")

# What if I manually compute STFT for frame 2 (centered at sample 0)
# With center=True, frame 0 is at sample -400, frame 1 at -200, frame 2 at 0
# So frame 2 uses samples from -400 to +399
# First 400 are zeros (padding), next 400 are ones
frame_data = np.concatenate([np.zeros(400), np.ones(400)])
windowed = frame_data * window
fft_result = np.fft.fft(windowed, n=800)
manual_dc = np.abs(fft_result[0])

print(f"\nManual computation for frame 2:")
print(f"  DC component: {manual_dc}")
print(f"  Matches librosa: {np.isclose(manual_dc, S[0, 2])}")

# Check the scaling
print(f"\nLibrosa / Manual ratio: {S[0, 2] / manual_dc}")
