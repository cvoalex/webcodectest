#!/usr/bin/env python3
"""
Understand librosa STFT scaling
"""

import numpy as np
import librosa

# Test with sine wave at known frequency
sr = 16000
t = np.arange(10240) / sr
freq = 440.0
sine = np.sin(2 * np.pi * freq * t).astype(np.float32)

# STFT
D_librosa = librosa.stft(sine, n_fft=800, hop_length=200, win_length=800, window='hann', center=True)
S_librosa = np.abs(D_librosa)

# Manual STFT for comparison (frame 2)
window = np.hanning(800)
# Frame 2 centered at sample 0
frame_start = 0 - 400
frame_end = frame_start + 800
# Need to pad
padded_sine = np.pad(sine, (400, 400), mode='constant')
frame = padded_sine[400:1200]  # This is frame 2
windowed = frame * window
fft_manual = np.fft.fft(windowed, n=800)
S_manual = np.abs(fft_manual[:401])

print("Sine wave STFT comparison (frame 2):")
print(f"  Librosa max: {S_librosa[:, 2].max()}")
print(f"  Manual max: {S_manual.max()}")
print(f"  Ratio: {S_librosa[:, 2].max() / S_manual.max()}")

# Find the peak bin
peak_bin_librosa = np.argmax(S_librosa[:, 2])
peak_bin_manual = np.argmax(S_manual)

print(f"\n  Librosa peak bin: {peak_bin_librosa}, value: {S_librosa[peak_bin_librosa, 2]}")
print(f"  Manual peak bin: {peak_bin_manual}, value: {S_manual[peak_bin_manual]}")
print(f"  Ratio at peak: {S_librosa[peak_bin_librosa, 2] / S_manual[peak_bin_manual]}")

# Check DC component
print(f"\n  Librosa DC: {S_librosa[0, 2]}")
print(f"  Manual DC: {S_manual[0]}")
if S_manual[0] > 1e-10:
    print(f"  Ratio at DC: {S_librosa[0, 2] / S_manual[0]}")

# The answer: librosa DOES NOT scale by 2 for the one-sided spectrum
# Let me check if my manual computation is wrong
print("\nChecking manual computation:")
print(f"  Frame mean: {frame.mean()}")
print(f"  Windowed sum: {windowed.sum()}")
print(f"  FFT DC (should be windowed sum): {fft_manual[0]}")
print(f"  Match: {np.isclose(windowed.sum(), fft_manual[0])}")
