#!/usr/bin/env python3
"""
Test STFT with constant signal to check normalization
"""

import numpy as np
import librosa

# Create constant signal
constant = np.ones(10240, dtype=np.float32)

print("Test: Constant signal (all 1.0)")
print(f"Input: {constant[:10]}")

stft_result = librosa.stft(
    constant,
    n_fft=800,
    hop_length=200,
    win_length=800,
    window='hann',
    center=False
)

magnitude = np.abs(stft_result)
print(f"\nSTFT magnitude shape: {magnitude.shape}")
print(f"First frame magnitude (bins 0-10): {magnitude[:11, 0]}")
print(f"First frame DC (bin 0): {magnitude[0, 0]}")

# Manual computation
frame = constant[0:800]
window = np.hanning(800)
windowed = frame * window

print(f"\nWindow stats:")
print(f"  Sum: {window.sum()}")
print(f"  Energy (sum of squares): {(window**2).sum()}")

# FFT
fft_result = np.fft.fft(windowed, n=800)
mag_manual = np.abs(fft_result[:401])

print(f"\nManual FFT:")
print(f"  DC (bin 0): {mag_manual[0]}")
print(f"  Bins 0-10: {mag_manual[:11]}")

print(f"\nComparison:")
print(f"  Librosa DC: {magnitude[0, 0]}")
print(f"  Manual DC: {mag_manual[0]}")
print(f"  Ratio: {mag_manual[0] / magnitude[0, 0] if magnitude[0, 0] != 0 else 'N/A'}")

# Test different normalizations
print(f"\nTrying different normalizations:")
print(f"  FFT / 1: {mag_manual[0]}")
print(f"  FFT / 800: {mag_manual[0] / 800}")
print(f"  FFT / 400: {mag_manual[0] / 400}")
print(f"  FFT / sqrt(800): {mag_manual[0] / np.sqrt(800)}")
print(f"  FFT / 2: {mag_manual[0] / 2}")
print(f"  Librosa: {magnitude[0, 0]}")

# Check the actual librosa code behavior
print(f"\n{'='*60}")
print("Checking if matches any normalization:")
for norm_factor in [1, 2, 400, 800, np.sqrt(800), window.sum(), np.sqrt((window**2).sum())]:
    normalized = mag_manual / norm_factor
    if np.allclose(normalized, magnitude[:, 0]):
        print(f"  MATCH! Normalization factor: {norm_factor}")
