#!/usr/bin/env python3
"""
Check librosa STFT normalization
"""

import numpy as np
import librosa

# Create simple test: impulse at start
impulse = np.zeros(10240, dtype=np.float32)
impulse[0] = 1.0

print("Test 1: Impulse at start")
print(f"Input: {impulse[:10]}")

stft_result = librosa.stft(
    impulse,
    n_fft=800,
    hop_length=200,
    win_length=800,
    window='hann',
    center=False
)

magnitude = np.abs(stft_result)
print(f"STFT magnitude shape: {magnitude.shape}")
print(f"First frame magnitude (first 10): {magnitude[:10, 0]}")
print(f"First frame max: {magnitude[:, 0].max()}")

# Now compute manually with different normalizations
frame = impulse[0:800]
window = np.hanning(800)
windowed = frame * window

# FFT with no normalization
fft_none = np.fft.fft(windowed, n=800)
mag_none = np.abs(fft_none[:401])
print(f"\nFFT (no norm) first frame max: {mag_none.max()}")

# FFT with normalization by n
fft_norm_n = np.fft.fft(windowed, n=800) / 800
mag_norm_n = np.abs(fft_norm_n[:401])
print(f"FFT (norm by n=800) first frame max: {mag_norm_n.max()}")

# FFT with normalization by sqrt(n)
fft_norm_sqrt = np.fft.fft(windowed, n=800) / np.sqrt(800)
mag_norm_sqrt = np.abs(fft_norm_sqrt[:401])
print(f"FFT (norm by sqrt(n)) first frame max: {mag_norm_sqrt.max()}")

# Check which matches librosa
print(f"\nLibrosa max: {magnitude[:, 0].max()}")
print(f"Matches no norm: {np.allclose(mag_none, magnitude[:, 0])}")
print(f"Matches norm n: {np.allclose(mag_norm_n, magnitude[:, 0])}")
print(f"Matches norm sqrt(n): {np.allclose(mag_norm_sqrt, magnitude[:, 0])}")

# Check window normalization
window_sum = window.sum()
window_energy = (window ** 2).sum()
print(f"\nWindow sum: {window_sum}")
print(f"Window energy (sum of squares): {window_energy}")
print(f"sqrt(window energy): {np.sqrt(window_energy)}")

# Try normalizing by window energy
fft_norm_win = np.fft.fft(windowed, n=800) / np.sqrt(window_energy)
mag_norm_win = np.abs(fft_norm_win[:401])
print(f"FFT (norm by sqrt(window energy)) max: {mag_norm_win.max()}")
print(f"Matches librosa: {np.allclose(mag_norm_win, magnitude[:, 0])}")
