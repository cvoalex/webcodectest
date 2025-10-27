#!/usr/bin/env python3
"""
Check actual librosa source and scipy behavior
"""

import numpy as np
from scipy import signal
import librosa

# Constant signal
constant = np.ones(800, dtype=np.float32)

print("Testing scipy.stft vs manual FFT")
print("="*60)

# Scipy STFT
f, t, Zxx = signal.stft(
    constant,
    fs=16000,
    window='hann',
    nperseg=800,
    noverlap=600,  # hop = 200
    nfft=800,
    boundary=None,  # No padding
    padded=False
)

print(f"Scipy STFT result shape: {Zxx.shape}")
print(f"DC value: {np.abs(Zxx[0, 0])}")
print(f"First 10 bins: {np.abs(Zxx[:10, 0])}")

# Manual FFT
window = signal.get_window('hann', 800)
print(f"\nWindow sum: {window.sum()}")
windowed = constant * window
fft_result = np.fft.fft(windowed)
print(f"Manual FFT DC: {np.abs(fft_result[0])}")

# Normalized manual FFT (by window sum)
fft_normalized = np.fft.fft(windowed) / window.sum() * len(window)
print(f"Manual FFT DC (normalized): {np.abs(fft_normalized[0])}")

print("\n" + "="*60)
print("Testing librosa.stft")
print("="*60)

# Librosa STFT
librosa_stft = librosa.stft(
    constant,
    n_fft=800,
    hop_length=200,
    win_length=800,
    window='hann',
    center=False
)

print(f"Librosa STFT shape: {librosa_stft.shape}")
print(f"DC value: {np.abs(librosa_stft[0, 0])}")
print(f"First 10 bins: {np.abs(librosa_stft[:10, 0])}")

# Check the source - librosa uses scipy internally
# But applies scaling differently
print("\n" + "="*60)
print("Check window scaling")
print("="*60)

# scipy.signal.stft applies scaling
# Default is to scale by 1/sqrt(n) for forward FFT, but that's for FFT itself
# For STFT, it's more complex

# Let's look at what happens with the window
# scipy scales the window by a factor so that the overall gain is 1.0 for a constant input
window_hann = np.hanning(800)
scale_factor = np.sqrt(1.0 / (window_hann ** 2).sum())
print(f"Hanning window energy: {(window_hann**2).sum()}")
print(f"Scale factor (1/sqrt(energy)): {scale_factor}")
print(f"Scaled window sum: {(window_hann * scale_factor).sum()}")

# Try this scaling
windowed_scaled = constant * window_hann * scale_factor
fft_scaled = np.fft.fft(windowed_scaled)
print(f"Scaled FFT DC: {np.abs(fft_scaled[0])}")

# Librosa default is to use 'window' scaling which normalizes by window
# But actually, it doesn't normalize the FFT at all
# Let me check the actual values more carefully

print("\n" + "="*60)
print("Detailed comparison")
print("="*60)

# Simple sine wave test
freq = 440.0
sr = 16000
duration = 800 / sr
t_sine = np.linspace(0, duration, 800, endpoint=False)
sine = np.sin(2 * np.pi * freq * t_sine).astype(np.float32)

librosa_sine_stft = librosa.stft(sine, n_fft=800, hop_length=200, win_length=800, window='hann', center=False)
print(f"Sine wave STFT max: {np.abs(librosa_sine_stft[:, 0]).max()}")
print(f"Sine wave STFT argmax: {np.argmax(np.abs(librosa_sine_stft[:, 0]))}")

# Manual
sine_windowed = sine * np.hanning(800)
sine_fft = np.fft.fft(sine_windowed)
print(f"Manual sine FFT max: {np.abs(sine_fft[:401]).max()}")
print(f"Manual sine FFT argmax: {np.argmax(np.abs(sine_fft[:401]))}")
