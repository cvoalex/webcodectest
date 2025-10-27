#!/usr/bin/env python3
"""
Test if librosa.stft has a DC bin issue
"""

import numpy as np
import librosa

# Create a simple test: DC offset signal
dc_signal = np.ones(10240, dtype=np.float32) * 0.1

print("Test 1: DC offset signal (all 0.1)")
print(f"Signal mean: {dc_signal.mean()}")
print(f"Signal sum (first 800): {dc_signal[:800].sum()}")

# Compute STFT
stft = librosa.stft(dc_signal, n_fft=800, hop_length=200, win_length=800, window='hann', center=False)
magnitude = np.abs(stft)

print(f"\nSTFT DC bin (first frame): {magnitude[0, 0]}")
print(f"Expected (approx window_sum * 0.1): {400 * 0.1}")

# Manual FFT
frame = dc_signal[:800]
window = np.hanning(800)
windowed = frame * window
fft_manual = np.fft.fft(windowed, n=800)
print(f"Manual FFT DC: {np.abs(fft_manual[0])}")

# Test 2: Our actual signal
print("\n" + "="*60)
print("Test 2: Actual audio signal")

import json
with open('audio_test_data/reference_data.json', 'r') as f:
    data = json.load(f)
    audio = np.array(data['audio'], dtype=np.float32)

# Pre-emphasis
audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

# Check if there's anything special about the audio
print(f"Audio length: {len(audio_pre)}")
print(f"First frame (0-800) mean: {audio_pre[:800].mean()}")

# Librosa STFT - check with different parameters
print("\nTesting different librosa parameters:")

# Default
stft1 = librosa.stft(audio_pre, n_fft=800, hop_length=200, win_length=800, window='hann', center=False)
print(f"  Default: DC = {np.abs(stft1[0, 0]):.6e}")

# Try with center=True
stft2 = librosa.stft(audio_pre, n_fft=800, hop_length=200, win_length=800, window='hann', center=True)
print(f"  center=True: DC = {np.abs(stft2[0, 0]):.6e}")

# Try without window
stft3 = librosa.stft(audio_pre, n_fft=800, hop_length=200, win_length=800, window=None, center=False)
print(f"  No window: DC = {np.abs(stft3[0, 0]):.6e}")

# Manual computation to verify
frame_manual = audio_pre[:800]
window_manual = np.hanning(800)
windowed_manual = frame_manual * window_manual
fft_manual = np.fft.fft(windowed_manual, n=800)
print(f"  Manual FFT: DC = {np.abs(fft_manual[0]):.6e}")

print(f"\nwindowed sum (DC): {windowed_manual.sum():.6e}")
