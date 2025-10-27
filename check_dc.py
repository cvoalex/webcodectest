#!/usr/bin/env python3
"""
Check DC component of windowed signal
"""

import numpy as np
import json

# Load the audio
with open('audio_test_data/reference_data.json', 'r') as f:
    data = json.load(f)
    audio = np.array(data['audio'], dtype=np.float32)

# Pre-emphasis
audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

# First frame
frame = audio_pre[0:800]

# Window
window = np.hanning(800)

# Windowed
windowed = frame * window

print("DC component check:")
print(f"Sum of windowed signal: {windowed.sum()}")
print(f"Mean of windowed signal: {windowed.mean()}")
print(f"Expected DC bin (should be sum): {windowed.sum()}")

# FFT
fft_result = np.fft.fft(windowed, n=800)
print(f"\nFFT DC bin [0]: {fft_result[0]}")
print(f"FFT DC magnitude: {np.abs(fft_result[0])}")

# Compare with librosa
import librosa
stft = librosa.stft(audio_pre, n_fft=800, hop_length=200, win_length=800, window='hann', center=False)
print(f"\nLibrosa DC bin [0]: {stft[0, 0]}")
print(f"Librosa DC magnitude: {np.abs(stft[0, 0])}")

# The DC should be the sum of the signal
# Let's verify manually
dc_expected = windowed.sum()
print(f"\nExpected DC (sum of windowed): {dc_expected}")
print(f"Manual FFT DC: {fft_result[0]}")
print(f"Match: {np.isclose(dc_expected, fft_result[0])}")
