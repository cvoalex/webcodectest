#!/usr/bin/env python3
"""
Debug mel-spectrogram differences between Go and Python
"""

import numpy as np
import librosa
import json

def preemphasis(wav, k=0.97):
    """Apply pre-emphasis filter"""
    return np.append(wav[0], wav[1:] - k * wav[:-1])

# Load the same audio that Go is using
with open('audio_test_data/reference_data.json', 'r') as f:
    data = json.load(f)
    audio = np.array(data['audio'], dtype=np.float32)

print(f"Audio shape: {audio.shape}")
print(f"Audio first 10 samples: {audio[:10]}")
print(f"Audio stats: min={audio.min()}, max={audio.max()}, mean={audio.mean()}")

# Apply pre-emphasis
audio_preemphasized = preemphasis(audio, k=0.97)
print(f"\nAfter pre-emphasis:")
print(f"  First 10 samples: {audio_preemphasized[:10]}")
print(f"  Stats: min={audio_preemphasized.min()}, max={audio_preemphasized.max()}")

# Compute STFT
n_fft = 800
hop_length = 200
win_length = 800

print(f"\nSTFT parameters:")
print(f"  n_fft: {n_fft}")
print(f"  hop_length: {hop_length}")
print(f"  win_length: {win_length}")
print(f"  center: False (no padding)")

D = librosa.stft(
    audio_preemphasized,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window='hann',
    center=False  # This is important!
)

print(f"\nSTFT shape: {D.shape}")
print(f"Number of frames: {D.shape[1]}")
print(f"Number of freq bins: {D.shape[0]}")

# Magnitude
S = np.abs(D)
print(f"\nMagnitude spectrogram:")
print(f"  First frame (first 10 bins): {S[:10, 0]}")
print(f"  Stats: min={S.min()}, max={S.max()}")

# Check first frame magnitude sum
print(f"  First frame total energy: {np.sum(S[:, 0])}")
print(f"  Second frame total energy: {np.sum(S[:, 1])}")
