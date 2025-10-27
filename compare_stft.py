#!/usr/bin/env python3
"""
Compare STFT output: Python vs what Go should compute
"""

import numpy as np
import librosa
import json

def preemphasis(wav, k=0.97):
    """Apply pre-emphasis filter"""
    return np.append(wav[0], wav[1:] - k * wav[:-1])

# Load the reference audio
with open('audio_test_data/reference_data.json', 'r') as f:
    data = json.load(f)
    audio = np.array(data['audio'], dtype=np.float32)

print("="*60)
print("AUDIO INPUT")
print("="*60)
print(f"Shape: {audio.shape}")
print(f"First 10 samples: {audio[:10]}")
print(f"Sample [0]: {audio[0]}")
print(f"Sample [800]: {audio[800]}")

# Apply pre-emphasis
audio_preemphasized = preemphasis(audio, k=0.97)
print(f"\nAfter pre-emphasis:")
print(f"First 10 samples: {audio_preemphasized[:10]}")

# Compute STFT
n_fft = 800
hop_length = 200
win_length = 800

# Librosa STFT
stft = librosa.stft(
    audio_preemphasized,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window='hann',
    center=False  # CRITICAL: No padding
)

print("\n" + "="*60)
print("STFT OUTPUT (librosa)")
print("="*60)
print(f"Shape: {stft.shape} (freq_bins, time_frames)")
print(f"First frame shape: {stft[:, 0].shape}")
print(f"First frame (first 10 complex values):")
for i in range(10):
    print(f"  [{i}]: {stft[i, 0]}")

magnitude = np.abs(stft)
print(f"\nMagnitude spectrogram shape: {magnitude.shape}")
print(f"First frame magnitude (first 10):")
for i in range(10):
    print(f"  [{i}]: {magnitude[i, 0]}")

print(f"\nFirst frame magnitude stats:")
print(f"  Min: {magnitude[:, 0].min()}")
print(f"  Max: {magnitude[:, 0].max()}")
print(f"  Mean: {magnitude[:, 0].mean()}")
print(f"  Sum: {magnitude[:, 0].sum()}")

# Now let's manually compute the first frame to understand what's happening
print("\n" + "="*60)
print("MANUAL FIRST FRAME COMPUTATION")
print("="*60)

# Extract first frame (samples 0-799)
frame = audio_preemphasized[0:800]
print(f"Frame samples: {len(frame)}")
print(f"Frame first 10: {frame[:10]}")

# Apply Hanning window
window = np.hanning(800)
print(f"\nHanning window first 10: {window[:10]}")
print(f"Hanning window [399-400] (center): {window[399:401]}")

windowed = frame * window
print(f"\nWindowed frame first 10: {windowed[:10]}")
print(f"Windowed frame sum: {windowed.sum()}")

# Compute FFT manually
fft_result = np.fft.fft(windowed, n=800)
print(f"\nFFT result shape: {fft_result.shape}")
print(f"FFT first 10 complex values:")
for i in range(10):
    print(f"  [{i}]: {fft_result[i]}")

fft_magnitude = np.abs(fft_result[:401])
print(f"\nFFT magnitude (first 10):")
for i in range(10):
    print(f"  [{i}]: {fft_magnitude[i]}")

print(f"\nCompare with librosa:")
print(f"  Manual matches librosa: {np.allclose(fft_magnitude, magnitude[:, 0])}")
print(f"  Max difference: {np.abs(fft_magnitude - magnitude[:, 0]).max()}")
