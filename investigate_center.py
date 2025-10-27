#!/usr/bin/env python3
"""
Investigate librosa center parameter
"""

import numpy as np
import librosa

# Load audio
import json
with open('audio_test_data/reference_data.json', 'r') as f:
    data = json.load(f)
    audio = np.array(data['audio'], dtype=np.float32)

# Pre-emphasis
audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

print("Comparing center=True vs center=False")
print("="*60)

# center=False
stft_no_center = librosa.stft(audio_pre, n_fft=800, hop_length=200, win_length=800, window='hann', center=False)
print(f"center=False:")
print(f"  Shape: {stft_no_center.shape}")
print(f"  DC (frame 0): {np.abs(stft_no_center[0, 0]):.6e}")
print(f"  DC (frame 1): {np.abs(stft_no_center[0, 1]):.6e}")
print(f"  DC (frame 2): {np.abs(stft_no_center[0, 2]):.6e}")

# center=True
stft_center = librosa.stft(audio_pre, n_fft=800, hop_length=200, win_length=800, window='hann', center=True)
print(f"\ncenter=True:")
print(f"  Shape: {stft_center.shape}")
print(f"  DC (frame 0): {np.abs(stft_center[0, 0]):.6e}")
print(f"  DC (frame 1): {np.abs(stft_center[0, 1]):.6e}")
print(f"  DC (frame 2): {np.abs(stft_center[0, 2]):.6e}")

# Manual STFT for first frame
print("\n" + "="*60)
print("Manual STFT for first frame (no padding):")
frame = audio_pre[0:800]
window = np.hanning(800)
windowed = frame * window
fft_result = np.fft.fft(windowed, n=800)
print(f"  DC: {np.abs(fft_result[0]):.6e}")
print(f"  Windowed sum: {windowed.sum():.6e}")

# What if we pad?
print("\n" + "="*60)
print("Manual STFT with padding (like center=True):")
# center=True pads with n_fft//2 on each side
pad_len = 800 // 2
audio_padded = np.pad(audio_pre, (pad_len, pad_len), mode='constant')
print(f"  Padded length: {len(audio_padded)}")
print(f"  First frame starts at: {0} (padded), {0-pad_len} (original)")

# Extract first frame from padded
frame_padded = audio_padded[0:800]
windowed_padded = frame_padded * window
fft_padded = np.fft.fft(windowed_padded, n=800)
print(f"  DC with padding: {np.abs(fft_padded[0]):.6e}")

# What about the actual first frame of audio (after unpadding)?
# With center=True, frame 0 is centered at sample -400 (before signal starts)
# Frame 1 is centered at sample -400 + 200 = -200
# Frame 2 is centered at sample 0 (start of actual audio)
print(f"\n  Frame 2 DC (should be first real audio): {np.abs(stft_center[0, 2]):.6e}")

# Compare: frame 2 of centered vs frame 0 of non-centered
# These should be similar but not identical because of window positioning
frame_0_original = audio_pre[0:800]
window_0 = np.hanning(800)
windowed_0 = frame_0_original * window_0
fft_0 = np.fft.fft(windowed_0, n=800)
print(f"  Manual frame 0 (no padding): {np.abs(fft_0[0]):.6e}")
print(f"  Matches no-center frame 0: {np.allclose(np.abs(fft_0[0]), np.abs(stft_no_center[0, 0]))}")
