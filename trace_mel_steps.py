#!/usr/bin/env python3
"""
Step-by-step comparison of mel-spectrogram computation
"""

import sys
sys.path.insert(0, 'data_utils/ave')

import audio
from hparams import hparams as hp
import numpy as np
import json
import librosa

# Load test audio
with open('audio_test_data/reference_data.json', 'r') as f:
    data = json.load(f)
    wav = np.array(data['audio'], dtype=np.float32)

print("="*70)
print("Step-by-Step Mel-Spectrogram Computation")
print("="*70)

# Step 1: Pre-emphasis
from scipy import signal
preemphasized = signal.lfilter([1, -hp.preemphasis], [1], wav)
print(f"\n1. Pre-emphasis:")
print(f"   First 10: {preemphasized[:10]}")

# Step 2: STFT
D = librosa.stft(y=preemphasized, n_fft=hp.n_fft, hop_length=audio.get_hop_size(), win_length=hp.win_size)
print(f"\n2. STFT:")
print(f"   Shape: {D.shape}")  # (freq_bins, time_frames)
print(f"   First frame DC: {D[0, 0]}")

# Step 3: Magnitude
S = np.abs(D)
print(f"\n3. Magnitude:")
print(f"   First frame (first 10 bins): {S[:10, 0]}")
print(f"   Max: {S[:, 0].max()}")

# Step 4: Mel filterbank
mel_basis = librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)
print(f"\n4. Mel filterbank:")
print(f"   Shape: {mel_basis.shape}")  # (n_mels, freq_bins)

# Step 5: Apply mel filterbank
mel_S = np.dot(mel_basis, S)
print(f"\n5. Mel magnitude:")
print(f"   Shape: {mel_S.shape}")  # (n_mels, time_frames)
print(f"   First frame (first 10 mels): {mel_S[:10, 0]}")

# Step 6: Convert to dB
min_level = np.exp(hp.min_level_db / 20 * np.log(10))
mel_S_db = 20 * np.log10(np.maximum(min_level, mel_S))
print(f"\n6. Convert to dB:")
print(f"   Min level: {min_level}")
print(f"   First frame (first 10 mels): {mel_S_db[:10, 0]}")

# Step 7: Subtract reference level
mel_S_db_ref = mel_S_db - hp.ref_level_db
print(f"\n7. Subtract ref_level_db ({hp.ref_level_db}):")
print(f"   First frame (first 10 mels): {mel_S_db_ref[:10, 0]}")

# Step 8: Normalize
if hp.symmetric_mels:
    normalized = (2 * hp.max_abs_value) * ((mel_S_db_ref - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    normalized = np.clip(normalized, -hp.max_abs_value, hp.max_abs_value)
else:
    normalized = hp.max_abs_value * ((mel_S_db_ref - hp.min_level_db) / (-hp.min_level_db))
    normalized = np.clip(normalized, 0, hp.max_abs_value)

print(f"\n8. Normalize (symmetric, max_abs_value={hp.max_abs_value}):")
print(f"   First frame (first 10 mels): {normalized[:10, 0]}")

# Transpose to [frames, mels]
result = normalized.T

print(f"\n9. Transpose to [frames, mels]:")
print(f"   Shape: {result.shape}")
print(f"   First frame (first 10 mels): {result[0, :10]}")

# Compare with original function
mel_original = audio.melspectrogram(wav)
print(f"\n10. Compare with original melspectrogram function:")
print(f"   Shape: {mel_original.shape}")
print(f"   First frame (first 10 mels): {mel_original[:10, 0]}")
print(f"   Transposed first frame: {mel_original.T[0, :10]}")
print(f"   Matches manual: {np.allclose(result, mel_original.T)}")
