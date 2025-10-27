#!/usr/bin/env python3
"""
Test the ORIGINAL mel-spectrogram code from data_utils
"""

import sys
sys.path.insert(0, 'data_utils/ave')

import audio
from hparams import hparams as hp
import numpy as np
import json

# Load test audio
with open('audio_test_data/reference_data.json', 'r') as f:
    data = json.load(f)
    wav = np.array(data['audio'], dtype=np.float32)

print("="*70)
print("Testing ORIGINAL mel-spectrogram code")
print("="*70)

print(f"\nInput audio:")
print(f"  Shape: {wav.shape}")
print(f"  Duration: {len(wav) / hp.sample_rate * 1000:.1f} ms")

# Compute mel using original code
mel = audio.melspectrogram(wav)

print(f"\nOriginal mel-spectrogram:")
print(f"  Shape: {mel.shape}")
print(f"  Min: {mel.min():.6f}")
print(f"  Max: {mel.max():.6f}")
print(f"  Mean: {mel.mean():.6f}")
print(f"  Std: {mel.std():.6f}")

# Transpose to [frames, mels] like Go expects
mel_transposed = mel.T

print(f"\nTransposed mel-spectrogram [frames, mels]:")
print(f"  Shape: {mel_transposed.shape}")

# Check first frame
print(f"\nFirst frame (first 10 values):")
print(f"  {mel_transposed[0, :10]}")

# Save this as the correct reference
output_data = {
    'audio': wav.tolist(),
    'mel_spectrogram': mel_transposed.tolist(),
    'shape': list(mel_transposed.shape),
    'stats': {
        'min': float(mel_transposed.min()),
        'max': float(mel_transposed.max()),
        'mean': float(mel_transposed.mean()),
        'std': float(mel_transposed.std())
    }
}

with open('audio_test_data/reference_data_correct.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Saved correct reference to audio_test_data/reference_data_correct.json")

# Also save as numpy
np.save('audio_test_data/reference_mel_spec_correct.npy', mel_transposed)
print(f"✅ Saved numpy array to audio_test_data/reference_mel_spec_correct.npy")
