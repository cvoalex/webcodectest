#!/usr/bin/env python3
"""
Save librosa mel filterbank for Go to use
"""

import librosa
import numpy as np
import json

# Create mel filterbank with same parameters
mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=0, fmax=8000)

print(f"Mel filterbank shape: {mel_basis.shape}")
print(f"First filter sum: {mel_basis[0].sum()}")
print(f"Last filter sum: {mel_basis[79].sum()}")

# Save as numpy
np.save('audio_test_data/librosa_mel_filters.npy', mel_basis)
print("✅ Saved to audio_test_data/librosa_mel_filters.npy")

# Also save as JSON for easier inspection
# Convert to list format
mel_filters_list = mel_basis.tolist()

with open('audio_test_data/librosa_mel_filters.json', 'w') as f:
    json.dump({
        'filters': mel_filters_list,
        'shape': list(mel_basis.shape),
        'params': {
            'sr': 16000,
            'n_fft': 800,
            'n_mels': 80,
            'fmin': 0,
            'fmax': 8000
        }
    }, f)

print("✅ Saved to audio_test_data/librosa_mel_filters.json")
