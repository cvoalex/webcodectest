#!/usr/bin/env python3
"""
Check exact librosa mel filter normalization
"""

import librosa
import numpy as np

# Get mel filters
mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=0, fmax=8000)

# Get frequency points
mel_f = librosa.mel_frequencies(n_mels=80+2, fmin=0, fmax=8000)
hz_f = mel_f

print("Mel frequency points:")
for i in range(min(5, len(hz_f))):
    print(f"  [{i}]: {hz_f[i]:.2f} Hz")

print("\nFirst filter (bin 0):")
filter_0 = mel_basis[0]
nonzero = np.nonzero(filter_0)[0]
print(f"  Non-zero bins: {nonzero}")
print(f"  Values: {filter_0[nonzero]}")
print(f"  Sum: {filter_0.sum()}")
print(f"  Max: {filter_0.max()}")

# Calculate expected normalization
# Slaney norm: 2.0 / (f[i+2] - f[i]) where f is in Hz
expected_norm = 2.0 / (hz_f[2] - hz_f[0])
print(f"\nExpected Slaney norm factor: {expected_norm}")
print(f"  Range: {hz_f[2] - hz_f[0]:.2f} Hz")

# Check if it matches
unnormalized_sum = filter_0.sum() / expected_norm
print(f"  Unnormalized sum would be: {unnormalized_sum}")

# Print filter values in detail
print("\nFilter 0 detailed:")
for bin_idx in nonzero:
    freq = bin_idx * 16000 / 800  # FFT bin to Hz
    print(f"  Bin {bin_idx} ({freq:.1f} Hz): {filter_0[bin_idx]}")
