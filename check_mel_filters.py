#!/usr/bin/env python3
"""
Check librosa mel filterbank normalization
"""

import librosa
import numpy as np

mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=0, fmax=8000)

print("Mel filterbank shape:", mel_basis.shape)
print("\nFirst filter (mel bin 0):")
print(f"  Non-zero values: {np.count_nonzero(mel_basis[0])}")
print(f"  Max value: {mel_basis[0].max()}")
print(f"  Sum: {mel_basis[0].sum()}")

print("\nMid filter (mel bin 40):")
print(f"  Non-zero values: {np.count_nonzero(mel_basis[40])}")
print(f"  Max value: {mel_basis[40].max()}")
print(f"  Sum: {mel_basis[40].sum()}")

print("\nLast filter (mel bin 79):")
print(f"  Non-zero values: {np.count_nonzero(mel_basis[79])}")
print(f"  Max value: {mel_basis[79].max()}")
print(f"  Sum: {mel_basis[79].sum()}")

# Check the documentation
print("\nLibrosa mel filters use 'slaney' normalization by default")
print("This normalizes each filter to have area = 1.0")

# Verify
areas = [mel_basis[i].sum() for i in range(80)]
print(f"\nFilter areas (sums):")
print(f"  Min: {min(areas)}")
print(f"  Max: {max(areas)}")
print(f"  Mean: {np.mean(areas)}")

# Check without normalization
mel_basis_no_norm = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=0, fmax=8000, norm=None)
print("\nWithout normalization:")
print(f"  First filter max: {mel_basis_no_norm[0].max()}")
print(f"  First filter sum: {mel_basis_no_norm[0].sum()}")
