"""Test if Go's Hanning window matches NumPy/SciPy"""
import numpy as np

# NumPy's Hanning window
size = 800
np_hann = np.hanning(size)

# Manual calculation (what Go does)
manual_hann = np.zeros(size, dtype=np.float32)
for i in range(size):
    manual_hann[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (size - 1)))

print("Hanning Window Comparison (size=800)")
print("="*70)
print(f"NumPy hanning:  {np_hann[:5]}")
print(f"Manual formula: {manual_hann[:5]}")
print(f"\nDifference:")
print(f"  Max diff: {np.abs(np_hann - manual_hann).max():.12f}")
print(f"  Are they equal? {np.allclose(np_hann, manual_hann, atol=1e-7)}")

# Save for Go comparison
manual_hann.tofile('test_output/python_hann_window.bin')
print(f"\n✓ Saved hanning window to test_output/python_hann_window.bin")

# Also check what librosa uses
import librosa
D = librosa.stft(y=np.ones(1600), n_fft=800, hop_length=200, win_length=800, window='hann')
print(f"\nLibrosa STFT shape with dummy input: {D.shape}")
