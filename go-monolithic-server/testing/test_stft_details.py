"""
Test EXACTLY what librosa.stft does with a simple signal.
This will help us replicate it perfectly in Go.
"""
import numpy as np
import librosa

# Create a simple test signal - a sine wave
sr = 16000
freq = 440  # A4 note
duration = 0.1  # 100ms
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
test_signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

print("Test Signal")
print("="*70)
print(f"Duration: {duration}s, Samples: {len(test_signal)}")
print(f"Frequency: {freq} Hz")
print(f"First 10 samples: {test_signal[:10]}")
print(f"Mean: {test_signal.mean():.8f}, Std: {test_signal.std():.8f}")

# Apply librosa STFT
D = librosa.stft(y=test_signal, n_fft=800, hop_length=200, win_length=800, window='hann', center=True)
mag = np.abs(D)

print(f"\nLibrosa STFT")
print(f"  Output shape: {D.shape} (freq_bins, time_frames)")
print(f"  Magnitude mean: {mag.mean():.8f}")
print(f"  Magnitude max: {mag.max():.8f}")
print(f"  Peak frequency bin: {mag[:, 0].argmax()} (expected around {440 * 800 / 16000:.1f})")

# Now do it manually to understand exactly what's happening
print(f"\n" + "="*70)
print("Manual STFT (to match librosa exactly)")
print("="*70)

# 1. Pad signal (center=True)
pad_len = 800 // 2
padded = np.pad(test_signal, (pad_len, pad_len), mode='constant')
print(f"1. Padded signal: {len(padded)} samples (was {len(test_signal)})")

# 2. Create Hanning window
window = np.hanning(800)
print(f"2. Hanning window sum: {window.sum():.6f}")

# 3. Extract first frame and apply window
frame_0 = padded[0:800]
windowed_0 = frame_0 * window

print(f"3. First frame:")
print(f"   Frame samples: {frame_0[:5]}")
print(f"   After windowing: {windowed_0[:5]}")
print(f"   Windowed RMS: {np.sqrt(np.mean(windowed_0**2)):.8f}")

# 4. FFT
fft_result = np.fft.rfft(windowed_0, n=800)
mag_0 = np.abs(fft_result)

print(f"4. FFT:")
print(f"   FFT output length: {len(fft_result)} (n_fft//2 + 1 = 401)")
print(f"   Magnitude first 5: {mag_0[:5]}")
print(f"   Magnitude mean: {mag_0.mean():.8f}")
print(f"   Magnitude max: {mag_0.max():.8f}")

# Compare with librosa
print(f"\n5. Comparison:")
print(f"   Librosa frame 0 mag: {mag[:, 0][:5]}")
print(f"   Manual frame 0 mag:  {mag_0[:5]}")
print(f"   Match? {np.allclose(mag[:, 0], mag_0, atol=1e-6)}")

# Check scaling
print(f"\n6. Scaling check:")
print(f"   Librosa uses fft.fft() which does NOT normalize by default")
print(f"   NumPy's rfft also does NOT normalize by default")
print(f"   Both should match!")

# Save for Go comparison
test_signal.tofile('test_output/test_sine_signal.bin')
mag_0.astype(np.float32).tofile('test_output/test_sine_stft_frame0.bin')
print(f"\n✓ Saved test signal and expected STFT to test_output/")
