"""
Step-by-step comparison of Python vs Go mel-spec generation.
This will help identify exactly where the bug is.
"""
import numpy as np
import librosa
from scipy import signal
import struct

# Load audio
wav_path = r'D:\Projects\webcodecstest\aud.wav'
wav, sr = librosa.load(wav_path, sr=16000)

print("="*70)
print("STEP-BY-STEP MEL-SPECTROGRAM COMPARISON")
print("="*70)

# Step 1: Preemphasis
def preemphasis_python(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)

wav_pre = preemphasis_python(wav, 0.97)

print(f"\n1. PREEMPHASIS (k=0.97)")
print(f"   Input samples: {len(wav)}")
print(f"   First 5 values: {wav[:5]}")
print(f"   After pre-emphasis:")
print(f"     First 5 values: {wav_pre[:5]}")
print(f"     Mean: {wav_pre.mean():.8f}")
print(f"     Std:  {wav_pre.std():.8f}")

# Step 2: STFT with padding (librosa center=True)
print(f"\n2. STFT (n_fft=800, hop_length=200, center=True)")

# Manual padding (what librosa does with center=True)
pad_size = 800 // 2
padded = np.pad(wav_pre, (pad_size, pad_size), mode='constant')
print(f"   Padded length: {len(padded)} (original {len(wav_pre)} + 2×{pad_size})")
print(f"   Padded first 5: {padded[:5]} (should be zeros)")
print(f"   Padded at pad boundary: {padded[pad_size:pad_size+5]}")

# STFT
D = librosa.stft(y=wav_pre, n_fft=800, hop_length=200, win_length=800, window='hann', center=True)
print(f"   STFT shape: {D.shape} (freq_bins={800//2+1}, time_frames)")
print(f"   Complex dtype: {D.dtype}")

# Magnitude
mag = np.abs(D)
print(f"   Magnitude stats:")
print(f"     Mean: {mag.mean():.8f}")
print(f"     Max:  {mag.max():.8f}")
print(f"     Min:  {mag.min():.8f}")

# Check first frame magnitude
print(f"   First frame (time=0) magnitudes:")
print(f"     First 5 freq bins: {mag[:5, 0]}")
print(f"     Mean: {mag[:, 0].mean():.8f}")

# Step 3: Mel filterbank
mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
print(f"\n3. MEL FILTERBANK")
print(f"   Shape: {mel_basis.shape} (n_mels=80, n_fft//2+1=401)")
print(f"   Filter 0 sum: {mel_basis[0].sum():.8f}")
print(f"   Filter 0 non-zero: {(mel_basis[0] != 0).sum()} bins")
print(f"   Filter 0 max value: {mel_basis[0].max():.8f}")

# Apply mel filterbank
S_mel = np.dot(mel_basis, mag)
print(f"   Mel-spec (linear) shape: {S_mel.shape}")
print(f"   Mel-spec stats:")
print(f"     Mean: {S_mel.mean():.8f}")
print(f"     Max:  {S_mel.max():.8f}")
print(f"     Min:  {S_mel.min():.8f}")

# Check first frame
print(f"   First mel frame:")
print(f"     First 5 mel bins: {S_mel[:5, 0]}")
print(f"     Mean: {S_mel[:, 0].mean():.8f}")

# Step 4: Amp to dB
min_level = np.exp(-5 * np.log(10))
S_db = 20 * np.log10(np.maximum(min_level, S_mel)) - 20

print(f"\n4. AMP TO DB")
print(f"   Min level: {min_level:.12f}")
print(f"   After dB conversion:")
print(f"     Mean: {S_db.mean():.6f}")
print(f"     Max:  {S_db.max():.6f}")
print(f"     Min:  {S_db.min():.6f}")
print(f"   First mel frame (dB):")
print(f"     First 5: {S_db[:5, 0]}")

# Step 5: Normalize
S_norm = np.clip((2 * 4.) * ((S_db - -100) / (--100)) - 4., -4., 4.)

print(f"\n5. NORMALIZE (clip to [-4, 4])")
print(f"   Formula: clip(8 * ((S + 100) / 100) - 4, -4, 4)")
print(f"   After normalization:")
print(f"     Mean: {S_norm.mean():.6f}")
print(f"     Max:  {S_norm.max():.6f}")
print(f"     Min:  {S_norm.min():.6f}")
print(f"     Count at -4.0: {(S_norm == -4.0).sum()}")
print(f"     Count at +4.0: {(S_norm == 4.0).sum()}")

# Transpose to [time, freq]
mel_spec_final = S_norm.T

print(f"\n6. FINAL MEL-SPEC")
print(f"   Shape: {mel_spec_final.shape} [time, freq]")

# Extract frame 8 window
start_idx = int(80. * (8 / float(25)))
end_idx = start_idx + 16
window = mel_spec_final[start_idx:end_idx, :]

print(f"\n7. FRAME 8 WINDOW [mel[{start_idx}:{end_idx}], :]")
print(f"   Shape: {window.shape}")
print(f"   Mean: {window.mean():.6f}")
print(f"   First row, first 10 values: {window[0, :10]}")
print(f"   Value at [0, 10]: {window[0, 10]:.6f}")
print(f"   Value at [1, 10]: {window[1, 10]:.6f}")

# Save diagnostic data for Go comparison
np.save('test_output/python_preemphasis.npy', wav_pre[:1000])  # First 1000 samples
np.save('test_output/python_stft_mag.npy', mag[:, :10])  # First 10 time frames
np.save('test_output/python_mel_linear.npy', S_mel[:, :10])  # First 10 frames
np.save('test_output/python_mel_db.npy', S_db[:, :10])
np.save('test_output/python_mel_normalized.npy', S_norm[:, :10])

print(f"\n✓ Saved diagnostic arrays to test_output/python_*.npy")
print(f"\nNow run Go with debug output to compare!")
