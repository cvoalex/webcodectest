"""
Detailed diagnostic: trace through each mel-spec generation step
Compare Python vs Go at each stage.
"""
import numpy as np
import librosa
from scipy import signal

# Load audio
wav_path = r'D:\Projects\webcodecstest\aud.wav'
wav, sr = librosa.load(wav_path, sr=16000)

print(f"Audio: {len(wav)} samples")
print(f"First 10 samples: {wav[:10]}")
print(f"Mean: {wav.mean():.6f}, Std: {wav.std():.6f}\n")

# Step 1: Preemphasis
def preemphasis(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)

wav_pre = preemphasis(wav, 0.97)
print(f"After preemphasis:")
print(f"  Mean: {wav_pre.mean():.6f}, Std: {wav_pre.std():.6f}")
print(f"  First 10: {wav_pre[:10]}\n")

# Step 2: STFT
D = librosa.stft(y=wav_pre, n_fft=800, hop_length=200, win_length=800)
print(f"STFT shape: {D.shape}  (freq_bins, time_frames)")
print(f"STFT is complex: {D.dtype}")
print(f"Magnitude mean: {np.abs(D).mean():.6f}\n")

# Step 3: Mel filterbank
mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
print(f"Mel filterbank shape: {mel_basis.shape}  (n_mels, n_fft//2+1)")
print(f"Mel filterbank sum per mel bin (first 5): {mel_basis[:5].sum(axis=1)}\n")

S = np.dot(mel_basis, np.abs(D))
print(f"Mel-spec (linear) shape: {S.shape}")
print(f"Mel-spec mean: {S.mean():.6f}, max: {S.max():.6f}\n")

# Step 4: Amp to dB
min_level = np.exp(-5 * np.log(10))
print(f"Min level: {min_level:.10f}")
S_db = 20 * np.log10(np.maximum(min_level, S)) - 20
print(f"After amp→dB:")
print(f"  Mean: {S_db.mean():.6f}, Min: {S_db.min():.6f}, Max: {S_db.max():.6f}\n")

# Step 5: Normalize
S_norm = np.clip((2 * 4.) * ((S_db - -100) / (--100)) - 4., -4., 4.)
print(f"After normalization:")
print(f"  Mean: {S_norm.mean():.6f}, Min: {S_norm.min():.6f}, Max: {S_norm.max():.6f}\n")

# Check frame 8 window
mel_spec_full = S_norm.T  # Transpose to [time, freq]
start_idx = int(80. * (8 / float(25)))
end_idx = start_idx + 16
window = mel_spec_full[start_idx:end_idx, :]

print(f"Frame 8 window [mel[{start_idx}:{end_idx}], :]:")
print(f"  Shape: {window.shape}")
print(f"  Mean: {window.mean():.6f}")
print(f"  Min: {window.min():.6f}")
print(f"  Max: {window.max():.6f}")
print(f"  Values at [0,10]: {window[0, 10]:.6f}")
print(f"  Values at [1,10]: {window[1, 10]:.6f}")
print(f"  Number of values == -4.0: {(window == -4.0).sum()}")
