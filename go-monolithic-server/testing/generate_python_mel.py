"""
Compare mel-spectrogram generation between PyTorch and Go.
We'll generate a mel-spec in Python and compare with Go's output.
"""
import numpy as np
import librosa
from scipy import signal

def preemphasis(wav, k=0.97):
    return signal.lfilter([1, -k], [1], wav)

def melspectrogram_python(wav):
    """Python implementation matching test_single_frame_pth.py"""
    # Preemphasis
    wav = preemphasis(wav, 0.97)
    
    # STFT
    D = librosa.stft(y=wav, n_fft=800, hop_length=200, win_length=800)
    
    # Mel filterbank
    mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    S = np.dot(mel_basis, np.abs(D))
    
    # Amp to dB
    min_level = np.exp(-5 * np.log(10))
    S_db = 20 * np.log10(np.maximum(min_level, S)) - 20
    
    # Normalize
    S_norm = np.clip((2 * 4.) * ((S_db - -100) / (--100)) - 4., -4., 4.)
    
    return S_norm.T  # [time, freq]

def crop_audio_window(spec, start_frame):
    """Extract 16-frame mel window for a given video frame"""
    start_idx = int(80. * (start_frame / float(25)))
    end_idx = start_idx + 16
    if end_idx > spec.shape[0]:
        end_idx = spec.shape[0]
        start_idx = end_idx - 16
    return spec[start_idx:end_idx, :]

# Load audio
wav_path = r'D:\Projects\webcodecstest\aud.wav'
wav, sr = librosa.load(wav_path, sr=16000)

print(f"Loaded audio: {len(wav)} samples at {sr} Hz ({len(wav)/sr:.2f}s)")

# Generate mel-spectrogram
mel_spec = melspectrogram_python(wav)
print(f"Mel-spectrogram shape: {mel_spec.shape} [time, freq]")
print(f"Mel-spec stats: min={mel_spec.min():.3f}, max={mel_spec.max():.3f}, mean={mel_spec.mean():.3f}")

# Extract windows for first few frames
print("\nMel windows for first frames:")
for frame_idx in range(min(10, 20)):
    window = crop_audio_window(mel_spec, frame_idx)
    start_idx = int(80. * (frame_idx / float(25)))
    end_idx = start_idx + 16
    print(f"Frame {frame_idx:2d}: mel[{start_idx:3d}:{end_idx:3d}] shape={window.shape}, mean={window.mean():.4f}")

# Save first few mel windows for comparison
import os
os.makedirs('test_output/mel_windows_python', exist_ok=True)
for frame_idx in range(10):
    window = crop_audio_window(mel_spec, frame_idx)
    np.save(f'test_output/mel_windows_python/frame_{frame_idx}.npy', window)

print(f"\n✓ Saved mel windows to test_output/mel_windows_python/")

# Also save the full mel-spec
np.save('test_output/mel_spec_python_full.npy', mel_spec)
print(f"✓ Saved full mel-spec: {mel_spec.shape}")
