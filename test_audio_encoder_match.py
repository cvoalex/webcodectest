"""
Test if our ONNX audio encoder matches the expected output format
"""
import sys
sys.path.append('.')
sys.path.append('data_utils')
sys.path.append('data_utils/ave')

import numpy as np
import torch
import onnxruntime as ort
import librosa

# Import mel computation from our audio utils
from data_utils.ave import audio as ave_audio

print("🎵 Testing Audio Encoder Output Format")
print("=" * 60)

# Load a sample audio file
audio_path = "aud.wav"
audio, sr = librosa.load(audio_path, sr=16000, mono=True)
print(f"Loaded audio: {len(audio)} samples at {sr} Hz")

# Take 640ms of audio (16 frames × 40ms)
samples_640ms = int(0.640 * sr)  # 10,240 samples
audio_chunk = audio[:samples_640ms]
print(f"Using {len(audio_chunk)} samples (640ms)")

# Compute mel-spectrogram
mel = ave_audio.melspectrogram(audio_chunk)  # Should be [80, time_frames]
print(f"Mel spectrogram shape: {mel.shape}")

# Transpose to [time_frames, 80]
mel_t = mel.T
print(f"Transposed mel shape: {mel_t.shape}")

# Extract middle 16 frames (like the training code does)
if mel_t.shape[0] >= 16:
    start_idx = (mel_t.shape[0] - 16) // 2
    mel_16frames = mel_t[start_idx:start_idx+16, :]  # [16, 80]
else:
    mel_16frames = np.pad(mel_t, ((0, 16 - mel_t.shape[0]), (0, 0)), 'constant')

print(f"16-frame mel window shape: {mel_16frames.shape}")
print(f"  Min: {mel_16frames.min():.3f}, Max: {mel_16frames.max():.3f}, Mean: {mel_16frames.mean():.3f}")

# Run through ONNX audio encoder
onnx_path = "go-monolithic-server/audio_encoder.onnx"
session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Reshape for ONNX: [1, 1, 80, 16]
mel_input = mel_16frames.T[np.newaxis, np.newaxis, :, :]  # [80, 16] -> [1, 1, 80, 16]
print(f"ONNX input shape: {mel_input.shape}")

# Run inference
outputs = session.run(None, {'mel_spectrogram': mel_input.astype(np.float32)})
features = outputs[0][0]  # [512]

print(f"\n✅ Audio encoder output:")
print(f"  Shape: {features.shape}")
print(f"  Min: {features.min():.3f}, Max: {features.max():.3f}, Mean: {features.mean():.3f}")
print(f"  First 10 values: {features[:10]}")

# Compare with aud_ave.npy from sanders model
aud_ave_path = "old/old_minimal_server/models/sanders/aud_ave.npy"
aud_ave = np.load(aud_ave_path)
print(f"\n📊 Reference aud_ave.npy:")
print(f"  Shape: {aud_ave.shape}")
print(f"  Min: {aud_ave.min():.3f}, Max: {aud_ave.max():.3f}, Mean: {aud_ave.mean():.3f}")
print(f"  Frame 0 first 10: {aud_ave[0, :10]}")

# Test the reshape logic
print(f"\n🔄 Testing reshape [512] -> [32, 16, 16]:")
reshaped = features.reshape(32, 16, 1).repeat(16, axis=2)  # [32, 16, 16]
print(f"  Output shape: {reshaped.shape}")
print(f"  Element [0,0,0:3]: {reshaped[0,0,:3]}")
print(f"  Element [0,1,0:3]: {reshaped[0,1,:3]}")
print(f"  Element [1,0,0:3]: {reshaped[1,0,:3]}")

# Verify it matches PyTorch logic
torch_features = torch.from_numpy(features)
torch_reshaped = torch_features.view(32, 16).unsqueeze(-1).repeat(1, 1, 16)
print(f"\n🔄 PyTorch reshape verification:")
print(f"  Output shape: {torch_reshaped.shape}")
print(f"  Element [0,0,0:3]: {torch_reshaped[0,0,:3].numpy()}")
print(f"  Match: {np.allclose(reshaped, torch_reshaped.numpy())}")

print(f"\n✅ Test complete!")
