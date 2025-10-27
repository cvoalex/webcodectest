"""
Process real audio through Python pipeline and save the exact audio tensor
that would be fed to the model
"""
import sys
sys.path.append('.')
sys.path.append('data_utils')
sys.path.append('data_utils/ave')

import numpy as np
import torch
import librosa
from data_utils.ave import audio as ave_audio
from data_utils.ave.test_w2l_audio import AudDataset, model as audio_encoder

print("🎵 Processing real audio through Python pipeline")
print("=" * 60)

# Load same audio file as Go test
audio_path = "aud.wav"
audio_samples, sr = librosa.load(audio_path, sr=16000, mono=True)
print(f"Loaded: {len(audio_samples)} samples at {sr} Hz")

# Take first 640ms (same as Go test for frame 0)
samples_640ms = int(0.640 * sr)
audio_chunk = audio_samples[:samples_640ms]
print(f"Using first 640ms: {len(audio_chunk)} samples")

# Compute mel spectrogram (same as Go)
mel = ave_audio.melspectrogram(audio_chunk)  # [80, time_frames]
print(f"Mel shape: {mel.shape}")

# Transpose to [time_frames, 80]
mel_t = mel.T
print(f"Transposed mel: {mel_t.shape}")

# Extract 16-frame window (centered)
if mel_t.shape[0] >= 16:
    start_idx = (mel_t.shape[0] - 16) // 2
    mel_16 = mel_t[start_idx:start_idx+16, :]
else:
    mel_16 = np.pad(mel_t, ((0, 16 - mel_t.shape[0]), (0, 0)), 'constant')

print(f"16-frame mel window: {mel_16.shape}")

# Prepare for audio encoder: [batch, 1, 80, 16]
mel_input = torch.from_numpy(mel_16.T).unsqueeze(0).unsqueeze(0).float().cuda()
print(f"Audio encoder input shape: {mel_input.shape}")

# Run through audio encoder
with torch.no_grad():
    audio_features = audio_encoder(mel_input)  # Should output [1, 512]

print(f"\nAudio encoder output: {audio_features.shape}")
print(f"Min/Max/Mean: {audio_features.min():.3f} / {audio_features.max():.3f} / {audio_features.mean():.3f}")
print(f"First 10 features: {audio_features[0,:10].cpu().numpy()}")

# Now apply the EXACT transformation from inference_engine.py line 147-148
audio_slice = audio_features  # [1, 512]
audio_tensor = audio_slice.unsqueeze(0)  # [1, 1, 512]
audio_reshaped = audio_tensor.view(1, 32, 16).unsqueeze(-1).repeat(1, 1, 1, 16)  # [1, 32, 16, 16]

print(f"\nFinal audio tensor for UNet: {audio_reshaped.shape}")
print(f"Element [0,0,0,0:5]: {audio_reshaped[0,0,0,:5].cpu().numpy()}")
print(f"Element [0,0,1,0:5]: {audio_reshaped[0,0,1,:5].cpu().numpy()}")
print(f"Element [0,1,0,0:5]: {audio_reshaped[0,1,0,:5].cpu().numpy()}")

# Save the flattened audio tensor
flat = audio_reshaped.flatten().cpu().numpy()
np.save("python_audio_tensor_frame0.npy", flat)
print(f"\n✅ Saved Python audio tensor to python_audio_tensor_frame0.npy")
print(f"   Shape: {flat.shape}")
print(f"   This is the EXACT tensor that should go to UNet for frame 0")
