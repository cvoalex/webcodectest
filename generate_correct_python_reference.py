"""
Generate CORRECT audio tensor reference using SyncTalk_2D approach
This matches inference_328.py line 142: audio_feat.reshape(32,16,16)
"""
import sys
sys.path.append('.')
sys.path.append('SyncTalk_2D')
sys.path.append('data_utils')
sys.path.append('data_utils/ave')

import numpy as np
import torch
import librosa
from data_utils.ave import audio as ave_audio
from data_utils.ave.test_w2l_audio import model as audio_encoder

print("🎵 Generating CORRECT Python reference (SyncTalk_2D method)")
print("=" * 80)

# Load same audio file as Go test
audio_path = "aud.wav"
audio_samples, sr = librosa.load(audio_path, sr=16000, mono=True)
print(f"✅ Loaded: {len(audio_samples)} samples at {sr} Hz")

# Compute full mel spectrogram
mel = ave_audio.melspectrogram(audio_samples)  # [80, time_frames]
mel_t = mel.T  # [time_frames, 80]
print(f"✅ Mel-spectrogram: {mel_t.shape} (time_frames x 80)")

# For EACH frame, we need to encode a 16-frame window of mel-spectrogram
# This matches SyncTalk_2D/utils.py AudDataset.__getitem__()
# For frame i at 25fps: start_idx = int(80 * (i / 25))

frame_idx = 0  # We'll generate reference for frame 0
fps = 25
start_idx = int(80.0 * (frame_idx / float(fps)))
end_idx = start_idx + 16

if end_idx > mel_t.shape[0]:
    end_idx = mel_t.shape[0]
    start_idx = end_idx - 16

# Extract 16-frame window
mel_window = mel_t[start_idx:end_idx, :]  # [16, 80]
print(f"✅ Frame {frame_idx} mel window: [{start_idx}:{end_idx}] = {mel_window.shape}")

# Prepare for audio encoder: transpose to [80, 16] then add batch dim
mel_input = torch.from_numpy(mel_window.T).unsqueeze(0).float().cuda()  # [1, 80, 16]
print(f"✅ Audio encoder input: {mel_input.shape}")

# Encode this ONE frame's window to get [1, 512]
with torch.no_grad():
    frame_features = audio_encoder(mel_input)  # [1, 512]

print(f"✅ Audio encoder output (frame {frame_idx}): {frame_features.shape}")
print(f"   Min/Max/Mean: {frame_features.min():.3f} / {frame_features.max():.3f} / {frame_features.mean():.3f}")
print(f"   First 10: {frame_features[0,:10].cpu().numpy()}")

# Now we need to encode ALL frames to build the feature array
# For demonstration, encode first 4 frames (matching Go test batch_size=1)
all_features = []
for i in range(4):
    start_idx = int(80.0 * (i / float(fps)))
    end_idx = start_idx + 16
    if end_idx > mel_t.shape[0]:
        end_idx = mel_t.shape[0]
        start_idx = end_idx - 16
    
    mel_window = mel_t[start_idx:end_idx, :]
    mel_input = torch.from_numpy(mel_window.T).unsqueeze(0).float().cuda()
    
    with torch.no_grad():
        features = audio_encoder(mel_input).cpu()
    
    all_features.append(features[0].numpy())  # Store as [512]

all_features = np.array(all_features)  # [num_frames, 512]
print(f"\n✅ Encoded {len(all_features)} frames, shape: {all_features.shape}")

# Now apply get_audio_features() for output frame 0
# This extracts frames from index-8 to index+7 (16 frames total)
# Reference: SyncTalk_2D/utils.py lines 66-84

def get_audio_features_numpy(features, index):
    """Numpy version of get_audio_features from SyncTalk_2D/utils.py"""
    left = index - 8
    right = index + 8  # Note: range is [left, right) so this gives 16 frames
    
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    
    auds = features[left:right]  # [n, 512]
    
    if pad_left > 0:
        zeros = np.zeros((pad_left, 512), dtype=auds.dtype)
        auds = np.concatenate([zeros, auds], axis=0)
    if pad_right > 0:
        zeros = np.zeros((pad_right, 512), dtype=auds.dtype)
        auds = np.concatenate([auds, zeros], axis=0)
    
    return auds  # [16, 512]

# Get features for output frame 0
audio_feat = get_audio_features_numpy(all_features, 0)  # [16, 512]
print(f"\n✅ get_audio_features(index=0): {audio_feat.shape}")
print(f"   This is 16 frames (index -8 to +7), each with 512 features")

# Apply the CORRECT reshape from SyncTalk_2D/inference_328.py line 142
# audio_feat = audio_feat.reshape(32, 16, 16)
audio_reshaped = audio_feat.reshape(32, 16, 16)  # Simple linear reshape
print(f"\n✅ Reshaped to: {audio_reshaped.shape} (CORRECT SyncTalk_2D method)")
print(f"   This is a simple linear reinterpretation of 8,192 values")

# Add batch dimension for UNet
audio_tensor = audio_reshaped[np.newaxis, ...]  # [1, 32, 16, 16]
print(f"✅ Final tensor for UNet: {audio_tensor.shape}")

# Save flattened tensor for comparison
flat = audio_tensor.flatten()
np.save("python_audio_tensor_frame0_CORRECT.npy", flat)
print(f"\n✅ Saved CORRECT Python audio tensor to python_audio_tensor_frame0_CORRECT.npy")
print(f"   Shape: {flat.shape}")
print(f"   First 20 values: {flat[:20]}")
print(f"\n📊 Comparison info:")
print(f"   Min: {flat.min():.3f}")
print(f"   Max: {flat.max():.3f}")
print(f"   Mean: {flat.mean():.3f}")
