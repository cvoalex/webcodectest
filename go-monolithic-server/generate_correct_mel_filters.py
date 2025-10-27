"""
Generate correct mel filterbank for Go to use
This matches the exact parameters from SyncTalk_2D
"""
import librosa
import json
import numpy as np

# Parameters from SyncTalk_2D/utils.py
sr = 16000
n_fft = 800
n_mels = 80
fmin = 55
fmax = 7600

print("Generating mel filterbank with parameters:")
print(f"  Sample rate: {sr} Hz")
print(f"  FFT size: {n_fft}")
print(f"  Mel bins: {n_mels}")
print(f"  Frequency range: {fmin} - {fmax} Hz")

# Generate mel filterbank using librosa
mel_basis = librosa.filters.mel(
    sr=sr,
    n_fft=n_fft,
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax
)

print(f"\nGenerated mel filterbank shape: {mel_basis.shape}")
print(f"Expected shape: (80, 401)")

# Convert to regular Python list for JSON serialization
mel_filters_list = mel_basis.tolist()

# Create JSON structure
mel_data = {
    "sr": sr,
    "n_fft": n_fft,
    "n_mels": n_mels,
    "fmin": fmin,
    "fmax": fmax,
    "filters": mel_filters_list
}

# Save to the location Go reads from
output_path = "audio_test_data/mel_filters.json"
with open(output_path, 'w') as f:
    json.dump(mel_data, f)

print(f"\n✅ Saved correct mel filterbank to: {output_path}")

# Also save to main project directory
output_path2 = "../audio_test_data/mel_filters.json"
try:
    with open(output_path2, 'w') as f:
        json.dump(mel_data, f)
    print(f"✅ Also saved to: {output_path2}")
except:
    print(f"⚠️  Could not save to {output_path2}")

# Verify
print("\nFirst filter (first 10 values):", mel_basis[0, :10])
print("Filter 5 (first 20 values):", mel_basis[5, :20])
print("\nFilter statistics:")
print(f"  Non-zero elements: {np.count_nonzero(mel_basis)}")
print(f"  Max value: {mel_basis.max():.8f}")
print(f"  Min value: {mel_basis.min():.8f}")
