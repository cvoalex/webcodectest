#!/usr/bin/env python3
"""
Generate reference mel-spectrogram using Python/librosa
to validate Go implementation
"""

import numpy as np
import librosa
import json
import os

def preemphasis(wav, k=0.97):
    """Apply pre-emphasis filter"""
    return np.append(wav[0], wav[1:] - k * wav[:-1])

def generate_test_audio(sample_rate=16000, duration_ms=640, frequency=440.0):
    """Generate test sine wave"""
    num_samples = (sample_rate * duration_ms) // 1000
    t = np.arange(num_samples) / sample_rate
    audio = 0.5 * np.sin(2.0 * np.pi * frequency * t)
    return audio.astype(np.float32)

def compute_mel_spectrogram(audio, sample_rate=16000):
    """Compute mel-spectrogram matching Go implementation"""
    
    # Parameters matching Go implementation
    n_fft = 800
    hop_length = 200
    win_length = 800
    n_mels = 80
    fmin = 0.0
    fmax = sample_rate / 2.0
    ref_level_db = 20.0
    min_level_db = -100.0
    
    # Pre-emphasis
    audio_preemphasized = preemphasis(audio, k=0.97)
    
    # Compute STFT
    D = librosa.stft(
        audio_preemphasized,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True  # IMPORTANT: Default behavior (adds padding)
    )
    
    # Magnitude spectrogram
    S = np.abs(D)
    
    # Mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Apply mel filterbank
    mel_S = np.dot(mel_basis, S)
    
    # Convert to dB
    mel_S_db = 10.0 * np.log10(mel_S**2 + 1e-10) - ref_level_db
    
    # Normalize
    mel_S_normalized = np.clip(mel_S_db, min_level_db, 0) - min_level_db
    mel_S_normalized = mel_S_normalized / -min_level_db
    
    # Transpose to [frames, mels]
    return mel_S_normalized.T

def main():
    print("=" * 70)
    print("Python Reference Mel-Spectrogram Generator")
    print("=" * 70)
    
    # Generate test audio
    print("\n1️⃣  Generating test audio (440 Hz sine wave, 640ms)...")
    audio = generate_test_audio()
    print(f"   Audio shape: {audio.shape}")
    print(f"   Sample rate: 16000 Hz")
    print(f"   Duration: 640 ms")
    
    # Compute mel-spectrogram
    print("\n2️⃣  Computing mel-spectrogram...")
    mel_spec = compute_mel_spectrogram(audio)
    print(f"   Mel-spectrogram shape: {mel_spec.shape}")
    
    # Save audio for Go to use
    print("\n3️⃣  Saving reference data...")
    output_dir = "audio_test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save audio
    audio_file = os.path.join(output_dir, "test_audio.npy")
    np.save(audio_file, audio)
    print(f"   Saved audio: {audio_file}")
    
    # Save mel-spectrogram
    mel_file = os.path.join(output_dir, "reference_mel_spec.npy")
    np.save(mel_file, mel_spec)
    print(f"   Saved mel-spec: {mel_file}")
    
    # Save as JSON for easy Go loading
    json_file = os.path.join(output_dir, "reference_data.json")
    data = {
        "audio": audio.tolist(),
        "mel_spec": mel_spec.tolist(),
        "shape": list(mel_spec.shape),
        "stats": {
            "mel_min": float(mel_spec.min()),
            "mel_max": float(mel_spec.max()),
            "mel_mean": float(mel_spec.mean()),
            "mel_std": float(mel_spec.std())
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   Saved JSON: {json_file}")
    
    # Print statistics
    print("\n4️⃣  Mel-spectrogram statistics:")
    print(f"   Shape: {mel_spec.shape}")
    print(f"   Min: {mel_spec.min():.6f}")
    print(f"   Max: {mel_spec.max():.6f}")
    print(f"   Mean: {mel_spec.mean():.6f}")
    print(f"   Std: {mel_spec.std():.6f}")
    
    # Print first few values for manual verification
    print("\n5️⃣  First frame (first 10 values):")
    print(f"   {mel_spec[0, :10]}")
    
    print("\n" + "=" * 70)
    print("✅ Reference data generated successfully!")
    print("=" * 70)
    print(f"\nFiles saved in: {output_dir}/")
    print("Use these files to validate Go implementation")
    print("=" * 70)

if __name__ == "__main__":
    main()
