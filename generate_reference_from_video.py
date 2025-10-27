#!/usr/bin/env python3
"""
Generate reference mel-spectrogram from a video file
"""

import sys
sys.path.insert(0, 'data_utils/ave')

import audio
from hparams import hparams as hp
import numpy as np
import json
import librosa
import os

def generate_reference_from_video(video_path, duration_ms=640):
    """
    Extracts audio from a video, computes mel-spectrogram, and saves reference files.
    """
    print(f"\nProcessing video: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ ERROR: Video file not found at {video_path}")
        return

    try:
        # Load audio from video file
        wav, sr = librosa.load(video_path, sr=hp.sample_rate)
        print(f"✅ Loaded audio from video (sr={sr}, duration={len(wav)/sr:.2f}s)")
    except Exception as e:
        print(f"❌ ERROR: Could not load audio from video. Make sure ffmpeg is installed.")
        print(f"   Error details: {e}")
        return

    # Take the first `duration_ms`
    num_samples = (hp.sample_rate * duration_ms) // 1000
    if len(wav) < num_samples:
        print(f"⚠️ WARNING: Audio is shorter than {duration_ms}ms. Using full audio.")
        wav_segment = wav
    else:
        wav_segment = wav[:num_samples]
    
    print(f"   Using first {len(wav_segment)} samples ({len(wav_segment)/sr*1000:.1f} ms)")

    # Compute mel-spectrogram using the original project's function
    mel = audio.melspectrogram(wav_segment)
    mel_transposed = mel.T
    
    print(f"   Computed mel-spectrogram with shape: {mel_transposed.shape}")

    # Save the new reference data
    output_filename = 'reference_data_video.json'
    output_path = os.path.join('audio_test_data', output_filename)
    
    output_data = {
        'audio': wav_segment.tolist(),
        'mel_spectrogram': mel_transposed.tolist(),
        'shape': list(mel_transposed.shape),
        'stats': {
            'min': float(mel_transposed.min()),
            'max': float(mel_transposed.max()),
            'mean': float(mel_transposed.mean()),
            'std': float(mel_transposed.std())
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved new video reference to {output_path}")

def main():
    print("=" * 70)
    print("Reference Mel-Spectrogram Generator (from Video)")
    print("=" * 70)
    
    video_file = os.path.join('model_videos', 'test_optimized_package_fixed_1.mp4')
    generate_reference_from_video(video_file)

if __name__ == '__main__':
    # Change to script directory to ensure correct paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
