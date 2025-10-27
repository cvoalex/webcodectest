"""
Step-by-step comparison of Python vs Go mel-spectrogram processing
This script will compare each stage of the audio processing pipeline
"""

import numpy as np
import librosa
import librosa.filters
from scipy import signal
import json
import sys
import os

# Add SyncTalk_2D to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SyncTalk_2D'))

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)

def python_stft(y):
    """Exact Python STFT from SyncTalk_2D"""
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800, center=True)

def python_linear_to_mel(spectrogram):
    """Exact Python linear to mel conversion"""
    mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    return np.dot(mel_basis, spectrogram)

def python_amp_to_db(x):
    """Exact Python amp to dB conversion"""
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def python_normalize(S):
    """Exact Python normalization"""
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)

def python_melspectrogram(wav):
    """Complete Python mel-spectrogram pipeline"""
    # Step 1: Pre-emphasis
    emphasized = preemphasis(wav, 0.97)
    
    # Step 2: STFT
    D = python_stft(emphasized)
    
    # Step 3: Magnitude
    mag = np.abs(D)
    
    # Step 4: Linear to Mel
    mel = python_linear_to_mel(mag)
    
    # Step 5: Amp to dB
    mel_db = python_amp_to_db(mel) - 20.0
    
    # Step 6: Normalize
    normalized = python_normalize(mel_db)
    
    return normalized.T  # Transpose to [frames, mels]

def compare_arrays(name, python_arr, go_arr, tolerance=1e-5):
    """Compare two arrays and report differences"""
    print(f"\n{'='*80}")
    print(f"Comparing: {name}")
    print(f"{'='*80}")
    
    # Shape check
    print(f"Python shape: {python_arr.shape}")
    print(f"Go shape:     {go_arr.shape}")
    
    if python_arr.shape != go_arr.shape:
        print("❌ SHAPES DON'T MATCH!")
        return False
    
    # Flatten for easier comparison
    py_flat = python_arr.flatten()
    go_flat = go_arr.flatten()
    
    # Statistics
    print(f"\nPython - min: {py_flat.min():.8f}, max: {py_flat.max():.8f}, mean: {py_flat.mean():.8f}")
    print(f"Go     - min: {go_flat.min():.8f}, max: {go_flat.max():.8f}, mean: {go_flat.mean():.8f}")
    
    # Difference
    diff = np.abs(py_flat - go_flat)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\nDifference - max: {max_diff:.8f}, mean: {mean_diff:.8f}")
    
    # Find where max difference occurs
    max_idx = np.argmax(diff)
    max_idx_2d = np.unravel_index(max_idx, python_arr.shape)
    print(f"Max diff at index {max_idx_2d}: Python={py_flat[max_idx]:.8f}, Go={go_flat[max_idx]:.8f}")
    
    # Show first few values
    print(f"\nFirst 10 values:")
    print(f"Python: {py_flat[:10]}")
    print(f"Go:     {go_flat[:10]}")
    
    # Check tolerance
    if max_diff < tolerance:
        print(f"✅ PASS (max diff {max_diff:.8f} < {tolerance})")
        return True
    else:
        print(f"❌ FAIL (max diff {max_diff:.8f} >= {tolerance})")
        
        # Show problematic values
        problem_indices = np.where(diff > tolerance)[0]
        num_problems = len(problem_indices)
        print(f"\n{num_problems} values exceed tolerance ({num_problems/len(diff)*100:.2f}%)")
        
        if num_problems <= 20:
            print("\nAll problematic values:")
            for idx in problem_indices[:20]:
                idx_2d = np.unravel_index(idx, python_arr.shape)
                print(f"  Index {idx_2d}: Python={py_flat[idx]:.8f}, Go={go_flat[idx]:.8f}, diff={diff[idx]:.8f}")
        else:
            print("\nFirst 20 problematic values:")
            for idx in problem_indices[:20]:
                idx_2d = np.unravel_index(idx, python_arr.shape)
                print(f"  Index {idx_2d}: Python={py_flat[idx]:.8f}, Go={go_flat[idx]:.8f}, diff={diff[idx]:.8f}")
        
        return False

def save_intermediate_steps(audio_samples, output_dir="debug_output"):
    """Save all intermediate steps from Python processing"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PYTHON PROCESSING - Saving intermediate steps")
    print("="*80)
    
    # Step 0: Original audio
    print("\nStep 0: Original audio")
    print(f"  Shape: {audio_samples.shape}")
    print(f"  Min: {audio_samples.min():.8f}, Max: {audio_samples.max():.8f}, Mean: {audio_samples.mean():.8f}")
    print(f"  First 10: {audio_samples[:10]}")
    np.save(f"{output_dir}/python_step0_original.npy", audio_samples.astype(np.float32))
    
    # Step 1: Pre-emphasis
    emphasized = preemphasis(audio_samples, 0.97)
    print("\nStep 1: Pre-emphasis (k=0.97)")
    print(f"  Shape: {emphasized.shape}")
    print(f"  Min: {emphasized.min():.8f}, Max: {emphasized.max():.8f}, Mean: {emphasized.mean():.8f}")
    print(f"  First 10: {emphasized[:10]}")
    np.save(f"{output_dir}/python_step1_preemphasis.npy", emphasized.astype(np.float32))
    
    # Step 2: STFT (complex)
    D = python_stft(emphasized)
    print("\nStep 2: STFT (complex)")
    print(f"  Shape: {D.shape} (freq_bins, frames)")
    print(f"  First frame first 5 (real): {np.real(D[:5, 0])}")
    print(f"  First frame first 5 (imag): {np.imag(D[:5, 0])}")
    # Save real and imaginary parts separately
    np.save(f"{output_dir}/python_step2_stft_real.npy", np.real(D).T.astype(np.float32))  # Transpose to [frames, freq]
    np.save(f"{output_dir}/python_step2_stft_imag.npy", np.imag(D).T.astype(np.float32))
    
    # Step 3: Magnitude
    mag = np.abs(D)
    print("\nStep 3: Magnitude")
    print(f"  Shape: {mag.shape} (freq_bins, frames)")
    print(f"  Min: {mag.min():.8f}, Max: {mag.max():.8f}, Mean: {mag.mean():.8f}")
    print(f"  First frame first 10: {mag[:10, 0]}")
    np.save(f"{output_dir}/python_step3_magnitude.npy", mag.T.astype(np.float32))  # Transpose to [frames, freq]
    
    # Step 4: Linear to Mel
    mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    mel = python_linear_to_mel(mag)
    print("\nStep 4: Linear to Mel")
    print(f"  Mel basis shape: {mel_basis.shape}")
    print(f"  Mel shape: {mel.shape} (mel_bins, frames)")
    print(f"  Min: {mel.min():.8f}, Max: {mel.max():.8f}, Mean: {mel.mean():.8f}")
    print(f"  First frame first 10: {mel[:10, 0]}")
    np.save(f"{output_dir}/python_step4_mel.npy", mel.T.astype(np.float32))  # Transpose to [frames, mels]
    
    # Step 5: Amp to dB (before ref_level_db subtraction)
    mel_db_raw = python_amp_to_db(mel)
    print("\nStep 5a: Amp to dB (before ref subtraction)")
    print(f"  Shape: {mel_db_raw.shape}")
    print(f"  Min: {mel_db_raw.min():.8f}, Max: {mel_db_raw.max():.8f}, Mean: {mel_db_raw.mean():.8f}")
    print(f"  First frame first 10: {mel_db_raw[:10, 0]}")
    np.save(f"{output_dir}/python_step5a_db_raw.npy", mel_db_raw.T.astype(np.float32))
    
    # Step 5b: After ref_level_db subtraction
    mel_db = mel_db_raw - 20.0
    print("\nStep 5b: Amp to dB (after -20 ref_level_db)")
    print(f"  Shape: {mel_db.shape}")
    print(f"  Min: {mel_db.min():.8f}, Max: {mel_db.max():.8f}, Mean: {mel_db.mean():.8f}")
    print(f"  First frame first 10: {mel_db[:10, 0]}")
    np.save(f"{output_dir}/python_step5b_db_adjusted.npy", mel_db.T.astype(np.float32))
    
    # Step 6: Normalize
    normalized = python_normalize(mel_db)
    print("\nStep 6: Normalize")
    print(f"  Shape: {normalized.shape}")
    print(f"  Min: {normalized.min():.8f}, Max: {normalized.max():.8f}, Mean: {normalized.mean():.8f}")
    print(f"  First frame first 10: {normalized[:10, 0]}")
    print(f"  Count at -4.0: {np.sum(normalized == -4.0)}")
    print(f"  Count at +4.0: {np.sum(normalized == 4.0)}")
    np.save(f"{output_dir}/python_step6_normalized.npy", normalized.T.astype(np.float32))  # Transpose to [frames, mels]
    
    # Also save mel filters for Go to use
    print("\nSaving mel filterbank...")
    mel_filters_dict = {
        "filters": mel_basis.tolist(),
        "sr": 16000,
        "n_fft": 800,
        "n_mels": 80,
        "fmin": 55,
        "fmax": 7600
    }
    with open(f"{output_dir}/mel_filters_python.json", 'w') as f:
        json.dump(mel_filters_dict, f, indent=2)
    
    print(f"\n✅ All intermediate steps saved to {output_dir}/")
    
    return normalized.T  # Return [frames, mels]

def main():
    print("="*80)
    print("STEP-BY-STEP PYTHON vs GO MEL-SPECTROGRAM COMPARISON")
    print("="*80)
    
    # Load audio
    audio_path = "../aud.wav"
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"\n📂 Loading audio: {audio_path}")
    audio_samples = load_wav(audio_path, 16000)
    print(f"Loaded {len(audio_samples)} samples at 16kHz ({len(audio_samples)/16000:.2f} seconds)")
    
    # Process with Python and save all steps
    print("\n" + "="*80)
    print("PROCESSING WITH PYTHON")
    print("="*80)
    python_result = save_intermediate_steps(audio_samples, "debug_output")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Build and run the Go comparison program:")
    print("   cd go-monolithic-server")
    print("   go run test_step_by_step.go")
    print()
    print("2. The Go program will load the same audio and save its intermediate steps")
    print()
    print("3. This script will then compare each step automatically")
    print()
    print("Python processing complete! Intermediate steps saved to debug_output/")

if __name__ == "__main__":
    main()
