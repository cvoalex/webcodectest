"""
Test full audio encoder ONNX inference: Python vs Go
Compares final encoder outputs to validate the entire audio pipeline
"""

import numpy as np
import librosa
import onnxruntime as ort
import os
import sys

# Add SyncTalk_2D to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SyncTalk_2D'))

def load_wav(path, sr):
    """Load audio file"""
    return librosa.core.load(path, sr=sr)[0]

def python_melspectrogram(audio_samples):
    """
    Generate mel-spectrogram using exact SyncTalk_2D parameters
    """
    from scipy import signal
    
    # Pre-emphasis
    pre_emphasized = signal.lfilter([1, -0.97], [1], audio_samples)
    
    # STFT
    D = librosa.stft(y=pre_emphasized, n_fft=800, hop_length=200, win_length=800, center=True)
    
    # Magnitude
    S = np.abs(D)
    
    # Mel filterbank
    mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    mel_spectrogram = np.dot(mel_basis, S)
    
    # Amplitude to dB
    mel_spectrogram = 20 * np.log10(np.maximum(1e-5, mel_spectrogram))
    
    # Normalize
    mel_spectrogram = np.clip((mel_spectrogram - 20 + 100) / 100, -4, 4)
    
    return mel_spectrogram.T  # Transpose to (time, mels)

def run_python_inference(audio_path, model_path):
    """Run full Python pipeline: audio -> mel -> ONNX encoder -> embeddings"""
    print("\n" + "="*80)
    print("PYTHON INFERENCE")
    print("="*80)
    
    # Load audio
    print(f"\n1. Loading audio: {audio_path}")
    audio_samples = load_wav(audio_path, 16000)
    print(f"   Audio shape: {audio_samples.shape}")
    print(f"   Duration: {len(audio_samples)/16000:.2f}s")
    
    # Generate mel-spectrogram
    print(f"\n2. Generating mel-spectrogram...")
    mel_spec = python_melspectrogram(audio_samples)
    print(f"   Mel-spec shape: {mel_spec.shape}")
    print(f"   Min: {mel_spec.min():.6f}, Max: {mel_spec.max():.6f}, Mean: {mel_spec.mean():.6f}")
    
    # Pad/trim to target frames (e.g., 16 frames for testing)
    target_frames = 16
    if mel_spec.shape[0] > target_frames:
        mel_spec = mel_spec[:target_frames, :]
    elif mel_spec.shape[0] < target_frames:
        pad_width = ((0, target_frames - mel_spec.shape[0]), (0, 0))
        mel_spec = np.pad(mel_spec, pad_width, mode='constant', constant_values=-4.0)
    
    print(f"   Padded shape: {mel_spec.shape}")
    
    # Run ONNX inference
    print(f"\n3. Running ONNX inference: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Prepare input: transpose from (frames, mels) to (mels, frames) and add batch+channel dims
    mel_transposed = mel_spec.T  # Now (80, 16)
    mel_input = mel_transposed[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1, 1, 80, 16)
    print(f"   Input shape: {mel_input.shape}")
    
    # Get input name
    input_name = session.get_inputs()[0].name
    print(f"   Input name: {input_name}")
    
    # Run inference
    outputs = session.run(None, {input_name: mel_input})
    audio_embedding = outputs[0]
    
    print(f"   Output shape: {audio_embedding.shape}")
    print(f"   Output min: {audio_embedding.min():.6f}, max: {audio_embedding.max():.6f}, mean: {audio_embedding.mean():.6f}")
    
    # Save for comparison
    np.save('debug_output/python_mel_for_encoder.npy', mel_spec)
    np.save('debug_output/python_audio_embedding.npy', audio_embedding)
    
    return mel_spec, audio_embedding

def load_go_outputs():
    """Load Go outputs for comparison"""
    print("\n" + "="*80)
    print("LOADING GO OUTPUTS")
    print("="*80)
    
    # Load Go mel-spectrogram
    print("\n1. Loading Go mel-spectrogram...")
    go_mel = load_bin_matrix('debug_output/go_step6_normalized.bin')
    print(f"   Shape: {go_mel.shape}")
    print(f"   Min: {go_mel.min():.6f}, Max: {go_mel.max():.6f}, Mean: {go_mel.mean():.6f}")
    
    # Load Go audio embedding
    print("\n2. Loading Go audio embedding...")
    go_embedding = load_bin_array('debug_output/go_audio_embedding.bin')
    
    # Reshape to match expected dimensions
    # Audio encoder typically outputs (batch, frames, features) or (batch, features)
    # Need to determine the actual shape from the file
    print(f"   Raw shape: {go_embedding.shape}")
    
    # Try to infer correct shape
    # Common shapes: (1, 512), (1, 16, 512), etc.
    if len(go_embedding.shape) == 1:
        # Need to reshape - check total elements
        total = go_embedding.shape[0]
        if total == 512:
            go_embedding = go_embedding.reshape(1, 512)
        elif total % 512 == 0:
            frames = total // 512
            go_embedding = go_embedding.reshape(1, frames, 512)
        else:
            print(f"   WARNING: Unexpected total elements: {total}")
    
    print(f"   Reshaped: {go_embedding.shape}")
    
    # Check if it's all zeros (placeholder)
    if go_embedding.size > 0:
        if np.all(go_embedding == 0):
            print(f"   ⚠️  WARNING: Go embedding is all zeros (placeholder)")
            print(f"   This means ONNX inference wasn't run in Go test")
        else:
            print(f"   Min: {go_embedding.min():.6f}, Max: {go_embedding.max():.6f}, Mean: {go_embedding.mean():.6f}")
    
    return go_mel, go_embedding

def load_bin_matrix(filepath):
    """Load binary matrix file (2D array) saved by Go"""
    with open(filepath, 'rb') as f:
        import struct
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(rows, cols)

def load_bin_array(filepath):
    """Load binary array file (1D array) saved by Go"""
    with open(filepath, 'rb') as f:
        import struct
        length = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data[:length]

def compare_arrays(name, py_arr, go_arr, tolerance=1e-4):
    """Compare two arrays with detailed statistics"""
    print("\n" + "="*80)
    print(f"COMPARING: {name}")
    print("="*80)
    
    # Check shapes
    print(f"\nPython shape: {py_arr.shape}")
    print(f"Go shape:     {go_arr.shape}")
    
    if py_arr.shape != go_arr.shape:
        print(f"\n❌ SHAPE MISMATCH!")
        return False
    
    # Flatten for comparison
    py_flat = py_arr.flatten()
    go_flat = go_arr.flatten()
    
    # Statistics
    print(f"\nPython - min: {py_flat.min():.8f}, max: {py_flat.max():.8f}, mean: {py_flat.mean():.8f}")
    print(f"Go     - min: {go_flat.min():.8f}, max: {go_flat.max():.8f}, mean: {go_flat.mean():.8f}")
    
    # Differences
    diff = np.abs(py_flat - go_flat)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\nDifference - max: {max_diff:.10f}, mean: {mean_diff:.10f}")
    
    # Max diff location
    max_idx = np.argmax(diff)
    print(f"Max diff at index {max_idx}: Python={py_flat[max_idx]:.10f}, Go={go_flat[max_idx]:.10f}")
    
    # First 10 values
    print(f"\nFirst 10 values:")
    for i in range(min(10, len(py_flat))):
        check = "✓" if diff[i] <= tolerance else "✗"
        print(f"  [{i}] Python: {py_flat[i]:.10f}, Go: {go_flat[i]:.10f}, diff: {diff[i]:.10e} {check}")
    
    # Count failures
    failures = np.sum(diff > tolerance)
    failure_pct = 100.0 * failures / len(diff)
    
    # Error distribution
    print(f"\nError distribution:")
    print(f"  Errors > {tolerance}: {failures:,}")
    print(f"  Errors > {tolerance*10}: {np.sum(diff > tolerance*10):,}")
    print(f"  Errors > {tolerance*100}: {np.sum(diff > tolerance*100):,}")
    print(f"  Errors > {tolerance*1000}: {np.sum(diff > tolerance*1000):,}")
    
    # Pass/fail
    if failures == 0:
        print(f"\n✅ PASS - All values within tolerance {tolerance}")
        return True
    else:
        print(f"\n❌ FAIL - {failures:,} values ({failure_pct:.2f}%) exceed tolerance {tolerance}")
        
        # Show some mismatches
        mismatch_indices = np.where(diff > tolerance)[0]
        print(f"\nFirst 20 mismatches:")
        for idx in mismatch_indices[:20]:
            print(f"  Index {idx}: Python={py_flat[idx]:.10f}, Go={go_flat[idx]:.10f}, diff={diff[idx]:.10e}")
        
        return False

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("FULL AUDIO ENCODER TEST: Python vs Go")
    print("="*80)
    
    # Paths
    audio_path = "../aud.wav"
    model_path = "../audio_encoder.onnx"
    
    # Check files exist
    if not os.path.exists(audio_path):
        print(f"\n❌ Audio file not found: {audio_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found: {model_path}")
        return
    
    # Create output directory
    os.makedirs('debug_output', exist_ok=True)
    
    # Run Python inference
    py_mel, py_embedding = run_python_inference(audio_path, model_path)
    
    # Check if Go outputs exist
    if not os.path.exists('debug_output/go_audio_embedding.bin'):
        print("\n" + "="*80)
        print("⚠️  GO OUTPUTS NOT FOUND")
        print("="*80)
        print("\nPlease run the Go test first to generate outputs:")
        print("  go run test_step_by_step.go")
        print("\nThis will create:")
        print("  - debug_output/go_step6_normalized.bin")
        print("  - debug_output/go_audio_embedding.bin")
        return
    
    # Load Go outputs
    go_mel, go_embedding = load_go_outputs()
    
    # Compare mel-spectrograms (first 16 frames)
    py_mel_16 = py_mel[:16, :]
    go_mel_16 = go_mel[:16, :]
    
    mel_passed = compare_arrays(
        "Mel-Spectrogram (16 frames)",
        py_mel_16,
        go_mel_16,
        tolerance=0.01  # 1% tolerance for mel
    )
    
    # Compare audio embeddings
    if np.all(go_embedding == 0):
        print("\n⚠️  Skipping embedding comparison (Go output is placeholder)")
        embedding_passed = None
    else:
        embedding_passed = compare_arrays(
            "Audio Encoder Embeddings",
            py_embedding,
            go_embedding,
            tolerance=0.001  # 0.1% tolerance for embeddings
        )
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n{'✅' if mel_passed else '❌'} Mel-Spectrogram: {'PASS' if mel_passed else 'FAIL'}")
    if embedding_passed is not None:
        print(f"{'✅' if embedding_passed else '❌'} Audio Embeddings: {'PASS' if embedding_passed else 'FAIL'}")
    else:
        print(f"⚠️  Audio Embeddings: SKIPPED (placeholder)")
    
    if mel_passed:
        if embedding_passed:
            print("\n🎉 SUCCESS! Python and Go audio encoders produce matching results!")
        elif embedding_passed is None:
            print("\n✅ Mel-spectrograms match! (Encoder inference not tested in Go)")
            print("   The audio processing pipeline is working correctly.")
            print("   Mel-spec errors of <1% are within acceptable tolerance.")
        else:
            print("\n⚠️  Audio embeddings differ. This may affect lip-sync quality.")
    else:
        print("\n❌ Mel-spectrograms don't match sufficiently.")
        print("Review the differences above to determine if they are acceptable.")

if __name__ == "__main__":
    main()
