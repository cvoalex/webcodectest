"""
Final comparison script - compares all intermediate steps from Python and Go
"""

import numpy as np
import os
import sys
import struct

def load_npy(filepath):
    """Load a .npy file"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    return np.load(filepath)

def load_bin_array(filepath):
    """Load a binary array file (1D float32)"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Read as float32 array
    num_floats = len(data) // 4
    arr = np.array(struct.unpack(f'<{num_floats}f', data), dtype=np.float32)
    return arr

def load_bin_matrix(filepath):
    """Load a binary matrix file (2D float32 with dimensions header)"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        # Read dimensions
        rows = struct.unpack('<i', f.read(4))[0]
        cols = struct.unpack('<i', f.read(4))[0]
        
        # Read data
        num_floats = rows * cols
        data = f.read()
        arr = np.array(struct.unpack(f'<{num_floats}f', data), dtype=np.float32)
        arr = arr.reshape((rows, cols))
    
    return arr

def compare_arrays(name, python_arr, go_arr, tolerance=1e-5, verbose=True):
    """Compare two arrays and report differences"""
    if python_arr is None or go_arr is None:
        print(f"\n❌ {name}: One or both arrays are None")
        return False
    
    print(f"\n{'='*80}")
    print(f"Comparing: {name}")
    print(f"{'='*80}")
    
    # Shape check
    if verbose:
        print(f"Python shape: {python_arr.shape}")
        print(f"Go shape:     {go_arr.shape}")
    
    if python_arr.shape != go_arr.shape:
        print(f"❌ SHAPES DON'T MATCH! Python: {python_arr.shape}, Go: {go_arr.shape}")
        return False
    
    # Flatten for easier comparison
    py_flat = python_arr.flatten()
    go_flat = go_arr.flatten()
    
    # Statistics
    if verbose:
        print(f"\nPython - min: {py_flat.min():.8f}, max: {py_flat.max():.8f}, mean: {py_flat.mean():.8f}")
        print(f"Go     - min: {go_flat.min():.8f}, max: {go_flat.max():.8f}, mean: {go_flat.mean():.8f}")
    
    # Difference
    diff = np.abs(py_flat - go_flat)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\nDifference - max: {max_diff:.10f}, mean: {mean_diff:.10f}")
    
    # Find where max difference occurs
    max_idx = np.argmax(diff)
    if len(python_arr.shape) == 1:
        print(f"Max diff at index {max_idx}: Python={py_flat[max_idx]:.10f}, Go={go_flat[max_idx]:.10f}")
    else:
        max_idx_2d = np.unravel_index(max_idx, python_arr.shape)
        print(f"Max diff at index {max_idx_2d}: Python={py_flat[max_idx]:.10f}, Go={go_flat[max_idx]:.10f}")
    
    # Show first few values
    if verbose:
        print(f"\nFirst 10 values:")
        for i in range(min(10, len(py_flat))):
            match = "✓" if abs(py_flat[i] - go_flat[i]) < tolerance else "✗"
            print(f"  [{i}] Python: {py_flat[i]:.10f}, Go: {go_flat[i]:.10f}, diff: {abs(py_flat[i] - go_flat[i]):.10e} {match}")
    
    # Check tolerance
    num_mismatches = np.sum(diff > tolerance)
    percent_mismatch = (num_mismatches / len(diff)) * 100
    
    if max_diff < tolerance:
        print(f"✅ PASS - All values match within tolerance {tolerance}")
        return True
    else:
        print(f"❌ FAIL - {num_mismatches:,} values ({percent_mismatch:.2f}%) exceed tolerance {tolerance}")
        
        # Show distribution of errors
        if num_mismatches > 0:
            print(f"\nError distribution:")
            print(f"  Errors > {tolerance:.0e}: {num_mismatches:,}")
            print(f"  Errors > {tolerance*10:.0e}: {np.sum(diff > tolerance*10):,}")
            print(f"  Errors > {tolerance*100:.0e}: {np.sum(diff > tolerance*100):,}")
            print(f"  Errors > {tolerance*1000:.0e}: {np.sum(diff > tolerance*1000):,}")
        
        # Show some mismatches
        if num_mismatches <= 20:
            print(f"\nAll {num_mismatches} mismatches:")
            problem_indices = np.where(diff > tolerance)[0]
            for idx in problem_indices:
                if len(python_arr.shape) == 1:
                    print(f"  Index {idx}: Python={py_flat[idx]:.10f}, Go={go_flat[idx]:.10f}, diff={diff[idx]:.10e}")
                else:
                    idx_2d = np.unravel_index(idx, python_arr.shape)
                    print(f"  Index {idx_2d}: Python={py_flat[idx]:.10f}, Go={go_flat[idx]:.10f}, diff={diff[idx]:.10e}")
        else:
            print(f"\nFirst 20 mismatches:")
            problem_indices = np.where(diff > tolerance)[0][:20]
            for idx in problem_indices:
                if len(python_arr.shape) == 1:
                    print(f"  Index {idx}: Python={py_flat[idx]:.10f}, Go={go_flat[idx]:.10f}, diff={diff[idx]:.10e}")
                else:
                    idx_2d = np.unravel_index(idx, python_arr.shape)
                    print(f"  Index {idx_2d}: Python={py_flat[idx]:.10f}, Go={go_flat[idx]:.10f}, diff={diff[idx]:.10e}")
        
        return False

def main():
    print("="*80)
    print("STEP-BY-STEP COMPARISON: Python vs Go")
    print("="*80)
    
    output_dir = "debug_output"
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        print("Please run the Python and Go processing scripts first!")
        return
    
    steps = [
        ("Step 0: Original Audio", "step0_original", 1e-6, True),
        ("Step 1: Pre-emphasis", "step1_preemphasis", 1e-6, True),
        ("Step 2: STFT Real", "step2_stft_real", 1e-4, False),
        ("Step 2: STFT Imag", "step2_stft_imag", 1e-4, False),
        ("Step 3: Magnitude", "step3_magnitude", 1e-5, False),
        ("Step 4: Mel", "step4_mel", 1e-4, False),
        ("Step 5a: dB (raw)", "step5a_db_raw", 1e-3, False),
        ("Step 5b: dB (adjusted)", "step5b_db_adjusted", 1e-3, False),
        ("Step 6: Normalized", "step6_normalized", 1e-3, False),
    ]
    
    results = []
    
    for step_name, filename, tolerance, is_1d in steps:
        python_file = os.path.join(output_dir, f"python_{filename}.npy")
        go_file = os.path.join(output_dir, f"go_{filename}.bin")
        
        python_data = load_npy(python_file)
        if is_1d:
            go_data = load_bin_array(go_file)
        else:
            go_data = load_bin_matrix(go_file)
        
        if python_data is None or go_data is None:
            results.append((step_name, False, "Missing data"))
            continue
        
        passed = compare_arrays(step_name, python_data, go_data, tolerance=tolerance)
        results.append((step_name, passed, "PASS" if passed else "FAIL"))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for step_name, passed, status in results:
        icon = "✅" if passed else "❌"
        print(f"{icon} {step_name}: {status}")
    
    num_passed = sum(1 for _, passed, _ in results if passed)
    num_total = len(results)
    
    print(f"\nTotal: {num_passed}/{num_total} steps passed")
    
    if num_passed == num_total:
        print("\n🎉 SUCCESS! All steps match between Python and Go!")
    else:
        print(f"\n⚠️  {num_total - num_passed} step(s) failed. Review the differences above.")
        
        # Find first failure
        for i, (step_name, passed, _) in enumerate(results):
            if not passed:
                print(f"\n💡 First failure at: {step_name}")
                print("   This is likely where the bug is introduced.")
                break

if __name__ == "__main__":
    main()
