"""
Compare audio tensors between PyTorch reference and Go implementation.
This will help identify exactly where the discrepancy occurs.
"""
import numpy as np
import json
import sys

def load_pytorch_reference():
    """Load PyTorch reference tensors from SyncTalk2D"""
    base_path = r'D:\Projects\SyncTalk2D\dataset\sanders\test_output_pth'
    
    # Load audio input tensor [1, 32, 16, 16]
    audio_pth = np.load(f'{base_path}\\tensors_npy\\frame_8_audio_input.npy')
    
    # Load metadata
    with open(f'{base_path}\\frame_8_tensors.json', 'r') as f:
        metadata = json.load(f)
    
    print("=" * 70)
    print("PyTorch Reference (Frame 8)")
    print("=" * 70)
    print(f"Audio input shape: {audio_pth.shape}")
    print(f"Audio input mean: {audio_pth.mean():.6f}")
    print(f"Audio input std: {audio_pth.std():.6f}")
    print(f"Audio input min/max: [{audio_pth.min():.6f}, {audio_pth.max():.6f}]")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    print()
    
    return audio_pth, metadata

def load_go_tensor(frame_idx=8):
    """Load Go tensor from binary file"""
    tensor_path = f'test_output/audio_tensor_frame_{frame_idx}.bin'
    
    try:
        # Read binary file (32*16*16 = 8192 float32 values)
        audio_go = np.fromfile(tensor_path, dtype=np.float32)
        
        # Reshape to [1, 32, 16, 16] to match PyTorch
        audio_go = audio_go.reshape(1, 32, 16, 16)
        
        print("=" * 70)
        print(f"Go Implementation (Frame {frame_idx})")
        print("=" * 70)
        print(f"Audio input shape: {audio_go.shape}")
        print(f"Audio input mean: {audio_go.mean():.6f}")
        print(f"Audio input std: {audio_go.std():.6f}")
        print(f"Audio input min/max: [{audio_go.min():.6f}, {audio_go.max():.6f}]")
        print()
        
        return audio_go
    except FileNotFoundError:
        print(f"ERROR: Go tensor file not found: {tensor_path}")
        print("Please run Go server with tensor saving enabled first!")
        return None

def compare_tensors(pth_tensor, go_tensor):
    """Detailed comparison of tensors"""
    print("=" * 70)
    print("Tensor Comparison")
    print("=" * 70)
    
    # Overall statistics
    diff = np.abs(pth_tensor - go_tensor)
    print(f"Mean absolute difference: {diff.mean():.6f}")
    print(f"Max absolute difference: {diff.max():.6f}")
    print(f"Percentage of exact matches: {(diff == 0).sum() / diff.size * 100:.2f}%")
    print(f"Percentage within 0.001: {(diff < 0.001).sum() / diff.size * 100:.2f}%")
    print(f"Percentage within 0.01: {(diff < 0.01).sum() / diff.size * 100:.2f}%")
    print()
    
    # Check if tensors are identical
    if np.allclose(pth_tensor, go_tensor, atol=1e-5):
        print("✓ Tensors are IDENTICAL (within tolerance 1e-5)!")
        return
    
    # Analyze differences by frame (in the 16-frame window)
    print("Differences by frame in 16-frame window:")
    print("-" * 70)
    # Reshape to [16, 512] for easier analysis
    pth_frames = pth_tensor.reshape(16, 512)
    go_frames = go_tensor.reshape(16, 512)
    
    for i in range(16):
        frame_diff = np.abs(pth_frames[i] - go_frames[i])
        mean_diff = frame_diff.mean()
        max_diff = frame_diff.max()
        nonzero_pth = (pth_frames[i] != 0).sum()
        nonzero_go = (go_frames[i] != 0).sum()
        
        status = "✓" if mean_diff < 0.001 else "✗"
        print(f"Frame {i:2d}: mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}, "
              f"nonzero(pth={nonzero_pth:3d}, go={nonzero_go:3d}) {status}")
    print()
    
    # Find largest differences
    flat_diff = diff.flatten()
    top_indices = np.argsort(flat_diff)[-10:][::-1]
    
    print("Top 10 largest differences:")
    print("-" * 70)
    for idx in top_indices:
        pth_val = pth_tensor.flatten()[idx]
        go_val = go_tensor.flatten()[idx]
        diff_val = flat_diff[idx]
        
        # Convert flat index to position
        pos = np.unravel_index(idx, pth_tensor.shape)
        print(f"Position {pos}: PyTorch={pth_val:.6f}, Go={go_val:.6f}, diff={diff_val:.6f}")
    print()
    
    # Analyze zero patterns
    pth_zeros = (pth_tensor == 0).sum()
    go_zeros = (go_tensor == 0).sum()
    print(f"Zero values: PyTorch={pth_zeros}, Go={go_zeros}")
    
    # Check if first frames are zero (should be for frame 8)
    print("\nFirst 3 frames analysis (should have zeros for early frames):")
    for i in range(3):
        pth_is_zero = (pth_frames[i] == 0).all()
        go_is_zero = (go_frames[i] == 0).all()
        print(f"  Frame {i}: PyTorch all-zero={pth_is_zero}, Go all-zero={go_is_zero}")

def main():
    print("\n" + "=" * 70)
    print("Audio Tensor Comparison: PyTorch Reference vs Go Implementation")
    print("=" * 70)
    print()
    
    # Load PyTorch reference
    pth_tensor, metadata = load_pytorch_reference()
    
    # Load Go tensor
    go_tensor = load_go_tensor(frame_idx=8)
    
    if go_tensor is None:
        print("\n⚠ Cannot compare - Go tensor not found!")
        print("\nNext steps:")
        print("1. Modify Go server to save audio tensors")
        print("2. Run server and generate frame 8")
        print("3. Re-run this comparison script")
        return 1
    
    # Compare
    compare_tensors(pth_tensor, go_tensor)
    
    print("=" * 70)
    print("Comparison Complete")
    print("=" * 70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
