"""Compare mel-spectrogram windows between Python and Go"""
import numpy as np

frame_idx = 8

# Load Python reference
mel_python = np.load(f'test_output/mel_windows_python/frame_{frame_idx}.npy')
print(f"Python mel window shape: {mel_python.shape}")  # Should be [16, 80]

# Load Go implementation
mel_go = np.fromfile(f'test_output/mel_windows_go/frame_{frame_idx}.bin', dtype=np.float32)
mel_go = mel_go.reshape(16, 80)
print(f"Go mel window shape: {mel_go.shape}")

# Compare
print("\n" + "="*70)
print(f"Mel-Spectrogram Comparison (Frame {frame_idx})")
print("="*70)

print(f"\nPython stats:")
print(f"  Mean: {mel_python.mean():.6f}")
print(f"  Std:  {mel_python.std():.6f}")
print(f"  Min:  {mel_python.min():.6f}")
print(f"  Max:  {mel_python.max():.6f}")

print(f"\nGo stats:")
print(f"  Mean: {mel_go.mean():.6f}")
print(f"  Std:  {mel_go.std():.6f}")
print(f"  Min:  {mel_go.min():.6f}")
print(f"  Max:  {mel_go.max():.6f}")

# Difference
diff = np.abs(mel_python - mel_go)
print(f"\nDifference:")
print(f"  Mean abs diff: {diff.mean():.6f}")
print(f"  Max abs diff:  {diff.max():.6f}")
print(f"  Exact matches: {(diff == 0).sum()} / {diff.size} ({(diff == 0).sum()/diff.size*100:.1f}%)")
print(f"  Within 0.001:  {(diff < 0.001).sum()} / {diff.size} ({(diff < 0.001).sum()/diff.size*100:.1f}%)")
print(f"  Within 0.01:   {(diff < 0.01).sum()} / {diff.size} ({(diff < 0.01).sum()/diff.size*100:.1f}%)")

if diff.max() > 0.01:
    print(f"\n❌ MEL-SPECTROGRAMS ARE SIGNIFICANTLY DIFFERENT!")
    print(f"   This explains why audio encoder outputs differ.")
    print(f"\n   Top 5 largest differences:")
    flat_diff = diff.flatten()
    top_indices = np.argsort(flat_diff)[-5:][::-1]
    for idx in top_indices:
        py_val = mel_python.flatten()[idx]
        go_val = mel_go.flatten()[idx]
        diff_val = flat_diff[idx]
        pos = np.unravel_index(idx, mel_python.shape)
        print(f"     Position {pos}: Python={py_val:.4f}, Go={go_val:.4f}, diff={diff_val:.4f}")
elif diff.max() > 0.001:
    print(f"\n⚠ Mel-spectrograms have small differences (max {diff.max():.6f})")
else:
    print(f"\n✓ Mel-spectrograms are nearly identical!")

# Check first few time steps
print(f"\nFirst 3 time steps comparison:")
for t in range(min(3, 16)):
    diff_t = np.abs(mel_python[t] - mel_go[t]).mean()
    print(f"  Time {t}: mean_diff={diff_t:.6f}")
