"""
Detailed analysis of audio feature structure
"""
import numpy as np

# Load audio tensor
audio = np.fromfile('debug_go_audio_tensor.bin', dtype=np.float32)

# View as [16 frames, 512 features]
frames = audio.reshape(16, 512)

print("=== Analyzing [16, 512] structure ===")
print()

# Check each frame
for f in range(16):
    frame_features = frames[f]
    non_zero = np.count_nonzero(frame_features)
    unique_vals = len(np.unique(frame_features))
    
    print(f"Frame {f:2d}: non-zero={non_zero:3d}/512, unique_vals={unique_vals:3d}, "
          f"min={frame_features.min():.4f}, max={frame_features.max():.4f}, "
          f"mean={frame_features.mean():.4f}")
    
    # Show first 32 values
    if f < 3:
        print(f"    First 32 vals: {frame_features[:32]}")
        print()

print()
print("=== Checking for feature duplication across frames ===")
print()

# Compare frames to see if any are duplicates
for i in range(16):
    for j in range(i+1, 16):
        if np.array_equal(frames[i], frames[j]):
            print(f"⚠️  Frame {i} == Frame {j} (DUPLICATE!)")

print()            
print("=== Analyzing which features vary ===")
print()

# For each feature position (0-511), check if it varies across frames
varying_features = []
constant_features = []

for feat_idx in range(512):
    feature_across_frames = frames[:, feat_idx]
    if np.all(feature_across_frames == feature_across_frames[0]):
        constant_features.append(feat_idx)
    else:
        varying_features.append(feat_idx)

print(f"Varying features: {len(varying_features)}/512")
print(f"Constant features: {len(constant_features)}/512")

if len(varying_features) < 512:
    print(f"\nFirst 20 varying feature indices: {varying_features[:20]}")
    print(f"First 20 constant feature indices: {constant_features[:20]}")

# Most important: check if features are structured in a weird way
print()
print("=== Checking for block structure in 512 features ===")
print()

# The 512 features might be structured as chunks
# Let's see if there are blocks of 16
for block_idx in range(0, 512, 16):
    block = frames[0, block_idx:block_idx+16]
    non_zero = np.count_nonzero(block)
    print(f"Feature block [{block_idx:3d}:{block_idx+16:3d}]: "
          f"non-zero={non_zero:2d}/16, "
          f"min={block.min():.4f}, max={block.max():.4f}")

print()
print("="*70)

