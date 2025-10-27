"""
Simple analysis of Go audio tensor without matplotlib
"""
import numpy as np

# Load the Go audio tensor (8192 floats = 32768 bytes)
go_audio = np.fromfile('debug_go_audio_tensor.bin', dtype=np.float32)
print(f"Go audio tensor shape: {go_audio.shape}")
print(f"Go audio tensor min/max: {go_audio.min():.6f} / {go_audio.max():.6f}")
print(f"Go audio tensor mean/std: {go_audio.mean():.6f} / {go_audio.std():.6f}")
print()

# The tensor should be [32, 16, 16] = 8192 elements
# Reshape as [32, 16, 16]
audio_3d = go_audio.reshape(32, 16, 16)

# Now let's see what the CORRECT Python reshape should look like
# According to the doc: get_audio_features returns [16, 512]
# Then: audio_feat.reshape(32, 16, 16)

# Simulate what Python does:
# Start with [16, 512] (16 frames, 512 features each)
frames_16x512 = go_audio.reshape(16, 512)

print(f"As [16, 512]:")
print(f"  Shape: {frames_16x512.shape}")
print(f"  First frame (512 features): min={frames_16x512[0].min():.6f}, max={frames_16x512[0].max():.6f}")
print()

# Python's reshape to [32, 16, 16] reinterprets the SAME linear memory
python_reshape = frames_16x512.reshape(32, 16, 16)

print(f"After Python-style reshape to [32, 16, 16]:")
print(f"  Shape: {python_reshape.shape}")
print()

# Check if they're identical (they should be!)
if np.array_equal(audio_3d, python_reshape):
    print("✅ Go audio tensor matches expected Python reshape!")
else:
    print("❌ Go audio tensor does NOT match Python reshape!")
    print(f"   Difference: max_diff = {np.abs(audio_3d - python_reshape).max()}")
print()

# Now let's understand the STRIPE pattern
# Check if there's a repeating pattern across channels
print("=== Analyzing for STRIPE patterns ===")
print("Checking if channels have repeating data...")
print()

# Compare channels to see if they're duplicates
unique_channels = []
channel_map = {}  # Maps channel index to which unique channel it matches
for i in range(32):
    is_unique = True
    for j, uch in enumerate(unique_channels):
        if np.array_equal(audio_3d[i], uch):
            print(f"Channel {i:2d} is DUPLICATE of unique channel {j}")
            channel_map[i] = j
            is_unique = False
            break
    if is_unique:
        unique_channels.append(audio_3d[i])
        channel_map[i] = len(unique_channels) - 1
        print(f"Channel {i:2d} is UNIQUE (unique channel {len(unique_channels)-1})")

print()
print(f"Total unique channels: {len(unique_channels)} out of 32")

if len(unique_channels) < 32:
    print("⚠️  WARNING: Channels are duplicated! This would cause stripes!")
    print()
    print("Duplication pattern:")
    for i in range(32):
        if i % 4 == 0:
            print()
        print(f"  Ch{i:2d} -> UniqueIdx{channel_map[i]:2d}", end="  ")
    print()
else:
    print("✅ All channels are unique")
    
print()
# Check for row-wise patterns within channels
print("Checking for row-wise repetition...")
for ch_idx in [0, 1, 15, 31]:  # Sample a few channels
    channel = audio_3d[ch_idx]
    unique_rows = []
    for i in range(16):
        is_unique = True
        for j, ur in enumerate(unique_rows):
            if np.array_equal(channel[i], ur):
                is_unique = False
                break
        if is_unique:
            unique_rows.append(channel[i])
    print(f"  Channel {ch_idx:2d}: {len(unique_rows):2d}/16 unique rows")

print()
print("="*70)

# Print first few values from a few channels to see the pattern
print("\nFirst 8 values from selected channels:")
for ch in [0, 1, 2, 15, 16, 31]:
    print(f"  Ch{ch:2d}: {audio_3d[ch, 0, :8]}")

