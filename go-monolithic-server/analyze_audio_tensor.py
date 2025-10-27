"""
Analyze the Go audio tensor to understand the stripes issue
"""
import numpy as np
import matplotlib.pyplot as plt

# Load the Go audio tensor (8192 floats = 32768 bytes)
go_audio = np.fromfile('debug_go_audio_tensor.bin', dtype=np.float32)
print(f"Go audio tensor shape: {go_audio.shape}")
print(f"Go audio tensor min/max: {go_audio.min():.6f} / {go_audio.max():.6f}")
print(f"Go audio tensor mean/std: {go_audio.mean():.6f} / {go_audio.std():.6f}")

# The tensor should be [32, 16, 16] = 8192 elements
# Reshape as [32, 16, 16]
audio_3d = go_audio.reshape(32, 16, 16)

# Visualize all 32 channels
fig, axes = plt.subplots(4, 8, figsize=(20, 10))
fig.suptitle('Go Audio Tensor: All 32 Channels (16x16 each)', fontsize=16)

for i in range(32):
    row = i // 8
    col = i % 8
    ax = axes[row, col]
    im = ax.imshow(audio_3d[i], cmap='viridis', aspect='auto')
    ax.set_title(f'Ch {i}', fontsize=8)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('test_output/go_audio_channels.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved visualization to: test_output/go_audio_channels.png")

# Now let's see what the CORRECT Python reshape should look like
# According to the doc: get_audio_features returns [16, 512]
# Then: audio_feat.reshape(32, 16, 16)

# Simulate what Python does:
# Start with [16, 512] (16 frames, 512 features each)
frames_16x512 = go_audio.reshape(16, 512)

print(f"\nAs [16, 512]:")
print(f"  Shape: {frames_16x512.shape}")
print(f"  First frame (512 features): min={frames_16x512[0].min():.6f}, max={frames_16x512[0].max():.6f}")

# Python's reshape to [32, 16, 16] reinterprets the SAME linear memory
python_reshape = frames_16x512.reshape(32, 16, 16)

print(f"\nAfter Python-style reshape to [32, 16, 16]:")
print(f"  Shape: {python_reshape.shape}")

# Check if they're identical (they should be!)
if np.array_equal(audio_3d, python_reshape):
    print("✅ Go audio tensor matches expected Python reshape!")
else:
    print("❌ Go audio tensor does NOT match Python reshape!")
    print(f"   Difference: max_diff = {np.abs(audio_3d - python_reshape).max()}")

# Visualize the difference
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# Channel 0 from both
axes2[0].imshow(audio_3d[0], cmap='viridis', aspect='auto')
axes2[0].set_title('Go Audio: Channel 0')
axes2[0].axis('off')

axes2[1].imshow(python_reshape[0], cmap='viridis', aspect='auto')
axes2[1].set_title('Python Reshape: Channel 0')
axes2[1].axis('off')

diff = audio_3d[0] - python_reshape[0]
im = axes2[2].imshow(diff, cmap='RdBu', aspect='auto', vmin=-0.1, vmax=0.1)
axes2[2].set_title(f'Difference (max={np.abs(diff).max():.6f})')
axes2[2].axis('off')
plt.colorbar(im, ax=axes2[2])

plt.tight_layout()
plt.savefig('test_output/go_vs_python_reshape.png', dpi=150, bbox_inches='tight')
print("✅ Saved comparison to: test_output/go_vs_python_reshape.png")

# Now let's understand the STRIPE pattern
# Check if there's a repeating pattern across channels
print("\n=== Analyzing for STRIPE patterns ===")
print("Checking if channels have repeating data...")

# Compare channels to see if they're duplicates
unique_channels = []
for i in range(32):
    is_unique = True
    for j, uch in enumerate(unique_channels):
        if np.array_equal(audio_3d[i], uch):
            print(f"Channel {i} is DUPLICATE of unique channel {j}")
            is_unique = False
            break
    if is_unique:
        unique_channels.append(audio_3d[i])

print(f"\nTotal unique channels: {len(unique_channels)} out of 32")

if len(unique_channels) < 32:
    print("⚠️  WARNING: Channels are duplicated! This would cause stripes!")
    
# Check for row-wise patterns within channels
print("\nChecking for row-wise repetition...")
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
    print(f"  Channel {ch_idx}: {len(unique_rows)}/16 unique rows")

print("\n" + "="*60)
plt.show()
