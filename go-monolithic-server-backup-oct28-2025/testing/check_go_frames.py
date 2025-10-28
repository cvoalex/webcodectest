"""Check if first two frames are identical in Go implementation"""
import numpy as np

# Load Go tensor
go_tensor = np.fromfile(r'test_output/audio_tensor_frame_8.bin', dtype=np.float32).reshape(1, 32, 16, 16)

# Reshape to [16, 512] for easier analysis
frames = go_tensor.reshape(16, 512)

print("Checking if frames are identical...")
for i in range(16):
    for j in range(i+1, 16):
        if np.array_equal(frames[i], frames[j]):
            print(f"Frame {i} == Frame {j}: IDENTICAL")
        elif np.allclose(frames[i], frames[j], atol=1e-6):
            print(f"Frame {i} ≈ Frame {j}: Very close (max diff: {np.abs(frames[i]-frames[j]).max()})")

print("\nChecking individual frame stats:")
for i in range(min(4, 16)):
    nonzero = (frames[i] != 0).sum()
    mean = frames[i].mean()
    print(f"Frame {i}: mean={mean:.6f}, nonzero={nonzero}/512, all_zero={nonzero==0}")
