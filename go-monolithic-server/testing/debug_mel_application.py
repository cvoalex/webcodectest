"""
Debug mel filterbank application
"""
import numpy as np
import struct

# Load Python magnitude
py_mag = np.load("debug_output/python_step3_magnitude.npy")  # [frames, freq]

# Load Go magnitude (binary format)
with open("debug_output/go_step3_magnitude.bin", 'rb') as f:
    rows = struct.unpack('<i', f.read(4))[0]
    cols = struct.unpack('<i', f.read(4))[0]
    data = f.read()
    go_mag = np.array(struct.unpack(f'<{rows*cols}f', data), dtype=np.float32).reshape((rows, cols))

print(f"Python magnitude shape: {py_mag.shape}")
print(f"Go magnitude shape: {go_mag.shape}")
print(f"\nPython first frame (first 10 freq bins): {py_mag[0, :10]}")
print(f"Go first frame (first 10 freq bins): {go_mag[0, :10]}")

# Load mel filters
import json
with open("debug_output/mel_filters_python.json", 'r') as f:
    mel_data = json.load(f)
    mel_filters = np.array(mel_data['filters'], dtype=np.float32)

print(f"\nMel filterbank shape: {mel_filters.shape}")  # Should be (80, 401)

# Manually apply mel filterbank to Python magnitude (first frame only)
py_frame0 = py_mag[0, :]  # Shape: (401,)
print(f"\nPython frame 0 shape: {py_frame0.shape}")

# Apply mel filterbank: mel_result[i] = sum(mel_filters[i, j] * magnitude[j])
py_mel_manual = np.dot(mel_filters, py_frame0)
print(f"Python mel (manual) first 10: {py_mel_manual[:10]}")

# Load actual Python mel output
py_mel = np.load("debug_output/python_step4_mel.npy")
print(f"Python mel (from file) first frame first 10: {py_mel[0, :10]}")

# Load Go mel output
with open("debug_output/go_step4_mel.bin", 'rb') as f:
    rows = struct.unpack('<i', f.read(4))[0]
    cols = struct.unpack('<i', f.read(4))[0]
    data = f.read()
    go_mel = np.array(struct.unpack(f'<{rows*cols}f', data), dtype=np.float32).reshape((rows, cols))

print(f"Go mel (from file) first frame first 10: {go_mel[0, :10]}")

# Now manually apply to Go magnitude
go_frame0 = go_mag[0, :]
go_mel_manual = np.dot(mel_filters, go_frame0)
print(f"\nGo mel (manual) first 10: {go_mel_manual[:10]}")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"Python mel manual == Python mel file: {np.allclose(py_mel_manual, py_mel[0, :])}")
print(f"Go mel manual == Go mel file: {np.allclose(go_mel_manual, go_mel[0, :])}")

print(f"\nDifference between Python and Go magnitudes (frame 0):")
mag_diff = np.abs(py_frame0 - go_frame0)
print(f"  Max: {mag_diff.max():.8f}, Mean: {mag_diff.mean():.8f}")
print(f"  Indices with biggest differences:")
for idx in np.argsort(mag_diff)[-10:]:
    print(f"    [{idx}] Python={py_frame0[idx]:.8f}, Go={go_frame0[idx]:.8f}, diff={mag_diff[idx]:.8f}")
