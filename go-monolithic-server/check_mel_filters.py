"""
Check if mel filters match between Python and Go
"""
import numpy as np
import json
import struct

# Load Python mel filters
with open("debug_output/mel_filters_python.json", 'r') as f:
    py_mel_data = json.load(f)
    py_mel_filters = np.array(py_mel_data['filters'], dtype=np.float32)

# Load Go mel filters
with open("audio_test_data/mel_filters.json", 'r') as f:
    go_mel_data = json.load(f)
    go_mel_filters = np.array(go_mel_data['filters'], dtype=np.float32)

print(f"Python mel filters shape: {py_mel_filters.shape}")
print(f"Go mel filters shape: {go_mel_filters.shape}")

print(f"\nPython first filter (first 10 values): {py_mel_filters[0, :10]}")
print(f"Go first filter (first 10 values): {go_mel_filters[0, :10]}")

print(f"\nPython filter 5 (first 20 values): {py_mel_filters[5, :20]}")
print(f"Go filter 5 (first 20 values): {go_mel_filters[5, :20]}")

# Check if they're identical
if np.allclose(py_mel_filters, go_mel_filters):
    print("\n✅ Mel filters are IDENTICAL")
else:
    print("\n❌ Mel filters are DIFFERENT!")
    diff = np.abs(py_mel_filters - go_mel_filters)
    print(f"Max difference: {diff.max()}")
    print(f"Mean difference: {diff.mean()}")
    
    # Find where they differ most
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"Max diff at {max_idx}: Python={py_mel_filters[max_idx]}, Go={go_mel_filters[max_idx]}")
