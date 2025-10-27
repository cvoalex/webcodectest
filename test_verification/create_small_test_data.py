#!/usr/bin/env python3
"""
Create a small 5-frame test dataset from the existing 100-frame dataset
ONE-TIME SETUP - NOT FOR INFERENCE!
"""
import numpy as np
import os

print("================================================================================")
print("🔬 CREATING 5-FRAME TEST DATASET")
print("================================================================================")

# Load existing 100-frame data
data_dir = "d:/Projects/webcodecstest/test_data_sanders_for_go"
visual_data = np.fromfile(os.path.join(data_dir, "visual_input.bin"), dtype=np.float32)
audio_data = np.fromfile(os.path.join(data_dir, "audio_input.bin"), dtype=np.float32)

print(f"Loaded 100-frame dataset:")
print(f"  Visual: {visual_data.shape} ({visual_data.nbytes / (1024*1024):.2f} MB)")
print(f"  Audio: {audio_data.shape} ({audio_data.nbytes / (1024*1024):.2f} MB)")

# Extract first 5 frames
visual_frame_size = 6 * 320 * 320
audio_frame_size = 32 * 16 * 16

visual_5 = visual_data[:5 * visual_frame_size]
audio_5 = audio_data[:5 * audio_frame_size]

print(f"\nExtracted 5 frames:")
print(f"  Visual: {visual_5.shape} ({visual_5.nbytes / (1024*1024):.2f} MB)")
print(f"  Audio: {audio_5.shape} ({audio_5.nbytes / (1024*1024):.2f} MB)")

# Create output directory
output_dir = "d:/Projects/webcodecstest/go-onnx-inference/test_data_5_frames"
os.makedirs(output_dir, exist_ok=True)

# Save
visual_5.tofile(os.path.join(output_dir, "visual_input.bin"))
audio_5.tofile(os.path.join(output_dir, "audio_input.bin"))

# Create metadata
import json
metadata = {
    "num_frames": 5,
    "visual_shape": [5, 6, 320, 320],
    "audio_shape": [5, 32, 16, 16],
    "format": "BGR",
    "description": "Small 5-frame test dataset for parallel testing"
}

with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Created 5-frame dataset in: {output_dir}/")
print(f"   visual_input.bin: {visual_5.nbytes / (1024*1024):.2f} MB")
print(f"   audio_input.bin: {audio_5.nbytes / (1024*1024):.2f} MB")
print(f"   metadata.json")
print("\n🚀 Now run with 5 workers for 1 frame per worker!")
print("================================================================================")
