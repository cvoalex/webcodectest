#!/usr/bin/env python3
"""
Debug: Print out exactly what inputs we're feeding to the model
Compare with what the reference code would produce
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

def load_frame_data_our_way(frame_id, package_dir="models/default_model"):
    """Our current approach"""
    
    # Load face regions
    face_cap = cv2.VideoCapture(os.path.join(package_dir, "face_regions_320.mp4"))
    masked_cap = cv2.VideoCapture(os.path.join(package_dir, "masked_regions_320.mp4"))
    
    for _ in range(frame_id):
        face_cap.read()
        masked_cap.read()
    
    ret1, face_frame = face_cap.read()
    ret2, masked_frame = masked_cap.read()
    
    face_cap.release()
    masked_cap.release()
    
    # Convert BGR to RGB
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and transpose
    face_norm = face_frame.astype(np.float32) / 255.0
    masked_norm = masked_frame.astype(np.float32) / 255.0
    
    face_tensor = np.transpose(face_norm, (2, 0, 1))
    masked_tensor = np.transpose(masked_norm, (2, 0, 1))
    
    # Concatenate
    visual_input = np.concatenate([face_tensor, masked_tensor], axis=0)
    visual_input = np.expand_dims(visual_input, axis=0)
    
    return visual_input, face_frame, masked_frame

print("=" * 80)
print("INPUT PREPROCESSING DEBUG")
print("=" * 80)
print()

frame_id = 50
package_dir = "d:/Projects/webcodecstest/fast_service/models/default_model"

visual, face_img, masked_img = load_frame_data_our_way(frame_id, package_dir)

print(f"Frame {frame_id}:")
print(f"  Visual input shape: {visual.shape}")
print(f"  Visual input range: [{visual.min():.6f}, {visual.max():.6f}]")
print(f"  Visual input mean: {visual.mean():.6f}")
print()

print("Channel-wise analysis:")
print(f"  Channels 0-2 (face RGB):")
print(f"    Min: {visual[0, 0:3].min():.6f}")
print(f"    Max: {visual[0, 0:3].max():.6f}")
print(f"    Mean: {visual[0, 0:3].mean():.6f}")
print()

print(f"  Channels 3-5 (masked face RGB):")
print(f"    Min: {visual[0, 3:6].min():.6f}")
print(f"    Max: {visual[0, 3:6].max():.6f}")
print(f"    Mean: {visual[0, 3:6].mean():.6f}")
print()

# Check if masked region is actually different
diff = np.abs(visual[0, 0:3] - visual[0, 3:6]).mean()
print(f"Difference between face and masked: {diff:.6f}")
if diff < 0.01:
    print("  ⚠️ WARNING: Face and masked are nearly identical!")
    print("  This suggests masked region might not be properly masked")
print()

# Check the actual pixel values
print("Sample pixel values (center of image):")
h, w = 160, 160
print(f"  Face RGB at ({h},{w}): {visual[0, 0:3, h, w]}")
print(f"  Masked RGB at ({h},{w}): {visual[0, 3:6, h, w]}")
print()

# Save debug images
debug_dir = "debug_inputs"
os.makedirs(debug_dir, exist_ok=True)

cv2.imwrite(f"{debug_dir}/face.png", cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{debug_dir}/masked.png", cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

print(f"✅ Debug images saved to {debug_dir}/")
print()

print("=" * 80)
print("REFERENCE: What test_single_frame_pth.py does:")
print("=" * 80)
print()
print("From reference code:")
print("  roi_img = cv2.imread(roi_path)")
print("  masked_img = cv2.imread(masked_path)")
print()
print("  img_real_ex = roi_img.transpose(2, 0, 1).astype(np.float32) / 255.0")
print("  img_masked = masked_img.transpose(2, 0, 1).astype(np.float32) / 255.0")
print("  img_concat_T = torch.from_numpy(np.concatenate([img_real_ex, img_masked], axis=0))[None]")
print()
print("This is EXACTLY what we're doing!")
print()
print("Note: Reference loads from:")
print("  roi_path = os.path.join(cache_dir, 'rois_320', f'{frame_index}.jpg')")
print("  masked_path = os.path.join(cache_dir, 'model_inputs', f'{frame_index}_masked.jpg')")
print()
print("We load from:")
print("  face_regions_320.mp4")
print("  masked_regions_320.mp4")
print()
print("These should be the SAME data, just in video format!")
