#!/usr/bin/env python3
"""
Export preprocessed test data in a format Go can easily read
Saves as raw binary files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

def export_test_data(
    package_dir="models/default_model",
    output_dir="test_data_for_go",
    num_frames=10,
    start_frame=100
):
    """Export test data as raw binary files for Go"""
    
    print(f"\n📦 Exporting test data for Go")
    print(f"   Source: {package_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Frames: {start_frame} to {start_frame + num_frames - 1}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load face regions video
    face_video_path = os.path.join(package_dir, "face_regions_320.mp4")
    masked_video_path = os.path.join(package_dir, "masked_regions_320.mp4")
    audio_features_path = os.path.join(package_dir, "aud_ave.npy")
    
    print(f"\n🎥 Loading face regions...")
    face_cap = cv2.VideoCapture(face_video_path)
    masked_cap = cv2.VideoCapture(masked_video_path)
    
    # Skip to start frame
    for _ in range(start_frame):
        face_cap.read()
        masked_cap.read()
    
    # Read frames
    face_frames = []
    masked_frames = []
    
    for i in range(num_frames):
        ret1, face_frame = face_cap.read()
        ret2, masked_frame = masked_cap.read()
        
        if not ret1 or not ret2:
            print(f"⚠️  Could only read {i} frames")
            break
            
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        
        face_frames.append(face_frame)
        masked_frames.append(masked_frame)
    
    face_cap.release()
    masked_cap.release()
    
    # Convert to tensors and normalize
    face_frames = np.array(face_frames).astype(np.float32) / 255.0
    face_frames = (face_frames - 0.5) * 2.0
    
    masked_frames = np.array(masked_frames).astype(np.float32) / 255.0
    masked_frames = (masked_frames - 0.5) * 2.0
    
    # Transpose to [N, C, H, W]
    face_frames = np.transpose(face_frames, (0, 3, 1, 2))
    masked_frames = np.transpose(masked_frames, (0, 3, 1, 2))
    
    # Concatenate
    visual_input = np.concatenate([face_frames, masked_frames], axis=1)
    
    print(f"✅ Loaded {len(face_frames)} visual frames")
    print(f"   Visual input shape: {visual_input.shape}")
    
    # Load and process audio
    print(f"\n🎵 Loading audio features...")
    audio_features_full = np.load(audio_features_path)
    audio_frames = audio_features_full[start_frame:start_frame + num_frames]
    
    # Reshape audio
    audio_input = []
    for audio_frame in audio_frames:
        audio_reshaped = audio_frame[:512].reshape(32, 16)
        audio_tiled = np.tile(audio_reshaped[:, :, np.newaxis], (1, 1, 16))
        audio_input.append(audio_tiled)
    
    audio_input = np.array(audio_input).astype(np.float32)
    print(f"✅ Processed {len(audio_input)} audio frames")
    print(f"   Audio input shape: {audio_input.shape}")
    
    # Save as binary files
    print(f"\n💾 Saving binary files...")
    
    # Save visual input [N, 6, 320, 320]
    visual_path = os.path.join(output_dir, "visual_input.bin")
    visual_input.tofile(visual_path)
    print(f"   ✅ Visual: {visual_path}")
    print(f"      Shape: {visual_input.shape}, dtype: float32")
    
    # Save audio input [N, 32, 16, 16]
    audio_path = os.path.join(output_dir, "audio_input.bin")
    audio_input.tofile(audio_path)
    print(f"   ✅ Audio: {audio_path}")
    print(f"      Shape: {audio_input.shape}, dtype: float32")
    
    # Save metadata
    metadata = {
        'num_frames': num_frames,
        'start_frame': start_frame,
        'visual_shape': list(visual_input.shape),
        'audio_shape': list(audio_input.shape),
        'dtype': 'float32'
    }
    
    import json
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata: {metadata_path}")
    
    # Also save original face frames for comparison
    face_original = ((face_frames + 1) * 127.5).clip(0, 255).astype(np.uint8)
    for i in range(num_frames):
        frame = np.transpose(face_original[i], (1, 2, 0))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"input_face_{i:04d}.png"), frame_bgr)
    print(f"   ✅ Saved {num_frames} original face images")
    
    print(f"\n✅ Export complete!")
    print(f"\n📋 Usage in Go:")
    print(f"   1. Read visual_input.bin as [{num_frames}, 6, 320, 320] float32 array")
    print(f"   2. Read audio_input.bin as [{num_frames}, 32, 16, 16] float32 array")
    print(f"   3. Process each frame through ONNX model")
    print(f"   4. Compare outputs with input_face_XXXX.png images")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export test data for Go')
    parser.add_argument('--package', type=str, default='models/default_model',
                        help='Path to preprocessed package directory')
    parser.add_argument('--output', type=str, default='test_data_for_go',
                        help='Output directory')
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames to export')
    parser.add_argument('--start', type=int, default=100,
                        help='Start frame index')
    
    args = parser.parse_args()
    
    try:
        export_test_data(
            package_dir=args.package,
            output_dir=args.output,
            num_frames=args.frames,
            start_frame=args.start
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
