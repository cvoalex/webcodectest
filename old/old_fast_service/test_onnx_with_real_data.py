#!/usr/bin/env python3
"""
Test ONNX inference with REAL preprocessed data
Uses actual face/masked frames and audio features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import time
import onnxruntime as ort

def load_test_data(package_dir="models/default_model", num_frames=10, start_frame=100):
    """Load real test data from preprocessed package"""
    
    print(f"\n📦 Loading test data from: {package_dir}")
    print(f"   Frames: {start_frame} to {start_frame + num_frames - 1}")
    
    # Load face regions video
    face_video_path = os.path.join(package_dir, "face_regions_320.mp4")
    masked_video_path = os.path.join(package_dir, "masked_regions_320.mp4")
    audio_features_path = os.path.join(package_dir, "aud_ave.npy")
    
    print(f"\n🎥 Loading face regions: {face_video_path}")
    face_cap = cv2.VideoCapture(face_video_path)
    
    print(f"🎭 Loading masked regions: {masked_video_path}")
    masked_cap = cv2.VideoCapture(masked_video_path)
    
    print(f"🎵 Loading audio features: {audio_features_path}")
    audio_features_full = np.load(audio_features_path)
    print(f"   Audio features shape: {audio_features_full.shape}")
    
    # Read frames
    face_frames = []
    masked_frames = []
    
    # Skip to start frame
    for _ in range(start_frame):
        face_cap.read()
        masked_cap.read()
    
    # Read the frames we want
    for i in range(num_frames):
        ret1, face_frame = face_cap.read()
        ret2, masked_frame = masked_cap.read()
        
        if not ret1 or not ret2:
            print(f"⚠️  Could only read {i} frames")
            break
            
        # Convert BGR to RGB
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        
        face_frames.append(face_frame)
        masked_frames.append(masked_frame)
    
    face_cap.release()
    masked_cap.release()
    
    # Convert to tensors and normalize to [-1, 1]
    face_frames = np.array(face_frames).astype(np.float32) / 255.0  # [0, 1]
    face_frames = (face_frames - 0.5) * 2.0  # [-1, 1]
    
    masked_frames = np.array(masked_frames).astype(np.float32) / 255.0
    masked_frames = (masked_frames - 0.5) * 2.0
    
    # Transpose to [N, C, H, W]
    face_frames = np.transpose(face_frames, (0, 3, 1, 2))  # [N, 3, 320, 320]
    masked_frames = np.transpose(masked_frames, (0, 3, 1, 2))
    
    # Concatenate face and masked (visual input is 6 channels)
    visual_input = np.concatenate([face_frames, masked_frames], axis=1)  # [N, 6, 320, 320]
    
    print(f"\n✅ Loaded {len(face_frames)} frames")
    print(f"   Face frames shape: {face_frames.shape}")
    print(f"   Masked frames shape: {masked_frames.shape}")
    print(f"   Visual input shape: {visual_input.shape}")
    
    # Extract corresponding audio features
    # Audio features are [3318, 512], we need to reshape to [N, 32, 16, 16]
    audio_frames = audio_features_full[start_frame:start_frame + num_frames]
    print(f"   Audio frames shape (raw): {audio_frames.shape}")
    
    # Reshape audio: [N, 512] -> [N, 32, 16] -> [N, 32, 16, 16]
    # Need to pad or reshape appropriately
    audio_input = []
    for audio_frame in audio_frames:
        # Reshape [512] to [32, 16]
        audio_reshaped = audio_frame[:512].reshape(32, 16)
        # Tile to [32, 16, 16]
        audio_tiled = np.tile(audio_reshaped[:, :, np.newaxis], (1, 1, 16))
        audio_input.append(audio_tiled)
    
    audio_input = np.array(audio_input).astype(np.float32)
    print(f"   Audio input shape: {audio_input.shape}")
    
    return visual_input, audio_input, face_frames

def test_onnx_inference(
    model_path="models/default_model/models/99.onnx",
    package_dir="models/default_model",
    num_frames=10,
    start_frame=100,
    output_dir="output_onnx_real_data"
):
    """Test ONNX inference with real preprocessed data"""
    
    print("\n" + "="*80)
    print("🔬 ONNX INFERENCE TEST WITH REAL DATA")
    print("="*80)
    
    # Check CUDA
    providers = ort.get_available_providers()
    print(f"\nAvailable providers: {providers}")
    
    use_cuda = 'CUDAExecutionProvider' in providers
    if use_cuda:
        print("✅ Using CUDA execution provider")
        session_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        print("⚠️ CUDA not available, using CPU")
        session_providers = ['CPUExecutionProvider']
    
    # Load ONNX model
    print(f"\n📦 Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=session_providers)
    
    # Get input/output info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"   Input names: {input_names}")
    print(f"   Output names: {output_names}")
    
    # Load real test data
    visual_input, audio_input, original_faces = load_test_data(
        package_dir=package_dir,
        num_frames=num_frames,
        start_frame=start_frame
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference on each frame
    print(f"\n🚀 Running ONNX inference on {num_frames} frames...")
    print("="*80)
    
    inference_times = []
    output_frames = []
    
    for i in range(num_frames):
        # Prepare single frame inputs
        visual_np = visual_input[i:i+1]  # [1, 6, 320, 320]
        audio_np = audio_input[i:i+1]   # [1, 32, 16, 16]
        
        # Run inference
        start_time = time.perf_counter()
        
        outputs = session.run(
            output_names,
            {
                input_names[0]: visual_np,
                input_names[1]: audio_np
            }
        )
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
        # Get output [1, 3, 320, 320]
        output_frame = outputs[0][0]  # Remove batch dimension [3, 320, 320]
        output_frames.append(output_frame)
        
        print(f"   Frame {i+1}/{num_frames}: {inference_time:.2f}ms")
        
        # Save output frame
        # Convert from [3, 320, 320] to [320, 320, 3]
        frame = np.transpose(output_frame, (1, 2, 0))
        
        # Convert from [-1, 1] to [0, 255]
        frame = ((frame + 1) * 127.5).clip(0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(f"{output_dir}/output_frame_{i:04d}.png", frame_bgr)
        
        # Also save original face for comparison
        orig_face = np.transpose(original_faces[i], (1, 2, 0))
        orig_face = ((orig_face + 1) * 127.5).clip(0, 255).astype(np.uint8)
        orig_face_bgr = cv2.cvtColor(orig_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_dir}/input_face_{i:04d}.png", orig_face_bgr)
        
        # Create side-by-side comparison
        comparison = np.hstack([orig_face_bgr, frame_bgr])
        cv2.imwrite(f"{output_dir}/comparison_{i:04d}.png", comparison)
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    
    print(f"\n{'='*80}")
    print("📊 INFERENCE STATISTICS")
    print(f"{'='*80}")
    print(f"Total frames:     {num_frames}")
    print(f"Mean time:        {np.mean(inference_times):.3f} ms")
    print(f"Median time:      {np.median(inference_times):.3f} ms")
    print(f"Min time:         {np.min(inference_times):.3f} ms")
    print(f"Max time:         {np.max(inference_times):.3f} ms")
    print(f"Average FPS:      {1000/np.mean(inference_times):.1f}")
    
    # Analyze output quality
    output_frames = np.array(output_frames)
    print(f"\n📈 OUTPUT QUALITY")
    print(f"{'='*80}")
    print(f"Output shape:     {output_frames.shape}")
    print(f"Output mean:      {np.mean(output_frames):.6f}")
    print(f"Output std:       {np.std(output_frames):.6f}")
    print(f"Output min:       {np.min(output_frames):.6f}")
    print(f"Output max:       {np.max(output_frames):.6f}")
    
    # Check if output is mostly in valid range [-1, 1]
    in_range = np.sum((output_frames >= -1.1) & (output_frames <= 1.1)) / output_frames.size * 100
    print(f"Values in [-1.1, 1.1]: {in_range:.2f}%")
    
    print(f"\n✅ Output saved to: {output_dir}/")
    print(f"   - output_frame_XXXX.png: Model outputs")
    print(f"   - input_face_XXXX.png: Original face inputs")
    print(f"   - comparison_XXXX.png: Side-by-side comparison")
    
    return output_frames

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ONNX with real preprocessed data')
    parser.add_argument('--model', type=str, default='models/default_model/models/99.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--package', type=str, default='models/default_model',
                        help='Path to preprocessed package directory')
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames to process')
    parser.add_argument('--start', type=int, default=100,
                        help='Start frame index')
    parser.add_argument('--output', type=str, default='output_onnx_real_data',
                        help='Output directory')
    
    args = parser.parse_args()
    
    try:
        outputs = test_onnx_inference(
            model_path=args.model,
            package_dir=args.package,
            num_frames=args.frames,
            start_frame=args.start,
            output_dir=args.output
        )
        
        print("\n✅ Test completed successfully!")
        print("\n👀 Check the output images to verify quality!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
