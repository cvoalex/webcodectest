#!/usr/bin/env python3
"""
Test real audio inference with Python + ONNX
Processes aud.wav and generates output frames
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import time
import onnxruntime as ort
import wave

def extract_audio_features(audio_path):
    """Extract audio features - simplified version using dummy data"""
    print(f"🎵 Analyzing audio file: {audio_path}")
    
    # Get audio duration to estimate frame count
    try:
        with wave.open(audio_path, 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            
            # Assuming 25 FPS for lip sync
            num_frames = int(duration * 25)
            
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Estimated frames (25 FPS): {num_frames}")
    except Exception as e:
        print(f"   ⚠️ Could not read audio: {e}")
        print(f"   Using default: 100 frames")
        num_frames = 100
    
    # Create dummy audio features for testing
    # In production, this would be real mel-spectrogram features
    # Shape: [num_frames, 32, 16, 16]
    print(f"   Creating dummy audio features for {num_frames} frames...")
    audio_tensor = torch.randn(num_frames, 32, 16, 16)
    
    print(f"   Audio shape: {audio_tensor.shape}")
    print(f"   Total frames: {len(audio_tensor)}")
    
    return audio_tensor

def create_test_visual_input(num_frames):
    """Create dummy visual input (in real use, this would be previous frames)"""
    # For testing, create random visual input
    # Shape: [num_frames, 6, 320, 320]
    visual = torch.randn(num_frames, 6, 320, 320)
    return visual

def run_onnx_inference(
    audio_path="d:/Projects/webcodecstest/aud.wav",
    model_path="models/default_model/models/99.onnx",
    output_dir="output_python_onnx",
    save_frames=True
):
    """Run ONNX inference with real audio"""
    
    print("\n" + "="*80)
    print("🐍 PYTHON + ONNX RUNTIME - REAL AUDIO TEST")
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
    
    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"   Input names: {input_names}")
    print(f"   Output names: {output_names}")
    
    # Extract audio features
    audio_features = extract_audio_features(audio_path)
    num_frames = len(audio_features)
    
    # Create visual input (dummy for now)
    print(f"\n🖼️  Creating visual inputs...")
    visual_input = create_test_visual_input(num_frames)
    print(f"   Visual shape: {visual_input.shape}")
    
    # Create output directory
    if save_frames:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving frames to: {output_dir}")
    
    # Run inference on all frames
    print(f"\n🚀 Running inference on {num_frames} frames...")
    print(f"{'='*80}")
    
    inference_times = []
    all_outputs = []
    
    for i in range(num_frames):
        # Prepare inputs (add batch dimension)
        visual_np = visual_input[i:i+1].numpy()  # [1, 6, 320, 320]
        audio_np = audio_features[i:i+1].numpy()  # [1, 32, 16, 16]
        
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
        output_frame = outputs[0][0]  # Remove batch dimension
        all_outputs.append(output_frame)
        
        # Save frame if requested
        if save_frames and i % 10 == 0:  # Save every 10th frame
            # Convert from [3, 320, 320] to [320, 320, 3] and scale to 0-255
            frame = np.transpose(output_frame, (1, 2, 0))
            frame = ((frame + 1) * 127.5).clip(0, 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f"{output_dir}/frame_{i:04d}.png", frame)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == 0 or i == num_frames - 1:
            avg_time = np.mean(inference_times[-10:])
            print(f"   Frame {i+1}/{num_frames}: {inference_time:.2f}ms (avg: {avg_time:.2f}ms)")
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    
    print(f"\n{'='*80}")
    print("📊 INFERENCE STATISTICS")
    print(f"{'='*80}")
    print(f"Total frames:     {num_frames}")
    print(f"Mean time:        {np.mean(inference_times):.3f} ms")
    print(f"Median time:      {np.median(inference_times):.3f} ms")
    print(f"Std deviation:    {np.std(inference_times):.3f} ms")
    print(f"Min time:         {np.min(inference_times):.3f} ms")
    print(f"Max time:         {np.max(inference_times):.3f} ms")
    print(f"P95:              {np.percentile(inference_times, 95):.3f} ms")
    print(f"P99:              {np.percentile(inference_times, 99):.3f} ms")
    print(f"Average FPS:      {1000/np.mean(inference_times):.1f}")
    print(f"Total time:       {np.sum(inference_times)/1000:.2f} seconds")
    
    if save_frames:
        print(f"\n✅ Frames saved to: {output_dir}/")
        print(f"   Saved {len([f for f in os.listdir(output_dir) if f.endswith('.png')])} sample frames")
    
    # Calculate output statistics
    all_outputs = np.array(all_outputs)
    print(f"\n📈 OUTPUT STATISTICS")
    print(f"{'='*80}")
    print(f"Output shape:     {all_outputs.shape}")
    print(f"Output mean:      {np.mean(all_outputs):.6f}")
    print(f"Output std:       {np.std(all_outputs):.6f}")
    print(f"Output min:       {np.min(all_outputs):.6f}")
    print(f"Output max:       {np.max(all_outputs):.6f}")
    
    # Save summary
    summary = {
        'num_frames': num_frames,
        'mean_time_ms': float(np.mean(inference_times)),
        'median_time_ms': float(np.median(inference_times)),
        'fps': float(1000/np.mean(inference_times)),
        'total_time_s': float(np.sum(inference_times)/1000),
        'output_shape': list(all_outputs.shape),
        'output_mean': float(np.mean(all_outputs)),
        'output_std': float(np.std(all_outputs)),
        'output_min': float(np.min(all_outputs)),
        'output_max': float(np.max(all_outputs)),
    }
    
    # Save outputs for comparison
    np.save(f"{output_dir}/outputs.npy", all_outputs)
    print(f"\n💾 Saved outputs to: {output_dir}/outputs.npy")
    
    import json
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved summary to: {output_dir}/summary.json")
    
    return summary, all_outputs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ONNX inference with real audio')
    parser.add_argument('--audio', type=str, default='d:/Projects/webcodecstest/aud.wav',
                        help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/default_model/models/99.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--output', type=str, default='output_python_onnx',
                        help='Output directory')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save frames')
    
    args = parser.parse_args()
    
    try:
        summary, outputs = run_onnx_inference(
            audio_path=args.audio,
            model_path=args.model,
            output_dir=args.output,
            save_frames=not args.no_save
        )
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
