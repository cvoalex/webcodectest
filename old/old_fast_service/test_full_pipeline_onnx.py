#!/usr/bin/env python3
"""
Test ONNX with FULL pipeline including compositing
This should produce photorealistic lip-synced output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
import onnxruntime as ort

def load_frame_data(frame_id, package_dir="models/default_model"):
    """Load all data needed for one frame"""
    
    # Load face regions video
    face_cap = cv2.VideoCapture(os.path.join(package_dir, "face_regions_320.mp4"))
    masked_cap = cv2.VideoCapture(os.path.join(package_dir, "masked_regions_320.mp4"))
    
    # Skip to frame
    for _ in range(frame_id):
        face_cap.read()
        masked_cap.read()
    
    # Read frame
    ret1, face_frame = face_cap.read()
    ret2, masked_frame = masked_cap.read()
    
    face_cap.release()
    masked_cap.release()
    
    if not ret1 or not ret2:
        raise ValueError(f"Could not read frame {frame_id}")
    
    # Convert and normalize
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    
    # CORRECT: Model expects [0, 1] range, NOT [-1, 1]
    # Reference: img_real_ex = roi_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    face_norm = face_frame.astype(np.float32) / 255.0
    masked_norm = masked_frame.astype(np.float32) / 255.0
    
    # Transpose to [C, H, W]
    face_tensor = np.transpose(face_norm, (2, 0, 1))
    masked_tensor = np.transpose(masked_norm, (2, 0, 1))
    
    # Concatenate to 6 channels
    visual_input = np.concatenate([face_tensor, masked_tensor], axis=0)
    visual_input = np.expand_dims(visual_input, axis=0)  # [1, 6, 320, 320]
    
    # Load audio features
    audio_features = np.load(os.path.join(package_dir, "aud_ave.npy"))
    
    # CRITICAL FIX: Use 16-frame window (8 before + current + 7 after)
    # Just like get_audio_window in the reference code
    left = frame_id - 8
    right = frame_id + 8
    
    # Handle boundaries with padding
    if left < 0:
        pad_left = -left
        left = 0
    else:
        pad_left = 0
    
    if right > len(audio_features):
        pad_right = right - len(audio_features)
        right = len(audio_features)
    else:
        pad_right = 0
    
    # Get the window of 16 frames
    audio_window = audio_features[left:right]  # [num_frames, 512]
    
    # Pad if needed
    if pad_left > 0:
        audio_window = np.concatenate([
            np.tile(audio_features[0:1], (pad_left, 1)),
            audio_window
        ], axis=0)
    if pad_right > 0:
        audio_window = np.concatenate([
            audio_window,
            np.tile(audio_features[-1:], (pad_right, 1))
        ], axis=0)
    
    # Flatten and reshape: [16, 512] -> [8192] -> [32, 16, 16]
    audio_flat = audio_window.flatten()  # [8192]
    audio_reshaped = audio_flat.reshape(32, 16, 16)  # [32, 16, 16]
    audio_input = np.expand_dims(audio_reshaped, axis=0).astype(np.float32)  # [1, 32, 16, 16]
    
    # Load bounds
    bounds = np.load(os.path.join(package_dir, "face_bounds", f"{frame_id}.npy"))
    
    # Load original full video frame
    video_cap = cv2.VideoCapture(os.path.join(package_dir, "video.mp4"))
    for _ in range(frame_id):
        video_cap.read()
    ret, original_frame = video_cap.read()
    video_cap.release()
    
    if not ret:
        raise ValueError(f"Could not read original video frame {frame_id}")
    
    return visual_input, audio_input, bounds, original_frame, face_frame

def composite_prediction(prediction, bounds, original_frame, roi_image):
    """
    Composite the model prediction onto the original frame
    
    Args:
        prediction: [3, 320, 320] model output in range [0, 1]
        bounds: [xmin, ymin, xmax, ymax, width] face bounds
        original_frame: Full resolution original video frame
        roi_image: Original 320x320 ROI image (for border preservation)
    
    Returns:
        Composited frame
    """
    
    # Convert prediction from [C, H, W] to [H, W, C]
    prediction_hwc = np.transpose(prediction, (1, 2, 0))
    
    # Convert from [0, 1] to [0, 255]
    prediction_255 = (prediction_hwc * 255).clip(0, 255).astype(np.uint8)
    
    # CRITICAL: Resize original ROI to 328x328 (preserves 4px border from original crop)
    face_crop_328 = cv2.resize(roi_image, (328, 328), interpolation=cv2.INTER_CUBIC)
    
    # Place prediction in center (320x320 -> center of 328x328, replacing interior)
    face_crop_328[4:324, 4:324] = prediction_255
    
    # Parse bounds
    xmin, ymin, xmax, ymax, width = bounds
    height = ymax - ymin
    
    # Resize to original face size
    final_crop = cv2.resize(face_crop_328, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
    
    # Composite onto original frame
    result_frame = original_frame.copy()
    result_frame[int(ymin):int(ymax), int(xmin):int(xmax)] = final_crop
    
    return result_frame

def test_full_pipeline(
    frame_id=100,
    model_path="models/default_model/models/99.onnx",
    package_dir="models/default_model",
    output_dir="output_full_pipeline"
):
    """Test ONNX with full pipeline including compositing"""
    
    print("\n" + "="*80)
    print("🎬 FULL PIPELINE TEST: ONNX + COMPOSITING")
    print("="*80)
    
    # Setup ONNX
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
    
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"   Input names: {input_names}")
    print(f"   Output names: {output_names}")
    
    # Load frame data
    print(f"\n📊 Loading frame {frame_id} data...")
    visual_input, audio_input, bounds, original_frame, face_320 = load_frame_data(
        frame_id, package_dir
    )
    
    print(f"   Visual input: {visual_input.shape}")
    print(f"   Audio input: {audio_input.shape}")
    print(f"   Bounds: {bounds}")
    print(f"   Original frame: {original_frame.shape}")
    
    # Run inference
    print(f"\n🚀 Running ONNX inference...")
    start_time = time.perf_counter()
    
    outputs = session.run(
        output_names,
        {
            input_names[0]: visual_input,
            input_names[1]: audio_input
        }
    )
    
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000
    
    prediction = outputs[0][0]  # [3, 320, 320]
    
    print(f"   Inference time: {inference_time:.2f}ms")
    print(f"   Prediction shape: {prediction.shape}")
    print(f"   Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    
    # Composite onto original frame
    print(f"\n🎨 Compositing prediction onto original frame...")
    final_frame = composite_prediction(prediction, bounds, original_frame, face_320)
    
    print(f"   Final frame shape: {final_frame.shape}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original frame
    cv2.imwrite(f"{output_dir}/1_original_frame.png", original_frame)
    
    # Save face region only (320x320)
    prediction_img = np.transpose(prediction, (1, 2, 0))
    prediction_img = (prediction_img * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/2_generated_face_320.png", cv2.cvtColor(prediction_img, cv2.COLOR_RGB2BGR))
    
    # Save input face (for comparison)
    cv2.imwrite(f"{output_dir}/3_input_face_320.png", cv2.cvtColor(face_320, cv2.COLOR_RGB2BGR))
    
    # Save final composited result
    cv2.imwrite(f"{output_dir}/4_final_composited.png", final_frame)
    
    # Create side-by-side comparison
    comparison = np.hstack([original_frame, final_frame])
    cv2.imwrite(f"{output_dir}/5_comparison_before_after.png", comparison)
    
    print(f"\n✅ Results saved to {output_dir}/")
    print(f"   1_original_frame.png - Original full video frame")
    print(f"   2_generated_face_320.png - Generated face region (320x320)")
    print(f"   3_input_face_320.png - Input face region (320x320)")
    print(f"   4_final_composited.png - Final result with compositing")
    print(f"   5_comparison_before_after.png - Before/After comparison")
    
    print(f"\n🎯 This should look PHOTOREALISTIC with lip-synced mouth!")
    
    return final_frame

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test full ONNX pipeline with compositing')
    parser.add_argument('--frame', type=int, default=100, help='Frame ID to test')
    parser.add_argument('--model', type=str, default='models/default_model/models/99.onnx', help='ONNX model path')
    parser.add_argument('--package', type=str, default='models/default_model', help='Package directory')
    parser.add_argument('--output', type=str, default='output_full_pipeline', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        result = test_full_pipeline(
            frame_id=args.frame,
            model_path=args.model,
            package_dir=args.package,
            output_dir=args.output
        )
        
        print("\n✅ Test completed successfully!")
        print("👀 Check the output images - the final result should be photorealistic!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
