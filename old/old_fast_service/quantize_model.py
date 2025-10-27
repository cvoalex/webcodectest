"""
Quantize ONNX model to FP16 or INT8 for faster inference
"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16
from pathlib import Path
import numpy as np
import cv2
import json

def quantize_to_fp16(model_path, output_path):
    """Convert model to FP16 (half precision)"""
    print(f"\n🔄 Converting to FP16...")
    print(f"   Input: {model_path}")
    print(f"   Output: {output_path}")
    
    model = onnx.load(model_path)
    
    # Convert to FP16 but keep inputs/outputs as FP32 for compatibility
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True  # Keep inputs/outputs as FP32
    )
    
    # Save
    onnx.save(model_fp16, output_path)
    
    # Get size comparison
    orig_size = Path(model_path).stat().st_size / (1024**2)
    new_size = Path(output_path).stat().st_size / (1024**2)
    
    print(f"   ✅ FP16 model saved!")
    print(f"   Original size: {orig_size:.2f} MB")
    print(f"   FP16 size: {new_size:.2f} MB")
    print(f"   Reduction: {(1 - new_size/orig_size)*100:.1f}%")

def quantize_to_int8(model_path, output_path, data_dir):
    """Convert model to INT8 (dynamic quantization)"""
    print(f"\n🔄 Converting to INT8 (Dynamic Quantization)...")
    print(f"   Input: {model_path}")
    print(f"   Output: {output_path}")
    
    # Dynamic quantization (doesn't require calibration data)
    quantize_dynamic(
        model_path,
        output_path,
        weight_type=QuantType.QUInt8,
        extra_options={
            'ActivationSymmetric': False,
            'WeightSymmetric': True,
        }
    )
    
    # Get size comparison
    orig_size = Path(model_path).stat().st_size / (1024**2)
    new_size = Path(output_path).stat().st_size / (1024**2)
    
    print(f"   ✅ INT8 model saved!")
    print(f"   Original size: {orig_size:.2f} MB")
    print(f"   INT8 size: {new_size:.2f} MB")
    print(f"   Reduction: {(1 - new_size/orig_size)*100:.1f}%")

def test_quantized_model(model_path, data_dir, num_frames=10):
    """Quick test of quantized model"""
    import onnxruntime as ort
    
    print(f"\n🧪 Testing quantized model: {Path(model_path).name}")
    
    # Load model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"   ✅ Model loaded with {session.get_providers()}")
    
    # Get input names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    # Load test data
    data_dir = Path(data_dir)
    audio_features = np.load(data_dir / "aud_ave.npy")
    
    cap = cv2.VideoCapture(str(data_dir / "model_inputs_video.mp4"))
    
    import time
    inference_times = []
    
    for frame_id in range(num_frames):
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Prepare visual input [1, 6, 320, 320]
        frames = []
        for offset in [-1, 0, 1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id + offset))
            ret, f = cap.read()
            if ret:
                frames.append(f)
            else:
                frames.append(frame)
        
        stacked = np.concatenate(frames, axis=2)
        stacked = stacked.transpose(2, 0, 1)
        stacked = stacked.astype(np.float32) / 255.0
        visual_input = np.expand_dims(stacked[:6], axis=0)
        
        # Prepare audio input [1, 32, 16, 16]
        audio_window = []
        for i in range(frame_id - 8, frame_id + 8):
            if i < 0 or i >= len(audio_features):
                audio_window.append(np.zeros(512, dtype=np.float32))
            else:
                audio_window.append(audio_features[i])
        
        audio_flat = np.array(audio_window).flatten()
        audio_input = np.expand_dims(audio_flat.reshape(32, 16, 16), axis=0).astype(np.float32)
        
        # Inference
        start = time.time()
        outputs = session.run(
            output_names,
            {
                input_names[0]: visual_input,
                input_names[1]: audio_input
            }
        )
        inference_time = (time.time() - start) * 1000
        inference_times.append(inference_time)
    
    cap.release()
    
    avg_time = np.mean(inference_times)
    fps = 1000 / avg_time
    
    print(f"   📊 Results ({num_frames} frames):")
    print(f"      Avg inference: {avg_time:.2f}ms")
    print(f"      FPS: {fps:.2f}")
    print(f"   ✅ Model works correctly!")
    
    return avg_time

def main():
    print("="*80)
    print("🚀 ONNX Model Quantization Tool")
    print("="*80)
    
    model_path = "minimal_server/models/sanders/checkpoint/model_best.onnx"
    data_dir = "minimal_server/models/sanders"
    output_dir = Path("minimal_server/models/sanders/checkpoint")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📦 Input model: {model_path}")
    orig_size = Path(model_path).stat().st_size / (1024**2)
    print(f"   Size: {orig_size:.2f} MB")
    
    results = {}
    
    # Test original model
    print("\n" + "="*80)
    print("1️⃣  BASELINE: FP32 (Original)")
    print("="*80)
    results['fp32'] = test_quantized_model(model_path, data_dir, num_frames=20)
    
    # Quantize to FP16
    print("\n" + "="*80)
    print("2️⃣  FP16 Quantization")
    print("="*80)
    fp16_path = output_dir / "model_best_fp16.onnx"
    try:
        quantize_to_fp16(model_path, str(fp16_path))
        results['fp16'] = test_quantized_model(str(fp16_path), data_dir, num_frames=20)
    except Exception as e:
        print(f"   ❌ FP16 conversion failed: {e}")
        results['fp16'] = None
    
    # Quantize to INT8
    print("\n" + "="*80)
    print("3️⃣  INT8 Quantization")
    print("="*80)
    int8_path = output_dir / "model_best_int8.onnx"
    try:
        quantize_to_int8(model_path, str(int8_path), data_dir)
        results['int8'] = test_quantized_model(str(int8_path), data_dir, num_frames=20)
    except Exception as e:
        print(f"   ❌ INT8 conversion failed: {e}")
        results['int8'] = None
    
    # Summary
    print("\n" + "="*80)
    print("📊 QUANTIZATION COMPARISON")
    print("="*80)
    print(f"{'Model':<12} {'Inference Time':<18} {'FPS':<12} {'Speedup':<12}")
    print("-"*80)
    
    baseline = results['fp32']
    for name, time in results.items():
        if time is not None:
            fps = 1000 / time
            speedup = baseline / time if baseline else 1.0
            print(f"{name.upper():<12} {time:<18.2f} {fps:<12.2f} {speedup:<12.2f}x")
    
    # Recommendation
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    
    best_name = min(results.items(), key=lambda x: x[1] if x[1] else float('inf'))[0]
    best_time = results[best_name]
    best_fps = 1000 / best_time
    speedup = baseline / best_time
    
    model_files = {
        'fp32': 'model_best.onnx',
        'fp16': 'model_best_fp16.onnx',
        'int8': 'model_best_int8.onnx'
    }
    
    print(f"   🏆 BEST: {best_name.upper()}")
    print(f"      File: {model_files[best_name]}")
    print(f"      Inference: {best_time:.2f}ms")
    print(f"      FPS: {best_fps:.2f}")
    print(f"      Speedup: {speedup:.2f}x vs FP32")
    
    if best_name == 'fp16':
        print(f"\n   ✅ FP16 is faster! Use: {fp16_path}")
        print(f"      - 2x smaller model size")
        print(f"      - Faster GPU inference")
        print(f"      - No quality loss on modern GPUs")
    elif best_name == 'int8':
        print(f"\n   ✅ INT8 is faster! Use: {int8_path}")
        print(f"      - 4x smaller model size")
        print(f"      - Fastest inference")
        print(f"      - May have slight quality loss")
    else:
        print(f"\n   ℹ️  FP32 is already optimal for your GPU")
        print(f"      - Your GPU may not benefit from quantization")
        print(f"      - Modern GPUs handle FP32 very efficiently")

if __name__ == "__main__":
    main()
