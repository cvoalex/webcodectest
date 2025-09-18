#!/usr/bin/env python3
"""
Test script for the inference-only endpoint.
Only returns prediction + bounds for client-side compositing.
Expected to be much smaller and faster than full frame generation.
"""

import time
import requests
import base64
import numpy as np
import cv2

def test_inference_only():
    base_url = "http://localhost:8000"
    model_name = "test_optimized_package_fixed_3"  # Use the successfully loaded model
    
    print("🚀 Inference-Only Service Test")
    print("=" * 50)
    print()
    print("⚡ Testing inference-only endpoint...")
    print()
    
    times = []
    processing_times = []
    sizes = []
    
    for frame_id in range(5):
        print(f"--- Frame {frame_id} ---")
        
        start_time = time.time()
        
        try:
            # Call the inference-only endpoint
            response = requests.get(f"{base_url}/generate/inference/{model_name}/{frame_id}")
            
            if response.status_code == 200:
                total_time = (time.time() - start_time) * 1000
                
                # Get timing from headers
                processing_time = int(response.headers.get("X-Processing-Time-Ms", 0))
                prediction_size = int(response.headers.get("X-Prediction-Size", 0))
                bounds_data = response.headers.get("X-Bounds-Data", "")
                bounds_shape = response.headers.get("X-Bounds-Shape", "0")
                prediction_shape = response.headers.get("X-Prediction-Shape", "0x0x0")
                auto_loaded = response.headers.get("X-Auto-Loaded", "False")
                
                # Parse bounds data
                if bounds_data:
                    bounds_bytes = base64.b64decode(bounds_data)
                    bounds_length = int(bounds_shape)
                    bounds = np.frombuffer(bounds_bytes, dtype=np.float32)
                    bounds_info = f"bounds[{bounds_length}]"
                else:
                    bounds_info = "no bounds"
                
                times.append(total_time)
                processing_times.append(processing_time)
                sizes.append(prediction_size)
                
                print(f"   ✅ Success")
                print(f"   📊 Total time: {total_time:.1f}ms")
                print(f"   🖥️  Server processing: {processing_time}ms")
                print(f"   🗂️  Prediction size: {prediction_size:,} bytes")
                print(f"   📐 Prediction shape: {prediction_shape}")
                print(f"   📏 {bounds_info}")
                print(f"   🤖 Auto-loaded: {auto_loaded}")
                
                # Save prediction image for verification
                with open(f"prediction_frame_{frame_id}.jpg", "wb") as f:
                    f.write(response.content)
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
        
        print()
    
    if times:
        avg_total = sum(times) / len(times)
        avg_processing = sum(processing_times) / len(processing_times)
        avg_size = sum(sizes) / len(sizes)
        network_overhead = avg_total - avg_processing
        overhead_percentage = (network_overhead / avg_total) * 100
        
        print("📊 Inference-Only Performance Summary:")
        print(f"   Average total time: {avg_total:.1f}ms")
        print(f"   Average processing time: {avg_processing:.1f}ms")
        print(f"   Network/overhead time: {network_overhead:.1f}ms")
        print(f"   Total FPS: {1000/avg_total:.1f}")
        print(f"   Processing FPS: {1000/avg_processing:.1f}")
        print(f"   Average size: {avg_size:,} bytes ({avg_size/1024:.1f} KB)")
        print(f"   Overhead percentage: {overhead_percentage:.1f}%")
        print()
        
        # Compare with previous benchmarks
        print("🔥 Performance Comparison:")
        print(f"   Full frame service: ~2837ms (0.4 FPS, ~2.7MB)")
        print(f"   Inference only: {avg_total:.1f}ms ({1000/avg_total:.1f} FPS, {avg_size/1024:.1f}KB)")
        
        if avg_total > 0:
            speedup = 2837 / avg_total
            size_reduction = (2700 * 1024) / avg_size  # 2.7MB vs current
            print(f"   Speedup: {speedup:.1f}x faster!")
            print(f"   Size reduction: {size_reduction:.1f}x smaller!")

if __name__ == "__main__":
    test_inference_only()
