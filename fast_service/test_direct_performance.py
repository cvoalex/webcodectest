#!/usr/bin/env python3
"""
Direct Performance Test - Bypasses network overhead to test pure inference speed
"""

import time
import torch
import numpy as np
from multi_model_engine import multi_model_engine

async def test_direct_inference():
    """Test inference engine directly without network overhead"""
    
    print("🚀 Direct Inference Performance Test")
    print("=" * 50)
    
    # Ensure model is loaded
    model_name = "default_model"
    print(f"📥 Loading model '{model_name}'...")
    
    load_result = await multi_model_engine.load_model(model_name)
    if load_result["status"] not in ["success", "already_loaded", "loaded"]:
        print(f"❌ Failed to load model: {load_result}")
        return
    
    print(f"✅ Model loaded: {load_result['status']}")
    print(f"📊 Model info: {load_result.get('total_frames', 'N/A')} frames, device: {load_result.get('device', 'N/A')}")
    print(f"⚡ Initialization time: {load_result.get('initialization_time_ms', 'N/A')}ms")
    
    # Test single frame generation
    print(f"\n🎯 Testing single frame generation...")
    
    frame_times = []
    for i in range(5):
        start_time = time.time()
        
        try:
            frame, metadata = await multi_model_engine.generate_frame(model_name, i)
            frame_time = (time.time() - start_time) * 1000
            frame_times.append(frame_time)
            
            print(f"   Frame {i}: {frame_time:.1f}ms")
            
        except Exception as e:
            print(f"   Frame {i}: ❌ Error - {e}")
    
    if frame_times:
        avg_time = sum(frame_times) / len(frame_times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 Direct Inference Results:")
        print(f"   Average time: {avg_time:.1f}ms")
        print(f"   FPS: {fps:.1f}")
        print(f"   Min time: {min(frame_times):.1f}ms")
        print(f"   Max time: {max(frame_times):.1f}ms")
    
    # Test with cached frames (should be faster)
    print(f"\n🚀 Testing cached frame performance...")
    cached_times = []
    
    for i in range(5):
        start_time = time.time()
        
        try:
            frame, metadata = await multi_model_engine.generate_frame(model_name, i % 3)  # Reuse frames 0,1,2
            frame_time = (time.time() - start_time) * 1000
            cached_times.append(frame_time)
            
            print(f"   Cached Frame {i}: {frame_time:.1f}ms")
            
        except Exception as e:
            print(f"   Cached Frame {i}: ❌ Error - {e}")
    
    if cached_times:
        avg_cached_time = sum(cached_times) / len(cached_times)
        cached_fps = 1000 / avg_cached_time if avg_cached_time > 0 else 0
        
        print(f"\n📊 Cached Performance Results:")
        print(f"   Average time: {avg_cached_time:.1f}ms")
        print(f"   FPS: {cached_fps:.1f}")
        
        if frame_times:
            speedup = avg_time / avg_cached_time if avg_cached_time > 0 else 1
            print(f"   Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_direct_inference())
