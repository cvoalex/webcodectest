#!/usr/bin/env python3
"""
Minimal Service Test - Tests just the core generate_frame function
"""

import time
import asyncio
from multi_model_engine import multi_model_engine

async def test_minimal_service():
    """Test the minimal service layer without network/encoding overhead"""
    
    print("⚡ Minimal Service Layer Test")
    print("=" * 50)
    
    # Ensure model is loaded
    model_name = "default_model"
    
    # Load model if needed
    load_result = await multi_model_engine.load_model(model_name)
    print(f"✅ Model status: {load_result['status']}")
    
    # Test just the core inference (same as service calls)
    print(f"\n🎯 Testing core service inference...")
    
    times = []
    for i in range(5):
        start_time = time.time()
        
        try:
            # This is exactly what the service calls
            frame, metadata = await multi_model_engine.generate_frame(model_name, i + 200)
            
            frame_time = (time.time() - start_time) * 1000
            times.append(frame_time)
            
            print(f"   Frame {i}: {frame_time:.1f}ms")
            
        except Exception as e:
            print(f"   Frame {i}: ❌ Error - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"\n📊 Minimal Service Results:")
        print(f"   Average time: {avg_time:.1f}ms")
        print(f"   FPS: {fps:.1f}")
        
        # Compare with known direct performance
        direct_time = 24  # From our direct test
        service_overhead = avg_time - direct_time
        
        print(f"\n🔍 Service vs Direct Comparison:")
        print(f"   Direct inference: {direct_time}ms")
        print(f"   Service layer: {avg_time:.1f}ms")
        print(f"   Service overhead: {service_overhead:.1f}ms")
        print(f"   Overhead factor: {avg_time/direct_time:.1f}x")

if __name__ == "__main__":
    asyncio.run(test_minimal_service())
