#!/usr/bin/env python3
"""
Service Performance Test - Bypasses cache to test actual service performance
"""

import requests
import time
import json

SERVICE_URL = "http://localhost:8000"

def test_service_performance():
    """Test service performance without cache interference"""
    
    print("🔧 Service Performance Debug Test")
    print("=" * 50)
    
    # Clear cache first
    print("🧹 Clearing cache...")
    try:
        response = requests.delete(f"{SERVICE_URL}/cache")
        print(f"   Cache clear result: {response.status_code}")
    except Exception as e:
        print(f"   Cache clear failed: {e}")
    
    # Test with different frame IDs to avoid cache hits
    print(f"\n🎯 Testing fresh frame generation (no cache)...")
    
    frame_times = []
    processing_times = []
    
    for i in range(5):
        print(f"\n--- Frame {i} ---")
        
        start_time = time.time()
        
        try:
            response = requests.post(f"{SERVICE_URL}/generate/frame", json={
                "model_name": "default_model",
                "frame_id": i + 100  # Use high frame IDs to avoid cache
            }, timeout=30)
            
            total_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"   ✅ Success")
                print(f"   📊 Total time: {total_time:.1f}ms")
                print(f"   🖥️  Server processing: {result.get('processing_time_ms', 'N/A')}ms")
                print(f"   💾 From cache: {result.get('from_cache', 'N/A')}")
                print(f"   🤖 Auto-loaded: {result.get('auto_loaded', 'N/A')}")
                
                frame_times.append(total_time)
                processing_times.append(result.get('processing_time_ms', 0))
                
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Calculate and display results
    if frame_times:
        avg_total = sum(frame_times) / len(frame_times)
        avg_processing = sum(processing_times) / len(processing_times)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average total time: {avg_total:.1f}ms")
        print(f"   Average processing time: {avg_processing:.1f}ms")
        print(f"   Network/overhead time: {avg_total - avg_processing:.1f}ms")
        print(f"   Total FPS: {1000 / avg_total if avg_total > 0 else 0:.1f}")
        print(f"   Processing FPS: {1000 / avg_processing if avg_processing > 0 else 0:.1f}")
        
        overhead_percent = ((avg_total - avg_processing) / avg_total * 100) if avg_total > 0 else 0
        print(f"   Overhead percentage: {overhead_percent:.1f}%")

if __name__ == "__main__":
    test_service_performance()
