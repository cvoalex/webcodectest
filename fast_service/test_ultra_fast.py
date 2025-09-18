#!/usr/bin/env python3
"""
Ultra-Fast Service Test - Tests the binary endpoint with minimal overhead
"""

import requests
import time

SERVICE_URL = "http://localhost:8000"

def test_ultra_fast_service():
    """Test the ultra-fast binary endpoint"""
    
    print("🚀 Ultra-Fast Service Test")
    print("=" * 50)
    
    print(f"\n⚡ Testing ultra-fast binary endpoint...")
    
    times = []
    processing_times = []
    
    for i in range(5):
        print(f"\n--- Frame {i} ---")
        
        start_time = time.time()
        
        try:
            # Use the fast binary endpoint
            response = requests.get(f"{SERVICE_URL}/generate/frame/fast/default_model/{i + 300}", timeout=30)
            
            total_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Get processing time from headers
                processing_time = int(response.headers.get('X-Processing-Time-Ms', 0))
                
                print(f"   ✅ Success")
                print(f"   📊 Total time: {total_time:.1f}ms")
                print(f"   🖥️  Server processing: {processing_time}ms")
                print(f"   🗂️  Response size: {len(response.content)} bytes")
                print(f"   🤖 Auto-loaded: {response.headers.get('X-Auto-Loaded', 'N/A')}")
                
                times.append(total_time)
                processing_times.append(processing_time)
                
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Calculate and display results
    if times:
        avg_total = sum(times) / len(times)
        avg_processing = sum(processing_times) / len(processing_times)
        
        print(f"\n📊 Ultra-Fast Performance Summary:")
        print(f"   Average total time: {avg_total:.1f}ms")
        print(f"   Average processing time: {avg_processing:.1f}ms")
        print(f"   Network/overhead time: {avg_total - avg_processing:.1f}ms")
        print(f"   Total FPS: {1000 / avg_total if avg_total > 0 else 0:.1f}")
        print(f"   Processing FPS: {1000 / avg_processing if avg_processing > 0 else 0:.1f}")
        
        overhead_percent = ((avg_total - avg_processing) / avg_total * 100) if avg_total > 0 else 0
        print(f"   Overhead percentage: {overhead_percent:.1f}%")
        
        # Compare with previous results
        print(f"\n🔥 Performance Comparison:")
        print(f"   JSON service: ~2715ms (0.4 FPS)")
        print(f"   Fast binary: {avg_total:.1f}ms ({1000/avg_total:.1f} FPS)")
        
        if avg_total > 0:
            speedup = 2715 / avg_total
            print(f"   Speedup: {speedup:.1f}x faster!")

if __name__ == "__main__":
    test_ultra_fast_service()
