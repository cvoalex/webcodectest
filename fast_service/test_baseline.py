#!/usr/bin/env python3
"""
Quick test script to verify the current FastAPI service works
Establishes baseline performance before optimizations
"""

import requests
import time
import json

SERVICE_URL = "http://localhost:8000"

def test_service_startup():
    """Test basic service functionality"""
    print("🚀 Testing SyncTalk2D FastAPI Service")
    print("=" * 50)
    
    # 1. Check health
    print("1. Checking service health...")
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Service is running")
        else:
            print(f"   ❌ Service responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to service. Is it running on port 8000?")
        print("   💡 Start with: python service.py")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # 2. Check models endpoint
    print("\n2. Checking models endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/models", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Models endpoint working")
            print(f"   📊 Loaded models: {result.get('total_loaded', 0)}")
            print(f"   📁 Local models: {result.get('total_local', 0)}")
        else:
            print(f"   ⚠️  Models endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ❌ Models endpoint error: {e}")
    
    # 3. Test single frame generation (will auto-load model)
    print("\n3. Testing single frame generation (auto-load)...")
    try:
        start_time = time.time()
        response = requests.post(f"{SERVICE_URL}/generate/frame", json={
            "model_name": "default_model",
            "frame_id": 1
        }, timeout=60)
        
        request_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"   ✅ Frame generation successful")
                print(f"   ⏱️  Total time: {request_time:.1f}ms")
                print(f"   🖥️  Server processing: {result.get('processing_time_ms', 0)}ms")
                print(f"   🤖 Auto-loaded: {result.get('auto_loaded', False)}")
                
                # Calculate baseline FPS
                fps = 1000 / request_time if request_time > 0 else 0
                print(f"   🎯 Baseline FPS: {fps:.1f}")
                
                return True
            else:
                print(f"   ❌ Frame generation failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Frame generation failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("   ⏰ Frame generation timed out (60s)")
    except Exception as e:
        print(f"   ❌ Frame generation error: {e}")
    
    return False

def test_batch_processing():
    """Test batch processing capability"""
    print("\n4. Testing batch processing...")
    
    try:
        start_time = time.time()
        response = requests.post(f"{SERVICE_URL}/generate/batch", json={
            "requests": [
                {"model_name": "default_model", "frame_id": 2},
                {"model_name": "default_model", "frame_id": 3},
                {"model_name": "default_model", "frame_id": 4}
            ]
        }, timeout=120)
        
        total_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                processed = result.get("processed_count", 0)
                cached = result.get("cached_count", 0)
                
                print(f"   ✅ Batch processing successful")
                print(f"   ⏱️  Total time: {total_time:.1f}ms")
                print(f"   🖥️  Server processing: {result.get('processing_time_ms', 0)}ms")
                print(f"   📊 Processed: {processed}, Cached: {cached}")
                
                if processed > 0:
                    per_frame_time = total_time / processed
                    fps = 1000 / per_frame_time
                    print(f"   🎯 Batch FPS: {fps:.1f}")
                    
                return True
            else:
                print(f"   ❌ Batch processing failed")
        else:
            print(f"   ❌ Batch processing failed with status {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Batch processing error: {e}")
    
    return False

def show_system_info():
    """Show system information"""
    print("\n📋 System Information")
    print("-" * 30)
    
    try:
        response = requests.get(f"{SERVICE_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"Memory usage: {stats.get('memory_mb', 'N/A')} MB")
            print(f"Requests served: {stats.get('total_requests', 'N/A')}")
            print(f"Cache hits: {stats.get('cache_hits', 'N/A')}")
        else:
            print("Stats endpoint not available")
    except:
        print("Could not retrieve system stats")

def main():
    """Main test function"""
    print("🧪 SyncTalk2D Service Baseline Test")
    print("=" * 50)
    print("This test establishes baseline performance before optimizations")
    print()
    
    # Test basic functionality
    if test_service_startup():
        test_batch_processing()
        show_system_info()
        
        print(f"\n✅ Baseline Testing Complete!")
        print(f"📝 Results logged for optimization comparison")
        print(f"🚀 Ready for performance optimization phase")
    else:
        print(f"\n❌ Baseline testing failed")
        print(f"💡 Fix service issues before proceeding with optimizations")

if __name__ == "__main__":
    main()
