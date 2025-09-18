#!/usr/bin/env python3
"""
Test script for the batch endpoint with multi-audio support
Demonstrates generating multiple frames with different audio for each frame
"""

import requests
import base64
import json
import time
import os

# Service URL
SERVICE_URL = "http://localhost:8000"

def encode_audio_file(audio_path):
    """Encode audio file to base64"""
    try:
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        print(f"Warning: Audio file {audio_path} not found, using None")
        return None

def test_batch_endpoint():
    """Test the batch endpoint with multi-audio support"""
    
    print("🎵 Testing Batch Processing with Multi-Audio")
    print("=" * 50)
    
    # For demo purposes, we'll use the same audio file but simulate different ones
    # In real usage, you'd have different audio files
    demo_audio = "../demo/talk_hb.wav"
    
    # Prepare requests with different models and frames
    # Using same audio for demo, but each could be different
    audio_encoded = encode_audio_file(demo_audio)
    
    requests_data = [
        {"model_name": "default_model", "frame_id": 17, "audio_override": audio_encoded},
        {"model_name": "default_model", "frame_id": 18, "audio_override": audio_encoded},
        {"model_name": "default_model", "frame_id": 19, "audio_override": audio_encoded},
        {"model_name": "enhanced_model", "frame_id": 42, "audio_override": audio_encoded},
        {"model_name": "fast_model", "frame_id": 99, "audio_override": audio_encoded}
    ]
    
    # Filter out requests with missing audio (for demo)
    if audio_encoded is None:
        print("No demo audio found, testing without audio override...")
        requests_data = [
            {"model_name": "default_model", "frame_id": 17},
            {"model_name": "default_model", "frame_id": 18},
            {"model_name": "default_model", "frame_id": 19}
        ]
    
    print(f"📋 Submitting {len(requests_data)} frame requests:")
    for i, req in enumerate(requests_data):
        audio_status = "with audio" if req.get("audio_override") else "no audio"
        print(f"  {i+1}. {req['model_name']} frame {req['frame_id']} ({audio_status})")
    
    # Make the request
    start_time = time.time()
    
    try:
        response = requests.post(f"{SERVICE_URL}/generate/batch", json={
            "requests": requests_data
        }, timeout=60)
        
        request_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Multi-batch completed in {request_time:.1f}ms")
            print(f"📊 Results Summary:")
            print(f"   Total requests: {result.get('total_requests', 0)}")
            print(f"   Processed: {result.get('processed_count', 0)}")
            print(f"   From cache: {result.get('cached_count', 0)}")
            print(f"   Failed: {result.get('failed_count', 0)}")
            print(f"   Auto-loaded models: {result.get('auto_loaded_models', [])}")
            print(f"   Server processing time: {result.get('processing_time_ms', 0)}ms")
            
            print(f"\n🖼️  Individual Frame Results:")
            for key, frame_result in result.get("results", {}).items():
                if frame_result.get("success"):
                    frame_size = len(frame_result.get("frame", "")) if frame_result.get("frame") else 0
                    cache_status = "cached" if frame_result.get("from_cache") else "generated"
                    auto_load = "auto-loaded" if frame_result.get("auto_loaded") else "pre-loaded"
                    print(f"   ✅ {key}: {cache_status}, {auto_load}, {frame_size} bytes")
                else:
                    error = frame_result.get("error", "Unknown error")
                    print(f"   ❌ {key}: Failed - {error}")
                    
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to service. Is it running on port 8000?")
        print("Start with: python service.py")
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 60 seconds")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_service_health():
    """Check if service is running"""
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Service is running")
            return True
        else:
            print(f"⚠️  Service responded with status {response.status_code}")
            return False
    except:
        print("❌ Service is not responding")
        return False

def compare_with_single_requests():
    """Compare multi-batch vs individual requests performance"""
    
    print("\n🏃 Performance Comparison")
    print("=" * 50)
    
    # Test data
    demo_audio = "../demo/talk_hb.wav"
    audio_encoded = encode_audio_file(demo_audio)
    
    test_requests = [
        {"model_name": "default_model", "frame_id": 25},
        {"model_name": "default_model", "frame_id": 26},
        {"model_name": "default_model", "frame_id": 27}
    ]
    
    if audio_encoded:
        for req in test_requests:
            req["audio_override"] = audio_encoded
    
    # Test individual requests
    print("Testing individual requests...")
    start_time = time.time()
    individual_results = []
    
    for req in test_requests:
        try:
            response = requests.post(f"{SERVICE_URL}/generate/frame", json=req, timeout=30)
            if response.status_code == 200:
                individual_results.append(response.json())
        except Exception as e:
            print(f"Individual request failed: {e}")
    
    individual_time = (time.time() - start_time) * 1000
    
    # Test batch request
    print("Testing batch request...")
    start_time = time.time()
    
    try:
        response = requests.post(f"{SERVICE_URL}/generate/batch", json={
            "requests": test_requests
        }, timeout=30)
        
        batch_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            batch_result = response.json()
            
            print(f"\n📊 Performance Results:")
            print(f"   Individual requests: {individual_time:.1f}ms ({len(individual_results)} frames)")
            print(f"   Multi-batch request: {batch_time:.1f}ms ({batch_result.get('processed_count', 0)} frames)")
            
            if individual_time > 0:
                speedup = individual_time / batch_time
                print(f"   Speedup: {speedup:.2f}x faster")
        
    except Exception as e:
        print(f"Batch request failed: {e}")

if __name__ == "__main__":
    print("🚀 SyncTalk2D Multi-Audio Batch Test")
    print("=" * 50)
    
    # Check service health
    if not test_service_health():
        exit(1)
    
    # Run tests
    test_batch_endpoint()
    compare_with_single_requests()
    
    print(f"\n✨ Test completed!")
