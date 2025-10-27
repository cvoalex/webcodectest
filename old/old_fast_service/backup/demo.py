#!/usr/bin/env python3
"""
SyncTalk2D Dynamic Model Loading Demonstration

This script demonstrates the key features of the dynamic model loading system:
1. Automatic model downloading from registry
2. Model extraction and loading 
3. Frame generation with naming convention modelname_framenumber
4. Audio override functionality
5. Multi-model support
"""

import asyncio
import aiohttp
import json
import base64
import time
from pathlib import Path

async def demonstrate_dynamic_loading():
    """Demonstrate the dynamic model loading system"""
    
    print("🎯 SyncTalk2D Dynamic Model Loading Demonstration")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        
        print("\n📊 1. Check Service Status")
        async with session.get(f"{base_url}/status") as response:
            if response.status == 200:
                status = await response.json()
                print(f"✅ Service running: {status.get('service', 'Unknown')}")
                print(f"   Loaded models: {status.get('total_models', 0)}")
                print(f"   Redis connected: {status.get('redis_connected', False)}")
            else:
                print("❌ Service not available. Please start the service first:")
                print("   python service.py")
                return
        
        print("\n🌐 2. Check Available Models in Registry")
        async with session.get(f"{base_url}/models/registry") as response:
            registry = await response.json()
            if registry.get('success'):
                print(f"Registry URL: {registry.get('registry_url')}")
                for model in registry.get('registry_models', []):
                    print(f"  📦 {model['name']} v{model['version']} ({model['size_mb']} MB)")
                    print(f"     {model['description']}")
        
        print("\n📁 3. Check Local Models")
        async with session.get(f"{base_url}/models") as response:
            models = await response.json()
            if models.get('success'):
                print(f"Loaded models: {models['total_loaded']}")
                print(f"Local models: {models['total_local']}")
                
                for model in models.get('local_models', {}).get('extracted', []):
                    print(f"  ✅ {model['name']} (ready)")
                
                for model in models.get('local_models', {}).get('zipped', []):
                    print(f"  📦 {model['name']} (needs extraction)")
        
        print("\n🚀 4. Demonstrate Automatic Model Loading")
        print("Testing naming convention: modelname_framenumber")
        
        test_cases = [
            {"model": "default_model", "frame": 17, "desc": "default_model_17"},
            {"model": "enhanced_model", "frame": 42, "desc": "enhanced_model_42"}, 
            {"model": "fast_model", "frame": 0, "desc": "fast_model_0"}
        ]
        
        for case in test_cases:
            print(f"\n🎬 Testing: {case['desc']}")
            
            payload = {
                "model_name": case["model"],
                "frame_id": case["frame"]
            }
            
            start_time = time.time()
            async with session.post(f"{base_url}/generate/frame", json=payload) as response:
                total_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get('success'):
                        print(f"  ✅ Generated successfully!")
                        print(f"     Processing time: {result.get('processing_time_ms', 0):.1f}ms")
                        print(f"     Total time: {total_time*1000:.1f}ms")
                        print(f"     From cache: {result.get('from_cache', False)}")
                        print(f"     Auto-loaded: {result.get('auto_loaded', False)}")
                        
                        # Test second request (should be much faster)
                        start_time = time.time()
                        async with session.post(f"{base_url}/generate/frame", json=payload) as response2:
                            total_time2 = time.time() - start_time
                            result2 = await response2.json()
                            
                            if result2.get('success'):
                                print(f"  🔄 Second request: {result2.get('processing_time_ms', 0):.1f}ms (total: {total_time2*1000:.1f}ms)")
                                print(f"     From cache: {result2.get('from_cache', False)}")
                    else:
                        print(f"  ❌ Failed: {result.get('error')}")
                else:
                    error_text = await response.text()
                    print(f"  ❌ HTTP {response.status}: {error_text}")
        
        print("\n🎵 5. Test Audio Override")
        # Note: This would require a real audio file
        print("Audio override allows per-request custom audio:")
        print("  POST /generate/frame")
        print("  {")
        print('    "model_name": "default_model",')
        print('    "frame_id": 99,')
        print('    "audio_override": "base64_encoded_wav_file"')
        print("  }")
        
        print("\n📊 6. Final Statistics")
        async with session.get(f"{base_url}/stats") as response:
            if response.status == 200:
                stats = await response.json()
                
                if stats.get('success'):
                    cache_stats = stats.get('cache_stats', {}).get('overall_stats', {})
                    engine_stats = stats.get('engine_stats', {})
                    
                    print("Cache Performance:")
                    print(f"  Total hits: {cache_stats.get('total_cache_hits', 0)}")
                    print(f"  Total misses: {cache_stats.get('total_cache_misses', 0)}")
                    print(f"  Hit ratio: {cache_stats.get('overall_hit_ratio', 0):.2%}")
                    print(f"  Cached frames: {cache_stats.get('total_cached_frames', 0)}")
                    
                    print("\nModel Performance:")
                    for model_name, model_stats in engine_stats.items():
                        print(f"  {model_name}:")
                        print(f"    Requests: {model_stats.get('total_requests', 0)}")
                        print(f"    Avg time: {model_stats.get('average_inference_time_ms', 0):.1f}ms")
        
        print("\n✅ Dynamic Model Loading Demonstration Complete!")
        print("\nKey Features Demonstrated:")
        print("  🔄 Automatic model downloading from registry")
        print("  📦 Model extraction and loading on-demand")
        print("  🎯 Naming convention: modelname_framenumber")
        print("  💾 Redis caching with significant speedup")
        print("  📊 Comprehensive performance monitoring")
        print("  🎵 Audio override capability")


async def demonstrate_batch_processing():
    """Demonstrate batch frame generation"""
    
    print("\n🚀 Batch Processing Demonstration")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        
        # Generate multiple frames in batch
        payload = {
            "model_name": "default_model",
            "frame_ids": [0, 1, 2, 3, 4, 5, 10, 15, 20]
        }
        
        print(f"Generating batch: {payload['frame_ids']}")
        
        start_time = time.time()
        async with session.post(f"{base_url}/generate/batch", json=payload) as response:
            total_time = time.time() - start_time
            
            if response.status == 200:
                result = await response.json()
                
                if result.get('success'):
                    print(f"✅ Batch completed!")
                    print(f"   Total frames: {result.get('total_frames', 0)}")
                    print(f"   Cached: {result.get('cached_count', 0)}")
                    print(f"   Generated: {result.get('generated_count', 0)}")
                    print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")
                    print(f"   Total time: {total_time*1000:.1f}ms")
                else:
                    print(f"❌ Batch failed: {result.get('error')}")
            else:
                error_text = await response.text()
                print(f"❌ HTTP {response.status}: {error_text}")


def print_usage_examples():
    """Print usage examples for different scenarios"""
    
    print("\n📚 Usage Examples")
    print("=" * 30)
    
    print("\n1. Basic Frame Generation (Python):")
    print("""
import requests

response = requests.post("http://localhost:8000/generate/frame", json={
    "model_name": "default_model",
    "frame_id": 17
})

result = response.json()
if result["success"]:
    frame_base64 = result["frame"]
    # Decode and process frame
""")
    
    print("\n2. WebSocket Streaming (JavaScript):")
    print("""
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    ws.send(JSON.stringify({
        model_name: "default_model", 
        frame_id: 42
    }));
};

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    if (result.success) {
        const img = new Image();
        img.src = 'data:image/jpeg;base64,' + result.frame;
        document.body.appendChild(img);
    }
};
""")
    
    print("\n3. Audio Override (Python):")
    print("""
import base64

with open("custom_audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/generate/frame", json={
    "model_name": "enhanced_model",
    "frame_id": 99,
    "audio_override": audio_b64
})
""")


if __name__ == "__main__":
    print("🎬 SyncTalk2D Dynamic Model Loading System")
    print("🔥 Real-time lip-sync with automatic model management")
    print()
    
    # Check if service is likely running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ Service detected at http://localhost:8000")
        else:
            print("⚠️  Service responding but may have issues")
    except:
        print("❌ Service not running. Please start with: python service.py")
        print()
    
    # Run demonstrations
    try:
        asyncio.run(demonstrate_dynamic_loading())
        asyncio.run(demonstrate_batch_processing())
        print_usage_examples()
        
    except KeyboardInterrupt:
        print("\n👋 Demonstration stopped by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Make sure the service is running: python service.py")
    
    print("\n🎯 To start the service:")
    print("   cd fast_service")
    print("   python service.py")
    print("\n📚 For more info, see README.md")
