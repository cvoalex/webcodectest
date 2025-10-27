import asyncio
import aiohttp
import json
import base64
import time
import cv2
import numpy as np
from pathlib import Path

class MultiModelTestClient:
    """Enhanced test client for multi-model service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def encode_audio_file(self, audio_path: str) -> str:
        """Encode audio file to base64"""
        with open(audio_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def decode_frame_from_base64(self, frame_b64: str) -> np.ndarray:
        """Decode base64 frame to numpy array"""
        frame_data = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        return cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    
    async def health_check(self):
        """Check service health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def get_status(self):
        """Get service status"""
        async with self.session.get(f"{self.base_url}/status") as response:
            return await response.json()
    
    async def get_stats(self):
        """Get comprehensive stats"""
        async with self.session.get(f"{self.base_url}/stats") as response:
            return await response.json()
    
    async def load_model(self, model_name: str, package_path: str, audio_override: str = None):
        """Load a model package"""
        
        payload = {
            "model_name": model_name,
            "package_path": package_path
        }
        
        if audio_override:
            payload["audio_override"] = self.encode_audio_file(audio_override)
        
        async with self.session.post(f"{self.base_url}/models/load", json=payload) as response:
            return await response.json()
    
    async def unload_model(self, model_name: str):
        """Unload a model"""
        async with self.session.delete(f"{self.base_url}/models/{model_name}") as response:
            return await response.json()
    
    async def list_models(self):
        """List all loaded models"""
        async with self.session.get(f"{self.base_url}/models") as response:
            return await response.json()
    
    async def generate_frame(self, model_name: str, frame_id: int, audio_override: str = None):
        """Generate a single frame"""
        
        payload = {
            "model_name": model_name,
            "frame_id": frame_id
        }
        
        if audio_override:
            payload["audio_override"] = self.encode_audio_file(audio_override)
        
        async with self.session.post(f"{self.base_url}/generate/frame", json=payload) as response:
            return await response.json()
    
    async def generate_batch(self, model_name: str, frame_ids: list, audio_override: str = None):
        """Generate multiple frames"""
        
        payload = {
            "model_name": model_name,
            "frame_ids": frame_ids
        }
        
        if audio_override:
            payload["audio_override"] = self.encode_audio_file(audio_override)
        
        async with self.session.post(f"{self.base_url}/generate/batch", json=payload) as response:
            return await response.json()
    
    async def preload_cache(self, model_name: str, start_frame: int = 0, end_frame: int = 100):
        """Preload frames into cache"""
        
        payload = {
            "model_name": model_name,
            "start_frame": start_frame,
            "end_frame": end_frame
        }
        
        async with self.session.post(f"{self.base_url}/cache/preload", json=payload) as response:
            return await response.json()
    
    async def clear_model_cache(self, model_name: str):
        """Clear cache for specific model"""
        async with self.session.delete(f"{self.base_url}/cache/{model_name}") as response:
            return await response.json()
    
    async def clear_all_cache(self):
        """Clear all cache"""
        async with self.session.delete(f"{self.base_url}/cache") as response:
            return await response.json()


async def test_multi_model_workflow():
    """Test the complete multi-model workflow"""
    
    print("🚀 Starting Multi-Model Service Test")
    
    # Test configuration
    test_models = [
        {
            "name": "model_a", 
            "package": r"D:\Projects\SyncTalk2D\result\optimized_package_v2.zip",
            "audio": r"D:\Projects\SyncTalk2D\demo\talk_hb.wav"
        },
        {
            "name": "model_b", 
            "package": r"D:\Projects\SyncTalk2D\result\optimized_package_v2.zip",  # Same package for test
            "audio": None  # Use default audio
        }
    ]
    
    async with MultiModelTestClient() as client:
        
        # 1. Health check
        print("\n📋 Health Check")
        health = await client.health_check()
        print(f"Health: {health}")
        
        # 2. Initial status
        print("\n📊 Initial Status")
        status = await client.get_status()
        print(f"Loaded models: {status.get('loaded_models', [])}")
        
        # 3. Load multiple models
        print("\n📦 Loading Models")
        for model_config in test_models:
            print(f"Loading {model_config['name']}...")
            result = await client.load_model(
                model_config["name"], 
                model_config["package"],
                model_config["audio"]
            )
            print(f"✅ {model_config['name']}: {result.get('success', False)}")
            if not result.get('success'):
                print(f"❌ Error: {result.get('error')}")
        
        # 4. List loaded models
        print("\n📋 Loaded Models")
        models_list = await client.list_models()
        print(f"Total models: {models_list.get('total_models', 0)}")
        for model in models_list.get('loaded_models', []):
            print(f"  - {model}")
        
        # 5. Test frame generation with different models
        print("\n🎬 Testing Frame Generation")
        
        test_frames = [0, 5, 10]
        
        for model_config in test_models:
            model_name = model_config["name"]
            print(f"\nTesting {model_name}:")
            
            # Single frame generation
            print(f"  Generating frame 0...")
            start_time = time.time()
            frame_result = await client.generate_frame(model_name, 0)
            gen_time = time.time() - start_time
            
            if frame_result.get('success'):
                print(f"  ✅ Frame 0: {frame_result['processing_time_ms']:.1f}ms (total: {gen_time*1000:.1f}ms)")
                print(f"     From cache: {frame_result.get('from_cache', False)}")
            else:
                print(f"  ❌ Frame 0 failed: {frame_result.get('error')}")
            
            # Batch frame generation
            print(f"  Generating batch {test_frames}...")
            start_time = time.time()
            batch_result = await client.generate_batch(model_name, test_frames)
            batch_time = time.time() - start_time
            
            if batch_result.get('success'):
                print(f"  ✅ Batch: {batch_result['processing_time_ms']:.1f}ms (total: {batch_time*1000:.1f}ms)")
                print(f"     Cached: {batch_result.get('cached_count', 0)}, Generated: {batch_result.get('generated_count', 0)}")
            else:
                print(f"  ❌ Batch failed: {batch_result.get('error')}")
        
        # 6. Test audio override functionality
        print("\n🎵 Testing Audio Override")
        if len(test_models) > 1:
            model_a = test_models[0]["name"]
            alt_audio = test_models[0]["audio"]  # Use model A's audio for model B
            
            if alt_audio:
                print(f"Generating frame with audio override...")
                override_result = await client.generate_frame(
                    test_models[1]["name"], 
                    20, 
                    alt_audio
                )
                
                if override_result.get('success'):
                    print(f"✅ Audio override: {override_result['processing_time_ms']:.1f}ms")
                else:
                    print(f"❌ Audio override failed: {override_result.get('error')}")
        
        # 7. Test cache preloading
        print("\n💾 Testing Cache Preloading")
        for model_config in test_models[:1]:  # Test with first model only
            model_name = model_config["name"]
            print(f"Preloading cache for {model_name}...")
            
            preload_result = await client.preload_cache(model_name, 0, 10)
            
            if preload_result.get('success'):
                result_details = preload_result['preload_result']
                print(f"✅ Preloaded {result_details.get('preloaded_count', 0)} frames")
                print(f"   Processing time: {result_details.get('processing_time_ms', 0)}ms")
            else:
                print(f"❌ Preload failed: {preload_result.get('error')}")
        
        # 8. Performance statistics
        print("\n📈 Performance Statistics")
        stats = await client.get_stats()
        
        if stats.get('success'):
            cache_stats = stats.get('cache_stats', {})
            engine_stats = stats.get('engine_stats', {})
            
            print("Cache Statistics:")
            overall = cache_stats.get('overall_stats', {})
            print(f"  Total hits: {overall.get('total_cache_hits', 0)}")
            print(f"  Total misses: {overall.get('total_cache_misses', 0)}")
            print(f"  Hit ratio: {overall.get('overall_hit_ratio', 0):.2%}")
            print(f"  Cached frames: {overall.get('total_cached_frames', 0)}")
            
            model_stats = cache_stats.get('model_stats', {})
            for model_name, stats_data in model_stats.items():
                print(f"  {model_name}: {stats_data}")
            
            print("\nEngine Statistics:")
            for model_name, model_engine_stats in engine_stats.items():
                print(f"  {model_name}: {model_engine_stats}")
        
        # 9. Test cache clearing
        print("\n🧹 Testing Cache Management")
        
        # Clear specific model cache
        if len(test_models) > 1:
            clear_result = await client.clear_model_cache(test_models[0]["name"])
            print(f"Cleared {test_models[0]['name']} cache: {clear_result.get('cleared_items', 0)} items")
        
        # 10. Final status
        print("\n📊 Final Status")
        final_status = await client.get_status()
        print(f"Final loaded models: {final_status.get('total_models', 0)}")
        print(f"Redis connected: {final_status.get('redis_connected', False)}")
        
        print("\n✅ Multi-Model Service Test Complete!")


async def test_naming_convention():
    """Test the modelname_framenumber,<audio> naming convention"""
    
    print("\n🔤 Testing Naming Convention: modelname_framenumber,<audio>")
    
    async with MultiModelTestClient() as client:
        
        # Load a model
        model_name = "testmodel"
        package_path = r"D:\Projects\SyncTalk2D\result\optimized_package_v2.zip"
        audio_path = r"D:\Projects\SyncTalk2D\demo\talk_hb.wav"
        
        print(f"Loading model: {model_name}")
        load_result = await client.load_model(model_name, package_path)
        
        if load_result.get('success'):
            print("✅ Model loaded successfully")
            
            # Test the naming convention pattern
            test_cases = [
                {"model": model_name, "frame": 17, "audio": audio_path, "desc": "testmodel_17,<audio>"},
                {"model": model_name, "frame": 42, "audio": None, "desc": "testmodel_42 (no audio)"},
                {"model": model_name, "frame": 99, "audio": audio_path, "desc": "testmodel_99,<audio>"}
            ]
            
            for case in test_cases:
                print(f"\nTesting: {case['desc']}")
                
                result = await client.generate_frame(
                    case["model"], 
                    case["frame"], 
                    case["audio"]
                )
                
                if result.get('success'):
                    print(f"✅ Generated frame {case['frame']} in {result['processing_time_ms']:.1f}ms")
                    if case["audio"]:
                        print(f"   With custom audio override")
                else:
                    print(f"❌ Failed: {result.get('error')}")
        
        else:
            print(f"❌ Failed to load model: {load_result.get('error')}")


if __name__ == "__main__":
    print("🎯 Multi-Model Service Test Suite")
    print("=" * 50)
    
    # Run main test
    asyncio.run(test_multi_model_workflow())
    
    # Run naming convention test
    asyncio.run(test_naming_convention())
    
    print("\n🏁 All tests completed!")
