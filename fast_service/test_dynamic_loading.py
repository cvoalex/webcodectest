import asyncio
import aiohttp
import json
import base64
import time
import cv2
import numpy as np
from pathlib import Path

class DynamicModelTestClient:
    """Test client for dynamic model loading system"""
    
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
    
    async def get_status(self):
        """Get service status"""
        async with self.session.get(f"{self.base_url}/status") as response:
            return await response.json()
    
    async def list_models(self):
        """List loaded and local models"""
        async with self.session.get(f"{self.base_url}/models") as response:
            return await response.json()
    
    async def list_registry_models(self):
        """List models available in registry"""
        async with self.session.get(f"{self.base_url}/models/registry") as response:
            return await response.json()
    
    async def download_model(self, model_name: str):
        """Download model from registry"""
        async with self.session.post(f"{self.base_url}/models/download", params={"model_name": model_name}) as response:
            return await response.json()
    
    async def generate_frame_auto(self, model_name: str, frame_id: int, audio_override: str = None):
        """Generate frame with automatic model loading"""
        
        payload = {
            "model_name": model_name,
            "frame_id": frame_id
        }
        
        if audio_override:
            payload["audio_override"] = self.encode_audio_file(audio_override)
        
        async with self.session.post(f"{self.base_url}/generate/frame", json=payload) as response:
            return await response.json()
    
    async def cleanup_local_model(self, model_name: str, remove_zip: bool = False):
        """Clean up local model files"""
        async with self.session.delete(f"{self.base_url}/models/local/{model_name}", params={"remove_zip": remove_zip}) as response:
            return await response.json()


async def test_dynamic_model_loading():
    """Test the dynamic model loading system"""
    
    print("🎯 Testing Dynamic Model Loading System")
    print("=" * 60)
    
    async with DynamicModelTestClient() as client:
        
        # 1. Check initial status
        print("\n📊 Initial Status")
        status = await client.get_status()
        print(f"Service: {status.get('service', 'Unknown')}")
        print(f"Loaded models: {status.get('total_models', 0)}")
        
        # 2. List local models
        print("\n📁 Local Models")
        local_models = await client.list_models()
        if local_models.get('success'):
            print(f"Loaded: {local_models['total_loaded']}")
            print(f"Local available: {local_models['total_local']}")
            
            for model in local_models.get('local_models', {}).get('extracted', []):
                print(f"  ✅ {model['name']} (ready)")
            
            for model in local_models.get('local_models', {}).get('zipped', []):
                print(f"  📦 {model['name']} (needs extraction)")
        
        # 3. List registry models
        print("\n🌐 Registry Models")
        registry = await client.list_registry_models()
        if registry.get('success'):
            print(f"Available in registry: {registry['total_available']}")
            for model in registry.get('registry_models', []):
                print(f"  🌍 {model['name']} v{model['version']} ({model['size_mb']} MB)")
                print(f"     {model['description']}")
        
        # 4. Test automatic model loading via frame generation
        print("\n🚀 Testing Automatic Model Loading")
        
        test_models = ["default_model", "enhanced_model", "nonexistent_model"]
        
        for model_name in test_models:
            print(f"\n🎬 Testing model: {model_name}")
            
            try:
                start_time = time.time()
                result = await client.generate_frame_auto(model_name, 0)
                total_time = time.time() - start_time
                
                if result.get('success'):
                    print(f"  ✅ Frame generated successfully!")
                    print(f"     Processing time: {result.get('processing_time_ms', 0):.1f}ms")
                    print(f"     Total time: {total_time*1000:.1f}ms")
                    print(f"     From cache: {result.get('from_cache', False)}")
                    print(f"     Auto-loaded: {result.get('auto_loaded', False)}")
                    
                    # Test same model again (should be cached/loaded)
                    start_time = time.time()
                    result2 = await client.generate_frame_auto(model_name, 1)
                    total_time2 = time.time() - start_time
                    
                    if result2.get('success'):
                        print(f"  🔄 Second frame: {result2.get('processing_time_ms', 0):.1f}ms (total: {total_time2*1000:.1f}ms)")
                        print(f"     From cache: {result2.get('from_cache', False)}")
                
                else:
                    print(f"  ❌ Failed: {result.get('error')}")
                    
            except Exception as e:
                print(f"  ❌ Exception: {e}")
        
        # 5. Test naming convention: modelname_framenumber
        print("\n🔤 Testing Naming Convention")
        
        test_cases = [
            {"model": "default_model", "frame": 17, "desc": "default_model_17"},
            {"model": "enhanced_model", "frame": 42, "desc": "enhanced_model_42"},
            {"model": "default_model", "frame": 99, "desc": "default_model_99"}
        ]
        
        for case in test_cases:
            print(f"\nTesting: {case['desc']}")
            
            try:
                result = await client.generate_frame_auto(case["model"], case["frame"])
                
                if result.get('success'):
                    print(f"  ✅ Generated {case['desc']} in {result['processing_time_ms']:.1f}ms")
                    print(f"     From cache: {result.get('from_cache', False)}")
                else:
                    print(f"  ❌ Failed: {result.get('error')}")
                    
            except Exception as e:
                print(f"  ❌ Exception: {e}")
        
        # 6. Test manual download
        print("\n📥 Testing Manual Download")
        download_result = await client.download_model("fast_model")
        
        if download_result.get('success'):
            print(f"✅ Manual download successful:")
            print(f"   Actions: {download_result.get('actions_taken', [])}")
            print(f"   Path: {download_result.get('model_path')}")
        else:
            print(f"❌ Manual download failed: {download_result.get('error')}")
        
        # 7. Final status check
        print("\n📊 Final Status")
        final_status = await client.list_models()
        if final_status.get('success'):
            print(f"Total loaded models: {final_status['total_loaded']}")
            print(f"Total local models: {final_status['total_local']}")
            
            print("Loaded models:")
            for model in final_status.get('loaded_models', []):
                print(f"  🔗 {model}")
        
        print("\n✅ Dynamic Model Loading Test Complete!")


async def test_model_cleanup():
    """Test model cleanup functionality"""
    
    print("\n🧹 Testing Model Cleanup")
    print("=" * 40)
    
    async with DynamicModelTestClient() as client:
        
        # Load a model first
        print("Loading model for cleanup test...")
        result = await client.generate_frame_auto("default_model", 0)
        
        if result.get('success'):
            print("✅ Model loaded successfully")
            
            # Test cleanup
            print("Testing cleanup...")
            cleanup_result = await client.cleanup_local_model("default_model", remove_zip=False)
            
            if cleanup_result.get('success'):
                print("✅ Cleanup successful:")
                cleanup_details = cleanup_result.get('cleanup_result', {})
                for item in cleanup_details.get('removed_items', []):
                    print(f"   Removed: {item}")
                
                for error in cleanup_details.get('errors', []):
                    print(f"   Error: {error}")
            else:
                print(f"❌ Cleanup failed: {cleanup_result.get('error')}")
        
        else:
            print(f"❌ Could not load model for cleanup test: {result.get('error')}")


if __name__ == "__main__":
    print("🎯 Dynamic Model Loading Test Suite")
    print("=" * 70)
    
    # Run main test
    asyncio.run(test_dynamic_model_loading())
    
    # Run cleanup test
    asyncio.run(test_model_cleanup())
    
    print("\n🏁 All dynamic loading tests completed!")
