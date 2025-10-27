#!/usr/bin/env python3
"""
Test client for SyncTalk2D Fast Inference Service
"""

import requests
import json
import time
import base64
import cv2
import numpy as np
from typing import List, Dict, Any
import asyncio
import websockets

class FastInferenceClient:
    """Client for interacting with the Fast Inference Service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def initialize(self, package_path: str, audio_path: str = None) -> Dict[str, Any]:
        """Initialize the service"""
        
        data = {"package_path": package_path}
        if audio_path:
            data["audio_path"] = audio_path
        
        response = self.session.post(f"{self.base_url}/initialize", json=data)
        response.raise_for_status()
        
        return response.json()
    
    def get_frame(self, frame_id: int) -> Dict[str, Any]:
        """Get a single frame"""
        
        response = self.session.get(f"{self.base_url}/frame/{frame_id}")
        response.raise_for_status()
        
        return response.json()
    
    def get_frame_batch(self, start_frame: int, end_frame: int) -> Dict[str, Any]:
        """Get a batch of frames"""
        
        response = self.session.get(f"{self.base_url}/frames/batch/{start_frame}/{end_frame}")
        response.raise_for_status()
        
        return response.json()
    
    def preload_frames(self, start_frame: int, end_frame: int) -> Dict[str, Any]:
        """Preload frames in background"""
        
        data = {"start_frame": start_frame, "end_frame": end_frame}
        response = self.session.post(f"{self.base_url}/cache/preload", json=data)
        response.raise_for_status()
        
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        """Get service health status"""
        
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        
        return response.json()
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear service cache"""
        
        response = self.session.delete(f"{self.base_url}/cache")
        response.raise_for_status()
        
        return response.json()
    
    def decode_frame(self, frame_data: str) -> np.ndarray:
        """Decode base64 frame to numpy array"""
        
        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return frame
    
    def save_frame(self, frame_data: str, output_path: str):
        """Save frame to file"""
        
        frame = self.decode_frame(frame_data)
        cv2.imwrite(output_path, frame)


async def websocket_test(ws_url: str = "ws://localhost:8000/stream"):
    """Test WebSocket streaming"""
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("🔌 Connected to WebSocket")
            
            # Request a few frames
            for frame_id in range(5):
                request = {
                    "type": "request_frame",
                    "frame_id": frame_id
                }
                
                await websocket.send(json.dumps(request))
                print(f"📤 Requested frame {frame_id}")
                
                response = await websocket.recv()
                data = json.loads(response)
                
                if data["type"] == "frame_ready":
                    print(f"📥 Received frame {data['frame_id']} (cached: {data['cached']}, time: {data['processing_time_ms']}ms)")
                elif data["type"] == "error":
                    print(f"❌ Error: {data['message']}")
            
            # Ping test
            await websocket.send(json.dumps({"type": "ping"}))
            response = await websocket.recv()
            data = json.loads(response)
            print(f"🏓 Ping response: {data['type']}")
            
    except Exception as e:
        print(f"WebSocket error: {e}")


def main():
    """Test the Fast Inference Service"""
    
    client = FastInferenceClient()
    
    print("🧪 Testing SyncTalk2D Fast Inference Service")
    print("=" * 50)
    
    # Test 1: Health check before initialization
    print("\n1. Health check (before init):")
    try:
        health = client.get_health()
        print(f"   Service status: {health['service']}")
        print(f"   Engine loaded: {health['inference_engine']['loaded']}")
        print(f"   Redis connected: {health['cache']['connected']}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Test 2: Initialize service
    print("\n2. Initializing service:")
    try:
        # Use relative path to test package
        init_result = client.initialize(
            package_path="../test_advanced_package_v4.zip",
            audio_path="../demo/talk_hb.wav"
        )
        print(f"   ✅ Initialization: {init_result['status']}")
        print(f"   ⏱️  Time: {init_result['initialization_time_ms']}ms")
        print(f"   🎬 Total frames: {init_result['total_frames']}")
        print(f"   🎵 Audio shape: {init_result['audio_features_shape']}")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return
    
    # Test 3: Single frame generation
    print("\n3. Single frame generation:")
    try:
        start_time = time.time()
        frame_result = client.get_frame(42)
        end_time = time.time()
        
        print(f"   ✅ Frame 42 generated")
        print(f"   ⏱️  Time: {frame_result['metadata']['processing_time_ms']}ms")
        print(f"   💾 Cached: {frame_result['metadata']['cached']}")
        print(f"   🖼️  Shape: {frame_result['metadata']['frame_shape']}")
        print(f"   🌐 Network time: {int((end_time - start_time) * 1000)}ms")
        
        # Save frame for inspection
        client.save_frame(frame_result['frame_data'], "test_frame_42.jpg")
        print(f"   💾 Saved: test_frame_42.jpg")
        
    except Exception as e:
        print(f"   ❌ Frame generation failed: {e}")
    
    # Test 4: Cache hit test (same frame)
    print("\n4. Cache hit test:")
    try:
        start_time = time.time()
        frame_result = client.get_frame(42)
        end_time = time.time()
        
        print(f"   ✅ Frame 42 retrieved")
        print(f"   ⏱️  Time: {frame_result['metadata']['processing_time_ms']}ms")
        print(f"   💾 Cached: {frame_result['metadata']['cached']}")
        print(f"   🌐 Network time: {int((end_time - start_time) * 1000)}ms")
        
    except Exception as e:
        print(f"   ❌ Cache test failed: {e}")
    
    # Test 5: Batch frame generation
    print("\n5. Batch frame generation:")
    try:
        start_time = time.time()
        batch_result = client.get_frame_batch(0, 4)
        end_time = time.time()
        
        batch_info = batch_result['batch_info']
        print(f"   ✅ Batch generated: frames {batch_info['start_frame']}-{batch_info['end_frame']}")
        print(f"   ⏱️  Time: {batch_info['total_processing_time_ms']}ms")
        print(f"   💾 Cached: {batch_info['cached_frames']}/{batch_info['total_frames']}")
        print(f"   🔥 Generated: {batch_info['generated_frames']}")
        print(f"   🌐 Network time: {int((end_time - start_time) * 1000)}ms")
        
    except Exception as e:
        print(f"   ❌ Batch generation failed: {e}")
    
    # Test 6: Preloading
    print("\n6. Background preloading:")
    try:
        start_time = time.time()
        preload_result = client.preload_frames(10, 19)
        end_time = time.time()
        
        print(f"   ✅ Preloaded: {preload_result['preloaded_count']}/{preload_result['total_frames']}")
        print(f"   ⏱️  Time: {preload_result['processing_time_ms']}ms")
        print(f"   🌐 Network time: {int((end_time - start_time) * 1000)}ms")
        
    except Exception as e:
        print(f"   ❌ Preloading failed: {e}")
    
    # Test 7: Metrics
    print("\n7. Performance metrics:")
    try:
        metrics = client.get_metrics()
        
        inference = metrics['inference']
        cache = metrics['cache']
        
        print(f"   📊 Total inferences: {inference['total_inferences']}")
        print(f"   ⚡ Avg inference time: {inference['avg_inference_time_ms']}ms")
        print(f"   💾 Cache hit ratio: {cache['hit_ratio']:.2%}")
        print(f"   🗃️  Cached frames: {cache['cached_frames']}")
        print(f"   💽 Redis memory: {cache['redis_memory_mb']:.1f}MB")
        
    except Exception as e:
        print(f"   ❌ Metrics failed: {e}")
    
    # Test 8: WebSocket streaming
    print("\n8. WebSocket streaming test:")
    try:
        asyncio.run(websocket_test())
    except Exception as e:
        print(f"   ❌ WebSocket test failed: {e}")
    
    # Test 9: Final health check
    print("\n9. Final health check:")
    try:
        health = client.get_health()
        print(f"   ✅ Service: {health['service']}")
        print(f"   🧠 Engine loaded: {health['inference_engine']['loaded']}")
        print(f"   💾 Cache connected: {health['cache']['connected']}")
        
        engine_stats = health['inference_engine']['stats']
        print(f"   📊 Total inferences: {engine_stats['total_inferences']}")
        print(f"   ⚡ Avg time: {engine_stats['avg_inference_time_ms']}ms")
        
    except Exception as e:
        print(f"   ❌ Final health check failed: {e}")
    
    print("\n🎉 Test completed!")


if __name__ == "__main__":
    main()
