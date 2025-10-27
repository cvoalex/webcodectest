#!/usr/bin/env python3
"""
🚀 gRPC Client for Ultra-Optimized Lip Sync Server
Test client for server-to-server communication.
"""

import asyncio
import time
import cv2
import numpy as np
from pathlib import Path

import grpc
from grpc import aio

try:
    import optimized_lipsyncsrv_pb2
    import optimized_lipsyncsrv_pb2_grpc
except ImportError:
    print("❌ Error: gRPC stubs not found!")
    print("Generate them with:")
    print("  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto")
    import sys
    sys.exit(1)


class OptimizedGRPCClient:
    """Client for optimized gRPC lip sync server"""
    
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.connected = False
    
    async def connect(self):
        """Connect to gRPC server"""
        print(f"🔌 Connecting to {self.server_address}...")
        
        self.channel = aio.insecure_channel(
            self.server_address,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
        )
        
        self.stub = optimized_lipsyncsrv_pb2_grpc.OptimizedLipSyncServiceStub(self.channel)
        self.connected = True
        
        print("✅ Connected to gRPC server!")
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.channel:
            await self.channel.close()
            self.connected = False
            print("⚡ Disconnected")
    
    async def health_check(self):
        """Check server health"""
        print("\n🏥 Health Check...")
        
        request = optimized_lipsyncsrv_pb2.HealthRequest()
        response = await self.stub.HealthCheck(request)
        
        print(f"   Status: {response.status}")
        print(f"   Healthy: {response.healthy}")
        print(f"   Loaded models: {response.loaded_models}")
        print(f"   Uptime: {response.uptime_seconds}s")
        
        return response
    
    async def list_models(self):
        """List loaded models"""
        print("\n📋 Listing Models...")
        
        request = optimized_lipsyncsrv_pb2.ListModelsRequest()
        response = await self.stub.ListModels(request)
        
        print(f"   Loaded models: {list(response.loaded_models)}")
        print(f"   Count: {response.count}")
        
        return response
    
    async def get_stats(self, model_name='sanders'):
        """Get model statistics"""
        print(f"\n📊 Getting Stats for '{model_name}'...")
        
        request = optimized_lipsyncsrv_pb2.StatsRequest(model_name=model_name)
        response = await self.stub.GetStats(request)
        
        print(f"   Total requests: {response.total_requests}")
        print(f"   Avg time: {response.avg_inference_time_ms:.2f}ms")
        print(f"   Min time: {response.min_inference_time_ms:.2f}ms")
        print(f"   Max time: {response.max_inference_time_ms:.2f}ms")
        print(f"   Frame count: {response.frame_count}")
        print(f"   Device: {response.device}")
        print(f"   Optimizations: {list(response.optimizations_active)}")
        
        return response
    
    async def generate_inference(self, model_name='sanders', frame_id=0, save_output=False):
        """Generate single frame inference"""
        
        start_time = time.time()
        
        request = optimized_lipsyncsrv_pb2.OptimizedInferenceRequest(
            model_name=model_name,
            frame_id=frame_id
        )
        
        response = await self.stub.GenerateInference(request)
        
        total_time = (time.time() - start_time) * 1000
        
        if response.success:
            print(f"✅ Frame {frame_id}:")
            print(f"   Server time: {response.processing_time_ms}ms")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Prepare: {response.prepare_time_ms:.1f}ms")
            print(f"   Inference: {response.inference_time_ms:.1f}ms")
            print(f"   Shape: {response.prediction_shape}")
            print(f"   Bounds: {list(response.bounds)}")
            print(f"   Image size: {len(response.prediction_data)} bytes")
            
            if save_output:
                # Decode and save image
                nparr = np.frombuffer(response.prediction_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                output_path = f"grpc_output_frame_{frame_id}.jpg"
                cv2.imwrite(output_path, img)
                print(f"   💾 Saved to: {output_path}")
        else:
            print(f"❌ Error: {response.error}")
        
        return response
    
    async def generate_batch(self, model_name='sanders', frame_ids=None):
        """Generate batch inference"""
        
        if frame_ids is None:
            frame_ids = [0, 10, 50, 100, 200]
        
        print(f"\n📦 Batch Inference: {len(frame_ids)} frames...")
        
        start_time = time.time()
        
        request = optimized_lipsyncsrv_pb2.BatchInferenceRequest(
            model_name=model_name,
            frame_ids=frame_ids
        )
        
        response = await self.stub.GenerateBatchInference(request)
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"✅ Batch complete:")
        print(f"   Frames: {len(response.responses)}")
        print(f"   Total time: {response.total_processing_time_ms}ms")
        print(f"   Avg per frame: {response.avg_frame_time_ms:.1f}ms")
        print(f"   Client total: {total_time:.1f}ms")
        
        successful = sum(1 for r in response.responses if r.success)
        print(f"   Successful: {successful}/{len(response.responses)}")
        
        return response
    
    async def stream_inference(self, model_name='sanders', frame_ids=None, max_frames=10):
        """Streaming inference test"""
        
        if frame_ids is None:
            frame_ids = list(range(max_frames))
        
        print(f"\n🌊 Streaming Inference: {len(frame_ids)} frames...")
        
        async def request_generator():
            for frame_id in frame_ids:
                yield optimized_lipsyncsrv_pb2.OptimizedInferenceRequest(
                    model_name=model_name,
                    frame_id=frame_id
                )
                # Small delay to simulate real-time
                await asyncio.sleep(0.02)  # 50 FPS
        
        start_time = time.time()
        response_count = 0
        total_processing = 0
        
        async for response in self.stub.StreamInference(request_generator()):
            response_count += 1
            total_processing += response.processing_time_ms
            
            if response.success:
                print(f"   Frame {response.frame_id}: {response.processing_time_ms}ms")
            else:
                print(f"   Frame {response.frame_id}: ERROR - {response.error}")
        
        total_time = (time.time() - start_time) * 1000
        avg_processing = total_processing / response_count if response_count > 0 else 0
        
        print(f"\n✅ Streaming complete:")
        print(f"   Frames: {response_count}")
        print(f"   Total time: {total_time:.1f}ms")
        print(f"   Avg processing: {avg_processing:.1f}ms")
        print(f"   Effective FPS: {response_count / (total_time/1000):.1f}")


async def main():
    """Test the gRPC client"""
    
    print("=" * 80)
    print("🧪 TESTING OPTIMIZED gRPC CLIENT")
    print("=" * 80)
    
    client = OptimizedGRPCClient()
    
    try:
        # Connect
        await client.connect()
        
        # Health check
        await client.health_check()
        
        # List models
        await client.list_models()
        
        # Single frame inference
        print("\n" + "=" * 80)
        print("🎬 SINGLE FRAME INFERENCE")
        print("=" * 80)
        
        for frame_id in [0, 10, 50, 100]:
            await client.generate_inference('sanders', frame_id, save_output=(frame_id == 0))
            await asyncio.sleep(0.1)
        
        # Batch inference
        print("\n" + "=" * 80)
        print("📦 BATCH INFERENCE")
        print("=" * 80)
        
        await client.generate_batch('sanders', [0, 10, 20, 30, 40, 50])
        
        # Streaming inference
        print("\n" + "=" * 80)
        print("🌊 STREAMING INFERENCE")
        print("=" * 80)
        
        await client.stream_inference('sanders', max_frames=20)
        
        # Get final stats
        print("\n" + "=" * 80)
        print("📊 FINAL STATISTICS")
        print("=" * 80)
        
        await client.get_stats('sanders')
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
