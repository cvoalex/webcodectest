#!/usr/bin/env python3
"""
Quick gRPC Server Test
Test single frame inference to verify server works
"""

import asyncio
import cv2
import numpy as np

try:
    import grpc
    from grpc import aio
    import optimized_lipsyncsrv_pb2
    import optimized_lipsyncsrv_pb2_grpc
except ImportError as e:
    print(f"❌ Error: {e}")
    print("\nMake sure to:")
    print("1. Install: pip install grpcio grpcio-tools")
    print("2. Generate stubs: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto")
    import sys
    sys.exit(1)


async def test_server():
    """Quick test of gRPC server"""
    
    print("=" * 60)
    print("🧪 Quick gRPC Server Test")
    print("=" * 60)
    
    try:
        # Connect
        print("\n1️⃣ Connecting to localhost:50051...")
        channel = aio.insecure_channel(
            'localhost:50051',
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
        )
        stub = optimized_lipsyncsrv_pb2_grpc.OptimizedLipSyncServiceStub(channel)
        print("   ✅ Connected!")
        
        # Health check
        print("\n2️⃣ Health Check...")
        health_req = optimized_lipsyncsrv_pb2.HealthRequest()
        health_resp = await stub.HealthCheck(health_req)
        print(f"   Status: {health_resp.status}")
        print(f"   Healthy: {health_resp.healthy}")
        print(f"   Loaded models: {health_resp.loaded_models}")
        
        # Generate frame
        print("\n3️⃣ Generating frame 50...")
        request = optimized_lipsyncsrv_pb2.OptimizedInferenceRequest(
            model_name='sanders',
            frame_id=50
        )
        
        response = await stub.GenerateInference(request)
        
        if response.success:
            print(f"   ✅ Success!")
            print(f"   Processing time: {response.processing_time_ms:.1f}ms")
            print(f"   Inference time: {response.inference_time_ms:.1f}ms")
            print(f"   Image size: {len(response.prediction_data):,} bytes")
            print(f"   Shape: {response.prediction_shape}")
            
            # Decode and save
            nparr = np.frombuffer(response.prediction_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            output_path = "test_grpc_output.jpg"
            cv2.imwrite(output_path, img)
            print(f"   💾 Saved to: {output_path}")
            
            print(f"\n✅ ALL TESTS PASSED!")
            print(f"   Server is working correctly! 🎉")
            
        else:
            print(f"   ❌ Error: {response.error}")
        
        # Clean up
        await channel.close()
        
    except grpc.aio.AioRpcError as e:
        print(f"\n❌ gRPC Error: {e.code()}")
        print(f"   Details: {e.details()}")
        print(f"\n💡 Make sure the server is running:")
        print(f"   python optimized_grpc_server.py")
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = asyncio.run(test_server())
    exit(0 if success else 1)
