"""Quick test script for the new gRPC video API"""
import asyncio
import grpc
import optimized_lipsyncsrv_pb2
import optimized_lipsyncsrv_pb2_grpc

async def test_api():
    print("🧪 Testing gRPC Video API...")
    print("=" * 60)
    
    # Connect to server
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = optimized_lipsyncsrv_pb2_grpc.OptimizedLipSyncServiceStub(channel)
    
    try:
        # Test 1: ListModels
        print("\n1️⃣ Testing ListModels...")
        request = optimized_lipsyncsrv_pb2.ListModelsRequest()
        response = await stub.ListModels(request)
        print(f"✅ Success!")
        print(f"   Models: {list(response.loaded_models)}")
        print(f"   Count: {response.count}")
        
        if response.count == 0:
            print("❌ No models loaded! Make sure sanders is loaded on server.")
            return
        
        model_name = response.loaded_models[0]
        print(f"\n📦 Using model: {model_name}")
        
        # Test 2: GetModelMetadata
        print(f"\n2️⃣ Testing GetModelMetadata for '{model_name}'...")
        request = optimized_lipsyncsrv_pb2.GetModelMetadataRequest(
            model_name=model_name
        )
        response = await stub.GetModelMetadata(request)
        print(f"✅ Success!")
        print(f"   Frame count: {response.frame_count}")
        print(f"   Available videos: {list(response.available_videos)}")
        print(f"   Bounds: {list(response.bounds)}")
        
        if response.frame_count == 0:
            print("❌ No frames in model!")
            return
        
        # Test 3: GetVideoFrame
        print(f"\n3️⃣ Testing GetVideoFrame (frame 0, full_body)...")
        request = optimized_lipsyncsrv_pb2.GetVideoFrameRequest(
            model_name=model_name,
            frame_id=0,
            video_type="full_body"
        )
        response = await stub.GetVideoFrame(request)
        print(f"✅ Success!")
        print(f"   Frame data size: {len(response.frame_data)} bytes")
        print(f"   Frame ID: {response.frame_id}")
        print(f"   Video type: {response.video_type}")
        
        # Save frame to file for verification
        with open('test_frame_0.jpg', 'wb') as f:
            f.write(response.frame_data)
        print(f"   💾 Saved to test_frame_0.jpg")
        
        print("\n" + "=" * 60)
        print("🎉 All API tests passed!")
        print("=" * 60)
        
    except grpc.RpcError as e:
        print(f"\n❌ gRPC Error: {e.code()}")
        print(f"   Details: {e.details()}")
        print("\n💡 Make sure the gRPC server is running:")
        print("   cd D:\\Projects\\webcodecstest\\minimal_server")
        print("   .\\start_grpc_single.bat")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await channel.close()

if __name__ == "__main__":
    asyncio.run(test_api())
