"""
Start the optimized gRPC server on port 50052 for testing
"""

import asyncio
import grpc
from concurrent import futures
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimized_lipsyncsrv_pb2
import optimized_lipsyncsrv_pb2_grpc
from optimized_grpc_server import OptimizedLipSyncServicer
from optimized_inference_engine import optimized_engine

async def serve():
    """Start the gRPC server on port 50052"""
    
    print("=" * 80)
    print("🚀 STARTING OPTIMIZED gRPC SERVER ON PORT 50052")
    print("=" * 80)
    print("   ⚡ gRPC with HTTP/2")
    print("   ⚡ Protocol Buffers serialization")
    print("   ⚡ Pre-loaded videos in RAM")
    print("   ⚡ Memory-mapped audio features")
    print("   ⚡ Zero I/O overhead")
    print("   ⚡ GPU Batch Processing")
    print("   ⚡ Audio Batch Optimization")
    print("=" * 80)
    
    # Auto-load optimized sanders model
    print("\n📦 Loading optimized sanders...")
    try:
        result = await optimized_engine.load_package(
            "sanders",
            "models/sanders"  # Path relative to minimal_server directory
        )
        
        if result["status"] == "success":
            print("\n✅ sanders loaded successfully!")
            print(f"   Frame count: {result['frame_count']}")
            print(f"   Initialization time: {result['initialization_time_s']:.2f}s")
            print(f"   Device: {result['device']}")
            print(f"   Videos loaded: {', '.join(result['videos_loaded'])}")
            print(f"   Audio features shape: {result['audio_features_shape']}")
            print(f"   Memory-mapped audio: {result['memory_mapped_audio']}")
        else:
            print(f"⚠️ Model load failed: {result.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        print(f"❌ Error loading sanders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.so_reuseport', 1),
            ('grpc.use_local_subchannel_pool', 1),
        ]
    )
    
    # Add servicer
    optimized_lipsyncsrv_pb2_grpc.add_OptimizedLipSyncServiceServicer_to_server(
        OptimizedLipSyncServicer(),
        server
    )
    
    # Bind to port 50052
    listen_addr = '[::]:50052'
    server.add_insecure_port(listen_addr)
    
    print("\n" + "=" * 80)
    print(f"✅ Server listening on {listen_addr}")
    print("=" * 80)
    print("\n🎯 Available RPC Methods:")
    print("   • GenerateFrame - Single frame generation")
    print("   • GenerateBatchInference - GPU batch processing")
    print("   • GenerateBatchWithAudio - Audio batch optimization")
    print("   • LoadModel - Dynamic model loading")
    print("   • ListModels - List loaded models")
    print("   • HealthCheck - Server health status")
    print("=" * 80)
    print("\n📊 Press Ctrl+C to stop the server\n")
    
    # Start server
    await server.start()
    
    # Wait for termination
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        await server.stop(grace=5)
        print("✅ Server stopped gracefully")

if __name__ == '__main__':
    asyncio.run(serve())
