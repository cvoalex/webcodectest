#!/usr/bin/env python3
"""
🚀 Ultra-Optimized gRPC Server for Server-to-Server Communication
Maximum performance with gRPC for production deployments.

Features:
- gRPC for high-performance RPC
- HTTP/2 multiplexing
- Binary serialization (Protocol Buffers)
- Streaming support for real-time
- Lower latency than WebSockets
- Better CPU efficiency
"""

import sys
import os
import asyncio
import time
import cv2
import numpy as np
from concurrent import futures
from typing import AsyncIterator

# gRPC imports
import grpc
from grpc import aio

# Import optimized engine
from optimized_inference_engine import optimized_engine

# Generate Python code from proto:
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto

try:
    import optimized_lipsyncsrv_pb2
    import optimized_lipsyncsrv_pb2_grpc
except ImportError:
    print("❌ Error: gRPC stubs not found!")
    print("Generate them with:")
    print("  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto")
    print("\nInstall grpcio-tools if needed:")
    print("  pip install grpcio grpcio-tools")
    sys.exit(1)


class OptimizedLipSyncServicer(optimized_lipsyncsrv_pb2_grpc.OptimizedLipSyncServiceServicer):
    """Optimized Lip Sync gRPC servicer"""
    
    def __init__(self):
        self.request_count = 0
        self.start_time = time.time()
        print("🚀 Optimized LipSync gRPC Servicer initialized")
    
    async def GenerateInference(
        self, 
        request: optimized_lipsyncsrv_pb2.OptimizedInferenceRequest,
        context: grpc.aio.ServicerContext
    ) -> optimized_lipsyncsrv_pb2.OptimizedInferenceResponse:
        """Generate single frame inference"""
        
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Validate model is loaded
            models = optimized_engine.list_models()
            if request.model_name not in models.get("loaded_models", []):
                return optimized_lipsyncsrv_pb2.OptimizedInferenceResponse(
                    success=False,
                    error=f"Model '{request.model_name}' not loaded. Available: {models.get('loaded_models', [])}",
                    frame_id=request.frame_id,
                    model_name=request.model_name,
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Generate inference
            prediction, bounds, metadata = await optimized_engine.generate_inference_only(
                request.model_name,
                request.frame_id
            )
            
            # Encode prediction to JPEG
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
            _, buffer = cv2.imencode('.jpg', prediction, encode_param)
            prediction_bytes = buffer.tobytes()
            
            processing_time = (time.time() - start_time) * 1000
            
            print(f"✅ gRPC inference: model={request.model_name}, frame={request.frame_id}, time={processing_time:.1f}ms")
            
            # Build response
            response = optimized_lipsyncsrv_pb2.OptimizedInferenceResponse(
                success=True,
                prediction_data=prediction_bytes,
                bounds=bounds.tolist() if hasattr(bounds, 'tolist') else list(bounds),
                processing_time_ms=int(processing_time),
                model_name=request.model_name,
                frame_id=request.frame_id,
                prediction_shape=f"{prediction.shape[1]}x{prediction.shape[0]}x{prediction.shape[2]}",
                prepare_time_ms=metadata.get('prepare_time_ms', 0),
                inference_time_ms=metadata.get('inference_time_ms', 0),
                composite_time_ms=metadata.get('composite_time_ms', 0),
                optimizations=[
                    "pre_loaded_videos",
                    "memory_mapped_audio",
                    "cached_metadata",
                    "zero_copy_access",
                    "grpc_protocol"
                ]
            )
            
            return response
            
        except Exception as e:
            print(f"❌ Inference error: {e}")
            import traceback
            traceback.print_exc()
            
            return optimized_lipsyncsrv_pb2.OptimizedInferenceResponse(
                success=False,
                error=str(e),
                frame_id=request.frame_id,
                model_name=request.model_name,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def GenerateBatchInference(
        self,
        request: optimized_lipsyncsrv_pb2.BatchInferenceRequest,
        context: grpc.aio.ServicerContext
    ) -> optimized_lipsyncsrv_pb2.BatchInferenceResponse:
        """Generate batch inference for multiple frames"""
        
        start_time = time.time()
        responses = []
        
        print(f"📦 Batch inference: model={request.model_name}, frames={len(request.frame_ids)}")
        
        # Process each frame
        for frame_id in request.frame_ids:
            inference_request = optimized_lipsyncsrv_pb2.OptimizedInferenceRequest(
                model_name=request.model_name,
                frame_id=frame_id
            )
            
            response = await self.GenerateInference(inference_request, context)
            responses.append(response)
        
        total_time = int((time.time() - start_time) * 1000)
        avg_time = total_time / len(request.frame_ids) if request.frame_ids else 0
        
        print(f"✅ Batch complete: {len(request.frame_ids)} frames in {total_time}ms (avg: {avg_time:.1f}ms/frame)")
        
        return optimized_lipsyncsrv_pb2.BatchInferenceResponse(
            responses=responses,
            total_processing_time_ms=total_time,
            avg_frame_time_ms=avg_time
        )
    
    async def StreamInference(
        self,
        request_iterator: AsyncIterator[optimized_lipsyncsrv_pb2.OptimizedInferenceRequest],
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[optimized_lipsyncsrv_pb2.OptimizedInferenceResponse]:
        """Streaming inference for real-time applications (50+ FPS capable)"""
        
        print("🌊 Stream inference started")
        
        async for request in request_iterator:
            response = await self.GenerateInference(request, context)
            yield response
    
    async def LoadPackage(
        self,
        request: optimized_lipsyncsrv_pb2.LoadPackageRequest,
        context: grpc.aio.ServicerContext
    ) -> optimized_lipsyncsrv_pb2.LoadPackageResponse:
        """Load optimized model package"""
        
        print(f"📦 Loading package: {request.package_name} from {request.package_dir}")
        
        result = await optimized_engine.load_package(
            request.package_name,
            request.package_dir
        )
        
        if result["status"] == "success":
            return optimized_lipsyncsrv_pb2.LoadPackageResponse(
                success=True,
                package_name=request.package_name,
                message="Package loaded successfully",
                initialization_time_ms=int(result.get("initialization_time_s", 0) * 1000),
                frame_count=result.get("frame_count", 0),
                device=result.get("device", "unknown"),
                videos_loaded=result.get("videos_loaded", []),
                audio_features_shape=result.get("audio_features_shape", []),
                memory_mapped_audio=result.get("memory_mapped_audio", False)
            )
        else:
            return optimized_lipsyncsrv_pb2.LoadPackageResponse(
                success=False,
                package_name=request.package_name,
                message="Failed to load package",
                error=result.get("error", "Unknown error")
            )
    
    async def GetStats(
        self,
        request: optimized_lipsyncsrv_pb2.StatsRequest,
        context: grpc.aio.ServicerContext
    ) -> optimized_lipsyncsrv_pb2.StatsResponse:
        """Get model statistics"""
        
        try:
            stats = optimized_engine.get_model_stats(request.model_name)
            
            return optimized_lipsyncsrv_pb2.StatsResponse(
                model_name=stats["model_name"],
                total_requests=stats["total_requests"],
                avg_inference_time_ms=stats["avg_inference_time_ms"],
                min_inference_time_ms=stats["min_inference_time_ms"],
                max_inference_time_ms=stats["max_inference_time_ms"],
                frame_count=stats["frame_count"],
                device=stats["device"],
                optimizations_active=stats["optimizations_active"]
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return optimized_lipsyncsrv_pb2.StatsResponse()
    
    async def ListModels(
        self,
        request: optimized_lipsyncsrv_pb2.ListModelsRequest,
        context: grpc.aio.ServicerContext
    ) -> optimized_lipsyncsrv_pb2.ListModelsResponse:
        """List loaded models"""
        
        models = optimized_engine.list_models()
        
        return optimized_lipsyncsrv_pb2.ListModelsResponse(
            loaded_models=models.get("loaded_models", []),
            count=models.get("count", 0)
        )
    
    async def HealthCheck(
        self,
        request: optimized_lipsyncsrv_pb2.HealthRequest,
        context: grpc.aio.ServicerContext
    ) -> optimized_lipsyncsrv_pb2.HealthResponse:
        """Health check endpoint"""
        
        models = optimized_engine.list_models()
        uptime = int(time.time() - self.start_time)
        
        return optimized_lipsyncsrv_pb2.HealthResponse(
            healthy=True,
            status="running",
            loaded_models=models.get("count", 0),
            uptime_seconds=uptime
        )
    
    async def GetVideoFrame(
        self,
        request: optimized_lipsyncsrv_pb2.GetVideoFrameRequest,
        context: grpc.aio.ServicerContext
    ) -> optimized_lipsyncsrv_pb2.GetVideoFrameResponse:
        """Get a specific video frame"""
        
        try:
            frame_data = await optimized_engine.get_video_frame(
                request.model_name,
                request.frame_id,
                request.video_type
            )
            
            return optimized_lipsyncsrv_pb2.GetVideoFrameResponse(
                success=True,
                frame_data=frame_data,
                frame_id=request.frame_id,
                video_type=request.video_type
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return optimized_lipsyncsrv_pb2.GetVideoFrameResponse(
                success=False,
                frame_id=request.frame_id,
                video_type=request.video_type,
                error=str(e)
            )
    
    async def GetModelMetadata(
        self,
        request: optimized_lipsyncsrv_pb2.GetModelMetadataRequest,
        context: grpc.aio.ServicerContext
    ) -> optimized_lipsyncsrv_pb2.GetModelMetadataResponse:
        """Get model metadata including available videos and frame count"""
        
        try:
            metadata = await optimized_engine.get_model_metadata(request.model_name)
            
            return optimized_lipsyncsrv_pb2.GetModelMetadataResponse(
                success=True,
                model_name=request.model_name,
                frame_count=metadata["frame_count"],
                available_videos=metadata["available_videos"],
                audio_path=metadata.get("audio_path", ""),
                bounds=metadata.get("bounds", [])
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return optimized_lipsyncsrv_pb2.GetModelMetadataResponse(
                success=False,
                model_name=request.model_name,
                frame_count=0,
                available_videos=[],
                audio_path="",
                error=str(e)
            )


async def serve():
    """Start the gRPC server"""
    
    print("=" * 80)
    print("🚀 STARTING ULTRA-OPTIMIZED gRPC SERVER")
    print("=" * 80)
    print("   ⚡ gRPC with HTTP/2")
    print("   ⚡ Protocol Buffers serialization")
    print("   ⚡ Pre-loaded videos in RAM")
    print("   ⚡ Memory-mapped audio features")
    print("   ⚡ Zero I/O overhead")
    print("=" * 80)
    
    # Auto-load optimized default_model
    print("\n📦 Loading optimized default_model...")
    try:
        result = await optimized_engine.load_package(
            "default_model",
            "models/default_model"
        )
        
        if result["status"] == "success":
            print("\n✅ default_model loaded successfully!")
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
        print(f"❌ Error loading default_model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create gRPC server
    server = aio.server(
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
    
    # Bind to port
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    print("\n" + "=" * 80)
    print(f"⚡ OPTIMIZED gRPC Server running on {listen_addr}")
    print("=" * 80)
    print("   🚀 Protocol: gRPC/HTTP2")
    print("   🚀 Serialization: Protocol Buffers")
    print("   🚀 Pre-loaded data: Zero I/O")
    print("   🚀 Streaming: Supported")
    print("=" * 80)
    print("\n💡 Press Ctrl+C to stop the server")
    print("\n📝 Available services:")
    print("   - GenerateInference: Single frame inference")
    print("   - GenerateBatchInference: Batch processing")
    print("   - StreamInference: Real-time streaming")
    print("   - LoadPackage: Load model packages")
    print("   - GetStats: Performance statistics")
    print("   - ListModels: List loaded models")
    print("   - HealthCheck: Server health")
    
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down gRPC server...")
        await server.stop(grace=5)
        print("✅ Server stopped gracefully")


def main():
    """Entry point"""
    asyncio.run(serve())


if __name__ == '__main__':
    main()
