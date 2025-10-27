#!/usr/bin/env python3
"""
High-Performance gRPC Server for Lip Sync Inference
Designed for maximum speed with binary streaming.
"""

import asyncio
import time
import cv2
import numpy as np
import grpc
from concurrent import futures
import base64

# Import generated gRPC code
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc

# Import reflection library
from grpc_reflection.v1alpha import reflection

# Import our existing engine
from multi_model_engine import multi_model_engine

class LipSyncServicer(lipsyncsrv_pb2_grpc.LipSyncServiceServicer):
    """High-performance gRPC servicer for lip sync inference"""
    
    def __init__(self):
        self.request_count = 0
        self.total_time = 0.0
    
    def GenerateInference(self, request, context):
        """Single frame inference with maximum speed"""
        
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Run inference using our optimized engine
            # Note: We need to run this in an event loop since our engine is async
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                prediction, bounds, metadata = loop.run_until_complete(
                    multi_model_engine.generate_inference_only(
                        request.model_name, 
                        request.frame_id, 
                        request.audio_override if request.audio_override else None
                    )
                )
            finally:
                loop.close()
            
            # Encode prediction to JPEG for smaller size
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
            _, prediction_bytes = cv2.imencode('.jpg', prediction, encode_param)
            
            processing_time = (time.time() - start_time) * 1000
            self.total_time += processing_time
            
            # Create response
            response = lipsyncsrv_pb2.InferenceResponse(
                success=True,
                prediction_data=prediction_bytes.tobytes(),
                bounds=bounds.tolist(),
                processing_time_ms=int(processing_time),
                model_name=request.model_name,
                frame_id=request.frame_id,
                auto_loaded=metadata.get("auto_loaded", False),
                prediction_shape=f"{prediction.shape[0]}x{prediction.shape[1]}x{prediction.shape[2]}"
            )
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            response = lipsyncsrv_pb2.InferenceResponse(
                success=False,
                error=str(e),
                processing_time_ms=int(processing_time),
                model_name=request.model_name,
                frame_id=request.frame_id
            )
            
            return response
    
    def GenerateBatchInference(self, request, context):
        """Batch inference for multiple frames"""
        
        start_time = time.time()
        responses = []
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                for frame_id in request.frame_ids:
                    try:
                        prediction, bounds, metadata = loop.run_until_complete(
                            multi_model_engine.generate_inference_only(
                                request.model_name, 
                                frame_id, 
                                request.audio_override if request.audio_override else None
                            )
                        )
                        
                        # Encode prediction to JPEG
                        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
                        _, prediction_bytes = cv2.imencode('.jpg', prediction, encode_param)
                        
                        response = lipsyncsrv_pb2.InferenceResponse(
                            success=True,
                            prediction_data=prediction_bytes.tobytes(),
                            bounds=bounds.tolist(),
                            processing_time_ms=int(metadata["processing_time_ms"]),
                            model_name=request.model_name,
                            frame_id=frame_id,
                            auto_loaded=metadata.get("auto_loaded", False),
                            prediction_shape=f"{prediction.shape[0]}x{prediction.shape[1]}x{prediction.shape[2]}"
                        )
                        
                        responses.append(response)
                        
                    except Exception as e:
                        error_response = lipsyncsrv_pb2.InferenceResponse(
                            success=False,
                            error=str(e),
                            model_name=request.model_name,
                            frame_id=frame_id
                        )
                        responses.append(error_response)
            finally:
                loop.close()
            
            total_processing_time = (time.time() - start_time) * 1000
            
            batch_response = lipsyncsrv_pb2.BatchInferenceResponse(
                responses=responses,
                total_processing_time_ms=int(total_processing_time)
            )
            
            return batch_response
            
        except Exception as e:
            # Return error for entire batch
            error_response = lipsyncsrv_pb2.InferenceResponse(
                success=False,
                error=str(e),
                model_name=request.model_name
            )
            
            batch_response = lipsyncsrv_pb2.BatchInferenceResponse(
                responses=[error_response],
                total_processing_time_ms=int((time.time() - start_time) * 1000)
            )
            
            return batch_response
    
    def StreamInference(self, request_iterator, context):
        """High-speed streaming inference"""
        
        for request in request_iterator:
            try:
                # Process each request as it comes in
                response = self.GenerateInference(request, context)
                yield response
                
            except Exception as e:
                error_response = lipsyncsrv_pb2.InferenceResponse(
                    success=False,
                    error=str(e),
                    model_name=request.model_name if request else "unknown",
                    frame_id=request.frame_id if request else -1
                )
                yield error_response
    
    def LoadModel(self, request, context):
        """Load a model"""
        
        start_time = time.time()
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    multi_model_engine.load_model(
                        request.model_name,
                        request.package_path,
                        request.audio_override if request.audio_override else None
                    )
                )
            finally:
                loop.close()
            
            processing_time = (time.time() - start_time) * 1000
            
            if result["status"] == "loaded" or result["status"] == "already_loaded":
                response = lipsyncsrv_pb2.LoadModelResponse(
                    success=True,
                    model_name=request.model_name,
                    message=f"Model {request.model_name} loaded successfully",
                    initialization_time_ms=int(processing_time)
                )
            else:
                response = lipsyncsrv_pb2.LoadModelResponse(
                    success=False,
                    model_name=request.model_name,
                    error=result.get("error", "Unknown error"),
                    initialization_time_ms=int(processing_time)
                )
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            response = lipsyncsrv_pb2.LoadModelResponse(
                success=False,
                model_name=request.model_name,
                error=str(e),
                initialization_time_ms=int(processing_time)
            )
            
            return response

def serve():
    """Start the high-performance gRPC server"""
    
    # Configure server for maximum performance
    options = [
        ('grpc.keepalive_time_ms', 30000),  # Send keepalive ping every 30 seconds
        ('grpc.keepalive_timeout_ms', 5000),  # Wait 5 seconds for ping ack
        ('grpc.keepalive_permit_without_calls', True),  # Allow keepalive without calls
        ('grpc.http2.max_pings_without_data', 0),  # Allow unlimited pings
        ('grpc.http2.min_time_between_pings_ms', 10000),  # Min time between pings
        ('grpc.http2.min_ping_interval_without_data_ms', 300000),  # Min ping interval
        ('grpc.max_receive_message_length', 16 * 1024 * 1024),  # 16MB max message
        ('grpc.max_send_message_length', 16 * 1024 * 1024),  # 16MB max message
    ]
    
    # Use thread pool for maximum concurrency
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=options
    )
    
    # Add our servicer
    lipsyncsrv_pb2_grpc.add_LipSyncServiceServicer_to_server(
        LipSyncServicer(), server
    )
    
    # Enable server reflection
    SERVICE_NAMES = (
        lipsyncsrv_pb2.DESCRIPTOR.services_by_name['LipSyncService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Listen on port 50051 (standard gRPC port)
    server.add_insecure_port('[::]:50051')
    
    print("🚀 Starting High-Performance gRPC Lip Sync Server...")
    print("📡 Listening on port 50051")
    print("💡 Server reflection enabled")
    print("⚡ Optimized for maximum speed with binary streaming")
    
    server.start()
    
    # Auto-load sanders model on startup
    print("🔄 Auto-loading sanders model...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                multi_model_engine.load_model(
                    "sanders",
                    "models/sanders",
                    None
                )
            )
            
            if result["status"] == "loaded" or result["status"] == "already_loaded":
                print("✅ Sanders model loaded successfully")
            else:
                print(f"❌ Failed to load sanders model: {result.get('error', 'Unknown error')}")
                
        finally:
            loop.close()
    except Exception as e:
        print(f"❌ Error loading sanders model: {e}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gRPC server...")
        server.stop(0)

if __name__ == '__main__':
    serve()
