#!/usr/bin/env python3
"""
gRPC-Web Enabled Server for Direct Browser Access
Adds gRPC-Web support to eliminate Go middleman
"""

import asyncio
import time
import base64
import grpc
from concurrent import futures
import numpy as np

# Import existing gRPC code
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc
from multi_model_engine import multi_model_engine

# gRPC-Web imports
from grpc_web import grpc_web_server

class LipSyncServicer(lipsyncsrv_pb2_grpc.LipSyncServiceServicer):
    """Optimized gRPC servicer with direct browser support"""
    
    def __init__(self):
        self.request_count = 0
        self.total_time = 0.0
        print("🎬 LipSync gRPC Servicer initialized")
    
    def GenerateInference(self, request, context):
        """Single frame inference - maximum speed, direct browser access"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Run inference using our optimized engine
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
            
            processing_time = (time.time() - start_time) * 1000
            self.total_time += processing_time
            
            print(f"⚡ Direct gRPC Inference #{self.request_count}: {processing_time:.1f}ms (avg: {self.total_time/self.request_count:.1f}ms)")
            
            # Convert prediction to bytes if needed
            if isinstance(prediction, np.ndarray):
                import cv2
                _, buffer = cv2.imencode('.jpg', prediction)
                prediction_bytes = buffer.tobytes()
            else:
                prediction_bytes = prediction
            
            response = lipsyncsrv_pb2.InferenceResponse(
                success=True,
                model_name=request.model_name,
                prediction_data=prediction_bytes,
                bounds=bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
                prediction_shape=str(prediction.shape) if hasattr(prediction, 'shape') else "",
                processing_time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            print(f"❌ gRPC Error: {e}")
            return lipsyncsrv_pb2.InferenceResponse(
                success=False,
                error=str(e),
                model_name=request.model_name
            )

async def main():
    """Main entry point with gRPC-Web support"""
    print("🚀 Starting gRPC-Web Enabled Lip Sync Server...")
    
    # Load default model
    print("📦 Loading default model...")
    await multi_model_engine.load_model("default_model", "/path/to/default/model")
    print("✅ Default model loaded successfully")
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = LipSyncServicer()
    lipsyncsrv_pb2_grpc.add_LipSyncServiceServicer_to_server(servicer, server)
    
    # Add both regular gRPC and gRPC-Web support
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    # Enable gRPC-Web support for direct browser access
    grpc_web_server.add_grpc_web_support(server)
    
    print("🔥 gRPC Server with Web Support running on port 50051")
    print("   - Regular gRPC: localhost:50051")
    print("   - gRPC-Web: http://localhost:50051 (for browsers)")
    print("   - CORS enabled for all origins")
    
    server.start()
    
    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.stop(0)

if __name__ == '__main__':
    asyncio.run(main())
