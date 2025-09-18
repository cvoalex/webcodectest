#!/usr/bin/env python3
"""
BACKUP - Current working gRPC server
This is the original working version before WebSocket optimization
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
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.total_time += processing_time
            
            print(f"⚡ gRPC Inference #{self.request_count}: {processing_time:.1f}ms (avg: {self.total_time/self.request_count:.1f}ms)")
            
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

    def GenerateBatchInference(self, request, context):
        """Batch inference for multiple frames"""
        start_time = time.time()
        responses = []
        
        try:
            # Run batch inference using our optimized engine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                batch_results = loop.run_until_complete(
                    multi_model_engine.generate_batch_inference(
                        request.model_name,
                        request.frame_ids,
                        request.audio_data
                    )
                )
            finally:
                loop.close()
            
            # Convert results to responses
            for result in batch_results:
                prediction, bounds, metadata = result
                
                if isinstance(prediction, np.ndarray):
                    _, buffer = cv2.imencode('.jpg', prediction)
                    prediction_bytes = buffer.tobytes()
                else:
                    prediction_bytes = prediction
                
                response = lipsyncsrv_pb2.InferenceResponse(
                    success=True,
                    model_name=request.model_name,
                    prediction_data=prediction_bytes,
                    bounds=bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
                    prediction_shape=str(prediction.shape) if hasattr(prediction, 'shape') else ""
                )
                responses.append(response)
            
            processing_time = (time.time() - start_time) * 1000
            print(f"⚡ Batch gRPC Inference: {len(request.frame_ids)} frames in {processing_time:.1f}ms")
            
            return lipsyncsrv_pb2.BatchInferenceResponse(responses=responses)
            
        except Exception as e:
            print(f"❌ Batch gRPC Error: {e}")
            error_response = lipsyncsrv_pb2.InferenceResponse(
                success=False,
                error=str(e),
                model_name=request.model_name
            )
            return lipsyncsrv_pb2.BatchInferenceResponse(responses=[error_response])

def serve():
    """Start the gRPC server"""
    print("🚀 Starting High-Performance gRPC Lip Sync Server...")
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = LipSyncServicer()
    lipsyncsrv_pb2_grpc.add_LipSyncServiceServicer_to_server(servicer, server)
    
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    
    print(f"🔥 gRPC Server listening on {listen_addr}")
    server.start()
    
    try:
        while True:
            time.sleep(86400)  # Keep server running
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gRPC server...")
        server.stop(0)

async def main():
    """Main entry point"""
    # Load default model at startup
    print("📦 Loading default model...")
    await multi_model_engine.load_model("default_model", "/path/to/default/model")
    print("✅ Default model loaded successfully")
    
    # Start the server
    serve()

if __name__ == '__main__':
    asyncio.run(main())
