#!/usr/bin/env python3
"""
Simple gRPC client script that can be called from Go
Calls the Python gRPC service and returns JSON response
"""

import sys
import json
import grpc
import asyncio
import base64

# Add the fast_service directory to the path
sys.path.append('../fast_service')

try:
    import lipsyncsrv_pb2
    import lipsyncsrv_pb2_grpc
except ImportError:
    print(json.dumps({"success": False, "error": "gRPC protobuf files not found"}))
    sys.exit(1)

def call_grpc_service(model_name, frame_id):
    """Call the Python gRPC service"""
    try:
        # Create gRPC channel
        channel = grpc.insecure_channel('localhost:50051')
        stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
        
        # Create request
        request = lipsyncsrv_pb2.InferenceRequest(
            model_name=model_name,
            frame_id=frame_id,
            audio_override=""  # Empty for now
        )
        
        # Make the call with timeout
        response = stub.GenerateInference(request, timeout=10.0)
        
        # Convert response to JSON-serializable format
        result = {
            "success": response.success,
            "prediction_data": base64.b64encode(response.prediction_data).decode('utf-8') if response.prediction_data else None,
            "bounds": list(response.bounds) if response.bounds else [],
            "processing_time_ms": response.processing_time_ms,
            "model_name": response.model_name,
            "frame_id": frame_id,
            "auto_loaded": response.auto_loaded,
            "prediction_shape": response.prediction_shape,
            "error": response.error if not response.success else None
        }
        
        return result
        
    except grpc.RpcError as e:
        return {
            "success": False,
            "error": f"gRPC error: {e.code()}: {e.details()}",
            "frame_id": frame_id,
            "model_name": model_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "frame_id": frame_id,
            "model_name": model_name
        }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"success": False, "error": "Usage: python grpc_client.py <model_name> <frame_id>"}))
        sys.exit(1)
    
    model_name = sys.argv[1]
    frame_id = int(sys.argv[2])
    
    result = call_grpc_service(model_name, frame_id)
    print(json.dumps(result))
