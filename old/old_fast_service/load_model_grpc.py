#!/usr/bin/env python3
"""
gRPC Model Loading Helper
Loads models into the gRPC service for testing.
"""

import grpc
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc

def load_model_grpc():
    """Load the test model via gRPC"""
    
    print("📦 Loading model via gRPC...")
    
    # Connect to gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    # Create load request
    request = lipsyncsrv_pb2.LoadModelRequest(
        model_name="test_optimized_package_fixed_3",
        package_path="../test_optimized_package_fixed.zip"
    )
    
    try:
        # Load model
        response = stub.LoadModel(request)
        
        if response.success:
            print(f"✅ {response.message}")
            print(f"⏱️  Initialization time: {response.initialization_time_ms}ms")
        else:
            print(f"❌ Failed to load model: {response.error}")
            
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    load_model_grpc()
