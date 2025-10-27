#!/usr/bin/env python3
"""
Load Multiple Models for Testing
Loads all 5 models into the gRPC service for multi-model testing.
"""

import grpc
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc
import time

def load_multiple_models():
    """Load all test models via gRPC"""
    
    print("📦 Loading Multiple Models via gRPC...")
    print("=" * 50)
    
    # Connect to gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    # Models to load
    models_to_load = [
        ("test_optimized_package_fixed_1", "test_optimized_package_fixed_1.zip"),
        ("test_optimized_package_fixed_2", "test_optimized_package_fixed_2.zip"),
        ("test_optimized_package_fixed_3", "../test_optimized_package_fixed.zip"),
        ("test_optimized_package_fixed_4", "test_optimized_package_fixed_4.zip"),
        ("test_optimized_package_fixed_5", "test_optimized_package_fixed_5.zip")
    ]
    
    loaded_models = []
    failed_models = []
    
    for model_name, package_path in models_to_load:
        print(f"📦 Loading {model_name}...")
        
        # Create load request
        request = lipsyncsrv_pb2.LoadModelRequest(
            model_name=model_name,
            package_path=package_path
        )
        
        try:
            # Load model
            start_time = time.time()
            response = stub.LoadModel(request)
            load_time = (time.time() - start_time) * 1000
            
            if response.success:
                print(f"   ✅ {response.message}")
                print(f"   ⏱️  Load time: {load_time:.1f}ms")
                print(f"   🔧 Initialization: {response.initialization_time_ms}ms")
                loaded_models.append(model_name)
            else:
                print(f"   ❌ Failed: {response.error}")
                failed_models.append(model_name)
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
            failed_models.append(model_name)
        
        # Brief pause between loads
        time.sleep(0.5)
    
    print(f"\n📊 Model Loading Summary:")
    print(f"   ✅ Successfully loaded: {len(loaded_models)}")
    print(f"   ❌ Failed to load: {len(failed_models)}")
    
    if loaded_models:
        print(f"   📦 Loaded models: {', '.join(loaded_models)}")
    
    if failed_models:
        print(f"   ❌ Failed models: {', '.join(failed_models)}")
    
    print(f"\n🚀 Ready for multi-model testing!")
    return loaded_models

if __name__ == "__main__":
    load_multiple_models()
