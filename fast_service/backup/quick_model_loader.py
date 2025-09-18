#!/usr/bin/env python3
"""
Quick Model Loader - List and load available models
"""

import os
import grpc
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc
import time

def list_available_models():
    """List all available model files"""
    print("🔍 Available Model Files:")
    print("=" * 40)
    
    models = []
    
    # Check fast_service directory
    for i in range(1, 6):
        model_file = f"test_optimized_package_fixed_{i}.zip"
        if os.path.exists(model_file):
            models.append((f"test_optimized_package_fixed_{i}", model_file))
            print(f"   ✅ {model_file}")
    
    # Check parent directory  
    parent_models = [
        ("test_optimized_package_fixed", "../test_optimized_package_fixed.zip"),
        ("test_optimized_package", "../test_optimized_package.zip"),
        ("test_optimized_package_with_model", "../test_optimized_package_with_model.zip")
    ]
    
    for model_name, path in parent_models:
        if os.path.exists(path):
            models.append((model_name, path))
            print(f"   ✅ {path}")
    
    print(f"\n📊 Total available models: {len(models)}")
    return models

def load_single_model(model_name, package_path):
    """Load a single model via gRPC"""
    try:
        # Connect to gRPC server
        channel = grpc.insecure_channel('localhost:50051')
        stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
        
        print(f"📦 Loading {model_name}...")
        
        # Create load request
        request = lipsyncsrv_pb2.LoadModelRequest(
            model_name=model_name,
            package_path=package_path
        )
        
        # Load model
        start_time = time.time()
        response = stub.LoadModel(request)
        load_time = (time.time() - start_time) * 1000
        
        if response.success:
            print(f"   ✅ Success: {response.message}")
            print(f"   ⏱️  Load time: {load_time:.1f}ms")
            return True
        else:
            print(f"   ❌ Failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"   💥 Exception: {e}")
        return False

def load_all_models():
    """Load all available models"""
    models = list_available_models()
    
    if not models:
        print("❌ No models found!")
        return
    
    print(f"\n🚀 Loading {len(models)} models...")
    print("=" * 40)
    
    loaded = 0
    failed = 0
    
    for model_name, package_path in models:
        if load_single_model(model_name, package_path):
            loaded += 1
        else:
            failed += 1
        print()
    
    print("📊 Loading Summary:")
    print(f"   ✅ Loaded: {loaded}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success rate: {(loaded/(loaded+failed)*100):.1f}%")

def main():
    print("🤖 SyncTalk2D Model Loader")
    print("=" * 40)
    
    choice = input("\nChoose action:\n1. List available models\n2. Load all models\n3. Load specific model\n\nEnter choice (1-3): ")
    
    if choice == "1":
        list_available_models()
    elif choice == "2":
        load_all_models()
    elif choice == "3":
        models = list_available_models()
        if models:
            print(f"\nSelect model to load:")
            for i, (name, path) in enumerate(models):
                print(f"{i+1}. {name}")
            
            try:
                idx = int(input(f"\nEnter number (1-{len(models)}): ")) - 1
                if 0 <= idx < len(models):
                    model_name, package_path = models[idx]
                    load_single_model(model_name, package_path)
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Invalid input!")
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
