#!/usr/bin/env python3
"""
REST API Model Loader - Load models via FastAPI service
"""

import os
import requests
import json
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

def load_model_via_rest(model_name, package_path, base_url="http://localhost:8000"):
    """Load a model via REST API"""
    
    print(f"📦 Loading {model_name} via REST API...")
    
    payload = {
        "model_name": model_name,
        "package_path": package_path
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/models/load", json=payload)
        load_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result.get('message', 'Model loaded')}")
            print(f"   ⏱️  Load time: {load_time:.1f}ms")
            return True
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"   💥 Exception: {e}")
        return False

def get_model_status(base_url="http://localhost:8000"):
    """Get current model status"""
    try:
        response = requests.get(f"{base_url}/models/status")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get model status: {response.status_code}")
            return None
    except Exception as e:
        print(f"💥 Error getting model status: {e}")
        return None

def test_model_inference(model_name, base_url="http://localhost:8000"):
    """Test inference with a loaded model"""
    print(f"🧪 Testing inference with {model_name}...")
    
    try:
        response = requests.get(f"{base_url}/generate/{model_name}/0")
        if response.status_code == 200:
            size = len(response.content)
            print(f"   ✅ Inference successful - {size:,} bytes")
            return True
        else:
            print(f"   ❌ Inference failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   💥 Inference error: {e}")
        return False

def main():
    print("🌐 SyncTalk2D REST API Model Loader")
    print("=" * 45)
    
    # Check if service is running
    try:
        response = requests.get("http://localhost:8000/")
        print("✅ FastAPI service detected")
    except:
        print("❌ FastAPI service not running!")
        print("   Start with: python service.py")
        return
    
    choice = input("\nChoose action:\n1. List available models\n2. Load all models\n3. Load specific model\n4. Show model status\n5. Load + test model\n\nEnter choice (1-5): ")
    
    if choice == "1":
        list_available_models()
        
    elif choice == "2":
        models = list_available_models()
        if models:
            print(f"\n🚀 Loading {len(models)} models...")
            print("=" * 40)
            
            loaded = 0
            for model_name, package_path in models:
                if load_model_via_rest(model_name, package_path):
                    loaded += 1
                print()
            
            print(f"📊 Loaded {loaded}/{len(models)} models successfully")
            
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
                    load_model_via_rest(model_name, package_path)
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Invalid input!")
                
    elif choice == "4":
        status = get_model_status()
        if status:
            print("\n📊 Model Status:")
            print(json.dumps(status, indent=2))
            
    elif choice == "5":
        models = list_available_models()
        if models:
            print(f"\nSelect model to load + test:")
            for i, (name, path) in enumerate(models):
                print(f"{i+1}. {name}")
            
            try:
                idx = int(input(f"\nEnter number (1-{len(models)}): ")) - 1
                if 0 <= idx < len(models):
                    model_name, package_path = models[idx]
                    if load_model_via_rest(model_name, package_path):
                        print()
                        test_model_inference(model_name)
                else:
                    print("❌ Invalid selection!")
            except ValueError:
                print("❌ Invalid input!")
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
