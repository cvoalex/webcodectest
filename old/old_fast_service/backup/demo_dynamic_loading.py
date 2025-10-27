"""
Dynamic Model Loading Demonstration Script

This script demonstrates the new dynamic model loading system:

1. Model Storage: Models are stored in models/ subdirectory
2. Auto-Loading: When you request a frame from model "mymodel_17", the system:
   - Checks if "mymodel" is loaded in memory
   - If not, checks models/mymodel/ directory
   - If not extracted, checks models/mymodel.zip
   - If not found locally, checks central registry and downloads
   - Extracts and loads the model automatically
   - Generates the requested frame

3. Naming Convention: "modelname_framenumber,<audio>" format supported
   - modelname_17 -> loads "modelname" and generates frame 17
   - audio override can be provided per request

Usage Examples:
   POST /generate/frame {"model_name": "default_model", "frame_id": 17}
   POST /generate/frame {"model_name": "enhanced_model", "frame_id": 42}
   
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fast_service.dynamic_model_manager import dynamic_model_manager
from fast_service.multi_model_engine import multi_model_engine


async def demonstrate_dynamic_loading():
    """Demonstrate the dynamic model loading process"""
    
    print("🎬 Dynamic Model Loading System Demonstration")
    print("=" * 60)
    
    # 1. Show models directory structure
    print("\n📁 Models Directory Structure:")
    models_dir = Path("models")
    
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_file() and item.suffix == ".zip":
                print(f"  📦 {item.name} ({item.stat().st_size / 1024 / 1024:.1f} MB)")
            elif item.is_dir():
                print(f"  📂 {item.name}/ (extracted)")
                
                # Show contents of extracted model
                required_files = ["video.mp4", "aud_ave.npy", "models/99.pth"]  # Updated to match actual structure
                for req_file in required_files:
                    file_path = item / req_file
                    if file_path.exists():
                        print(f"     ✅ {req_file}")
                    else:
                        print(f"     ❌ {req_file} (missing)")
    else:
        print("  📂 models/ directory not found")
    
    # 2. List local models
    print("\n📋 Local Models Status:")
    local_models = dynamic_model_manager.list_local_models()
    
    print(f"  Extracted: {len(local_models['extracted'])}")
    for model in local_models['extracted']:
        print(f"    ✅ {model['name']} -> {model['status']}")
    
    print(f"  Zipped: {len(local_models['zipped'])}")
    for model in local_models['zipped']:
        print(f"    📦 {model['name']} -> {model['status']}")
    
    # 3. Demonstrate auto-loading process
    print("\n🚀 Auto-Loading Process Demonstration:")
    
    test_models = ["default_model", "enhanced_model", "fast_model"]
    
    for model_name in test_models:
        print(f"\n🔍 Testing model: {model_name}")
        
        # Check if model is available
        print(f"   Checking availability...")
        result = await dynamic_model_manager.ensure_model_available(model_name)
        
        if result["success"]:
            print(f"   ✅ Model available: {result['model_path']}")
            print(f"   Actions taken: {result['actions_taken']}")
        else:
            print(f"   ❌ Model not available: {result['error']}")
            print(f"   Actions attempted: {result['actions_taken']}")
    
    # 4. Show how the system handles frame requests
    print("\n🎬 Frame Request Simulation:")
    
    test_cases = [
        {"model": "default_model", "frame": 17, "desc": "default_model_17"},
        {"model": "enhanced_model", "frame": 42, "desc": "enhanced_model_42 (will auto-download)"},
        {"model": "nonexistent_model", "frame": 5, "desc": "nonexistent_model_5 (will fail)"}
    ]
    
    for case in test_cases:
        print(f"\n📝 Simulating request: {case['desc']}")
        print(f"   Model: {case['model']}, Frame: {case['frame']}")
        
        # Check if model would be loaded
        is_loaded = multi_model_engine.is_model_loaded(case['model'])
        print(f"   Currently loaded: {is_loaded}")
        
        if not is_loaded:
            # Simulate the auto-loading process
            availability = await dynamic_model_manager.ensure_model_available(case['model'])
            
            if availability["success"]:
                print(f"   🔄 Would auto-load from: {availability['model_path']}")
                print(f"   📋 Actions: {availability['actions_taken']}")
            else:
                print(f"   ❌ Auto-load would fail: {availability['error']}")
    
    print("\n📖 System Features Summary:")
    print("  ✅ Automatic model downloading from registry")
    print("  ✅ Automatic extraction of model packages")
    print("  ✅ On-demand loading when frame requested")
    print("  ✅ Model name validation and error handling")
    print("  ✅ Support for audio override per request")
    print("  ✅ Efficient caching and memory management")
    
    print("\n🌐 Mock Registry Models:")
    registry_models = [
        {"name": "default_model", "version": "1.0.0", "size": "86.2 MB"},
        {"name": "enhanced_model", "version": "1.1.0", "size": "120.5 MB"},
        {"name": "fast_model", "version": "1.0.1", "size": "65.8 MB"}
    ]
    
    for model in registry_models:
        print(f"  🌍 {model['name']} v{model['version']} ({model['size']})")


def print_api_usage_examples():
    """Print API usage examples"""
    
    print("\n🔗 API Usage Examples:")
    print("=" * 40)
    
    examples = [
        {
            "title": "Basic Frame Generation (Auto-Load)",
            "method": "POST",
            "endpoint": "/generate/frame",
            "payload": {
                "model_name": "default_model",
                "frame_id": 17
            },
            "description": "System will auto-download and load 'default_model' if needed"
        },
        {
            "title": "Frame with Audio Override",
            "method": "POST", 
            "endpoint": "/generate/frame",
            "payload": {
                "model_name": "enhanced_model",
                "frame_id": 42,
                "audio_override": "<base64_encoded_audio>"
            },
            "description": "Uses custom audio instead of model's default"
        },
        {
            "title": "List Available Models",
            "method": "GET",
            "endpoint": "/models",
            "description": "Shows loaded models and locally available models"
        },
        {
            "title": "List Registry Models",
            "method": "GET",
            "endpoint": "/models/registry", 
            "description": "Shows models available for download"
        },
        {
            "title": "Manual Download",
            "method": "POST",
            "endpoint": "/models/download?model_name=fast_model",
            "description": "Manually download and extract a model"
        },
        {
            "title": "Batch Generation",
            "method": "POST",
            "endpoint": "/generate/batch",
            "payload": {
                "model_name": "default_model",
                "frame_ids": [0, 5, 10, 15, 20]
            },
            "description": "Generate multiple frames efficiently"
        }
    ]
    
    for example in examples:
        print(f"\n📌 {example['title']}")
        print(f"   {example['method']} {example['endpoint']}")
        
        if 'payload' in example:
            print(f"   Payload: {example['payload']}")
        
        print(f"   📝 {example['description']}")
    
    print("\n🎯 Key Benefits:")
    print("  • Zero manual model management")
    print("  • Automatic downloading from central registry")
    print("  • Efficient memory usage with on-demand loading")
    print("  • Support for multiple models simultaneously")
    print("  • Per-request audio customization")
    print("  • Comprehensive error handling and fallbacks")


if __name__ == "__main__":
    print("🎯 SyncTalk2D Dynamic Model Loading System")
    print("=" * 70)
    
    # Run demonstration
    asyncio.run(demonstrate_dynamic_loading())
    
    # Show API examples
    print_api_usage_examples()
    
    print("\n🏁 Demonstration Complete!")
    print("\nTo test the system:")
    print("1. Start the service: python -m uvicorn fast_service.service:app --reload")
    print("2. Run tests: python fast_service/test_dynamic_loading.py")
    print("3. Try API calls: curl -X POST http://localhost:8000/generate/frame \\")
    print("   -H 'Content-Type: application/json' \\")
    print("   -d '{\"model_name\": \"default_model\", \"frame_id\": 17}'")
