"""Quick test to verify batch engine loading"""
import asyncio
from optimized_inference_engine import optimized_engine

async def main():
    print("\n" + "="*70)
    print("🧪 BATCH ENGINE LOADING TEST")
    print("="*70)
    
    # Try loading with batch support
    print("\n📦 Loading model with batch support...")
    result = await optimized_engine.load_package(
        "sanders",
        "D:/Projects/webcodecstest/fast_service/models/default_model",
        enable_batching=True
    )
    
    print(f"\n✅ Load result: {result}")
    
    # Check if the package has batch methods
    package = optimized_engine.get_model("sanders")
    
    print(f"\n🔍 Package type: {type(package).__name__}")
    print(f"📋 Has generate_frames_batch? {hasattr(package, 'generate_frames_batch')}")
    print(f"📋 Has generate_frames_batch_inference_only? {hasattr(package, 'generate_frames_batch_inference_only')}")
    
    if hasattr(package, 'max_batch_size'):
        print(f"🎯 Max batch size: {package.max_batch_size}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    asyncio.run(main())
