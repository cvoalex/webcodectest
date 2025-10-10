"""
Test script for batch inference performance
Compare single-frame vs batched processing
"""

import asyncio
import time
import sys
import os

from batch_inference_engine import BatchInferenceEngine


async def test_single_frames(engine, frame_ids):
    """Test processing frames one at a time"""
    print(f"\n{'='*70}")
    print(f"🔹 SINGLE FRAME MODE - Processing {len(frame_ids)} frames sequentially")
    print(f"{'='*70}")
    
    start_time = time.time()
    results = []
    
    for frame_id in frame_ids:
        frame, metadata = await engine.generate_frame(frame_id)
        results.append((frame, metadata))
        print(f"  Frame {frame_id}: {metadata['inference_time_ms']:.2f}ms")
    
    total_time = time.time() - start_time
    fps = len(frame_ids) / total_time
    avg_inference = sum(r[1]['inference_time_ms'] for r in results) / len(results)
    
    print(f"\n📊 Single Frame Results:")
    print(f"   Total Time: {total_time*1000:.2f}ms")
    print(f"   FPS: {fps:.2f}")
    print(f"   Avg Inference: {avg_inference:.2f}ms per frame")
    
    return results, fps


async def test_batch_frames(engine, frame_ids, batch_size):
    """Test processing frames in batches"""
    print(f"\n{'='*70}")
    print(f"🔸 BATCH MODE - Processing {len(frame_ids)} frames in batches of {batch_size}")
    print(f"{'='*70}")
    
    start_time = time.time()
    all_results = []
    
    # Process in batches
    for i in range(0, len(frame_ids), batch_size):
        batch = frame_ids[i:i+batch_size]
        results = await engine.generate_frames_batch(batch)
        all_results.extend(results)
        
        batch_ids = [r[1]['frame_id'] for r in results]
        batch_time = results[0][1]['batch_inference_time_ms']
        per_frame = results[0][1]['per_frame_inference_ms']
        print(f"  Batch {batch_ids}: {batch_time:.2f}ms total ({per_frame:.2f}ms per frame)")
    
    total_time = time.time() - start_time
    fps = len(frame_ids) / total_time
    avg_inference = sum(r[1]['per_frame_inference_ms'] for r in all_results) / len(all_results)
    
    print(f"\n📊 Batch Results:")
    print(f"   Total Time: {total_time*1000:.2f}ms")
    print(f"   FPS: {fps:.2f}")
    print(f"   Avg Inference: {avg_inference:.2f}ms per frame")
    
    return all_results, fps


async def main():
    print("\n" + "="*70)
    print("🚀 BATCH INFERENCE PERFORMANCE TEST")
    print("="*70)
    
    # Configuration
    MODEL_DIR = "D:/Projects/webcodecstest/fast_service/models/default_model"
    BATCH_SIZE = 4
    TEST_FRAMES = list(range(95, 115))  # 20 frames for testing
    
    print(f"\n📁 Loading model from: {MODEL_DIR}")
    print(f"🎯 Batch Size: {BATCH_SIZE}")
    print(f"📊 Test Frames: {len(TEST_FRAMES)} frames ({TEST_FRAMES[0]}-{TEST_FRAMES[-1]})")
    
    # Initialize engine with batch support
    engine = BatchInferenceEngine(MODEL_DIR, max_batch_size=BATCH_SIZE)
    
    # Initialize the engine (load model, videos, etc.)
    print("\n⏳ Initializing engine...")
    await engine.initialize()
    print("✅ Engine ready!")
    
    # Test 1: Single frame processing
    single_results, single_fps = await test_single_frames(engine, TEST_FRAMES)
    
    # Cool down
    await asyncio.sleep(1)
    
    # Test 2: Batch processing
    batch_results, batch_fps = await test_batch_frames(engine, TEST_FRAMES, BATCH_SIZE)
    
    # Compare results
    print(f"\n{'='*70}")
    print("📈 PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    speedup = batch_fps / single_fps
    print(f"\n🔹 Single Frame Mode: {single_fps:.2f} FPS")
    print(f"🔸 Batch Mode ({BATCH_SIZE} frames): {batch_fps:.2f} FPS")
    print(f"⚡ Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    
    # Calculate user capacity
    print(f"\n{'='*70}")
    print("👥 USER CAPACITY ESTIMATE (25 FPS per user)")
    print(f"{'='*70}")
    
    users_single = single_fps / 25
    users_batch = batch_fps / 25
    
    print(f"\n🔹 Single Mode: {users_single:.2f} users")
    print(f"🔸 Batch Mode: {users_batch:.2f} users")
    print(f"📊 Additional capacity: +{users_batch - users_single:.2f} users")
    
    if users_batch >= 2:
        print(f"\n✅ SUCCESS! Can support 2+ users at 25 FPS with batching!")
    else:
        print(f"\n⚠️  Batching improves performance but may need further optimization for 2+ users")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
