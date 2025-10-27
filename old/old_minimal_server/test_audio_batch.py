"""
Test script for audio batch optimization
Demonstrates bandwidth savings with contiguous audio chunks
"""

import asyncio
import grpc
import time
import numpy as np

# Import generated protobuf
import optimized_lipsyncsrv_pb2 as pb2
import optimized_lipsyncsrv_pb2_grpc as pb2_grpc


async def test_audio_batch():
    print("\n" + "="*70)
    print("🎵 AUDIO BATCH INFERENCE TEST")
    print("="*70)
    
    # Test configuration
    MODEL_NAME = "sanders"
    START_FRAME = 100
    FRAME_COUNT = 4
    SERVER_ADDR = "localhost:50051"  # Using existing server on port 50051
    
    # Calculate audio chunks needed
    audio_chunks_needed = FRAME_COUNT + 15  # 8 before + N frames + 7 after
    old_method_chunks = FRAME_COUNT * 16
    savings_pct = (old_method_chunks - audio_chunks_needed) / old_method_chunks * 100
    
    print(f"\n📊 Test Configuration:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Frames: {START_FRAME}-{START_FRAME+FRAME_COUNT-1} ({FRAME_COUNT} frames)")
    print(f"   Audio chunks needed: {audio_chunks_needed}")
    print(f"   Old method: {old_method_chunks} chunks")
    print(f"   Savings: {old_method_chunks - audio_chunks_needed} chunks ({savings_pct:.1f}%)")
    
    # Connect to gRPC server
    print(f"\n🔌 Connecting to {SERVER_ADDR}...")
    
    async with grpc.aio.insecure_channel(SERVER_ADDR) as channel:
        stub = pb2_grpc.OptimizedLipSyncServiceStub(channel)
        
        print("✅ Connected!")
        
        # Generate dummy audio chunks (in real scenario, these would be actual audio)
        print(f"\n🎵 Generating {audio_chunks_needed} dummy audio chunks...")
        audio_chunks = []
        chunk_size = 16384  # ~16KB per 40ms chunk
        
        for i in range(audio_chunks_needed):
            # Create dummy audio data
            chunk = bytes([i % 256] * chunk_size)
            audio_chunks.append(chunk)
        
        total_audio_size = len(audio_chunks) * chunk_size
        print(f"   Generated {len(audio_chunks)} chunks ({total_audio_size / (1024*1024):.2f} MB)")
        
        # Create request
        request = pb2.BatchInferenceWithAudioRequest(
            model_name=MODEL_NAME,
            start_frame_id=START_FRAME,
            frame_count=FRAME_COUNT,
            audio_chunks=audio_chunks
        )
        
        print(f"\n🎯 Sending audio batch request...")
        print(f"   Request size: {total_audio_size / (1024*1024):.2f} MB")
        
        # Send request
        start_time = time.time()
        
        try:
            response = await stub.GenerateBatchWithAudio(request)
            
            elapsed = time.time() - start_time
            
            # Print results
            print(f"\n{'='*70}")
            print("📊 RESULTS")
            print(f"{'='*70}")
            
            print(f"\n✅ Received {len(response.responses)} responses\n")
            
            success_count = 0
            total_size = 0
            
            for i, r in enumerate(response.responses):
                frame_id = START_FRAME + i
                if r.success:
                    success_count += 1
                    size = len(r.prediction_data)
                    total_size += size
                    
                    print(f"  ✅ Frame {frame_id}: {r.processing_time_ms}ms "
                          f"({r.inference_time_ms:.2f}ms inference) - "
                          f"{size} bytes ({size/1024:.2f} KB)")
                else:
                    print(f"  ❌ Frame {frame_id}: ERROR - {r.error}")
            
            print(f"\n{'='*70}")
            print("📈 PERFORMANCE SUMMARY")
            print(f"{'='*70}")
            
            print(f"\n🎯 Batch Stats:")
            print(f"   Total Time: {elapsed*1000:.2f}ms")
            print(f"   Server Total: {response.total_processing_time_ms}ms")
            print(f"   Server Avg: {response.avg_frame_time_ms:.2f}ms per frame")
            print(f"   Success Rate: {success_count}/{FRAME_COUNT} frames")
            
            if elapsed > 0:
                fps = FRAME_COUNT / elapsed
                print(f"   Throughput: {fps:.2f} FPS")
            
            if total_size > 0:
                print(f"   Frame Data: {total_size} bytes ({total_size/(1024*1024):.2f} MB)")
                data_rate = (total_size / (1024*1024)) / elapsed
                print(f"   Data Rate: {data_rate:.2f} MB/s")
            
            # Bandwidth analysis
            print(f"\n{'='*70}")
            print("📊 BANDWIDTH ANALYSIS")
            print(f"{'='*70}")
            
            print(f"\n🎵 Audio Transfer:")
            print(f"   Chunks Sent: {audio_chunks_needed}")
            print(f"   Old Method: {old_method_chunks} chunks")
            print(f"   Savings: {old_method_chunks - audio_chunks_needed} chunks ({savings_pct:.1f}%)")
            print(f"   Audio Size: {total_audio_size/(1024*1024):.2f} MB")
            
            old_audio_size = old_method_chunks * chunk_size
            print(f"   Old Method Size: {old_audio_size/(1024*1024):.2f} MB")
            print(f"   Bandwidth Saved: {(old_audio_size - total_audio_size)/(1024*1024):.2f} MB ({savings_pct:.1f}%)")
            
            print(f"\n{'='*70}\n")
            
        except grpc.RpcError as e:
            print(f"\n❌ RPC Error: {e.code()}")
            print(f"   Details: {e.details()}")
            return
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return


async def test_comparison():
    """Compare old vs new method"""
    print("\n" + "="*70)
    print("📊 OLD METHOD vs NEW METHOD COMPARISON")
    print("="*70)
    
    test_cases = [
        (1, "Single frame"),
        (2, "2 frames"),
        (4, "4 frames"),
        (8, "8 frames"),
        (20, "20 frames"),
    ]
    
    print(f"\n{'Frames':<10} {'Old Chunks':<15} {'New Chunks':<15} {'Savings':<15} {'Savings %':<10}")
    print("-" * 70)
    
    for frame_count, desc in test_cases:
        old_chunks = frame_count * 16
        new_chunks = frame_count + 15
        savings = old_chunks - new_chunks
        savings_pct = (savings / old_chunks * 100) if old_chunks > 0 else 0
        
        print(f"{desc:<10} {old_chunks:<15} {new_chunks:<15} {savings:<15} {savings_pct:>6.1f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n🚀 AUDIO BATCH OPTIMIZATION TEST\n")
    
    # Show comparison table
    asyncio.run(test_comparison())
    
    # Run actual test
    asyncio.run(test_audio_batch())
