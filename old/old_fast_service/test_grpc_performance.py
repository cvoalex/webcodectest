#!/usr/bin/env python3
"""
High-Speed gRPC Client Test
Tests the performance of our gRPC inference service.
"""

import time
import grpc
import numpy as np
import base64

# Import generated gRPC code
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc

def test_grpc_performance():
    """Test gRPC inference performance"""
    
    print("🚀 gRPC High-Speed Performance Test")
    print("=" * 50)
    print()
    
    # Connect to gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    model_name = "test_optimized_package_fixed_3"
    
    print("⚡ Testing single frame inference...")
    print()
    
    times = []
    processing_times = []
    sizes = []
    
    for frame_id in range(5):
        print(f"--- Frame {frame_id} ---")
        
        start_time = time.time()
        
        try:
            # Create request
            request = lipsyncsrv_pb2.InferenceRequest(
                model_name=model_name,
                frame_id=frame_id
            )
            
            # Call gRPC service
            response = stub.GenerateInference(request)
            
            total_time = (time.time() - start_time) * 1000
            
            if response.success:
                times.append(total_time)
                processing_times.append(response.processing_time_ms)
                sizes.append(len(response.prediction_data))
                
                print(f"   ✅ Success")
                print(f"   📊 Total time: {total_time:.1f}ms")
                print(f"   🖥️  Server processing: {response.processing_time_ms}ms")
                print(f"   🗂️  Response size: {len(response.prediction_data):,} bytes")
                print(f"   📐 Prediction shape: {response.prediction_shape}")
                print(f"   📏 Bounds: {len(response.bounds)} values")
                print(f"   🤖 Auto-loaded: {response.auto_loaded}")
                
                # Save prediction for verification
                with open(f"grpc_prediction_frame_{frame_id}.jpg", "wb") as f:
                    f.write(response.prediction_data)
                
            else:
                print(f"   ❌ Error: {response.error}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
        
        print()
    
    if times:
        avg_total = sum(times) / len(times)
        avg_processing = sum(processing_times) / len(processing_times)
        avg_size = sum(sizes) / len(sizes)
        network_overhead = avg_total - avg_processing
        overhead_percentage = (network_overhead / avg_total) * 100
        
        print("📊 gRPC Performance Summary:")
        print(f"   Average total time: {avg_total:.1f}ms")
        print(f"   Average processing time: {avg_processing:.1f}ms")
        print(f"   Network/overhead time: {network_overhead:.1f}ms")
        print(f"   Total FPS: {1000/avg_total:.1f}")
        print(f"   Processing FPS: {1000/avg_processing:.1f}")
        print(f"   Average size: {avg_size:,} bytes ({avg_size/1024:.1f} KB)")
        print(f"   Overhead percentage: {overhead_percentage:.1f}%")
        print()
        
        print("🔥 Performance Comparison:")
        print(f"   HTTP REST API: ~2111ms (0.5 FPS, 16.3KB)")
        print(f"   gRPC Binary: {avg_total:.1f}ms ({1000/avg_total:.1f} FPS, {avg_size/1024:.1f}KB)")
        
        if avg_total > 0:
            speedup = 2111 / avg_total
            print(f"   gRPC Speedup: {speedup:.1f}x faster! 🚀")

def test_grpc_batch():
    """Test batch inference performance"""
    
    print("\n🚀 gRPC Batch Inference Test")
    print("=" * 50)
    print()
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    model_name = "test_optimized_package_fixed_3"
    frame_ids = list(range(5))  # Process 5 frames in one batch
    
    start_time = time.time()
    
    try:
        # Create batch request
        request = lipsyncsrv_pb2.BatchInferenceRequest(
            model_name=model_name,
            frame_ids=frame_ids
        )
        
        # Call batch inference
        response = stub.GenerateBatchInference(request)
        
        total_time = (time.time() - start_time) * 1000
        
        successful_frames = sum(1 for r in response.responses if r.success)
        total_data_size = sum(len(r.prediction_data) for r in response.responses if r.success)
        
        print(f"✅ Batch completed: {successful_frames}/{len(frame_ids)} frames")
        print(f"📊 Total time: {total_time:.1f}ms")
        print(f"🖥️  Server processing: {response.total_processing_time_ms}ms")
        print(f"🗂️  Total data size: {total_data_size:,} bytes ({total_data_size/1024:.1f} KB)")
        print(f"⚡ Frames per second: {(successful_frames * 1000) / total_time:.1f} FPS")
        print(f"📈 Time per frame: {total_time / successful_frames:.1f}ms")
        
        print("\n🔥 Batch vs Individual Comparison:")
        print(f"   Individual requests: ~{67 * len(frame_ids):.0f}ms processing + network overhead")
        print(f"   Batch request: {response.total_processing_time_ms}ms processing")
        print(f"   Batch efficiency: {(67 * len(frame_ids)) / response.total_processing_time_ms:.1f}x better processing!")
        
    except Exception as e:
        print(f"💥 Batch failed: {e}")

def test_grpc_streaming():
    """Test streaming inference performance"""
    
    print("\n🚀 gRPC Streaming Test")
    print("=" * 50)
    print()
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    model_name = "test_optimized_package_fixed_3"
    
    def generate_requests():
        """Generator for streaming requests"""
        for frame_id in range(5):
            yield lipsyncsrv_pb2.InferenceRequest(
                model_name=model_name,
                frame_id=frame_id
            )
    
    start_time = time.time()
    successful_frames = 0
    total_data_size = 0
    
    try:
        # Stream requests and responses
        for response in stub.StreamInference(generate_requests()):
            if response.success:
                successful_frames += 1
                total_data_size += len(response.prediction_data)
                print(f"📦 Frame {response.frame_id}: {len(response.prediction_data):,} bytes, {response.processing_time_ms}ms")
            else:
                print(f"❌ Frame {response.frame_id}: {response.error}")
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"\n✅ Streaming completed: {successful_frames} frames")
        print(f"📊 Total time: {total_time:.1f}ms")
        print(f"🗂️  Total data: {total_data_size:,} bytes ({total_data_size/1024:.1f} KB)")
        print(f"⚡ Streaming FPS: {(successful_frames * 1000) / total_time:.1f}")
        print(f"📈 Avg time per frame: {total_time / successful_frames:.1f}ms")
        
    except Exception as e:
        print(f"💥 Streaming failed: {e}")

if __name__ == "__main__":
    test_grpc_performance()
    test_grpc_batch()
    test_grpc_streaming()
