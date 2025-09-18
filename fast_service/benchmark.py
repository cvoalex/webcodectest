#!/usr/bin/env python3
"""
Performance benchmark for SyncTalk2D FastAPI service
Tests throughput under different scenarios
"""

import requests
import time
import base64
import statistics
import concurrent.futures
import threading

SERVICE_URL = "http://localhost:8000"

def encode_demo_audio():
    """Encode demo audio file"""
    try:
        with open("../demo/talk_hb.wav", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def single_frame_test(model_name, frame_id, audio_data):
    """Test single frame generation"""
    start_time = time.time()
    
    response = requests.post(f"{SERVICE_URL}/generate/frame", json={
        "model_name": model_name,
        "frame_id": frame_id,
        "audio_override": audio_data
    })
    
    total_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        result = response.json()
        return {
            "success": True,
            "total_time_ms": total_time,
            "server_time_ms": result.get("processing_time_ms", 0),
            "from_cache": result.get("from_cache", False),
            "auto_loaded": result.get("auto_loaded", False)
        }
    else:
        return {"success": False, "error": response.text}

def batch_frame_test(requests_data):
    """Test batch frame generation"""
    start_time = time.time()
    
    response = requests.post(f"{SERVICE_URL}/generate/batch", json={
        "requests": requests_data
    })
    
    total_time = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        result = response.json()
        return {
            "success": True,
            "total_time_ms": total_time,
            "server_time_ms": result.get("processing_time_ms", 0),
            "processed_count": result.get("processed_count", 0),
            "cached_count": result.get("cached_count", 0)
        }
    else:
        return {"success": False, "error": response.text}

def benchmark_single_frames():
    """Benchmark single frame requests"""
    print("\n🏃 Single Frame Benchmark")
    print("=" * 40)
    
    audio_data = encode_demo_audio()
    model_name = "default_model"
    
    # Test 20 different frames
    frame_ids = list(range(100, 120))
    times = []
    cache_hits = 0
    
    for frame_id in frame_ids:
        result = single_frame_test(model_name, frame_id, audio_data)
        
        if result["success"]:
            times.append(result["total_time_ms"])
            if result["from_cache"]:
                cache_hits += 1
            
            status = "cached" if result["from_cache"] else "generated"
            print(f"Frame {frame_id}: {result['total_time_ms']:.1f}ms ({status})")
        else:
            print(f"Frame {frame_id}: FAILED")
    
    if times:
        avg_time = statistics.mean(times)
        fps = 1000 / avg_time
        print(f"\n📊 Single Frame Results:")
        print(f"   Average time: {avg_time:.1f}ms")
        print(f"   FPS: {fps:.1f}")
        print(f"   Cache hits: {cache_hits}/{len(frame_ids)}")
        return fps
    
    return 0

def benchmark_batch_processing():
    """Benchmark batch processing"""
    print("\n🚀 Batch Processing Benchmark")
    print("=" * 40)
    
    audio_data = encode_demo_audio()
    model_name = "default_model"
    
    # Test batches of different sizes
    batch_sizes = [1, 3, 5, 10]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create batch request
        requests_data = []
        for i in range(batch_size):
            requests_data.append({
                "model_name": model_name,
                "frame_id": 200 + i,  # Use fresh frame IDs
                "audio_override": audio_data
            })
        
        result = batch_frame_test(requests_data)
        
        if result["success"]:
            total_time = result["total_time_ms"]
            per_frame_time = total_time / batch_size
            fps = 1000 / per_frame_time
            
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Per frame: {per_frame_time:.1f}ms")
            print(f"   Effective FPS: {fps:.1f}")
            print(f"   Processed: {result['processed_count']}")
            print(f"   Cached: {result['cached_count']}")

def benchmark_cache_performance():
    """Test cache performance by requesting same frames twice"""
    print("\n💾 Cache Performance Test")
    print("=" * 40)
    
    audio_data = encode_demo_audio()
    model_name = "default_model"
    frame_ids = [300, 301, 302, 303, 304]
    
    # First pass - generate frames
    print("First pass (generation):")
    first_pass_times = []
    for frame_id in frame_ids:
        result = single_frame_test(model_name, frame_id, audio_data)
        if result["success"]:
            first_pass_times.append(result["total_time_ms"])
            print(f"  Frame {frame_id}: {result['total_time_ms']:.1f}ms")
    
    # Second pass - should hit cache
    print("\nSecond pass (cache):")
    second_pass_times = []
    for frame_id in frame_ids:
        result = single_frame_test(model_name, frame_id, audio_data)
        if result["success"]:
            second_pass_times.append(result["total_time_ms"])
            status = "CACHED" if result["from_cache"] else "GENERATED"
            print(f"  Frame {frame_id}: {result['total_time_ms']:.1f}ms ({status})")
    
    if first_pass_times and second_pass_times:
        gen_avg = statistics.mean(first_pass_times)
        cache_avg = statistics.mean(second_pass_times)
        speedup = gen_avg / cache_avg
        
        print(f"\n📊 Cache Results:")
        print(f"   Generation avg: {gen_avg:.1f}ms ({1000/gen_avg:.1f} FPS)")
        print(f"   Cache avg: {cache_avg:.1f}ms ({1000/cache_avg:.1f} FPS)")
        print(f"   Cache speedup: {speedup:.1f}x faster")

def concurrent_requests_test():
    """Test concurrent request handling"""
    print("\n⚡ Concurrent Requests Test")
    print("=" * 40)
    
    audio_data = encode_demo_audio()
    model_name = "default_model"
    
    # Test with different concurrency levels
    for concurrency in [2, 4, 8]:
        print(f"\nTesting {concurrency} concurrent requests:")
        
        def make_request(frame_id):
            return single_frame_test(model_name, 400 + frame_id, audio_data)
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_request, i) for i in range(concurrency)]
            results = [f.result() for f in futures]
        
        total_time = (time.time() - start_time) * 1000
        successful = [r for r in results if r["success"]]
        
        if successful:
            avg_per_request = total_time / len(successful)
            throughput = len(successful) * 1000 / total_time
            
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Avg per request: {avg_per_request:.1f}ms")
            print(f"   Throughput: {throughput:.1f} requests/sec")
            print(f"   Successful: {len(successful)}/{concurrency}")

def main():
    print("🚀 SyncTalk2D Performance Benchmark")
    print("=" * 50)
    
    # Check service health
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Service not healthy")
            return
    except:
        print("❌ Cannot connect to service")
        return
    
    print("✅ Service is running")
    
    # Run benchmarks
    benchmark_single_frames()
    benchmark_batch_processing()
    benchmark_cache_performance()
    concurrent_requests_test()
    
    print(f"\n✨ Benchmark completed!")

if __name__ == "__main__":
    main()
