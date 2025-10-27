#!/usr/bin/env python3
"""
GPU Memory Usage Monitor
Checks GPU memory consumption with loaded models and during inference.
"""

import subprocess
import time
import grpc
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc
import psutil
import os

def get_gpu_memory_info():
    """Get detailed GPU memory information using nvidia-smi"""
    try:
        # Get detailed GPU memory info
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'total_memory': int(parts[2]),
                    'used_memory': int(parts[3]),
                    'free_memory': int(parts[4]),
                    'gpu_utilization': int(parts[5]),
                    'temperature': int(parts[6])
                })
        
        return gpu_info
    except Exception as e:
        print(f"❌ Error getting GPU info: {e}")
        return []

def get_process_memory():
    """Get current process memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'cpu_percent': process.cpu_percent(),
        'num_threads': process.num_threads()
    }

def test_inference_memory_impact():
    """Test memory usage during inference operations"""
    
    print("\n🧠 Testing Inference Memory Impact...")
    print("=" * 50)
    
    # Connect to gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    # Get available models
    models = [
        "test_optimized_package_fixed_1",
        "test_optimized_package_fixed_2", 
        "test_optimized_package_fixed_3",
        "test_optimized_package_fixed_4",
        "test_optimized_package_fixed_5"
    ]
    
    print("📊 Memory usage during inference operations:")
    print()
    
    for i, model_name in enumerate(models):
        print(f"🔄 Testing model {i+1}/5: {model_name}")
        
        # Memory before inference
        gpu_before = get_gpu_memory_info()
        process_before = get_process_memory()
        
        try:
            # Perform inference
            request = lipsyncsrv_pb2.InferenceRequest(
                model_name=model_name,
                frame_id=0
            )
            
            start_time = time.time()
            response = stub.GenerateInference(request)
            inference_time = (time.time() - start_time) * 1000
            
            # Memory after inference
            gpu_after = get_gpu_memory_info()
            process_after = get_process_memory()
            
            if response.success and gpu_before and gpu_after:
                gpu_used_diff = gpu_after[0]['used_memory'] - gpu_before[0]['used_memory']
                process_diff = process_after['rss'] - process_before['rss']
                
                print(f"   ✅ Inference: {inference_time:.1f}ms")
                print(f"   📊 GPU memory change: {gpu_used_diff:+d} MB")
                print(f"   💾 Process memory change: {process_diff:+.1f} MB")
                print(f"   🌡️ GPU utilization: {gpu_after[0]['gpu_utilization']}%")
                print(f"   🔥 GPU temperature: {gpu_after[0]['temperature']}°C")
            else:
                print(f"   ❌ Failed inference or unable to get memory info")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
        
        print()
        time.sleep(1)  # Brief pause between tests

def monitor_memory_during_batch():
    """Monitor memory usage during batch processing"""
    
    print("\n📦 Monitoring Memory During Batch Processing...")
    print("=" * 55)
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    # Memory before batch
    gpu_before = get_gpu_memory_info()
    process_before = get_process_memory()
    
    if gpu_before:
        print(f"📊 Before batch processing:")
        print(f"   GPU Memory: {gpu_before[0]['used_memory']}/{gpu_before[0]['total_memory']} MB ({gpu_before[0]['used_memory']/gpu_before[0]['total_memory']*100:.1f}%)")
        print(f"   Process Memory: {process_before['rss']:.1f} MB")
        print()
    
    try:
        # Large batch processing
        request = lipsyncsrv_pb2.BatchInferenceRequest(
            model_name="test_optimized_package_fixed_3",
            frame_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 5  # 50 frames
        )
        
        print("🔄 Processing 50-frame batch...")
        start_time = time.time()
        response = stub.GenerateBatchInference(request)
        batch_time = time.time() - start_time
        
        # Memory after batch
        gpu_after = get_gpu_memory_info()
        process_after = get_process_memory()
        
        successful_frames = sum(1 for r in response.responses if r.success)
        
        print(f"✅ Batch completed: {successful_frames}/50 frames in {batch_time:.1f}s")
        print()
        
        if gpu_after:
            gpu_diff = gpu_after[0]['used_memory'] - gpu_before[0]['used_memory']
            process_diff = process_after['rss'] - process_before['rss']
            
            print(f"📊 After batch processing:")
            print(f"   GPU Memory: {gpu_after[0]['used_memory']}/{gpu_after[0]['total_memory']} MB ({gpu_after[0]['used_memory']/gpu_after[0]['total_memory']*100:.1f}%)")
            print(f"   GPU Memory Change: {gpu_diff:+d} MB")
            print(f"   Process Memory: {process_after['rss']:.1f} MB")
            print(f"   Process Memory Change: {process_diff:+.1f} MB")
            print(f"   GPU Utilization: {gpu_after[0]['gpu_utilization']}%")
            
    except Exception as e:
        print(f"💥 Batch processing error: {e}")

def main():
    """Main memory monitoring function"""
    
    print("🧠 GPU Memory Usage Analysis")
    print("=" * 60)
    print("📊 Analyzing memory consumption with loaded models")
    print()
    
    # Get initial GPU state
    gpu_info = get_gpu_memory_info()
    process_info = get_process_memory()
    
    if gpu_info:
        gpu = gpu_info[0]  # Primary GPU
        print(f"🎮 GPU Information:")
        print(f"   Name: {gpu['name']}")
        print(f"   Total Memory: {gpu['total_memory']:,} MB ({gpu['total_memory']/1024:.1f} GB)")
        print(f"   Used Memory: {gpu['used_memory']:,} MB ({gpu['used_memory']/1024:.1f} GB)")
        print(f"   Free Memory: {gpu['free_memory']:,} MB ({gpu['free_memory']/1024:.1f} GB)")
        print(f"   Memory Usage: {gpu['used_memory']/gpu['total_memory']*100:.1f}%")
        print(f"   GPU Utilization: {gpu['gpu_utilization']}%")
        print(f"   Temperature: {gpu['temperature']}°C")
        print()
        
        print(f"💾 Process Information:")
        print(f"   RSS Memory: {process_info['rss']:.1f} MB")
        print(f"   Virtual Memory: {process_info['vms']:.1f} MB")
        print(f"   CPU Usage: {process_info['cpu_percent']:.1f}%")
        print(f"   Threads: {process_info['num_threads']}")
    else:
        print("❌ Could not retrieve GPU information")
        print("💡 Make sure nvidia-smi is available and you have NVIDIA GPUs")
        return
    
    # Test inference memory impact
    test_inference_memory_impact()
    
    # Monitor batch processing
    monitor_memory_during_batch()
    
    # Final memory state
    print("\n📊 Final Memory State:")
    print("=" * 25)
    
    final_gpu = get_gpu_memory_info()
    final_process = get_process_memory()
    
    if final_gpu and gpu_info:
        final_gpu_used = final_gpu[0]['used_memory']
        initial_gpu_used = gpu_info[0]['used_memory']
        total_gpu_change = final_gpu_used - initial_gpu_used
        
        print(f"🎮 GPU Memory:")
        print(f"   Current Usage: {final_gpu_used:,} MB ({final_gpu_used/final_gpu[0]['total_memory']*100:.1f}%)")
        print(f"   Total Change: {total_gpu_change:+d} MB")
        print()
        
        print(f"💾 Process Memory:")
        print(f"   Current RSS: {final_process['rss']:.1f} MB")
        print(f"   Total Change: {final_process['rss'] - process_info['rss']:+.1f} MB")
    
    print("\n🎯 Memory Analysis Summary:")
    print("=" * 30)
    
    if gpu_info:
        models_memory_estimate = gpu_info[0]['used_memory']
        print(f"📦 Estimated memory for 5 loaded models: ~{models_memory_estimate:,} MB ({models_memory_estimate/1024:.1f} GB)")
        print(f"💡 Average per model: ~{models_memory_estimate/5:.0f} MB")
        
        if models_memory_estimate > 0:
            efficiency = (process_info['rss'] / models_memory_estimate) * 100
            print(f"⚡ Memory efficiency: {efficiency:.1f}% (process/GPU ratio)")
        
        # Memory usage categories
        if models_memory_estimate < 4000:
            print("✅ Memory usage: Low (suitable for most GPUs)")
        elif models_memory_estimate < 8000:
            print("⚠️ Memory usage: Moderate (requires 8GB+ GPU)")
        else:
            print("❌ Memory usage: High (requires 16GB+ GPU)")

if __name__ == "__main__":
    main()
