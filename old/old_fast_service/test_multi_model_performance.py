#!/usr/bin/env python3
"""
Multi-Model Performance Test
Tests gRPC performance across multiple models with various scenarios:
1. Sequential model switching
2. Concurrent multi-model processing
3. Production batch processing across models
4. Model loading performance
5. Mixed workload scenarios
"""

import os
import time
import cv2
import numpy as np
import grpc
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

def create_test_directory():
    """Create a timestamped directory for test results"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"test_multi_model_{timestamp}"
    
    # Create main directory
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{test_dir}/predictions", exist_ok=True)
    os.makedirs(f"{test_dir}/analysis", exist_ok=True)
    os.makedirs(f"{test_dir}/grids", exist_ok=True)
    os.makedirs(f"{test_dir}/models", exist_ok=True)
    
    print(f"📁 Created test directory: {test_dir}")
    return test_dir

def get_available_models():
    """Get list of available model names"""
    models = []
    for i in range(1, 6):
        model_file = f"test_optimized_package_fixed_{i}.zip"
        if os.path.exists(model_file):
            models.append(f"test_optimized_package_fixed_{i}")
    
    # Also include the original model
    if os.path.exists("../test_optimized_package_fixed.zip"):
        models.append("test_optimized_package_fixed_3")  # This is already loaded
    
    return models

def test_model_loading_performance(test_dir, models):
    """Test model loading performance"""
    
    print(f"\n🔄 Testing Model Loading Performance ({len(models)} models)")
    print("=" * 60)
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    loading_times = {}
    
    for model_name in models:
        print(f"📦 Loading model: {model_name}")
        
        start_time = time.time()
        
        try:
            # Try to make a test inference to trigger model loading
            request = lipsyncsrv_pb2.InferenceRequest(
                model_name=model_name,
                frame_id=0
            )
            
            response = stub.GenerateInference(request)
            
            load_time = (time.time() - start_time) * 1000
            
            if response.success:
                loading_times[model_name] = load_time
                print(f"   ✅ Loaded in {load_time:.1f}ms")
                
                # Save a sample prediction
                filename = f"{test_dir}/models/sample_{model_name}.jpg"
                with open(filename, "wb") as f:
                    f.write(response.prediction_data)
            else:
                print(f"   ❌ Failed to load: {response.error}")
                
        except Exception as e:
            print(f"   💥 Exception loading {model_name}: {e}")
    
    if loading_times:
        avg_load_time = sum(loading_times.values()) / len(loading_times)
        print(f"\n📊 Model Loading Results:")
        print(f"   ✅ Successfully loaded: {len(loading_times)}/{len(models)} models")
        print(f"   ⏱️  Average load time: {avg_load_time:.1f}ms")
        print(f"   🏃 Fastest load: {min(loading_times.values()):.1f}ms")
        print(f"   🐌 Slowest load: {max(loading_times.values()):.1f}ms")
        
        return {
            "loaded_models": len(loading_times),
            "total_models": len(models),
            "avg_load_time": avg_load_time,
            "min_load_time": min(loading_times.values()),
            "max_load_time": max(loading_times.values()),
            "loading_times": loading_times
        }
    
    return None

def test_sequential_multi_model(test_dir, models, frames_per_model=20):
    """Test sequential processing across multiple models"""
    
    print(f"\n🔄 Testing Sequential Multi-Model Processing")
    print(f"   📦 {len(models)} models × {frames_per_model} frames = {len(models) * frames_per_model} total")
    print("=" * 70)
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    times = []
    processing_times = []
    model_performance = {model: [] for model in models}
    successful_frames = 0
    failed_frames = 0
    
    overall_start = time.time()
    
    for model_name in models:
        print(f"📷 Processing {frames_per_model} frames with model: {model_name}")
        
        for frame_id in range(frames_per_model):
            start_time = time.time()
            
            try:
                request = lipsyncsrv_pb2.InferenceRequest(
                    model_name=model_name,
                    frame_id=frame_id % 10
                )
                
                response = stub.GenerateInference(request)
                
                total_time = (time.time() - start_time) * 1000
                
                if response.success:
                    times.append(total_time)
                    processing_times.append(response.processing_time_ms)
                    model_performance[model_name].append(total_time)
                    successful_frames += 1
                    
                    # Save every 5th frame
                    if frame_id % 5 == 0:
                        filename = f"{test_dir}/predictions/seq_{model_name}_frame_{frame_id:02d}.jpg"
                        with open(filename, "wb") as f:
                            f.write(response.prediction_data)
                else:
                    failed_frames += 1
                    
            except Exception as e:
                failed_frames += 1
        
        # Print model summary
        if model_performance[model_name]:
            avg_time = sum(model_performance[model_name]) / len(model_performance[model_name])
            print(f"   ✅ {model_name}: {len(model_performance[model_name])} frames, avg {avg_time:.1f}ms")
    
    overall_time = time.time() - overall_start
    
    if times:
        avg_total = sum(times) / len(times)
        avg_processing = sum(processing_times) / len(processing_times)
        overall_fps = successful_frames / overall_time
        
        print(f"\n📊 Sequential Multi-Model Results:")
        print(f"   ✅ Successful frames: {successful_frames}/{len(models) * frames_per_model}")
        print(f"   ❌ Failed frames: {failed_frames}")
        print(f"   ⏱️  Total test time: {overall_time:.1f}s")
        print(f"   🎬 Overall FPS: {overall_fps:.1f}")
        print(f"   📊 Average frame time: {avg_total:.1f}ms")
        print(f"   🖥️  Average processing: {avg_processing:.1f}ms")
        
        return {
            "successful_frames": successful_frames,
            "failed_frames": failed_frames,
            "total_time": overall_time,
            "overall_fps": overall_fps,
            "avg_frame_time": avg_total,
            "avg_processing_time": avg_processing,
            "model_performance": model_performance
        }
    
    return None

def test_concurrent_multi_model(test_dir, models, frames_per_model=50):
    """Test concurrent processing across multiple models"""
    
    print(f"\n🚀 Testing Concurrent Multi-Model Processing")
    print(f"   📦 {len(models)} models × {frames_per_model} frames = {len(models) * frames_per_model} total")
    print(f"   🔄 {len(models)} concurrent threads (one per model)")
    print("=" * 75)
    
    def process_model_concurrent(model_name, frame_count):
        """Process frames for a specific model in a separate thread"""
        
        # Each thread gets its own channel
        channel = grpc.insecure_channel('localhost:50051')
        stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
        
        times = []
        successful = 0
        
        for frame_id in range(frame_count):
            try:
                start_time = time.time()
                
                request = lipsyncsrv_pb2.InferenceRequest(
                    model_name=model_name,
                    frame_id=frame_id % 10
                )
                
                response = stub.GenerateInference(request)
                
                total_time = (time.time() - start_time) * 1000
                
                if response.success:
                    times.append(total_time)
                    successful += 1
                    
                    # Save every 10th frame
                    if frame_id % 10 == 0:
                        filename = f"{test_dir}/predictions/conc_{model_name}_frame_{frame_id:02d}.jpg"
                        with open(filename, "wb") as f:
                            f.write(response.prediction_data)
                
            except Exception as e:
                pass  # Silent failure for cleaner output
        
        channel.close()
        return model_name, successful, times
    
    # Start concurrent processing
    overall_start = time.time()
    
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # Submit one thread per model
        futures = []
        for model_name in models:
            future = executor.submit(process_model_concurrent, model_name, frames_per_model)
            futures.append(future)
        
        # Collect results
        all_times = []
        total_successful = 0
        model_results = {}
        
        for future in as_completed(futures):
            model_name, successful, times = future.result()
            total_successful += successful
            all_times.extend(times)
            model_results[model_name] = {
                "successful": successful,
                "avg_time": sum(times) / len(times) if times else 0
            }
            print(f"   ✅ {model_name}: {successful}/{frames_per_model} frames completed")
    
    overall_time = time.time() - overall_start
    
    if all_times:
        avg_time = sum(all_times) / len(all_times)
        overall_fps = total_successful / overall_time
        
        print(f"\n📊 Concurrent Multi-Model Results:")
        print(f"   ✅ Total frames processed: {total_successful}/{len(models) * frames_per_model}")
        print(f"   📦 Models used: {len(models)}")
        print(f"   ⏱️  Total test time: {overall_time:.1f}s")
        print(f"   🎬 Overall FPS: {overall_fps:.1f}")
        print(f"   📊 Average frame time: {avg_time:.1f}ms")
        print(f"   🚀 Multi-model scaling: {overall_fps:.1f} FPS across {len(models)} models")
        
        return {
            "total_frames": total_successful,
            "total_time": overall_time,
            "overall_fps": overall_fps,
            "avg_frame_time": avg_time,
            "models_count": len(models),
            "model_results": model_results
        }
    
    return None

def test_production_multi_model_batch(test_dir, models, batches_per_model=20):
    """Test production batch processing across multiple models"""
    
    print(f"\n🏭 Testing Production Multi-Model Batch Processing")
    print(f"   📦 {len(models)} models × {batches_per_model} batches × 5 frames = {len(models) * batches_per_model * 5} total")
    print("=" * 80)
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    batch_times = []
    total_frames_processed = 0
    model_batch_performance = {model: [] for model in models}
    
    overall_start = time.time()
    
    for model_name in models:
        print(f"📦 Processing {batches_per_model} batches with model: {model_name}")
        
        for batch_num in range(batches_per_model):
            # Create batch of 5 frames
            frame_ids = [batch_num * 5 + i for i in range(5)]
            
            start_time = time.time()
            
            try:
                request = lipsyncsrv_pb2.BatchInferenceRequest(
                    model_name=model_name,
                    frame_ids=[fid % 10 for fid in frame_ids]  # Cycle through 0-9
                )
                
                response = stub.GenerateBatchInference(request)
                
                batch_time = (time.time() - start_time) * 1000
                batch_times.append(batch_time)
                model_batch_performance[model_name].append(batch_time)
                
                successful_in_batch = sum(1 for r in response.responses if r.success)
                total_frames_processed += successful_in_batch
                
                # Save sample predictions (every 5th batch)
                if batch_num % 5 == 0:
                    for i, frame_response in enumerate(response.responses):
                        if frame_response.success:
                            filename = f"{test_dir}/predictions/prod_batch_{model_name}_{batch_num:02d}_{i}.jpg"
                            with open(filename, "wb") as f:
                                f.write(frame_response.prediction_data)
                
            except Exception as e:
                print(f"   💥 Batch {batch_num} failed for {model_name}: {e}")
        
        # Print model batch summary
        if model_batch_performance[model_name]:
            avg_batch_time = sum(model_batch_performance[model_name]) / len(model_batch_performance[model_name])
            print(f"   ✅ {model_name}: {len(model_batch_performance[model_name])} batches, avg {avg_batch_time:.1f}ms per batch")
    
    overall_time = time.time() - overall_start
    
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        overall_fps = total_frames_processed / overall_time
        
        print(f"\n📊 Production Multi-Model Batch Results:")
        print(f"   ✅ Total frames processed: {total_frames_processed}/{len(models) * batches_per_model * 5}")
        print(f"   📦 Total batches: {len(batch_times)}")
        print(f"   ⏱️  Total test time: {overall_time:.1f}s")
        print(f"   🎬 Overall FPS: {overall_fps:.1f}")
        print(f"   📊 Average batch time: {avg_batch_time:.1f}ms")
        print(f"   🔄 Time per frame: {avg_batch_time/5:.1f}ms")
        print(f"   🏭 Production latency: {avg_batch_time:.1f}ms per 5-frame batch")
        
        return {
            "total_frames": total_frames_processed,
            "total_batches": len(batch_times),
            "total_time": overall_time,
            "overall_fps": overall_fps,
            "avg_batch_time": avg_batch_time,
            "time_per_frame": avg_batch_time/5,
            "model_batch_performance": model_batch_performance
        }
    
    return None

def test_mixed_workload(test_dir, models, duration_seconds=60):
    """Test mixed workload with random model selection"""
    
    print(f"\n🎲 Testing Mixed Workload Scenario")
    print(f"   ⏱️  Duration: {duration_seconds} seconds")
    print(f"   📦 Models: {len(models)} (randomly selected)")
    print(f"   🔄 Mix of single requests and batches")
    print("=" * 50)
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    
    single_requests = 0
    batch_requests = 0
    total_frames = 0
    failed_requests = 0
    times = []
    
    start_time = time.time()
    request_count = 0
    
    while (time.time() - start_time) < duration_seconds:
        request_count += 1
        
        # Randomly select model and request type
        model_name = random.choice(models)
        is_batch = random.choice([True, False])
        
        try:
            if is_batch:
                # Batch request (2-5 frames)
                batch_size = random.randint(2, 5)
                frame_ids = [random.randint(0, 9) for _ in range(batch_size)]
                
                req_start = time.time()
                request = lipsyncsrv_pb2.BatchInferenceRequest(
                    model_name=model_name,
                    frame_ids=frame_ids
                )
                response = stub.GenerateBatchInference(request)
                req_time = (time.time() - req_start) * 1000
                
                successful_in_batch = sum(1 for r in response.responses if r.success)
                if successful_in_batch > 0:
                    batch_requests += 1
                    total_frames += successful_in_batch
                    times.append(req_time)
                    
                    # Save random sample
                    if request_count % 20 == 0 and response.responses[0].success:
                        filename = f"{test_dir}/predictions/mixed_batch_{request_count:03d}.jpg"
                        with open(filename, "wb") as f:
                            f.write(response.responses[0].prediction_data)
                else:
                    failed_requests += 1
            else:
                # Single request
                frame_id = random.randint(0, 9)
                
                req_start = time.time()
                request = lipsyncsrv_pb2.InferenceRequest(
                    model_name=model_name,
                    frame_id=frame_id
                )
                response = stub.GenerateInference(request)
                req_time = (time.time() - req_start) * 1000
                
                if response.success:
                    single_requests += 1
                    total_frames += 1
                    times.append(req_time)
                    
                    # Save random sample
                    if request_count % 15 == 0:
                        filename = f"{test_dir}/predictions/mixed_single_{request_count:03d}.jpg"
                        with open(filename, "wb") as f:
                            f.write(response.prediction_data)
                else:
                    failed_requests += 1
                    
        except Exception as e:
            failed_requests += 1
        
        # Brief pause to simulate realistic load
        time.sleep(0.01)  # 10ms pause
    
    actual_duration = time.time() - start_time
    
    if times:
        avg_time = sum(times) / len(times)
        fps = total_frames / actual_duration
        
        print(f"\n📊 Mixed Workload Results:")
        print(f"   ⏱️  Actual duration: {actual_duration:.1f}s")
        print(f"   📞 Single requests: {single_requests}")
        print(f"   📦 Batch requests: {batch_requests}")
        print(f"   ✅ Total frames: {total_frames}")
        print(f"   ❌ Failed requests: {failed_requests}")
        print(f"   🎬 Overall FPS: {fps:.1f}")
        print(f"   📊 Average request time: {avg_time:.1f}ms")
        print(f"   🎲 Requests per second: {(single_requests + batch_requests) / actual_duration:.1f}")
        
        return {
            "duration": actual_duration,
            "single_requests": single_requests,
            "batch_requests": batch_requests,
            "total_frames": total_frames,
            "failed_requests": failed_requests,
            "overall_fps": fps,
            "avg_request_time": avg_time,
            "requests_per_second": (single_requests + batch_requests) / actual_duration
        }
    
    return None

def create_analysis_grids(test_dir):
    """Create analysis grids from the multi-model predictions"""
    
    print("\n🎨 Creating Multi-Model Analysis Grids...")
    print("=" * 40)
    
    predictions_dir = f"{test_dir}/predictions"
    
    # Find different types of prediction files
    sequential_files = []
    concurrent_files = []
    production_files = []
    mixed_files = []
    
    for filename in os.listdir(predictions_dir):
        if filename.startswith("seq_") and filename.endswith(".jpg"):
            sequential_files.append(filename)
        elif filename.startswith("conc_") and filename.endswith(".jpg"):
            concurrent_files.append(filename)
        elif filename.startswith("prod_batch_") and filename.endswith(".jpg"):
            production_files.append(filename)
        elif filename.startswith("mixed_") and filename.endswith(".jpg"):
            mixed_files.append(filename)
    
    sequential_files.sort()
    concurrent_files.sort()
    production_files.sort()
    mixed_files.sort()
    
    # Create grids for each type
    if len(sequential_files) >= 16:
        create_sample_grid(predictions_dir, sequential_files[:16], 
                          f"{test_dir}/grids/sequential_multi_model.jpg", 
                          "Sequential Multi-Model Processing", 4, 4)
    
    if len(concurrent_files) >= 16:
        create_sample_grid(predictions_dir, concurrent_files[:16], 
                          f"{test_dir}/grids/concurrent_multi_model.jpg", 
                          "Concurrent Multi-Model Processing", 4, 4)
    
    if len(production_files) >= 16:
        create_sample_grid(predictions_dir, production_files[:16], 
                          f"{test_dir}/grids/production_multi_model.jpg", 
                          "Production Multi-Model Batches", 4, 4)
    
    if len(mixed_files) >= 16:
        create_sample_grid(predictions_dir, mixed_files[:16], 
                          f"{test_dir}/grids/mixed_workload.jpg", 
                          "Mixed Workload Scenario", 4, 4)
    
    print(f"   ✅ Analysis grids saved to: {test_dir}/grids/")

def create_sample_grid(predictions_dir, filenames, output_path, title, rows, cols):
    """Create a grid of sample images"""
    
    if not filenames:
        return
    
    # Load first image to get dimensions
    first_img = cv2.imread(os.path.join(predictions_dir, filenames[0]))
    if first_img is None:
        return
    
    frame_height, frame_width = first_img.shape[:2]
    
    # Create grid
    grid_height = frame_height * rows + 60
    grid_width = frame_width * cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(grid, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Place frames in grid
    for i, filename in enumerate(filenames[:rows*cols]):
        row = i // cols
        col = i % cols
        
        img = cv2.imread(os.path.join(predictions_dir, filename))
        if img is not None:
            y_start = row * frame_height + 60
            y_end = y_start + frame_height
            x_start = col * frame_width
            x_end = x_start + frame_width
            
            grid[y_start:y_end, x_start:x_end] = img
            
            # Add filename label
            label = filename.split('.')[0][-8:]  # Last 8 chars
            cv2.putText(grid, label, (x_start + 5, y_start + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, grid)
    print(f"   📷 Created: {os.path.basename(output_path)}")

def save_analysis_report(test_dir, loading_results, sequential_results, concurrent_results, production_results, mixed_results, models):
    """Save detailed multi-model analysis report"""
    
    report_path = f"{test_dir}/analysis/multi_model_performance_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🚀 MULTI-MODEL gRPC Lip Sync Performance Test\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models Tested: {len(models)} models\n")
        f.write(f"Model Names: {', '.join(models)}\n\n")
        
        if loading_results:
            f.write("📦 MODEL LOADING PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Successfully loaded: {loading_results['loaded_models']}/{loading_results['total_models']} models\n")
            f.write(f"Average load time: {loading_results['avg_load_time']:.1f}ms\n")
            f.write(f"Fastest load: {loading_results['min_load_time']:.1f}ms\n")
            f.write(f"Slowest load: {loading_results['max_load_time']:.1f}ms\n\n")
        
        if sequential_results:
            f.write("📊 SEQUENTIAL MULTI-MODEL PROCESSING\n")
            f.write("-" * 40 + "\n")
            f.write(f"Successful frames: {sequential_results['successful_frames']}\n")
            f.write(f"Failed frames: {sequential_results['failed_frames']}\n")
            f.write(f"Total test time: {sequential_results['total_time']:.1f}s\n")
            f.write(f"Overall FPS: {sequential_results['overall_fps']:.1f}\n")
            f.write(f"Average frame time: {sequential_results['avg_frame_time']:.1f}ms\n")
            f.write(f"Average processing time: {sequential_results['avg_processing_time']:.1f}ms\n\n")
        
        if concurrent_results:
            f.write("🚀 CONCURRENT MULTI-MODEL PROCESSING\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total frames processed: {concurrent_results['total_frames']}\n")
            f.write(f"Models used concurrently: {concurrent_results['models_count']}\n")
            f.write(f"Total test time: {concurrent_results['total_time']:.1f}s\n")
            f.write(f"Overall FPS: {concurrent_results['overall_fps']:.1f}\n")
            f.write(f"Average frame time: {concurrent_results['avg_frame_time']:.1f}ms\n")
            f.write(f"Multi-model scaling: {concurrent_results['overall_fps']:.1f} FPS across {concurrent_results['models_count']} models\n\n")
        
        if production_results:
            f.write("🏭 PRODUCTION MULTI-MODEL BATCH PROCESSING\n")
            f.write("-" * 45 + "\n")
            f.write(f"Total frames processed: {production_results['total_frames']}\n")
            f.write(f"Total batches: {production_results['total_batches']}\n")
            f.write(f"Total test time: {production_results['total_time']:.1f}s\n")
            f.write(f"Overall FPS: {production_results['overall_fps']:.1f}\n")
            f.write(f"Average batch time: {production_results['avg_batch_time']:.1f}ms\n")
            f.write(f"Time per frame: {production_results['time_per_frame']:.1f}ms\n")
            f.write(f"Production latency: {production_results['avg_batch_time']:.1f}ms per 5-frame batch\n\n")
        
        if mixed_results:
            f.write("🎲 MIXED WORKLOAD SCENARIO\n")
            f.write("-" * 25 + "\n")
            f.write(f"Test duration: {mixed_results['duration']:.1f}s\n")
            f.write(f"Single requests: {mixed_results['single_requests']}\n")
            f.write(f"Batch requests: {mixed_results['batch_requests']}\n")
            f.write(f"Total frames: {mixed_results['total_frames']}\n")
            f.write(f"Failed requests: {mixed_results['failed_requests']}\n")
            f.write(f"Overall FPS: {mixed_results['overall_fps']:.1f}\n")
            f.write(f"Average request time: {mixed_results['avg_request_time']:.1f}ms\n")
            f.write(f"Requests per second: {mixed_results['requests_per_second']:.1f}\n\n")
        
        # Multi-model insights
        f.write("🔥 MULTI-MODEL INSIGHTS\n")
        f.write("-" * 25 + "\n")
        f.write(f"Models tested: {len(models)}\n")
        
        if sequential_results and concurrent_results:
            scaling_factor = concurrent_results['overall_fps'] / sequential_results['overall_fps']
            f.write(f"Concurrent scaling: {scaling_factor:.1f}x improvement\n")
        
        if production_results:
            f.write(f"Production readiness: {'✅ YES' if production_results['overall_fps'] >= 30 else '❌ NO'}\n")
            f.write(f"Multi-model production FPS: {production_results['overall_fps']:.1f}\n")
        
        f.write("\n🎯 MULTI-MODEL TEST HIGHLIGHTS\n")
        f.write("-" * 30 + "\n")
        f.write("- Multiple model loading and performance testing\n")
        f.write("- Sequential and concurrent multi-model processing\n")
        f.write("- Production batch processing across models\n")
        f.write("- Mixed workload scenario simulation\n")
        f.write("- Real-world multi-model deployment validation\n\n")
        
        f.write("📁 FILES GENERATED\n")
        f.write("-" * 15 + "\n")
        f.write("- predictions/ : Sample predictions from all models and scenarios\n")
        f.write("- grids/ : Visual comparisons of multi-model processing\n")
        f.write("- models/ : Sample predictions from each loaded model\n")
        f.write("- analysis/ : This comprehensive multi-model report\n")
    
    print(f"📄 Multi-model analysis report saved: {report_path}")

def main():
    """Run the comprehensive multi-model test"""
    
    print("🚀 MULTI-MODEL gRPC High-Speed Performance Test")
    print("=" * 70)
    print("🎯 Testing multiple models with various scenarios")
    print("📁 Results will be saved to a timestamped directory")
    print()
    
    # Create test directory
    test_dir = create_test_directory()
    
    # Get available models
    models = get_available_models()
    
    if not models:
        print("❌ No models found! Please ensure model files exist.")
        return
    
    print(f"📦 Found {len(models)} models: {', '.join(models)}")
    print()
    
    # Test model loading performance
    loading_results = test_model_loading_performance(test_dir, models)
    
    # Test sequential multi-model processing
    sequential_results = test_sequential_multi_model(test_dir, models, frames_per_model=20)
    
    # Test concurrent multi-model processing
    concurrent_results = test_concurrent_multi_model(test_dir, models, frames_per_model=30)
    
    # Test production multi-model batch processing
    production_results = test_production_multi_model_batch(test_dir, models, batches_per_model=10)
    
    # Test mixed workload scenario
    mixed_results = test_mixed_workload(test_dir, models, duration_seconds=30)
    
    # Create analysis grids
    create_analysis_grids(test_dir)
    
    # Save comprehensive analysis report
    save_analysis_report(test_dir, loading_results, sequential_results, concurrent_results, 
                        production_results, mixed_results, models)
    
    print(f"\n✅ MULTI-MODEL Test Complete!")
    print(f"📁 Results saved to: {test_dir}")
    print(f"📊 Check analysis/ for comprehensive multi-model report")
    print(f"🎨 Check grids/ for visual multi-model comparisons")
    print(f"📷 Check predictions/ for sample predictions from all scenarios")
    
    # Print summary
    print(f"\n🔥 MULTI-MODEL PERFORMANCE SUMMARY:")
    if loading_results:
        print(f"   📦 Model Loading: {loading_results['avg_load_time']:.1f}ms average")
    if sequential_results:
        print(f"   📊 Sequential: {sequential_results['overall_fps']:.1f} FPS")
    if concurrent_results:
        print(f"   🚀 Concurrent: {concurrent_results['overall_fps']:.1f} FPS ({concurrent_results['models_count']} models)")
    if production_results:
        print(f"   🏭 Production: {production_results['overall_fps']:.1f} FPS (batch processing)")
    if mixed_results:
        print(f"   🎲 Mixed Load: {mixed_results['overall_fps']:.1f} FPS ({mixed_results['requests_per_second']:.1f} req/s)")
    
    if concurrent_results and sequential_results:
        scaling = concurrent_results['overall_fps'] / sequential_results['overall_fps']
        print(f"   📈 Multi-model scaling: {scaling:.1f}x improvement with concurrency")

if __name__ == "__main__":
    main()
