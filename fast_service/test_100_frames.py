#!/usr/bin/env python3
"""
High-Speed 100 Frame Test
Tests gRPC performance with 100 frames and saves results for review.
"""

import os
import time
import cv2
import numpy as np
import grpc
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc
from datetime import datetime

def create_test_directory():
    """Create a timestamped directory for test results"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"test_100_frames_{timestamp}"
    
    # Create main directory
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{test_dir}/predictions", exist_ok=True)
    os.makedirs(f"{test_dir}/analysis", exist_ok=True)
    os.makedirs(f"{test_dir}/grids", exist_ok=True)
    
    print(f"📁 Created test directory: {test_dir}")
    return test_dir

def test_100_frames_individual(test_dir):
    """Test 100 individual frame requests"""
    
    print("\n🚀 Testing 100 Individual Frame Requests")
    print("=" * 50)
    
    # Connect to gRPC server
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    model_name = "test_optimized_package_fixed_3"
    
    times = []
    processing_times = []
    sizes = []
    failed_frames = []
    
    overall_start = time.time()
    
    for frame_id in range(100):
        if frame_id % 10 == 0:
            print(f"📷 Processing frame {frame_id}/100...")
        
        start_time = time.time()
        
        try:
            # Create request
            request = lipsyncsrv_pb2.InferenceRequest(
                model_name=model_name,
                frame_id=frame_id % 10  # Cycle through frames 0-9
            )
            
            # Call gRPC service
            response = stub.GenerateInference(request)
            
            total_time = (time.time() - start_time) * 1000
            
            if response.success:
                times.append(total_time)
                processing_times.append(response.processing_time_ms)
                sizes.append(len(response.prediction_data))
                
                # Save prediction
                filename = f"{test_dir}/predictions/frame_{frame_id:03d}.jpg"
                with open(filename, "wb") as f:
                    f.write(response.prediction_data)
                
            else:
                failed_frames.append(frame_id)
                print(f"   ❌ Frame {frame_id} failed: {response.error}")
                
        except Exception as e:
            failed_frames.append(frame_id)
            print(f"   💥 Frame {frame_id} exception: {e}")
    
    overall_time = time.time() - overall_start
    
    # Calculate statistics
    if times:
        avg_total = sum(times) / len(times)
        avg_processing = sum(processing_times) / len(processing_times)
        avg_size = sum(sizes) / len(sizes)
        network_overhead = avg_total - avg_processing
        overhead_percentage = (network_overhead / avg_total) * 100
        
        successful_frames = len(times)
        overall_fps = successful_frames / overall_time
        
        print(f"\n📊 100-Frame Individual Test Results:")
        print(f"   ✅ Successful frames: {successful_frames}/100")
        print(f"   ❌ Failed frames: {len(failed_frames)}")
        print(f"   ⏱️  Total test time: {overall_time:.1f}s")
        print(f"   🎬 Overall FPS: {overall_fps:.1f}")
        print(f"   📊 Average frame time: {avg_total:.1f}ms")
        print(f"   🖥️  Average processing: {avg_processing:.1f}ms")
        print(f"   🌐 Network overhead: {overhead_percentage:.1f}%")
        print(f"   🗂️  Average size: {avg_size:,} bytes ({avg_size/1024:.1f} KB)")
        
        return {
            "successful_frames": successful_frames,
            "failed_frames": len(failed_frames),
            "total_time": overall_time,
            "overall_fps": overall_fps,
            "avg_frame_time": avg_total,
            "avg_processing_time": avg_processing,
            "network_overhead_pct": overhead_percentage,
            "avg_size_bytes": avg_size
        }
    
    return None

def test_100_frames_batch(test_dir):
    """Test 100 frames in batches"""
    
    print("\n🚀 Testing 100 Frames in Batches (10 frames per batch)")
    print("=" * 50)
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
    model_name = "test_optimized_package_fixed_3"
    
    batch_times = []
    total_frames_processed = 0
    
    overall_start = time.time()
    
    for batch_num in range(10):  # 10 batches of 10 frames each
        print(f"📦 Processing batch {batch_num + 1}/10...")
        
        # Create batch of 10 frames
        frame_ids = [(batch_num * 10 + i) % 10 for i in range(10)]  # Cycle through 0-9
        
        start_time = time.time()
        
        try:
            # Create batch request
            request = lipsyncsrv_pb2.BatchInferenceRequest(
                model_name=model_name,
                frame_ids=frame_ids
            )
            
            # Call batch inference
            response = stub.GenerateBatchInference(request)
            
            batch_time = (time.time() - start_time) * 1000
            batch_times.append(batch_time)
            
            successful_in_batch = sum(1 for r in response.responses if r.success)
            total_frames_processed += successful_in_batch
            
            # Save batch predictions
            for i, frame_response in enumerate(response.responses):
                if frame_response.success:
                    frame_id = batch_num * 10 + i
                    filename = f"{test_dir}/predictions/batch_frame_{frame_id:03d}.jpg"
                    with open(filename, "wb") as f:
                        f.write(frame_response.prediction_data)
            
            print(f"   ✅ Batch {batch_num + 1}: {successful_in_batch}/10 frames in {batch_time:.1f}ms")
            
        except Exception as e:
            print(f"   💥 Batch {batch_num + 1} failed: {e}")
    
    overall_time = time.time() - overall_start
    
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        overall_fps = total_frames_processed / overall_time
        
        print(f"\n📊 100-Frame Batch Test Results:")
        print(f"   ✅ Total frames processed: {total_frames_processed}/100")
        print(f"   📦 Successful batches: {len(batch_times)}/10")
        print(f"   ⏱️  Total test time: {overall_time:.1f}s")
        print(f"   🎬 Overall FPS: {overall_fps:.1f}")
        print(f"   📊 Average batch time: {avg_batch_time:.1f}ms")
        print(f"   🔄 Time per frame: {avg_batch_time/10:.1f}ms")
        
        return {
            "total_frames": total_frames_processed,
            "successful_batches": len(batch_times),
            "total_time": overall_time,
            "overall_fps": overall_fps,
            "avg_batch_time": avg_batch_time,
            "time_per_frame": avg_batch_time/10
        }
    
    return None

def create_analysis_grids(test_dir):
    """Create grid visualizations of the results"""
    
    print("\n🎨 Creating Analysis Grids...")
    print("=" * 30)
    
    predictions_dir = f"{test_dir}/predictions"
    
    # Find all prediction files
    individual_files = []
    batch_files = []
    
    for filename in os.listdir(predictions_dir):
        if filename.startswith("frame_") and filename.endswith(".jpg"):
            individual_files.append(filename)
        elif filename.startswith("batch_frame_") and filename.endswith(".jpg"):
            batch_files.append(filename)
    
    individual_files.sort()
    batch_files.sort()
    
    # Create grid of first 20 individual frames
    if len(individual_files) >= 20:
        create_frame_grid(predictions_dir, individual_files[:20], 
                         f"{test_dir}/grids/individual_frames_grid_1-20.jpg", 
                         "Individual Frames 1-20", 4, 5)
    
    # Create grid of frames 21-40
    if len(individual_files) >= 40:
        create_frame_grid(predictions_dir, individual_files[20:40], 
                         f"{test_dir}/grids/individual_frames_grid_21-40.jpg", 
                         "Individual Frames 21-40", 4, 5)
    
    # Create grid of first 20 batch frames
    if len(batch_files) >= 20:
        create_frame_grid(predictions_dir, batch_files[:20], 
                         f"{test_dir}/grids/batch_frames_grid_1-20.jpg", 
                         "Batch Frames 1-20", 4, 5)
    
    # Create comparison grid (first 10 individual vs first 10 batch)
    if len(individual_files) >= 10 and len(batch_files) >= 10:
        create_comparison_grid(predictions_dir, 
                              individual_files[:10], batch_files[:10],
                              f"{test_dir}/grids/individual_vs_batch_comparison.jpg")
    
    print(f"   ✅ Analysis grids saved to: {test_dir}/grids/")

def create_frame_grid(predictions_dir, filenames, output_path, title, rows, cols):
    """Create a grid of frame images"""
    
    if not filenames:
        return
    
    # Load first image to get dimensions
    first_img = cv2.imread(os.path.join(predictions_dir, filenames[0]))
    if first_img is None:
        return
    
    frame_height, frame_width = first_img.shape[:2]
    
    # Create grid
    grid_height = frame_height * rows + 50  # Extra space for title
    grid_width = frame_width * cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(grid, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Place frames in grid
    for i, filename in enumerate(filenames[:rows*cols]):
        row = i // cols
        col = i % cols
        
        img = cv2.imread(os.path.join(predictions_dir, filename))
        if img is not None:
            y_start = row * frame_height + 50
            y_end = y_start + frame_height
            x_start = col * frame_width
            x_end = x_start + frame_width
            
            grid[y_start:y_end, x_start:x_end] = img
            
            # Add frame number
            frame_num = filename.split('_')[1].split('.')[0]
            cv2.putText(grid, f"#{frame_num}", (x_start + 5, y_start + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, grid)
    print(f"   📷 Created: {os.path.basename(output_path)}")

def create_comparison_grid(predictions_dir, individual_files, batch_files, output_path):
    """Create a comparison grid showing individual vs batch results"""
    
    # Load first image to get dimensions
    first_img = cv2.imread(os.path.join(predictions_dir, individual_files[0]))
    if first_img is None:
        return
    
    frame_height, frame_width = first_img.shape[:2]
    
    # Create grid (2 rows, 10 columns)
    grid_height = frame_height * 2 + 80  # Extra space for labels
    grid_width = frame_width * 10
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Add labels
    cv2.putText(grid, "Individual Requests (Top Row)", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(grid, "Batch Requests (Bottom Row)", (20, frame_height + 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Place individual frames (top row)
    for i, filename in enumerate(individual_files[:10]):
        img = cv2.imread(os.path.join(predictions_dir, filename))
        if img is not None:
            x_start = i * frame_width
            x_end = x_start + frame_width
            y_start = 50
            y_end = y_start + frame_height
            
            grid[y_start:y_end, x_start:x_end] = img
    
    # Place batch frames (bottom row)
    for i, filename in enumerate(batch_files[:10]):
        img = cv2.imread(os.path.join(predictions_dir, filename))
        if img is not None:
            x_start = i * frame_width
            x_end = x_start + frame_width
            y_start = frame_height + 80
            y_end = y_start + frame_height
            
            grid[y_start:y_end, x_start:x_end] = img
    
    cv2.imwrite(output_path, grid)
    print(f"   📷 Created: {os.path.basename(output_path)}")

def save_analysis_report(test_dir, individual_results, batch_results):
    """Save detailed analysis report"""
    
    report_path = f"{test_dir}/analysis/performance_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🚀 gRPC Lip Sync 100-Frame Performance Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if individual_results:
            f.write("📊 INDIVIDUAL FRAME REQUESTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Successful frames: {individual_results['successful_frames']}/100\n")
            f.write(f"Failed frames: {individual_results['failed_frames']}\n")
            f.write(f"Total test time: {individual_results['total_time']:.1f}s\n")
            f.write(f"Overall FPS: {individual_results['overall_fps']:.1f}\n")
            f.write(f"Average frame time: {individual_results['avg_frame_time']:.1f}ms\n")
            f.write(f"Average processing time: {individual_results['avg_processing_time']:.1f}ms\n")
            f.write(f"Network overhead: {individual_results['network_overhead_pct']:.1f}%\n")
            f.write(f"Average size: {individual_results['avg_size_bytes']:,} bytes\n\n")
        
        if batch_results:
            f.write("📦 BATCH REQUESTS (10 frames per batch)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total frames processed: {batch_results['total_frames']}/100\n")
            f.write(f"Successful batches: {batch_results['successful_batches']}/10\n")
            f.write(f"Total test time: {batch_results['total_time']:.1f}s\n")
            f.write(f"Overall FPS: {batch_results['overall_fps']:.1f}\n")
            f.write(f"Average batch time: {batch_results['avg_batch_time']:.1f}ms\n")
            f.write(f"Time per frame: {batch_results['time_per_frame']:.1f}ms\n\n")
        
        if individual_results and batch_results:
            f.write("🔥 PERFORMANCE COMPARISON\n")
            f.write("-" * 25 + "\n")
            individual_fps = individual_results['overall_fps']
            batch_fps = batch_results['overall_fps']
            speedup = batch_fps / individual_fps if individual_fps > 0 else 0
            f.write(f"Individual requests: {individual_fps:.1f} FPS\n")
            f.write(f"Batch requests: {batch_fps:.1f} FPS\n")
            f.write(f"Batch speedup: {speedup:.1f}x faster\n\n")
        
        f.write("📁 FILES GENERATED\n")
        f.write("-" * 15 + "\n")
        f.write("- predictions/ : All 100 prediction frames\n")
        f.write("- grids/ : Visual grid comparisons\n")
        f.write("- analysis/ : This performance report\n")
    
    print(f"📄 Performance report saved: {report_path}")

def main():
    """Run the comprehensive 100-frame test"""
    
    print("🚀 gRPC High-Speed 100-Frame Test")
    print("=" * 60)
    print("🎯 Testing both individual requests and batch processing")
    print("📁 All results will be saved to a timestamped directory")
    print()
    
    # Create test directory
    test_dir = create_test_directory()
    
    # Test individual requests
    individual_results = test_100_frames_individual(test_dir)
    
    # Test batch requests
    batch_results = test_100_frames_batch(test_dir)
    
    # Create analysis grids
    create_analysis_grids(test_dir)
    
    # Save analysis report
    save_analysis_report(test_dir, individual_results, batch_results)
    
    print(f"\n✅ 100-Frame Test Complete!")
    print(f"📁 All results saved to: {test_dir}")
    print(f"📊 Check the analysis/ folder for performance report")
    print(f"🎨 Check the grids/ folder for visual comparisons")
    print(f"📷 Check the predictions/ folder for all 100+ frames")

if __name__ == "__main__":
    main()
