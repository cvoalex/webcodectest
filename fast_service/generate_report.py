#!/usr/bin/env python3
"""
Generate the performance report for the completed test
"""

import os
from datetime import datetime

def save_analysis_report():
    """Save detailed analysis report"""
    
    # Use the existing directory from the test
    test_dir = "test_100_frames_20250905_152839"
    report_path = f"{test_dir}/analysis/performance_report.txt"
    
    # Results from the test output
    individual_results = {
        'successful_frames': 100,
        'failed_frames': 0,
        'total_time': 2.8,
        'overall_fps': 36.0,
        'avg_frame_time': 27.1,
        'avg_processing_time': 25.6,
        'network_overhead_pct': 5.9,
        'avg_size_bytes': 16765
    }
    
    batch_results = {
        'total_frames': 100,
        'successful_batches': 10,
        'total_time': 1.9,
        'overall_fps': 51.4,
        'avg_batch_time': 190.5,
        'time_per_frame': 19.0
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("🚀 gRPC Lip Sync 100-Frame Performance Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
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
        
        f.write("📦 BATCH REQUESTS (10 frames per batch)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total frames processed: {batch_results['total_frames']}/100\n")
        f.write(f"Successful batches: {batch_results['successful_batches']}/10\n")
        f.write(f"Total test time: {batch_results['total_time']:.1f}s\n")
        f.write(f"Overall FPS: {batch_results['overall_fps']:.1f}\n")
        f.write(f"Average batch time: {batch_results['avg_batch_time']:.1f}ms\n")
        f.write(f"Time per frame: {batch_results['time_per_frame']:.1f}ms\n\n")
        
        f.write("🔥 PERFORMANCE COMPARISON\n")
        f.write("-" * 25 + "\n")
        individual_fps = individual_results['overall_fps']
        batch_fps = batch_results['overall_fps']
        speedup = batch_fps / individual_fps if individual_fps > 0 else 0
        f.write(f"Individual requests: {individual_fps:.1f} FPS\n")
        f.write(f"Batch requests: {batch_fps:.1f} FPS\n")
        f.write(f"Batch speedup: {speedup:.1f}x faster\n\n")
        
        f.write("🎯 HIGHLIGHTS\n")
        f.write("-" * 12 + "\n")
        f.write("- 100% success rate (0 failed frames)\n")
        f.write("- Ultra-low latency: 27.1ms average per frame\n")
        f.write("- Excellent throughput: 51.4 FPS in batch mode\n")
        f.write("- Minimal network overhead: 5.9%\n")
        f.write("- Compact predictions: 16.4 KB per frame\n")
        f.write("- Production-ready real-time performance\n\n")
        
        f.write("📁 FILES GENERATED\n")
        f.write("-" * 15 + "\n")
        f.write("- predictions/ : All 200 prediction frames (100 individual + 100 batch)\n")
        f.write("- grids/ : Visual grid comparisons and analysis\n")
        f.write("- analysis/ : This performance report\n")
    
    print(f"📄 Performance report saved: {report_path}")

if __name__ == "__main__":
    save_analysis_report()
