#!/usr/bin/env python3
"""
Benchmark PyTorch vs ONNX Runtime inference speed
Tests on RTX 4090 with CUDA and TensorRT execution providers
"""

import torch
import numpy as np
import time
import sys
import os
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet_328 import Model

def benchmark_pytorch(
    model_path="models/default_model/models/99.pth",
    num_warmup=50,
    num_iterations=500,
    batch_size=1
):
    """
    Benchmark PyTorch model inference speed
    """
    print("\n" + "="*60)
    print("🔥 PYTORCH BENCHMARK")
    print("="*60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = Model(n_channels=6, mode='ave')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create test inputs
    test_visual = torch.randn(batch_size, 6, 320, 320, device=device)
    test_audio = torch.randn(batch_size, 32, 16, 16, device=device)
    
    # Warmup
    print(f"\n🔄 Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(test_visual, test_audio)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"⏱️  Running benchmark ({num_iterations} iterations)...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            output = model(test_visual, test_audio)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    times = np.array(times)
    results = {
        'mean': np.mean(times),
        'median': np.median(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
    
    print("\n📊 Results:")
    print(f"   Mean:   {results['mean']:.3f} ms")
    print(f"   Median: {results['median']:.3f} ms")
    print(f"   Std:    {results['std']:.3f} ms")
    print(f"   Min:    {results['min']:.3f} ms")
    print(f"   Max:    {results['max']:.3f} ms")
    print(f"   P95:    {results['p95']:.3f} ms")
    print(f"   P99:    {results['p99']:.3f} ms")
    print(f"   FPS:    {1000/results['mean']:.1f}")
    
    return results

def benchmark_onnx(
    model_path="models/default_model/models/99.onnx",
    num_warmup=50,
    num_iterations=500,
    batch_size=1,
    provider='CUDAExecutionProvider'
):
    """
    Benchmark ONNX Runtime model inference speed
    
    Args:
        provider: 'CUDAExecutionProvider' or 'TensorrtExecutionProvider'
    """
    import onnxruntime as ort
    
    print("\n" + "="*60)
    print(f"🚀 ONNX RUNTIME BENCHMARK ({provider})")
    print("="*60)
    
    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    
    if provider not in available_providers:
        print(f"❌ {provider} not available!")
        return None
    
    # Create session with specified provider
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # TensorRT specific options
    if provider == 'TensorrtExecutionProvider':
        provider_options = {
            'device_id': 0,
            'trt_max_workspace_size': 2147483648,  # 2GB
            'trt_fp16_enable': True,  # Enable FP16 for speed
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache'
        }
        providers = [(provider, provider_options)]
    else:
        providers = [provider]
    
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers
    )
    
    print(f"Using provider: {session.get_providers()[0]}")
    
    # Create test inputs
    test_visual = np.random.randn(batch_size, 6, 320, 320).astype(np.float32)
    test_audio = np.random.randn(batch_size, 32, 16, 16).astype(np.float32)
    
    visual_input_name = session.get_inputs()[0].name
    audio_input_name = session.get_inputs()[1].name
    
    # Warmup
    print(f"\n🔄 Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        _ = session.run(None, {visual_input_name: test_visual, audio_input_name: test_audio})
    
    # Benchmark
    print(f"⏱️  Running benchmark ({num_iterations} iterations)...")
    times = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        output = session.run(None, {visual_input_name: test_visual, audio_input_name: test_audio})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    times = np.array(times)
    results = {
        'mean': np.mean(times),
        'median': np.median(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
    
    print("\n📊 Results:")
    print(f"   Mean:   {results['mean']:.3f} ms")
    print(f"   Median: {results['median']:.3f} ms")
    print(f"   Std:    {results['std']:.3f} ms")
    print(f"   Min:    {results['min']:.3f} ms")
    print(f"   Max:    {results['max']:.3f} ms")
    print(f"   P95:    {results['p95']:.3f} ms")
    print(f"   P99:    {results['p99']:.3f} ms")
    print(f"   FPS:    {1000/results['mean']:.1f}")
    
    return results

def compare_results(pytorch_results, onnx_cuda_results, onnx_trt_results=None):
    """
    Compare and display results from all benchmarks
    """
    print("\n" + "="*60)
    print("📈 PERFORMANCE COMPARISON")
    print("="*60)
    
    baseline = pytorch_results['mean']
    
    print(f"\nPyTorch (baseline):           {pytorch_results['mean']:.3f} ms")
    print(f"ONNX + CUDA:                  {onnx_cuda_results['mean']:.3f} ms")
    
    speedup_cuda = baseline / onnx_cuda_results['mean']
    print(f"  → Speedup: {speedup_cuda:.2f}x ({((speedup_cuda-1)*100):.1f}% faster)")
    
    if onnx_trt_results:
        print(f"ONNX + TensorRT:              {onnx_trt_results['mean']:.3f} ms")
        speedup_trt = baseline / onnx_trt_results['mean']
        print(f"  → Speedup: {speedup_trt:.2f}x ({((speedup_trt-1)*100):.1f}% faster)")
        
        # TensorRT vs CUDA
        trt_vs_cuda = onnx_cuda_results['mean'] / onnx_trt_results['mean']
        print(f"\nTensorRT vs CUDA: {trt_vs_cuda:.2f}x faster")
    
    # FPS comparison
    print("\n📊 Frames Per Second (FPS):")
    print(f"PyTorch:       {1000/pytorch_results['mean']:.1f} FPS")
    print(f"ONNX + CUDA:   {1000/onnx_cuda_results['mean']:.1f} FPS")
    if onnx_trt_results:
        print(f"ONNX + TensorRT: {1000/onnx_trt_results['mean']:.1f} FPS")

def main():
    """
    Run complete benchmark suite
    """
    print("🎯 RTX 4090 Inference Benchmark: PyTorch vs ONNX Runtime")
    print("="*60)
    
    # Check if ONNX model exists
    onnx_path = "models/default_model/models/99.onnx"
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX model not found at: {onnx_path}")
        print("Please run export_to_onnx.py first!")
        return
    
    # Run benchmarks
    pytorch_results = benchmark_pytorch()
    onnx_cuda_results = benchmark_onnx(provider='CUDAExecutionProvider')
    
    # Try TensorRT if available
    try:
        import onnxruntime as ort
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            onnx_trt_results = benchmark_onnx(provider='TensorrtExecutionProvider')
        else:
            print("\n⚠️  TensorRT provider not available")
            onnx_trt_results = None
    except Exception as e:
        print(f"\n⚠️  Could not test TensorRT: {e}")
        onnx_trt_results = None
    
    # Compare results
    compare_results(pytorch_results, onnx_cuda_results, onnx_trt_results)
    
    print("\n✅ Benchmark complete!")

if __name__ == '__main__':
    main()
