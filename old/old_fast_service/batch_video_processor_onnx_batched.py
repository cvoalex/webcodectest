"""
ULTRA-FAST Batch Video Processor with Variable Batch Inference
- RAM caching (zero disk I/O)
- GPU batch inference (process multiple frames simultaneously)
- Variable batch sizes (1, 4, 8, 16, 32)
"""

import cv2
import numpy as np
import onnxruntime as ort
import json
import time
from pathlib import Path

class BatchedVideoProcessor:
    def __init__(self, model_path, data_dir):
        self.model_path = model_path
        self.data_dir = Path(data_dir)
        
        # Cached data
        self.audio_features = None
        self.crop_rectangles = None
        self.roi_frames = []
        self.model_input_frames = []
        self.crop_328_frames = []
        self.full_body_frames = []
        self.session = None
        
    def preload_all_data(self):
        """Load everything into RAM once"""
        print("\n📦 Preloading all data into RAM...")
        start = time.time()
        
        # Load audio
        print("   Loading audio features...")
        audio_path = self.data_dir / "aud_ave.npy"
        self.audio_features = np.load(audio_path)
        print(f"   ✅ Audio: {self.audio_features.shape}")
        
        # Load crop rectangles
        print("   Loading crop rectangles...")
        crop_rects_file = self.data_dir / "cache" / "crop_rectangles.json"
        with open(crop_rects_file, 'r') as f:
            self.crop_rectangles = json.load(f)
        print(f"   ✅ Crop rectangles: {len(self.crop_rectangles)} frames")
        
        # Load ALL video frames
        print("   Preloading ALL video frames...")
        
        # ROI frames
        print("      - rois_320_video.mp4...")
        cap = cv2.VideoCapture(str(self.data_dir / "rois_320_video.mp4"))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.roi_frames.append(frame)
        cap.release()
        print(f"        ✅ {len(self.roi_frames)} frames")
        
        # Model input frames
        print("      - model_inputs_video.mp4...")
        cap = cv2.VideoCapture(str(self.data_dir / "model_inputs_video.mp4"))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.model_input_frames.append(frame)
        cap.release()
        print(f"        ✅ {len(self.model_input_frames)} frames")
        
        # Crop 328 frames
        print("      - crops_328_video.mp4...")
        cap = cv2.VideoCapture(str(self.data_dir / "crops_328_video.mp4"))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.crop_328_frames.append(frame)
        cap.release()
        print(f"        ✅ {len(self.crop_328_frames)} frames")
        
        # Full body frames
        print("      - full_body_video.mp4...")
        cap = cv2.VideoCapture(str(self.data_dir / "full_body_video.mp4"))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.full_body_frames.append(frame)
        cap.release()
        print(f"        ✅ {len(self.full_body_frames)} frames")
        
        # Load ONNX model
        print("\n   Loading ONNX model...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        print(f"   ✅ Model loaded with {self.session.get_providers()}")
        
        # Print memory usage
        roi_size = len(self.roi_frames) * self.roi_frames[0].nbytes / (1024**2)
        model_input_size = len(self.model_input_frames) * self.model_input_frames[0].nbytes / (1024**2)
        crop_328_size = len(self.crop_328_frames) * self.crop_328_frames[0].nbytes / (1024**2)
        full_body_size = len(self.full_body_frames) * self.full_body_frames[0].nbytes / (1024**2)
        
        print(f"\n💾 Memory Usage:")
        print(f"   ROI frames: {roi_size:.1f} MB")
        print(f"   Model input frames: {model_input_size:.1f} MB")
        print(f"   Crop 328 frames: {crop_328_size:.1f} MB")
        print(f"   Full body frames: {full_body_size:.1f} MB")
        print(f"   Total: {roi_size + model_input_size + crop_328_size + full_body_size:.1f} MB")
        
        elapsed = time.time() - start
        print(f"\n⚡ Preload completed in {elapsed:.2f}s")
        print(f"   Ready to process {len(self.roi_frames)} frames with ZERO disk I/O!")
        
    def prepare_batch_inputs(self, frame_ids):
        """Prepare batched inputs for multiple frames"""
        batch_size = len(frame_ids)
        
        # Prepare visual inputs [batch_size, 6, 320, 320]
        visual_batch = []
        for frame_id in frame_ids:
            # Get 3 frames: frame-1, frame, frame+1
            frames = []
            for offset in [-1, 0, 1]:
                idx = max(0, min(len(self.model_input_frames) - 1, frame_id + offset))
                frame = self.model_input_frames[idx]
                frames.append(frame)
            
            # Stack and normalize: BGR [0,255] -> [0,1]
            stacked = np.concatenate(frames, axis=2)  # [320, 320, 9]
            stacked = stacked.transpose(2, 0, 1)  # [9, 320, 320]
            stacked = stacked.astype(np.float32) / 255.0
            
            # Take first 6 channels
            visual_batch.append(stacked[:6])
        
        visual_input = np.array(visual_batch, dtype=np.float32)  # [batch_size, 6, 320, 320]
        
        # Prepare audio inputs [batch_size, 32, 16, 16]
        audio_batch = []
        for frame_id in frame_ids:
            # 16-frame window centered on frame_id
            start_idx = max(0, frame_id - 8)
            end_idx = min(len(self.audio_features), frame_id + 8)
            
            # Get audio features and pad if needed
            audio_window = []
            for i in range(frame_id - 8, frame_id + 8):
                if i < 0 or i >= len(self.audio_features):
                    audio_window.append(np.zeros(512, dtype=np.float32))
                else:
                    audio_window.append(self.audio_features[i])
            
            audio_features = np.array(audio_window, dtype=np.float32)  # [16, 512]
            audio_features = audio_features.reshape(32, 16, 16)
            audio_batch.append(audio_features)
        
        audio_input = np.array(audio_batch, dtype=np.float32)  # [batch_size, 32, 16, 16]
        
        return visual_input, audio_input
    
    def run_batch_inference(self, frame_ids):
        """Run inference on a batch of frames"""
        visual_input, audio_input = self.prepare_batch_inputs(frame_ids)
        
        # Get input names dynamically
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]
        
        # Run inference
        start = time.time()
        outputs = self.session.run(
            output_names,
            {
                input_names[0]: visual_input,
                input_names[1]: audio_input
            }
        )
        inference_time = (time.time() - start) * 1000
        
        predictions = outputs[0]  # [batch_size, 3, 320, 320]
        
        # Convert back to BGR [0,255]
        predictions = (predictions * 255.0).astype(np.uint8)
        predictions = predictions.transpose(0, 2, 3, 1)  # [batch_size, 320, 320, 3]
        
        return predictions, inference_time
    
    def composite_frame(self, frame_id, prediction):
        """Composite one frame (from RAM)"""
        # Get crop_328 frame from RAM
        crop_328 = self.crop_328_frames[frame_id].copy()
        
        # Place prediction in center [4:324, 4:324]
        crop_328[4:324, 4:324] = prediction
        
        # Get full body frame from RAM
        full_frame = self.full_body_frames[frame_id].copy()
        
        # Get original crop rectangle
        rect = self.crop_rectangles[str(frame_id)]["rect"]
        x1, y1, x2, y2 = rect
        orig_width = x2 - x1
        orig_height = y2 - y1
        
        # Resize 328x328 back to original size
        crop_resized = cv2.resize(crop_328, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        
        # Composite at original position
        full_frame[y1:y2, x1:x2] = crop_resized
        
        return full_frame
    
    def process_batch_with_variable_batching(self, batch_size=8, num_frames=100):
        """Process frames with variable batch inference"""
        print(f"\n{'='*80}")
        print(f"⚡ ULTRA-FAST PROCESSING (Batch Size: {batch_size})")
        print(f"{'='*80}")
        
        output_dir = Path("output_batch_onnx_batched")
        output_dir.mkdir(exist_ok=True)
        
        total_inference_time = 0
        total_composite_time = 0
        frame_count = 0
        
        start_time = time.time()
        
        # Process in batches
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            frame_ids = list(range(batch_start, batch_end))
            current_batch_size = len(frame_ids)
            
            # Batch inference
            predictions, inference_time = self.run_batch_inference(frame_ids)
            total_inference_time += inference_time
            
            # Composite each frame
            composite_start = time.time()
            for i, frame_id in enumerate(frame_ids):
                full_frame = self.composite_frame(frame_id, predictions[i])
                
                # Save
                output_path = output_dir / f"frame_{frame_id:04d}.jpg"
                cv2.imwrite(str(output_path), full_frame)
                
                frame_count += 1
            
            composite_time = (time.time() - composite_start) * 1000
            total_composite_time += composite_time
            
            # Print progress
            if (batch_end) % 10 == 0 or batch_end == num_frames:
                avg_inf = inference_time / current_batch_size
                avg_comp = composite_time / current_batch_size
                print(f"   Processed {batch_end}/{num_frames} frames "
                      f"({avg_inf:.2f}ms inference/frame, {avg_comp:.2f}ms composite/frame)")
        
        total_time = time.time() - start_time
        
        # Statistics
        avg_inference = total_inference_time / frame_count
        avg_composite = total_composite_time / frame_count
        fps = frame_count / total_time
        
        print(f"\n📊 Performance Statistics (Batch Size {batch_size}):")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Frames processed: {frame_count}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Avg inference time: {avg_inference:.2f}ms/frame")
        print(f"   Avg composite time: {avg_composite:.2f}ms/frame")
        print(f"   Throughput (inference only): {1000/avg_inference:.1f} FPS")
        print(f"\n✅ Frames saved to {output_dir}/")
        
        return {
            'batch_size': batch_size,
            'total_time': total_time,
            'fps': fps,
            'avg_inference': avg_inference,
            'avg_composite': avg_composite
        }

def benchmark_batch_sizes():
    """Benchmark different batch sizes"""
    print("🚀 ULTRA-FAST Batched Processor - Variable Batch Size Benchmark")
    print("="*80)
    
    model_path = "minimal_server/models/sanders/checkpoint/model_best.onnx"
    data_dir = "minimal_server/models/sanders"
    
    processor = BatchedVideoProcessor(model_path, data_dir)
    processor.preload_all_data()
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16, 32]
    results = []
    
    print("\n" + "="*80)
    print("🎯 BATCH SIZE BENCHMARK")
    print("="*80)
    
    for batch_size in batch_sizes:
        result = processor.process_batch_with_variable_batching(
            batch_size=batch_size, 
            num_frames=100
        )
        results.append(result)
        print()
    
    # Summary table
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Batch Size':<12} {'FPS':<10} {'Inf/Frame':<15} {'Comp/Frame':<15} {'Total Time':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['batch_size']:<12} {r['fps']:<10.2f} {r['avg_inference']:<15.2f} "
              f"{r['avg_composite']:<15.2f} {r['total_time']:<12.2f}")
    
    # Find best
    best = max(results, key=lambda x: x['fps'])
    print("\n" + "="*80)
    print(f"🏆 BEST PERFORMANCE: Batch Size {best['batch_size']} @ {best['fps']:.2f} FPS")
    print(f"   Inference: {best['avg_inference']:.2f}ms/frame")
    print(f"   Composite: {best['avg_composite']:.2f}ms/frame")
    print("="*80)
    
    # Speedup comparison
    baseline = results[0]  # batch_size=1
    print(f"\n⚡ SPEEDUP vs Batch Size 1:")
    for r in results[1:]:
        speedup = r['fps'] / baseline['fps']
        inf_speedup = baseline['avg_inference'] / r['avg_inference']
        print(f"   Batch {r['batch_size']}: {speedup:.2f}x faster overall, "
              f"{inf_speedup:.2f}x faster inference")

if __name__ == "__main__":
    benchmark_batch_sizes()
