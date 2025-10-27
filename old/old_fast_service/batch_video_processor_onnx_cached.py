"""
BLAZING FAST Batch Video Processor with Aggressive Caching
Eliminates disk I/O bottleneck by preloading everything into RAM
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import onnxruntime as ort
import time
from pathlib import Path
import json

class CachedBatchProcessor:
    """Ultra-fast processor with everything cached in RAM"""
    
    def __init__(self, sanders_dir="d:/Projects/webcodecstest/minimal_server/models/sanders"):
        self.sanders_dir = sanders_dir
        
        # ONNX session
        self.session = None
        
        # Audio features
        self.audio_features = None
        
        # CACHED VIDEO FRAMES (loaded once into RAM)
        self.roi_frames = []           # 320x320 face ROIs
        self.model_input_frames = []   # 320x320 masked inputs
        self.crop_328_frames = []      # 328x328 original crops
        self.full_body_frames = []     # 1280x720 full frames
        
        # CACHED METADATA
        self.crop_rectangles = {}      # Original crop bounds
        
        print("🚀 Initializing BLAZING FAST cached processor...")
    
    def preload_all_data(self, max_frames=None):
        """Preload EVERYTHING into RAM for zero disk I/O during processing"""
        
        start_time = time.time()
        
        print("\n📦 Preloading all data into RAM...")
        
        # 1. Load audio features
        print("   Loading audio features...")
        audio_path = os.path.join(self.sanders_dir, "aud_ave.npy")
        self.audio_features = np.load(audio_path)
        print(f"   ✅ Audio: {self.audio_features.shape}")
        
        # 2. Load crop rectangles
        print("   Loading crop rectangles...")
        crop_rects_file = os.path.join(self.sanders_dir, "cache", "crop_rectangles.json")
        with open(crop_rects_file, 'r') as f:
            self.crop_rectangles = json.load(f)
        print(f"   ✅ Crop rectangles: {len(self.crop_rectangles)} frames")
        
        # 3. Preload ALL video frames
        print("   Preloading ALL video frames...")
        
        # ROI frames (320x320)
        print("      - rois_320_video.mp4...")
        roi_cap = cv2.VideoCapture(os.path.join(self.sanders_dir, "rois_320_video.mp4"))
        while True:
            ret, frame = roi_cap.read()
            if not ret or (max_frames and len(self.roi_frames) >= max_frames):
                break
            self.roi_frames.append(frame)
        roi_cap.release()
        print(f"        ✅ {len(self.roi_frames)} frames")
        
        # Model input frames (320x320 masked)
        print("      - model_inputs_video.mp4...")
        model_cap = cv2.VideoCapture(os.path.join(self.sanders_dir, "model_inputs_video.mp4"))
        while True:
            ret, frame = model_cap.read()
            if not ret or (max_frames and len(self.model_input_frames) >= max_frames):
                break
            self.model_input_frames.append(frame)
        model_cap.release()
        print(f"        ✅ {len(self.model_input_frames)} frames")
        
        # Crops 328x328 (original crops)
        print("      - crops_328_video.mp4...")
        crop_cap = cv2.VideoCapture(os.path.join(self.sanders_dir, "crops_328_video.mp4"))
        while True:
            ret, frame = crop_cap.read()
            if not ret or (max_frames and len(self.crop_328_frames) >= max_frames):
                break
            self.crop_328_frames.append(frame)
        crop_cap.release()
        print(f"        ✅ {len(self.crop_328_frames)} frames")
        
        # Full body frames (1280x720)
        print("      - full_body_video.mp4...")
        full_cap = cv2.VideoCapture(os.path.join(self.sanders_dir, "full_body_video.mp4"))
        while True:
            ret, frame = full_cap.read()
            if not ret or (max_frames and len(self.full_body_frames) >= max_frames):
                break
            self.full_body_frames.append(frame.copy())  # Important: copy!
        full_cap.release()
        print(f"        ✅ {len(self.full_body_frames)} frames")
        
        # 4. Initialize ONNX
        print("\n   Loading ONNX model...")
        model_path = os.path.join(self.sanders_dir, "checkpoint", "model_best.onnx")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"   ✅ Model loaded with {self.session.get_providers()}")
        
        preload_time = time.time() - start_time
        
        # Calculate memory usage
        roi_mem = len(self.roi_frames) * 320 * 320 * 3 / (1024**2)
        model_mem = len(self.model_input_frames) * 320 * 320 * 3 / (1024**2)
        crop_mem = len(self.crop_328_frames) * 328 * 328 * 3 / (1024**2)
        full_mem = len(self.full_body_frames) * 1280 * 720 * 3 / (1024**2)
        total_mem = roi_mem + model_mem + crop_mem + full_mem
        
        print(f"\n💾 Memory Usage:")
        print(f"   ROI frames: {roi_mem:.1f} MB")
        print(f"   Model input frames: {model_mem:.1f} MB")
        print(f"   Crop 328 frames: {crop_mem:.1f} MB")
        print(f"   Full body frames: {full_mem:.1f} MB")
        print(f"   Total: {total_mem:.1f} MB")
        print(f"\n⚡ Preload completed in {preload_time:.2f}s")
        print(f"   Ready to process {len(self.roi_frames)} frames with ZERO disk I/O!\n")
    
    def prepare_frame_input(self, frame_id):
        """Prepare visual and audio inputs for a frame (from RAM cache)"""
        
        # Get frames from RAM cache (instant!)
        roi_frame = self.roi_frames[frame_id]
        model_input_frame = self.model_input_frames[frame_id]
        
        # Normalize (keep BGR!)
        roi_norm = roi_frame.astype(np.float32) / 255.0
        model_input_norm = model_input_frame.astype(np.float32) / 255.0
        
        roi_tensor = np.transpose(roi_norm, (2, 0, 1))
        model_input_tensor = np.transpose(model_input_norm, (2, 0, 1))
        
        visual_input = np.concatenate([roi_tensor, model_input_tensor], axis=0)
        visual_input = np.expand_dims(visual_input, axis=0).astype(np.float32)
        
        # Get 16-frame audio window
        left = frame_id - 8
        right = frame_id + 8
        
        pad_left = max(0, -left)
        pad_right = max(0, right - len(self.audio_features))
        left = max(0, left)
        right = min(len(self.audio_features), right)
        
        audio_window = self.audio_features[left:right]
        
        if pad_left > 0:
            audio_window = np.concatenate([
                np.tile(self.audio_features[0:1], (pad_left, 1)),
                audio_window
            ], axis=0)
        if pad_right > 0:
            audio_window = np.concatenate([
                audio_window,
                np.tile(self.audio_features[-1:], (pad_right, 1))
            ], axis=0)
        
        audio_flat = audio_window.flatten()
        audio_reshaped = audio_flat.reshape(32, 16, 16)
        audio_input = np.expand_dims(audio_reshaped, axis=0).astype(np.float32)
        
        return visual_input, audio_input
    
    def infer_frame(self, visual_input, audio_input):
        """Run ONNX inference"""
        
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]
        
        outputs = self.session.run(
            output_names,
            {
                input_names[0]: visual_input,
                input_names[1]: audio_input
            }
        )
        
        return outputs[0][0]  # [3, 320, 320]
    
    def composite_frame_cached(self, prediction, frame_id):
        """Composite using CACHED data (blazing fast - no disk I/O!)"""
        
        # Convert prediction to image (keep BGR)
        pred_img = np.transpose(prediction, (1, 2, 0))
        pred_img = (pred_img * 255).clip(0, 255).astype(np.uint8)
        
        # Get crop_328 from RAM cache (instant!)
        crop_328 = self.crop_328_frames[frame_id].copy()
        
        # Place prediction in center (4-pixel border)
        crop_328[4:324, 4:324] = pred_img
        
        # Get full frame from RAM cache (instant!)
        full_frame = self.full_body_frames[frame_id].copy()
        
        # Get crop rectangle from RAM cache (instant!)
        if str(frame_id) in self.crop_rectangles:
            rect = self.crop_rectangles[str(frame_id)]["rect"]
            x1, y1, x2, y2 = rect
            orig_width = x2 - x1
            orig_height = y2 - y1
            
            # Resize to original size
            crop_resized = cv2.resize(crop_328, (orig_width, orig_height), 
                                     interpolation=cv2.INTER_CUBIC)
            
            # Ensure bounds are within frame
            H, W = full_frame.shape[:2]
            x1 = max(0, min(x1, W - orig_width))
            y1 = max(0, min(y1, H - orig_height))
            x2 = x1 + orig_width
            y2 = y1 + orig_height
            
            # Composite (fast!)
            full_frame[y1:y2, x1:x2] = crop_resized
            return full_frame
        
        # Fallback
        H, W = full_frame.shape[:2]
        x1 = (W - 328) // 2
        y1 = (H - 328) // 2
        x2 = x1 + 328
        y2 = y1 + 328
        full_frame[y1:y2, x1:x2] = crop_328
        
        return full_frame
    
    def process_batch_cached(self, num_frames, output_dir="output_batch_onnx_cached"):
        """Process batch with ALL data cached in RAM"""
        
        print("\n" + "="*80)
        print("⚡ BLAZING FAST BATCH PROCESSING (ZERO DISK I/O)")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        inference_times = []
        composite_times = []
        total_start = time.time()
        
        for frame_id in range(num_frames):
            # Prepare inputs (from RAM)
            visual_input, audio_input = self.prepare_frame_input(frame_id)
            
            # Inference
            inf_start = time.time()
            prediction = self.infer_frame(visual_input, audio_input)
            inf_time = (time.time() - inf_start) * 1000
            inference_times.append(inf_time)
            
            # Composite (from RAM)
            comp_start = time.time()
            final_frame = self.composite_frame_cached(prediction, frame_id)
            comp_time = (time.time() - comp_start) * 1000
            composite_times.append(comp_time)
            
            # Save
            output_path = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(output_path, final_frame)
            
            if (frame_id + 1) % 10 == 0:
                print(f"   Processed {frame_id + 1}/{num_frames} frames "
                      f"({inf_time:.2f}ms inference, {comp_time:.2f}ms composite)")
        
        total_time = time.time() - total_start
        
        print(f"\n📊 Performance Statistics:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Frames processed: {num_frames}")
        print(f"   FPS: {num_frames / total_time:.2f}")
        print(f"   Avg inference time: {np.mean(inference_times):.2f}ms")
        print(f"   Avg composite time: {np.mean(composite_times):.2f}ms")
        print(f"   Throughput (inference only): {1000.0 / np.mean(inference_times):.1f} FPS")
        print(f"\n✅ Frames saved to {output_dir}/")

def main():
    processor = CachedBatchProcessor()
    
    # Preload everything into RAM (one-time cost)
    processor.preload_all_data(max_frames=100)
    
    # Process with ZERO disk I/O (blazing fast!)
    processor.process_batch_cached(num_frames=100)
    
    print("\n⚡ Done! Compositing is now BLAZING FAST with zero disk I/O!")

if __name__ == "__main__":
    main()
