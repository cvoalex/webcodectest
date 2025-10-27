#!/usr/bin/env python3
"""
Complete Python ONNX batch video processor with CORRECT preprocessing
Generates full lip-synced video from sanders dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
import onnxruntime as ort
from pathlib import Path
import json

class BatchVideoProcessor:
    def __init__(self, sanders_dir, onnx_model_path):
        self.sanders_dir = sanders_dir
        self.onnx_model_path = onnx_model_path
        
        # Load ONNX session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        
        # Load audio features
        self.audio_features = np.load(os.path.join(sanders_dir, "aud_ave.npy"))
        self.num_frames = len(self.audio_features)
        
        print(f"✅ Initialized processor")
        print(f"   Model: {onnx_model_path}")
        print(f"   Audio frames: {self.num_frames}")
        print(f"   Providers: {self.session.get_providers()}")
    
    def load_frame_data(self, frame_id):
        """Load visual and audio data for one frame"""
        
        # Load ROI and model input videos
        roi_cap = cv2.VideoCapture(os.path.join(self.sanders_dir, "rois_320_video.mp4"))
        model_input_cap = cv2.VideoCapture(os.path.join(self.sanders_dir, "model_inputs_video.mp4"))
        
        # Skip to frame
        for _ in range(frame_id):
            roi_cap.read()
            model_input_cap.read()
        
        ret1, roi_frame = roi_cap.read()
        ret2, model_input_frame = model_input_cap.read()
        
        roi_cap.release()
        model_input_cap.release()
        
        if not ret1 or not ret2:
            raise ValueError(f"Could not read frame {frame_id}")
        
        # CRITICAL: Keep BGR format (don't convert to RGB!)
        # Model was trained with cv2.imread which returns BGR
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
        
        return visual_input, audio_input, roi_frame
    
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
    
    def composite_frame(self, prediction, roi_frame, frame_id):
        """Composite prediction onto full frame using ORIGINAL crop rectangles from preprocessing"""
        
        # Convert prediction to image (keep BGR)
        pred_img = np.transpose(prediction, (1, 2, 0))
        pred_img = (pred_img * 255).clip(0, 255).astype(np.uint8)
        
        # Load crops_328 frame as template (this was the ORIGINAL crop before resizing to 320)
        crop_328_cap = cv2.VideoCapture(os.path.join(self.sanders_dir, "crops_328_video.mp4"))
        for _ in range(frame_id):
            crop_328_cap.read()
        ret, crop_328 = crop_328_cap.read()
        crop_328_cap.release()
        
        if not ret:
            # Fallback to resizing ROI
            crop_328 = cv2.resize(roi_frame, (328, 328), interpolation=cv2.INTER_CUBIC)
        
        # Place prediction in center (4-pixel border) - this is the KEY step
        crop_328[4:324, 4:324] = pred_img
        
        # Load full body frame
        full_cap = cv2.VideoCapture(os.path.join(self.sanders_dir, "full_body_video.mp4"))
        for _ in range(frame_id):
            full_cap.read()
        ret, full_frame = full_cap.read()
        full_cap.release()
        
        if not ret:
            return crop_328  # Fallback to just the face
        
        # Load ORIGINAL crop rectangle from preprocessing cache
        crop_rects_file = os.path.join(self.sanders_dir, "cache", "crop_rectangles.json")
        if os.path.exists(crop_rects_file):
            import json
            with open(crop_rects_file, 'r') as f:
                crop_rects = json.load(f)
            
            if str(frame_id) in crop_rects:
                # Get the ORIGINAL rectangle where this 328x328 crop came from
                rect = crop_rects[str(frame_id)]["rect"]
                x1, y1, x2, y2 = rect
                orig_width = x2 - x1
                orig_height = y2 - y1
                
                # Resize 328x328 back to ORIGINAL size (before it was cropped and resized)
                crop_resized = cv2.resize(crop_328, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
                
                # Ensure bounds are within frame
                H, W = full_frame.shape[:2]
                x1 = max(0, min(x1, W - orig_width))
                y1 = max(0, min(y1, H - orig_height))
                x2 = x1 + orig_width
                y2 = y1 + orig_height
                
                # Composite resized crop onto full frame at ORIGINAL position
                full_frame[y1:y2, x1:x2] = crop_resized
                return full_frame
        
        # Fallback: center the 328x328 crop on frame
        H, W = full_frame.shape[:2]
        x1 = (W - 328) // 2
        y1 = (H - 328) // 2
        x2 = x1 + 328
        y2 = y1 + 328
        full_frame[y1:y2, x1:x2] = crop_328
        
        return full_frame
    
    def process_batch(self, start_frame, end_frame, output_dir):
        """Process a batch of frames"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {
            'frames': [],
            'total_time': 0,
            'inference_times': [],
            'composite_times': []
        }
        
        batch_start = time.perf_counter()
        
        for frame_id in range(start_frame, end_frame):
            frame_start = time.perf_counter()
            
            # Load data
            visual_input, audio_input, roi_frame = self.load_frame_data(frame_id)
            
            # Inference
            inference_start = time.perf_counter()
            prediction = self.infer_frame(visual_input, audio_input)
            inference_time = (time.perf_counter() - inference_start) * 1000
            
            # Composite
            composite_start = time.perf_counter()
            final_frame = self.composite_frame(prediction, roi_frame, frame_id)
            composite_time = (time.perf_counter() - composite_start) * 1000
            
            # Save frame
            cv2.imwrite(f"{output_dir}/frame_{frame_id:04d}.jpg", final_frame)
            
            frame_time = (time.perf_counter() - frame_start) * 1000
            
            stats['inference_times'].append(inference_time)
            stats['composite_times'].append(composite_time)
            stats['frames'].append({
                'id': frame_id,
                'total_time_ms': frame_time,
                'inference_time_ms': inference_time,
                'composite_time_ms': composite_time
            })
            
            if (frame_id - start_frame + 1) % 10 == 0:
                print(f"   Processed {frame_id - start_frame + 1}/{end_frame - start_frame} frames "
                      f"({inference_time:.2f}ms inference, {composite_time:.2f}ms composite)")
        
        batch_time = time.perf_counter() - batch_start
        stats['total_time'] = batch_time
        
        return stats

def main():
    sanders_dir = "d:/Projects/webcodecstest/minimal_server/models/sanders"
    onnx_model = os.path.join(sanders_dir, "checkpoint/model_best.onnx")
    output_dir = "output_batch_onnx"
    
    print("\n" + "="*80)
    print("🎬 PYTHON ONNX BATCH VIDEO PROCESSOR")
    print("="*80)
    
    processor = BatchVideoProcessor(sanders_dir, onnx_model)
    
    # Process first 100 frames for testing
    num_frames = min(100, processor.num_frames)
    
    print(f"\n🚀 Processing {num_frames} frames...")
    stats = processor.process_batch(0, num_frames, output_dir)
    
    # Calculate statistics
    avg_inference = np.mean(stats['inference_times'])
    avg_composite = np.mean(stats['composite_times'])
    fps = num_frames / stats['total_time']
    
    print(f"\n📊 Performance Statistics:")
    print(f"   Total time: {stats['total_time']:.2f}s")
    print(f"   Frames processed: {num_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Avg inference time: {avg_inference:.2f}ms")
    print(f"   Avg composite time: {avg_composite:.2f}ms")
    print(f"   Throughput: {1000/avg_inference:.1f} FPS (inference only)")
    
    # Save stats
    with open(f"{output_dir}/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Frames saved to {output_dir}/")
    print(f"📊 Statistics saved to {output_dir}/stats.json")
    
    # Assemble video
    print(f"\n🎥 Assembling video...")
    
    # Create frame list for ffmpeg
    with open(f"{output_dir}/frames.txt", 'w') as f:
        for i in range(num_frames):
            f.write(f"file 'frame_{i:04d}.jpg'\n")
            f.write(f"duration 0.04\n")  # 25 FPS
    
    print(f"✅ Frame list created: {output_dir}/frames.txt")
    print(f"\nTo create video, run:")
    print(f"  cd {output_dir}")
    print(f"  ffmpeg -f concat -safe 0 -i frames.txt -i {sanders_dir}/aud.wav -t {num_frames/25} -c:v libx264 -pix_fmt yuv420p -c:a aac output.mp4")

if __name__ == "__main__":
    main()
