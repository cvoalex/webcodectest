"""
🚀 BATCH INFERENCE ENGINE - Process Multiple Frames Simultaneously
Optimized for maximum GPU utilization by batching frame requests
"""

import torch
import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Any
from optimized_inference_engine import OptimizedModelPackage


class BatchInferenceEngine(OptimizedModelPackage):
    """Extended inference engine with batch processing capabilities"""
    
    def __init__(self, package_dir: str, max_batch_size: int = 4):
        super().__init__(package_dir)
        self.max_batch_size = max_batch_size
        print(f"🎯 Batch processing enabled: max_batch_size={max_batch_size}")
    
    def _prepare_image_tensor_batch(self, frame_ids: List[int]) -> torch.Tensor:
        """Prepare batched image tensors"""
        batch_tensors = []
        
        for frame_id in frame_ids:
            # Get model input frame (masked face)
            model_input = self.video_caches["model_inputs"].get_frame(frame_id)
            
            # Convert BGR to RGB
            model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            model_input = model_input.astype(np.float32) / 255.0
            
            # Transpose to [C, H, W]
            model_input = model_input.transpose(2, 0, 1)
            
            # Convert to tensor
            tensor = torch.from_numpy(model_input).to(self.device)
            batch_tensors.append(tensor)
        
        # Stack into batch [B, C, H, W]
        batch = torch.stack(batch_tensors, dim=0)
        return batch
    
    def _prepare_audio_tensor_batch(self, frame_ids: List[int]) -> torch.Tensor:
        """Prepare batched audio tensors"""
        batch_tensors = []
        
        for frame_id in frame_ids:
            # Get audio features (instant memory-mapped access!)
            audio_feat = self.audio_features[frame_id]  # Shape: [dim]
            audio_feat = audio_feat.reshape(1, -1, 1)  # [1, dim, 1]
            
            # Convert to tensor
            tensor = torch.from_numpy(audio_feat).float().to(self.device)
            batch_tensors.append(tensor)
        
        # Stack into batch [B, 1, dim, 1]
        batch = torch.stack(batch_tensors, dim=0).squeeze(1)  # [B, dim, 1]
        return batch
    
    async def generate_frames_batch(self, frame_ids: List[int]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Generate multiple frames in a single batch inference pass
        
        Args:
            frame_ids: List of frame IDs to generate
            
        Returns:
            List of (frame, metadata) tuples
        """
        
        if not frame_ids:
            return []
        
        batch_size = len(frame_ids)
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max {self.max_batch_size}")
        
        start_time = time.time()
        self.total_requests += batch_size
        
        # Validate all frame IDs
        max_frame = self.video_caches["model_inputs"].frame_count
        for frame_id in frame_ids:
            if frame_id >= max_frame:
                raise ValueError(f"Frame {frame_id} out of range [0, {max_frame-1}]")
        
        # Prepare batched inputs
        prepare_start = time.time()
        image_batch = self._prepare_image_tensor_batch(frame_ids)  # [B, 6, 320, 320]
        audio_batch = self._prepare_audio_tensor_batch(frame_ids)  # [B, dim, 1]
        prepare_time = (time.time() - prepare_start) * 1000
        
        # Run batched inference (CUDA will parallelize internally!)
        inference_start = time.time()
        with torch.no_grad():
            predictions = self.model(image_batch, audio_batch)  # [B, 3, 320, 320]
        
        inference_time = (time.time() - inference_start) * 1000
        per_frame_inference = inference_time / batch_size
        
        # Convert predictions to numpy
        predictions = predictions.cpu().numpy()  # [B, 3, 320, 320]
        predictions = (predictions * 255).astype(np.uint8)
        predictions = predictions.transpose(0, 2, 3, 1)  # [B, 320, 320, 3]
        
        # Composite each frame
        composite_start = time.time()
        results = []
        
        for idx, frame_id in enumerate(frame_ids):
            prediction = predictions[idx]  # [320, 320, 3]
            
            # Get full body frame
            full_body = self.video_caches["full_body"].get_frame(frame_id)
            
            # Get crop coordinates
            crop_info = self.crop_rectangles.get(str(frame_id), None)
            
            if crop_info:
                rect = crop_info["rect"]  # [x1, y1, x2, y2]
            else:
                # Fallback to center crop
                h, w = full_body.shape[:2]
                rect = [w//2 - 164, h//2 - 164, w//2 + 164, h//2 + 164]
            
            # Composite prediction onto full body
            final_frame = full_body.copy()
            x1, y1, x2, y2 = rect
            
            # Resize prediction to crop size
            crop_w, crop_h = x2 - x1, y2 - y1
            prediction_resized = cv2.resize(prediction, (crop_w, crop_h))
            
            # Paste onto full body
            final_frame[y1:y2, x1:x2] = prediction_resized
            
            # Metadata for this frame
            metadata = {
                "frame_id": frame_id,
                "model_name": self.model_name,
                "batch_size": batch_size,
                "batch_inference_time_ms": inference_time,
                "per_frame_inference_ms": per_frame_inference,
                "prepare_time_ms": prepare_time / batch_size,
            }
            
            results.append((final_frame, metadata))
        
        composite_time = (time.time() - composite_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.inference_times.append(total_time)
        
        print(f"✅ Batch {batch_size} frames: {total_time:.2f}ms total ({per_frame_inference:.2f}ms per frame)")
        
        return results
    
    async def generate_frames_batch_inference_only(self, frame_ids: List[int]) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """
        Generate batched inference results + bounds for client-side compositing
        
        Returns:
            List of (prediction, bounds, metadata) tuples
        """
        
        if not frame_ids:
            return []
        
        batch_size = len(frame_ids)
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max {self.max_batch_size}")
        
        start_time = time.time()
        
        # Validate all frame IDs
        max_frame = self.video_caches["model_inputs"].frame_count
        for frame_id in frame_ids:
            if frame_id >= max_frame:
                raise ValueError(f"Frame {frame_id} out of range [0, {max_frame-1}]")
        
        # Prepare batched inputs
        image_batch = self._prepare_image_tensor_batch(frame_ids)
        audio_batch = self._prepare_audio_tensor_batch(frame_ids)
        
        # Run batched inference
        inference_start = time.time()
        with torch.no_grad():
            predictions = self.model(image_batch, audio_batch)
        inference_time = (time.time() - inference_start) * 1000
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        predictions = (predictions * 255).astype(np.uint8)
        predictions = predictions.transpose(0, 2, 3, 1)  # [B, 320, 320, 3]
        
        # Prepare results
        results = []
        for idx, frame_id in enumerate(frame_ids):
            prediction = predictions[idx]
            
            # Get bounds
            crop_info = self.crop_rectangles.get(str(frame_id), None)
            if crop_info:
                bounds = np.array(crop_info["rect"], dtype=np.float32)
            else:
                h, w = 1080, 1920  # Default dimensions
                bounds = np.array([w//2 - 164, h//2 - 164, w//2 + 164, h//2 + 164], dtype=np.float32)
            
            metadata = {
                "frame_id": frame_id,
                "model_name": self.model_name,
                "batch_size": batch_size,
                "inference_time_ms": inference_time / batch_size,
                "processing_time_ms": inference_time / batch_size
            }
            
            results.append((prediction, bounds, metadata))
        
        total_time = (time.time() - start_time) * 1000
        self.inference_times.append(total_time)
        
        return results
