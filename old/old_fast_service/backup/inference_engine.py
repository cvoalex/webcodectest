import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import zipfile
import tempfile
import json
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from unet_328 import Model
from config import settings

# Import audio extraction if available
try:
    import sys
    import os
    # Add parent directory to path for data_utils import
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from data_utils.ave.audio import AudDataset
    AUDIO_EXTRACTION_AVAILABLE = True
except ImportError as e:
    AUDIO_EXTRACTION_AVAILABLE = False
    print(f"⚠️ Warning: Audio extraction modules not available: {e}")


class InferenceEngine:
    """Core inference engine for frame-by-frame generation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[Model] = None
        self.package_data: Optional[Dict] = None
        self.audio_features: Optional[torch.Tensor] = None
        self.is_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        
        # Performance tracking
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def initialize(self, package_path: str, audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Initialize the engine with model and data"""
        start_time = time.time()
        
        try:
            # Load preprocessed package
            self.package_data = await self._load_package(package_path)
            
            # Load model
            await self._load_model()
            
            # Load/extract audio features
            if audio_path:
                self.audio_features = await self._extract_audio_features(audio_path)
            else:
                self.audio_features = await self._load_package_audio()
            
            # Warmup model
            await self._warmup_model()
            
            self.is_loaded = True
            
            init_time = time.time() - start_time
            
            return {
                "status": "initialized",
                "initialization_time_ms": int(init_time * 1000),
                "total_frames": self.package_data["frame_count"],
                "audio_features_shape": list(self.audio_features.shape),
                "device": str(self.device),
                "model_loaded": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "initialization_time_ms": int((time.time() - start_time) * 1000)
            }
    
    async def generate_frame(self, frame_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a single frame"""
        if not self.is_loaded:
            raise RuntimeError("Engine not initialized")
            
        if frame_id >= self.package_data["frame_count"]:
            raise ValueError(f"Frame {frame_id} out of range (max: {self.package_data['frame_count']-1})")
        
        start_time = time.time()
        
        try:
            # Run inference in thread pool to avoid blocking
            frame = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._generate_frame_sync, frame_id
            )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.cache_misses += 1
            
            metadata = {
                "frame_id": frame_id,
                "processing_time_ms": int(inference_time * 1000),
                "cached": False,
                "frame_shape": frame.shape,
                "device": str(self.device)
            }
            
            return frame, metadata
            
        except Exception as e:
            error_metadata = {
                "frame_id": frame_id,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
            raise RuntimeError(f"Frame generation failed: {e}") from e
    
    def _generate_frame_sync(self, frame_id: int) -> np.ndarray:
        """Synchronous frame generation (runs in thread pool)"""
        
        # Load preprocessed data for this frame
        face_region = self._load_face_region(frame_id)
        masked_region = self._load_masked_region(frame_id)
        face_crop_328 = self._load_face_crop(frame_id)
        bounds = self._load_bounds(frame_id)
        
        # Get audio slice for this frame
        audio_slice = self.audio_features[frame_id:frame_id+1]  # Shape: (1, 512)
        
        # Prepare tensors
        face_tensor = torch.from_numpy(face_region).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        masked_tensor = torch.from_numpy(masked_region).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Reshape audio for model input
        audio_tensor = audio_slice.unsqueeze(0)  # Shape: (1, 1, 512)
        audio_reshaped = audio_tensor.view(1, 32, 16).unsqueeze(-1).repeat(1, 1, 1, 16)  # (1, 32, 16, 16)
        
        # Move to device
        face_tensor = face_tensor.to(self.device)
        masked_tensor = masked_tensor.to(self.device)
        audio_reshaped = audio_reshaped.to(self.device)
        
        # Run inference
        with torch.no_grad():
            if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
                face_tensor = face_tensor.half()
                masked_tensor = masked_tensor.half()
                audio_reshaped = audio_reshaped.half()
            
            # Concatenate inputs (face + masked + audio)
            model_input = torch.cat([face_tensor, masked_tensor], dim=1)  # (1, 6, 320, 320)
            
            # Generate prediction
            prediction = self.model(model_input, audio_reshaped)[0]  # (3, 320, 320)
            
            if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
                prediction = prediction.float()
        
        # Post-process prediction
        prediction_np = prediction.cpu().numpy().transpose(1, 2, 0)  # (320, 320, 3)
        prediction_np = np.clip(prediction_np * 255, 0, 255).astype(np.uint8)
        
        # Composite final frame
        final_frame = self._composite_frame(face_crop_328, prediction_np, bounds)
        
        return final_frame
    
    def _composite_frame(self, face_crop_328: np.ndarray, prediction: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Composite prediction onto original frame"""
        
        # Load original full frame
        original_frame = self._load_original_frame(0)  # For now, use frame 0 as template
        
        # Place prediction in center of 328x328 container
        face_crop_328[4:324, 4:324] = prediction
        
        # Resize to original face size
        if len(bounds) == 5:
            # Handle legacy format: [xmin, ymin, xmax, ymax, width]
            xmin, ymin, xmax, ymax, width = bounds
            height = ymax - ymin
        elif len(bounds) == 6:
            # Handle new format: [xmin, ymin, xmax, ymax, width, height]
            xmin, ymin, xmax, ymax, width, height = bounds
        else:
            raise ValueError(f"Unexpected bounds format: expected 5 or 6 values, got {len(bounds)}")
            
        final_crop = cv2.resize(face_crop_328, (int(width), int(height)))
        
        # Overlay onto full frame
        result_frame = original_frame.copy()
        result_frame[int(ymin):int(ymax), int(xmin):int(xmax)] = final_crop
        
        return result_frame
    
    async def _load_package(self, package_path: str) -> Dict[str, Any]:
        """Load and extract preprocessed package"""
        
        self.temp_dir = tempfile.mkdtemp(prefix="fast_service_")
        
        with zipfile.ZipFile(package_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        
        # Load package info
        with open(os.path.join(self.temp_dir, "package_info.json"), 'r') as f:
            package_info = json.load(f)
        
        return {
            "temp_dir": self.temp_dir,
            "frame_count": package_info["source_video"]["processed_frames"],
            "dataset_name": package_info["dataset_name"],
            "package_info": package_info
        }
    
    async def _load_model(self):
        """Load the neural network model"""
        
        # Find model checkpoint
        models_dir = os.path.join(self.package_data["temp_dir"], "models")
        video_model_path = os.path.join(models_dir, "99.pth")  # Use actual filename
        
        if not os.path.exists(video_model_path):
            # Fallback to any .pth file
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth') and f != 'audio_visual_encoder.pth']
            if not model_files:
                raise FileNotFoundError("No model checkpoint found in package")
            video_model_path = os.path.join(models_dir, model_files[0])
        
        # Load model
        self.model = Model(6, 'ave')  # 6 channels, AVE audio mode
        self.model.load_state_dict(torch.load(video_model_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()
        
        if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
            self.model = self.model.half()
    
    async def _extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """Extract audio features from custom audio file"""
        
        if not AUDIO_EXTRACTION_AVAILABLE:
            raise RuntimeError("Audio extraction not available")
        
        # Run in thread pool
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._extract_audio_sync, audio_path
        )
    
    def _extract_audio_sync(self, audio_path: str) -> torch.Tensor:
        """Synchronous audio extraction"""
        
        from data_utils.ave.audio import AudDataset
        
        # Extract audio features
        aud_dataset = AudDataset()
        aud_feat = aud_dataset.audio_feature_extraction(audio_path)
        
        return torch.from_numpy(aud_feat).float()
    
    async def _load_package_audio(self) -> torch.Tensor:
        """Load audio features from package"""
        
        aud_path = os.path.join(self.package_data["temp_dir"], "aud_ave.npy")
        aud_feat = np.load(aud_path)
        
        return torch.from_numpy(aud_feat).float()
    
    async def _warmup_model(self):
        """Warmup model with dummy inference"""
        
        print("🔥 Warming up model...")
        
        for i in range(settings.MODEL_WARMUP_FRAMES):
            dummy_face = torch.randn(1, 3, 320, 320, device=self.device)
            dummy_masked = torch.randn(1, 3, 320, 320, device=self.device)
            dummy_audio = torch.randn(1, 32, 16, 16, device=self.device)
            
            if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
                dummy_face = dummy_face.half()
                dummy_masked = dummy_masked.half()
                dummy_audio = dummy_audio.half()
            
            with torch.no_grad():
                model_input = torch.cat([dummy_face, dummy_masked], dim=1)
                _ = self.model(model_input, dummy_audio)
        
        print("✅ Model warmed up")
    
    def _load_face_region(self, frame_id: int) -> np.ndarray:
        """Load preprocessed face region"""
        # In real implementation, extract from face_regions_320.mp4
        # For now, return dummy data
        return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    
    def _load_masked_region(self, frame_id: int) -> np.ndarray:
        """Load preprocessed masked region"""
        # In real implementation, extract from masked_regions_320.mp4
        return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    
    def _load_face_crop(self, frame_id: int) -> np.ndarray:
        """Load preprocessed face crop"""
        # In real implementation, extract from face_crops_328.mp4
        return np.random.randint(0, 255, (328, 328, 3), dtype=np.uint8)
    
    def _load_bounds(self, frame_id: int) -> np.ndarray:
        """Load enhanced bounds for frame"""
        bounds_path = os.path.join(self.package_data["temp_dir"], "face_bounds", f"{frame_id}.npy")
        return np.load(bounds_path)
    
    def _load_original_frame(self, frame_id: int) -> np.ndarray:
        """Load original full frame"""
        # In real implementation, extract from video.mp4
        return np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        return {
            "is_loaded": self.is_loaded,
            "total_inferences": len(self.inference_times),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "avg_inference_time_ms": int(avg_inference_time * 1000),
            "device": str(self.device),
            "frame_count": self.package_data["frame_count"] if self.package_data else 0
        }


# Global inference engine instance
inference_engine = InferenceEngine()
