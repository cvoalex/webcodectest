"""
🚀 ULTRA-OPTIMIZED Inference Engine for Pre-processed Model Packages
Leverages sanders model format with maximum performance optimizations:
- Pre-loaded videos in RAM
- Memory-mapped audio features
- Cached metadata
- Minimal I/O operations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import json
import time
import asyncio
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from unet_328 import Model


class VideoCache:
    """Pre-loaded video frames cache for ultra-fast access"""
    
    def __init__(self, video_path: str, name: str):
        self.video_path = video_path
        self.name = name
        self.frames: List[np.ndarray] = []
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.is_loaded = False
        
    def load_all_frames(self):
        """Load all video frames into memory"""
        start_time = time.time()
        print(f"📹 Loading {self.name} into RAM...")
        
        cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  Loaded {frame_idx}/{self.frame_count} frames...")
        
        cap.release()
        
        load_time = time.time() - start_time
        memory_mb = sum(frame.nbytes for frame in self.frames) / (1024 * 1024)
        
        self.is_loaded = True
        print(f"✅ {self.name}: {self.frame_count} frames loaded ({memory_mb:.2f} MB) in {load_time:.2f}s")
        
        return self.frame_count
    
    def get_frame(self, frame_id: int) -> np.ndarray:
        """Get frame by ID (instant access from RAM)"""
        if not self.is_loaded:
            raise RuntimeError(f"Video {self.name} not loaded")
        
        if frame_id < 0 or frame_id >= len(self.frames):
            raise ValueError(f"Frame {frame_id} out of range [0, {len(self.frames)-1}]")
        
        return self.frames[frame_id]


class OptimizedModelPackage:
    """Optimized model package with pre-loaded resources"""
    
    def __init__(self, package_dir: str):
        self.package_dir = Path(package_dir)
        self.model_name = self.package_dir.name
        
        # Resources (will be loaded at init)
        self.video_caches: Dict[str, VideoCache] = {}
        self.audio_features: Optional[np.ndarray] = None  # Memory-mapped
        self.crop_rectangles: Optional[Dict] = None
        self.frame_metadata: Optional[Dict] = None
        self.package_info: Optional[Dict] = None
        
        # Model
        self.model: Optional[Model] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance tracking
        self.inference_times = []
        self.total_requests = 0
        
        print(f"🎯 Initializing Optimized Package: {self.model_name}")
        print(f"📁 Package directory: {self.package_dir}")
    
    async def initialize(self):
        """Initialize all resources with optimizations"""
        start_time = time.time()
        
        # Load package info
        await self._load_package_info()
        
        # Load metadata (small files, cache in memory)
        await self._load_metadata()
        
        # Memory-map audio features (instant access, no copy)
        await self._load_audio_features()
        
        # Pre-load all videos into RAM
        await self._preload_videos()
        
        # Load neural network model
        await self._load_model()
        
        init_time = time.time() - start_time
        print(f"\n✅ Package initialized in {init_time:.2f}s")
        print(f"🎯 Ready for ultra-fast inference!")
        
        return {
            "status": "success",
            "model_name": self.model_name,
            "frame_count": self.package_info["videos"]["model_inputs"]["frame_count"],
            "initialization_time_s": init_time,
            "device": str(self.device),
            "videos_loaded": list(self.video_caches.keys()),
            "audio_features_shape": self.audio_features.shape,
            "memory_mapped_audio": True
        }
    
    async def _load_package_info(self):
        """Load package info JSON"""
        info_path = self.package_dir / "package_info.json"
        
        if not info_path.exists():
            raise FileNotFoundError(f"Package info not found: {info_path}")
        
        with open(info_path, 'r') as f:
            self.package_info = json.load(f)
        
        print(f"📋 Package: {self.package_info['dataset_name']} v{self.package_info['package_version']}")
        print(f"📊 Frame count: {self.package_info['videos']['model_inputs']['frame_count']}")
    
    async def _load_metadata(self):
        """Load and cache metadata files in memory"""
        print(f"📝 Loading metadata...")
        
        # Load crop rectangles (used for compositing)
        crop_rect_path = self.package_dir / "cache" / "crop_rectangles.json"
        if crop_rect_path.exists():
            with open(crop_rect_path, 'r') as f:
                self.crop_rectangles = json.load(f)
            print(f"  ✅ Crop rectangles: {len(self.crop_rectangles)} frames cached")
        
        # Load frame metadata
        frame_meta_path = self.package_dir / "cache" / "frame_metadata.json"
        if frame_meta_path.exists():
            with open(frame_meta_path, 'r') as f:
                self.frame_metadata = json.load(f)
            print(f"  ✅ Frame metadata: {self.frame_metadata['processed_frames']} frames")
    
    async def _load_audio_features(self):
        """Memory-map audio features for instant zero-copy access"""
        print(f"🎵 Memory-mapping audio features...")
        
        audio_path = self.package_dir / "aud_ave.npy"
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio features not found: {audio_path}")
        
        # Memory-map the file (no copy, instant access!)
        self.audio_features = np.load(str(audio_path), mmap_mode='r')
        
        memory_mb = self.audio_features.nbytes / (1024 * 1024)
        print(f"✅ Audio features memory-mapped: {self.audio_features.shape} ({memory_mb:.2f} MB)")
        print(f"   - Type: {self.audio_features.dtype}")
        print(f"   - Range: [{self.audio_features.min():.4f}, {self.audio_features.max():.4f}]")
        print(f"   - Memory-mapped: Zero-copy access!")
    
    async def _preload_videos(self):
        """Pre-load all videos into RAM for instant frame access"""
        print(f"\n🎬 Pre-loading videos into RAM...")
        
        videos_to_load = [
            ("full_body", "full_body_video.mp4"),
            ("crops_328", "crops_328_video.mp4"),
            ("model_inputs", "model_inputs_video.mp4")
        ]
        
        # Note: rois_320_video.mp4 might be missing, handle gracefully
        if (self.package_dir / "rois_320_video.mp4").exists():
            videos_to_load.append(("rois_320", "rois_320_video.mp4"))
        
        total_memory = 0
        
        for name, filename in videos_to_load:
            video_path = self.package_dir / filename
            
            if not video_path.exists():
                print(f"⚠️  {filename} not found, skipping...")
                continue
            
            video_cache = VideoCache(str(video_path), name)
            video_cache.load_all_frames()
            
            self.video_caches[name] = video_cache
            total_memory += sum(frame.nbytes for frame in video_cache.frames) / (1024 * 1024)
        
        print(f"\n📊 Total video memory: {total_memory:.2f} MB")
        print(f"✅ All videos pre-loaded into RAM for instant access!")
    
    async def _load_model(self):
        """Load PyTorch model"""
        print(f"\n🧠 Loading neural network model...")
        
        checkpoint_path = self.package_dir / "checkpoint" / "best_trainloss.pth"
        
        if not checkpoint_path.exists():
            # Try alternative checkpoint name
            checkpoint_path = self.package_dir / "checkpoint" / "model_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found in {self.package_dir / 'checkpoint'}")
        
        # Create model
        self.model = Model(n_channels=6, mode='ave')
        self.model = self.model.to(self.device)
        
        # Load weights
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Warmup
        print(f"🔥 Warming up model...")
        with torch.no_grad():
            dummy_img = torch.zeros(1, 6, 320, 320, device=self.device)
            dummy_audio = torch.zeros(1, 32, 16, 16, device=self.device)
            
            for i in range(3):
                _ = self.model(dummy_img, dummy_audio)
        
        print(f"✅ Model loaded and warmed up on {self.device}")
    
    def _prepare_audio_tensor(self, frame_id: int) -> torch.Tensor:
        """Prepare audio tensor for model input (instant memory-mapped access)"""
        
        if frame_id >= len(self.audio_features):
            frame_id = len(self.audio_features) - 1
        
        # Get audio features (zero-copy from memory-mapped file!)
        audio_feat = self.audio_features[frame_id]  # Shape: (512,)
        
        # Reshape to model input format [1, 32, 16, 16]
        # AVE features need to be reshaped from (512,) to (32, 16) then expand
        audio_feat = audio_feat.reshape(32, 16)
        audio_feat = np.expand_dims(audio_feat, axis=-1)
        audio_feat = np.tile(audio_feat, (1, 1, 16))  # [32, 16, 16]
        audio_feat = np.expand_dims(audio_feat, axis=0)  # [1, 32, 16, 16]
        
        # Convert to tensor (minimal copy)
        audio_tensor = torch.from_numpy(audio_feat).float().to(self.device)
        
        return audio_tensor
    
    def _prepare_image_tensor(self, frame_id: int) -> torch.Tensor:
        """Prepare image tensor for model input (instant RAM access)"""
        
        # Get pre-masked model input from RAM cache (instant access!)
        model_input = self.video_caches["model_inputs"].get_frame(frame_id)
        
        # Model expects 320x320, check if resize needed
        if model_input.shape[0] != 320 or model_input.shape[1] != 320:
            model_input = cv2.resize(model_input, (320, 320))
        
        # Get corresponding crop for template
        crops_328 = self.video_caches["crops_328"].get_frame(frame_id)
        
        if crops_328.shape[0] != 320 or crops_328.shape[1] != 320:
            crops_328 = cv2.resize(crops_328, (320, 320))
        
        # Stack: [model_input (BGR), crops_328 (BGR)] = 6 channels
        combined = np.concatenate([model_input, crops_328], axis=2)  # [320, 320, 6]
        
        # Convert to tensor format [1, 6, 320, 320]
        combined = combined.transpose(2, 0, 1)  # [6, 320, 320]
        combined = np.expand_dims(combined, axis=0)  # [1, 6, 320, 320]
        combined = combined.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(combined).to(self.device)
        
        return image_tensor
    
    async def generate_frame(self, frame_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a single frame with ultra-fast inference"""
        
        if frame_id >= self.video_caches["model_inputs"].frame_count:
            raise ValueError(f"Frame {frame_id} out of range [0, {self.video_caches['model_inputs'].frame_count-1}]")
        
        start_time = time.time()
        self.total_requests += 1
        
        # Prepare inputs (all from RAM/memory-mapped data - instant!)
        prepare_start = time.time()
        image_tensor = self._prepare_image_tensor(frame_id)
        audio_tensor = self._prepare_audio_tensor(frame_id)
        prepare_time = (time.time() - prepare_start) * 1000
        
        # Run inference
        inference_start = time.time()
        with torch.no_grad():
            prediction = self.model(image_tensor, audio_tensor)
        
        # Convert to numpy
        prediction = prediction.squeeze(0).cpu().numpy()  # [3, 320, 320]
        prediction = (prediction * 255).astype(np.uint8)
        prediction = prediction.transpose(1, 2, 0)  # [320, 320, 3]
        
        inference_time = (time.time() - inference_start) * 1000
        
        # Get full body frame for compositing (instant RAM access!)
        full_body = self.video_caches["full_body"].get_frame(frame_id)
        
        # Get crop coordinates (instant cache access!)
        crop_info = self.crop_rectangles.get(str(frame_id), None)
        
        if crop_info:
            rect = crop_info["rect"]  # [x1, y1, x2, y2]
        else:
            # Fallback to center crop
            h, w = full_body.shape[:2]
            rect = [w//2 - 164, h//2 - 164, w//2 + 164, h//2 + 164]
        
        # Composite prediction onto full body
        composite_start = time.time()
        final_frame = full_body.copy()
        x1, y1, x2, y2 = rect
        
        # Resize prediction to crop size
        crop_w, crop_h = x2 - x1, y2 - y1
        prediction_resized = cv2.resize(prediction, (crop_w, crop_h))
        
        # Paste onto full body
        final_frame[y1:y2, x1:x2] = prediction_resized
        
        composite_time = (time.time() - composite_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        self.inference_times.append(total_time)
        
        metadata = {
            "frame_id": frame_id,
            "model_name": self.model_name,
            "total_time_ms": total_time,
            "prepare_time_ms": prepare_time,
            "inference_time_ms": inference_time,
            "composite_time_ms": composite_time,
            "device": str(self.device),
            "optimizations": [
                "pre_loaded_videos",
                "memory_mapped_audio",
                "cached_metadata",
                "zero_copy_access"
            ]
        }
        
        return final_frame, metadata
    
    async def generate_inference_only(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate inference result + bounds for client-side compositing"""
        
        if frame_id >= self.video_caches["model_inputs"].frame_count:
            raise ValueError(f"Frame {frame_id} out of range")
        
        start_time = time.time()
        self.total_requests += 1
        
        # Prepare inputs (instant access!)
        image_tensor = self._prepare_image_tensor(frame_id)
        audio_tensor = self._prepare_audio_tensor(frame_id)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(image_tensor, audio_tensor)
        
        # Convert to numpy
        prediction = prediction.squeeze(0).cpu().numpy()  # [3, 320, 320]
        prediction = (prediction * 255).astype(np.uint8)
        prediction = prediction.transpose(1, 2, 0)  # [320, 320, 3]
        
        # Get crop bounds
        crop_info = self.crop_rectangles.get(str(frame_id), None)
        
        if crop_info:
            bounds = np.array(crop_info["rect"], dtype=np.float32)
        else:
            # Fallback bounds
            bounds = np.array([0, 0, 328, 328], dtype=np.float32)
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        metadata = {
            "frame_id": frame_id,
            "model": self.model_name,
            "inference_time_ms": inference_time,
            "has_audio": True,
            "processing_time_ms": inference_time
        }
        
        return prediction, bounds, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        avg_time = np.mean(self.inference_times) if self.inference_times else 0
        min_time = np.min(self.inference_times) if self.inference_times else 0
        max_time = np.max(self.inference_times) if self.inference_times else 0
        
        return {
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "avg_inference_time_ms": avg_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "frame_count": self.video_caches["model_inputs"].frame_count if "model_inputs" in self.video_caches else 0,
            "device": str(self.device),
            "optimizations_active": [
                "pre_loaded_videos",
                "memory_mapped_audio", 
                "cached_metadata",
                "zero_copy_access"
            ]
        }


class OptimizedMultiModelEngine:
    """Multi-model engine for optimized packages"""
    
    def __init__(self):
        self.packages: Dict[str, OptimizedModelPackage] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        print("🚀 Optimized Multi-Model Engine initialized")
    
    async def load_package(self, package_name: str, package_dir: str) -> Dict[str, Any]:
        """Load an optimized model package"""
        
        if package_name in self.packages:
            return {
                "status": "already_loaded",
                "model_name": package_name
            }
        
        try:
            package = OptimizedModelPackage(package_dir)
            result = await package.initialize()
            
            self.packages[package_name] = package
            
            return result
            
        except Exception as e:
            print(f"❌ Error loading package '{package_name}': {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "model_name": package_name,
                "error": str(e)
            }
    
    async def generate_frame(self, model_name: str, frame_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate frame using specified model"""
        
        if model_name not in self.packages:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        return await self.packages[model_name].generate_frame(frame_id)
    
    async def generate_inference_only(self, model_name: str, frame_id: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate inference result only"""
        
        if model_name not in self.packages:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        return await self.packages[model_name].generate_inference_only(frame_id)
    
    def list_models(self) -> Dict[str, Any]:
        """List loaded models"""
        
        return {
            "loaded_models": list(self.packages.keys()),
            "count": len(self.packages)
        }
    
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a specific model"""
        
        if model_name not in self.packages:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        return self.packages[model_name].get_stats()


# Global optimized engine instance
optimized_engine = OptimizedMultiModelEngine()


async def test_optimized_engine():
    """Test the optimized engine"""
    print("=" * 80)
    print("🧪 TESTING OPTIMIZED INFERENCE ENGINE")
    print("=" * 80)
    
    # Load sanders package
    result = await optimized_engine.load_package(
        "sanders",
        "models/sanders"
    )
    
    print(f"\n📊 Load result: {json.dumps(result, indent=2)}")
    
    if result["status"] != "success":
        print("❌ Failed to load package")
        return
    
    print("\n" + "=" * 80)
    print("🎬 RUNNING PERFORMANCE TEST")
    print("=" * 80)
    
    # Test inference on multiple frames
    test_frames = [0, 10, 50, 100, 200, 300, 400, 500]
    
    for frame_id in test_frames:
        try:
            if frame_id >= result["frame_count"]:
                continue
            
            frame, metadata = await optimized_engine.generate_frame("sanders", frame_id)
            
            print(f"\n✅ Frame {frame_id}:")
            print(f"   Total: {metadata['total_time_ms']:.2f}ms")
            print(f"   - Prepare: {metadata['prepare_time_ms']:.2f}ms")
            print(f"   - Inference: {metadata['inference_time_ms']:.2f}ms")
            print(f"   - Composite: {metadata['composite_time_ms']:.2f}ms")
            
            # Save first frame as test
            if frame_id == 0:
                output_path = "test_optimized_output.jpg"
                cv2.imwrite(output_path, frame)
                print(f"   💾 Saved to: {output_path}")
        
        except Exception as e:
            print(f"❌ Error on frame {frame_id}: {e}")
    
    # Print final stats
    print("\n" + "=" * 80)
    print("📊 FINAL STATISTICS")
    print("=" * 80)
    
    stats = optimized_engine.get_model_stats("sanders")
    print(json.dumps(stats, indent=2))
    
    print("\n✅ Optimized engine test complete!")


if __name__ == "__main__":
    asyncio.run(test_optimized_engine())
