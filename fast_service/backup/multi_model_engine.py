import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import cv2
import zipfile
import tempfile
import json
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from unet_328 import Model
from config import settings
from dynamic_model_manager import dynamic_model_manager

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


class ModelInstance:
    """Single model instance with its data"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Optional[Model] = None
        self.package_data: Optional[Dict] = None
        self.temp_dir: Optional[str] = None
        self.is_loaded = False
        
        # Performance tracking per model
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_used = time.time()
        
        # GPU tensor pre-allocation for performance
        self.gpu_tensors_cache = {}
        
        # Frame caching
        self.face_frames_cache = None
        self.masked_frames_cache = None
    
    def update_last_used(self):
        """Update last used timestamp"""
        self.last_used = time.time()
    
    def get_or_create_gpu_tensor(self, key: str, shape: tuple, device: torch.device, dtype=torch.float32) -> torch.Tensor:
        """Get cached GPU tensor or create new one"""
        cache_key = f"{key}_{shape}_{dtype}"
        if cache_key not in self.gpu_tensors_cache:
            self.gpu_tensors_cache[cache_key] = torch.zeros(shape, device=device, dtype=dtype)
        return self.gpu_tensors_cache[cache_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this model"""
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "total_inferences": len(self.inference_times),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "avg_inference_time_ms": int(avg_inference_time * 1000),
            "last_used": self.last_used,
            "frame_count": self.package_data["frame_count"] if self.package_data else 0
        }


class MultiModelInferenceEngine:
    """Multi-model inference engine for frame-by-frame generation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, ModelInstance] = {}
        self.audio_cache: Dict[str, torch.Tensor] = {}  # Cache audio features by hash
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        
        # Global performance tracking
        self.total_requests = 0
        self.total_inference_time = 0
    
    async def load_model(self, model_name: str, package_path: Optional[str] = None, audio_override: Optional[str] = None) -> Dict[str, Any]:
        """Load a new model - with automatic downloading and extraction if needed"""
        start_time = time.time()
        
        try:
            print(f"🔄 Starting load_model for '{model_name}' with package_path='{package_path}'")
            
            if model_name in self.models:
                print(f"✅ Model '{model_name}' already loaded")
                return {"status": "already_loaded", "model_name": model_name}
            
            # Use dynamic model manager to ensure model is available
            if not package_path:
                print(f"🔍 Auto-resolving model '{model_name}'...")
                availability_result = await dynamic_model_manager.ensure_model_available(model_name)
                
                if not availability_result["success"]:
                    return {
                        "status": "error",
                        "model_name": model_name,
                        "error": availability_result["error"],
                        "actions_attempted": availability_result["actions_taken"]
                    }
                
                package_path = availability_result["model_path"]
                print(f"✅ Model '{model_name}' available at: {package_path}")
            
            print(f"📦 Creating model instance for '{model_name}'")
            
            # Create model instance
            model_instance = ModelInstance(model_name)
            
            print(f"📁 Loading package from: {package_path}")
            
            # Load preprocessed package
            await self._load_package(model_instance, package_path)
            
            print(f"🧠 Loading neural model...")
            
            # Load neural network model
            await self._load_neural_model(model_instance)
            
            print(f"🎬 Pre-loading video frames...")
            
            # Pre-load video frames for better performance
            await self._preload_video_frames(model_instance)
            
            print(f"🎵 Processing audio features...")
            
            # Load/extract audio features
            if audio_override:
                audio_hash = self._calculate_audio_hash(audio_override)
                if audio_hash not in self.audio_cache:
                    audio_features = await self._extract_audio_features(audio_override)
                    self.audio_cache[audio_hash] = audio_features
                model_instance.audio_hash = audio_hash
            else:
                # Use package audio
                audio_features = await self._load_package_audio(model_instance)
                audio_hash = f"package_{model_name}"
                self.audio_cache[audio_hash] = audio_features
                model_instance.audio_hash = audio_hash
            
            # Warmup model
            await self._warmup_model(model_instance)
            
            model_instance.is_loaded = True
            self.models[model_name] = model_instance
            
            print(f"✅ Model '{model_name}' successfully loaded into engine!")
            print(f"📊 Total models in engine: {len(self.models)}")
            print(f"📋 Available models: {list(self.models.keys())}")
            
            init_time = time.time() - start_time
            
            return {
                "status": "loaded",
                "model_name": model_name,
                "package_path": package_path if isinstance(package_path, str) else str(package_path),
                "initialization_time_ms": int(init_time * 1000),
                "total_frames": model_instance.package_data["frame_count"],
                "audio_features_shape": list(self.audio_cache[model_instance.audio_hash].shape),
                "device": str(self.device),
                "auto_resolved": not package_path
            }
            
        except Exception as e:
            print(f"❌ Error loading model '{model_name}': {e}")
            print(f"📊 Models in engine after error: {len(self.models)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "model_name": model_name,
                "error": str(e),
                "initialization_time_ms": int((time.time() - start_time) * 1000)
            }
    
    async def ensure_model_loaded(self, model_name: str) -> Dict[str, Any]:
        """Ensure a model is loaded, loading it automatically if needed"""
        
        if model_name in self.models:
            return {"status": "already_loaded", "model_name": model_name}
        
        print(f"🚀 Auto-loading model '{model_name}'...")
        return await self.load_model(model_name)
    
    async def generate_frame(self, model_name: str, frame_id: int, audio_override: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a single frame using specified model - with auto-loading"""
        
        # Auto-load model if not loaded
        if model_name not in self.models:
            print(f"🔄 Model '{model_name}' not loaded, auto-loading...")
            load_result = await self.ensure_model_loaded(model_name)
            
            if load_result["status"] == "error":
                raise ValueError(f"Failed to auto-load model '{model_name}': {load_result['error']}")
            
            print(f"✅ Model '{model_name}' auto-loaded successfully")
        
        model_instance = self.models[model_name]
        model_instance.update_last_used()
        
        if frame_id >= model_instance.package_data["frame_count"]:
            raise ValueError(f"Frame {frame_id} out of range for model '{model_name}' (max: {model_instance.package_data['frame_count']-1})")
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Select audio features
            if audio_override:
                audio_hash = self._calculate_audio_hash(audio_override)
                if audio_hash not in self.audio_cache:
                    audio_features = await self._extract_audio_features(audio_override)
                    self.audio_cache[audio_hash] = audio_features
                audio_to_use = self.audio_cache[audio_hash]
            else:
                audio_to_use = self.audio_cache[model_instance.audio_hash]
            
            # Run inference in thread pool
            frame = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._generate_frame_sync, model_instance, frame_id, audio_to_use
            )
            
            inference_time = time.time() - start_time
            model_instance.inference_times.append(inference_time)
            model_instance.cache_misses += 1
            self.total_inference_time += inference_time
            
            metadata = {
                "model_name": model_name,
                "frame_id": frame_id,
                "processing_time_ms": int(inference_time * 1000),
                "cached": False,
                "frame_shape": frame.shape,
                "device": str(self.device),
                "audio_override": audio_override is not None
            }
            
            return frame, metadata
            
        except Exception as e:
            error_metadata = {
                "model_name": model_name,
                "frame_id": frame_id,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
            raise RuntimeError(f"Frame generation failed for model '{model_name}': {e}") from e

    async def generate_inference_only(self, model_name: str, frame_id: int, audio_override: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate only the inference result + bounds for client-side compositing"""
        
        # Check if model is loaded (skip auto-loading for now to debug)
        if model_name not in self.models:
            print(f"� Model '{model_name}' not found in engine.")
            print(f"📋 Available models: {list(self.models.keys())}")
            print(f"📊 Total models in engine: {len(self.models)}")
            raise ValueError(f"Model '{model_name}' not loaded. Available models: {list(self.models.keys())}")
        
        model_instance = self.models[model_name]
        model_instance.update_last_used()
        
        if frame_id >= model_instance.package_data["frame_count"]:
            raise ValueError(f"Frame {frame_id} out of range for model '{model_name}' (max: {model_instance.package_data['frame_count']-1})")
        
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Select audio features
            if audio_override:
                audio_hash = self._calculate_audio_hash(audio_override)
                if audio_hash not in self.audio_cache:
                    audio_features = await self._extract_audio_features(audio_override)
                    self.audio_cache[audio_hash] = audio_features
                audio_to_use = self.audio_cache[audio_hash]
            else:
                audio_to_use = self.audio_cache[model_instance.audio_hash]
            
            # Run inference in thread pool - return prediction + bounds only
            prediction, bounds = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._generate_inference_only_sync, model_instance, frame_id, audio_to_use
            )
            
            inference_time = time.time() - start_time
            model_instance.inference_times.append(inference_time)
            model_instance.cache_misses += 1
            self.total_inference_time += inference_time
            
            metadata = {
                "model_name": model_name,
                "frame_id": frame_id,
                "processing_time_ms": int(inference_time * 1000),
                "cached": False,
                "prediction_shape": prediction.shape,
                "bounds_shape": bounds.shape,
                "device": str(self.device),
                "audio_override": audio_override is not None
            }
            
            return prediction, bounds, metadata
            
        except Exception as e:
            error_metadata = {
                "model_name": model_name,
                "frame_id": frame_id,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
            raise RuntimeError(f"Inference generation failed for model '{model_name}': {e}") from e
    
    def _generate_frame_sync(self, model_instance: ModelInstance, frame_id: int, audio_features: torch.Tensor) -> np.ndarray:
        """Synchronous frame generation (runs in thread pool)"""
        
        # Load preprocessed data for this frame
        face_region = self._load_face_region(model_instance, frame_id)
        masked_region = self._load_masked_region(model_instance, frame_id)
        face_crop_328 = self._load_face_crop(model_instance, frame_id)
        bounds = self._load_bounds(model_instance, frame_id)
        
        # Get audio slice for this frame
        audio_slice = audio_features[frame_id:frame_id+1]  # Shape: (1, 512)
        
        # Use pre-allocated GPU tensors for better performance
        face_tensor_gpu = model_instance.get_or_create_gpu_tensor("face", (1, 3, 320, 320), self.device)
        masked_tensor_gpu = model_instance.get_or_create_gpu_tensor("masked", (1, 3, 320, 320), self.device)
        audio_tensor_gpu = model_instance.get_or_create_gpu_tensor("audio", (1, 32, 16, 16), self.device)
        
        # Efficiently copy data to pre-allocated GPU tensors
        face_tensor_cpu = torch.from_numpy(face_region).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        masked_tensor_cpu = torch.from_numpy(masked_region).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        face_tensor_gpu.copy_(face_tensor_cpu)
        masked_tensor_gpu.copy_(masked_tensor_cpu)
        
        # Reshape audio efficiently
        audio_tensor = audio_slice.unsqueeze(0)  # Shape: (1, 1, 512)
        audio_reshaped = audio_tensor.view(1, 32, 16).unsqueeze(-1).repeat(1, 1, 1, 16)  # (1, 32, 16, 16)
        audio_tensor_gpu.copy_(audio_reshaped)
        
        # Run inference with pre-allocated tensors
        with torch.no_grad():
            if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
                face_tensor_gpu = face_tensor_gpu.half()
                masked_tensor_gpu = masked_tensor_gpu.half()
                audio_tensor_gpu = audio_tensor_gpu.half()
            
            # Concatenate inputs (face + masked)
            model_input = torch.cat([face_tensor_gpu, masked_tensor_gpu], dim=1)  # (1, 6, 320, 320)
            
            # Generate prediction
            prediction = model_instance.model(model_input, audio_tensor_gpu)[0]  # (3, 320, 320)
            
            if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
                prediction = prediction.float()
        
        # Post-process prediction
        prediction_np = prediction.cpu().numpy().transpose(1, 2, 0)  # (320, 320, 3)
        prediction_np = np.clip(prediction_np * 255, 0, 255).astype(np.uint8)
        
        # Composite final frame
        final_frame = self._composite_frame(model_instance, face_crop_328, prediction_np, bounds)
        
        return final_frame

    def _generate_inference_only_sync(self, model_instance: ModelInstance, frame_id: int, audio_features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous inference generation - returns only prediction + bounds (no compositing)"""
        
        # Load preprocessed data for this frame
        face_region = self._load_face_region(model_instance, frame_id)
        masked_region = self._load_masked_region(model_instance, frame_id)
        bounds = self._load_bounds(model_instance, frame_id)
        
        # Get audio slice for this frame
        audio_slice = audio_features[frame_id:frame_id+1]  # Shape: (1, 512)
        
        # Use pre-allocated GPU tensors for better performance
        face_tensor_gpu = model_instance.get_or_create_gpu_tensor("face", (1, 3, 320, 320), self.device)
        masked_tensor_gpu = model_instance.get_or_create_gpu_tensor("masked", (1, 3, 320, 320), self.device)
        audio_tensor_gpu = model_instance.get_or_create_gpu_tensor("audio", (1, 32, 16, 16), self.device)
        
        # Efficiently copy data to pre-allocated GPU tensors
        face_tensor_cpu = torch.from_numpy(face_region).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        masked_tensor_cpu = torch.from_numpy(masked_region).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        face_tensor_gpu.copy_(face_tensor_cpu)
        masked_tensor_gpu.copy_(masked_tensor_cpu)
        
        # Reshape audio efficiently
        audio_tensor = audio_slice.unsqueeze(0)  # Shape: (1, 1, 512)
        audio_reshaped = audio_tensor.view(1, 32, 16).unsqueeze(-1).repeat(1, 1, 1, 16)  # (1, 32, 16, 16)
        audio_tensor_gpu.copy_(audio_reshaped)
        
        # Run inference with pre-allocated tensors
        with torch.no_grad():
            if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
                face_tensor_gpu = face_tensor_gpu.half()
                masked_tensor_gpu = masked_tensor_gpu.half()
                audio_tensor_gpu = audio_tensor_gpu.half()
            
            # Concatenate inputs (face + masked)
            model_input = torch.cat([face_tensor_gpu, masked_tensor_gpu], dim=1)  # (1, 6, 320, 320)
            
            # Generate prediction
            prediction = model_instance.model(model_input, audio_tensor_gpu)[0]  # (3, 320, 320)
            
            if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
                prediction = prediction.float()
        
        # Post-process prediction (but don't composite)
        prediction_np = prediction.cpu().numpy().transpose(1, 2, 0)  # (320, 320, 3)
        prediction_np = np.clip(prediction_np * 255, 0, 255).astype(np.uint8)
        
        return prediction_np, bounds
    
    def _composite_frame(self, model_instance: ModelInstance, face_crop_328: np.ndarray, prediction: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Composite prediction onto original frame"""
        
        # Load original full frame (using first frame as template for now)
        original_frame = self._load_original_frame(model_instance, 0)
        
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
    
    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model to free memory"""
        
        if model_name not in self.models:
            return {"status": "error", "error": f"Model '{model_name}' not loaded"}
        
        try:
            model_instance = self.models[model_name]
            
            # Cleanup GPU memory
            if model_instance.model is not None:
                del model_instance.model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Cleanup temp directory
            if model_instance.temp_dir and os.path.exists(model_instance.temp_dir):
                import shutil
                shutil.rmtree(model_instance.temp_dir)
            
            # Remove from models dict
            del self.models[model_name]
            
            return {
                "status": "unloaded",
                "model_name": model_name,
                "stats": model_instance.get_stats()
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "model_name": model_name,
                "error": str(e)
            }
    
    def list_models(self) -> Dict[str, Any]:
        """List all loaded models and their stats"""
        
        models_info = {}
        for model_name, model_instance in self.models.items():
            models_info[model_name] = model_instance.get_stats()
        
        return {
            "loaded_models": list(self.models.keys()),
            "total_models": len(self.models),
            "models_info": models_info,
            "global_stats": {
                "total_requests": self.total_requests,
                "avg_request_time_ms": int((self.total_inference_time / self.total_requests * 1000)) if self.total_requests > 0 else 0,
                "device": str(self.device),
                "cached_audio_count": len(self.audio_cache)
            }
        }
    
    def _calculate_audio_hash(self, audio_path: str) -> str:
        """Calculate hash for audio file"""
        try:
            stat = os.stat(audio_path)
            hash_input = f"{audio_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            return hashlib.md5(audio_path.encode()).hexdigest()
    
    async def _load_package(self, model_instance: ModelInstance, package_path: str):
        """Load and extract preprocessed package"""
        
        # Check if package_path is a directory (already extracted) or a zip file
        if os.path.isdir(package_path):
            # Already extracted, use directly
            model_instance.temp_dir = package_path
            print(f"📁 Using pre-extracted model at: {package_path}")
        else:
            # Zip file, extract to temp directory
            model_instance.temp_dir = tempfile.mkdtemp(prefix=f"fast_service_{model_instance.model_name}_")
            
            with zipfile.ZipFile(package_path, 'r') as zip_ref:
                zip_ref.extractall(model_instance.temp_dir)
            print(f"📦 Extracted model to: {model_instance.temp_dir}")
        
        # Load package info
        package_info_path = os.path.join(model_instance.temp_dir, "package_info.json")
        with open(package_info_path, 'r') as f:
            package_info = json.load(f)
        
        model_instance.package_data = {
            "temp_dir": model_instance.temp_dir,
            "frame_count": package_info["source_video"]["processed_frames"],
            "dataset_name": package_info["dataset_name"],
            "package_info": package_info
        }
    
    async def _load_neural_model(self, model_instance: ModelInstance):
        """Load the neural network model"""
        
        # Find model checkpoint
        models_dir = os.path.join(model_instance.temp_dir, "models")
        video_model_path = os.path.join(models_dir, "99.pth")  # Use actual filename
        
        if not os.path.exists(video_model_path):
            # Fallback to any .pth file
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth') and f != 'audio_visual_encoder.pth']
            if not model_files:
                raise FileNotFoundError(f"No model checkpoint found in package for '{model_instance.model_name}'")
            video_model_path = os.path.join(models_dir, model_files[0])
        
        # Load model
        model_instance.model = Model(6, 'ave')  # 6 channels, AVE audio mode
        model_instance.model.load_state_dict(torch.load(video_model_path, map_location=self.device))
        model_instance.model = model_instance.model.to(self.device).eval()
        
        if settings.ENABLE_HALF_PRECISION and self.device.type == 'cuda':
            model_instance.model = model_instance.model.half()
    
    async def _extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """Extract audio features from custom audio file"""
        
        if not AUDIO_EXTRACTION_AVAILABLE:
            raise RuntimeError("Audio extraction not available")
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._extract_audio_sync, audio_path
        )
    
    def _extract_audio_sync(self, audio_path: str) -> torch.Tensor:
        """Synchronous audio extraction"""
        
        from data_utils.ave.audio import AudDataset
        
        aud_dataset = AudDataset()
        aud_feat = aud_dataset.audio_feature_extraction(audio_path)
        
        return torch.from_numpy(aud_feat).float()
    
    async def _load_package_audio(self, model_instance: ModelInstance) -> torch.Tensor:
        """Load audio features from package"""
        
        aud_path = os.path.join(model_instance.temp_dir, "aud_ave.npy")
        aud_feat = np.load(aud_path)
        
        return torch.from_numpy(aud_feat).float()
    
    async def _warmup_model(self, model_instance: ModelInstance):
        """Warmup model with dummy inference"""
        
        print(f"🔥 Warming up model '{model_instance.model_name}'...")
        
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
                _ = model_instance.model(model_input, dummy_audio)
        
        print(f"✅ Model '{model_instance.model_name}' warmed up")
    
    async def _preload_video_frames(self, model_instance: ModelInstance):
        """Pre-load all video frames at model startup for better performance"""
        print(f"🎬 Pre-loading video frames for model '{model_instance.model_name}'...")
        
        # Trigger loading of both video caches
        self._load_face_region(model_instance, 0)  # This will load all face frames
        self._load_masked_region(model_instance, 0)  # This will load all masked frames
        
        face_count = len(model_instance.face_frames_cache) if model_instance.face_frames_cache else 0
        masked_count = len(model_instance.masked_frames_cache) if model_instance.masked_frames_cache else 0
        
        print(f"✅ Pre-loaded {face_count} face frames and {masked_count} masked frames")
    
    # Enhanced data loading methods
    def _load_face_region(self, model_instance: ModelInstance, frame_id: int) -> np.ndarray:
        """Load preprocessed face region from video"""
        if model_instance.face_frames_cache is None:
            # Cache not initialized, load from video file
            video_path = os.path.join(model_instance.temp_dir, "face_regions_320.mp4")
            if os.path.exists(video_path):
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                model_instance.face_frames_cache = frames
                print(f"🎬 Loaded {len(frames)} face region frames for model '{model_instance.model_name}'")
            else:
                # Fallback to random data if video doesn't exist
                model_instance.face_frames_cache = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8) for _ in range(100)]
        
        # Return cached frame
        if frame_id < len(model_instance.face_frames_cache):
            return model_instance.face_frames_cache[frame_id]
        else:
            return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    
    def _load_masked_region(self, model_instance: ModelInstance, frame_id: int) -> np.ndarray:
        """Load preprocessed masked region from video"""
        if model_instance.masked_frames_cache is None:
            # Cache not initialized, load from video file
            video_path = os.path.join(model_instance.temp_dir, "masked_regions_320.mp4")
            if os.path.exists(video_path):
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                model_instance.masked_frames_cache = frames
                print(f"🎭 Loaded {len(frames)} masked region frames for model '{model_instance.model_name}'")
            else:
                # Fallback to random data if video doesn't exist
                model_instance.masked_frames_cache = [np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8) for _ in range(100)]
        
        # Return cached frame
        if frame_id < len(model_instance.masked_frames_cache):
            return model_instance.masked_frames_cache[frame_id]
        else:
            return np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    
    
    def _load_face_crop(self, model_instance: ModelInstance, frame_id: int) -> np.ndarray:
        """Load preprocessed face crop (328x328 version)"""
        # For now, resize the face region to 328x328
        face_region = self._load_face_region(model_instance, frame_id)
        import cv2
        return cv2.resize(face_region, (328, 328))
    
    def _load_bounds(self, model_instance: ModelInstance, frame_id: int) -> np.ndarray:
        """Load enhanced bounds for frame"""
        bounds_path = os.path.join(model_instance.temp_dir, "face_bounds", f"{frame_id}.npy")
        return np.load(bounds_path)
    
    def _load_original_frame(self, model_instance: ModelInstance, frame_id: int) -> np.ndarray:
        """Load original full frame"""
        return np.random.randint(0, 255, (1280, 720, 3), dtype=np.uint8)


# Global multi-model inference engine instance
multi_model_engine = MultiModelInferenceEngine()
