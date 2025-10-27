import redis
import json
import numpy as np
import cv2
from typing import Optional, Dict, Any, List
import pickle
import time
import hashlib
from config import settings


class MultiModelCacheManager:
    """Redis-based cache manager for multiple models"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(
            settings.REDIS_URL,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=False  # We need binary data for images
        )
        
        # Cache statistics per model
        self.model_stats = {}
        
        # Key prefixes with model support
        self.FRAME_PREFIX = "frame:"  # frame:model_name:frame_id
        self.AUDIO_PREFIX = "audio:"  # audio:hash
        self.METADATA_PREFIX = "meta:"  # meta:model_name:key
        self.STATS_PREFIX = "stats:"  # stats:model_name
        
    def _get_model_stats(self, model_name: str) -> Dict[str, int]:
        """Get or create stats for model"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_sets": 0
            }
        return self.model_stats[model_name]
    
    async def get_frame(self, model_name: str, frame_id: int) -> Optional[np.ndarray]:
        """Get cached frame by model and ID"""
        
        try:
            key = f"{self.FRAME_PREFIX}{model_name}:{frame_id}"
            cached_data = self.redis_client.get(key)
            
            stats = self._get_model_stats(model_name)
            
            if cached_data is not None:
                stats["cache_hits"] += 1
                # Deserialize frame data
                frame = pickle.loads(cached_data)
                return frame
            else:
                stats["cache_misses"] += 1
                return None
                
        except Exception as e:
            print(f"Cache get error for {model_name}:{frame_id}: {e}")
            self._get_model_stats(model_name)["cache_misses"] += 1
            return None
    
    async def set_frame(self, model_name: str, frame_id: int, frame: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache frame with model-specific key"""
        
        try:
            key = f"{self.FRAME_PREFIX}{model_name}:{frame_id}"
            
            # Serialize frame data
            serialized_frame = pickle.dumps(frame)
            
            # Set with TTL
            ttl = ttl or settings.FRAME_CACHE_TTL
            self.redis_client.setex(key, ttl, serialized_frame)
            
            self._get_model_stats(model_name)["cache_sets"] += 1
            return True
            
        except Exception as e:
            print(f"Cache set error for {model_name}:{frame_id}: {e}")
            return False
    
    async def get_frame_batch(self, model_name: str, frame_ids: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """Get multiple frames in batch for specific model"""
        
        keys = [f"{self.FRAME_PREFIX}{model_name}:{frame_id}" for frame_id in frame_ids]
        stats = self._get_model_stats(model_name)
        
        try:
            # Use pipeline for efficiency
            pipe = self.redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            
            cached_data = pipe.execute()
            
            results = {}
            for i, frame_id in enumerate(frame_ids):
                if cached_data[i] is not None:
                    try:
                        frame = pickle.loads(cached_data[i])
                        results[frame_id] = frame
                        stats["cache_hits"] += 1
                    except:
                        results[frame_id] = None
                        stats["cache_misses"] += 1
                else:
                    results[frame_id] = None
                    stats["cache_misses"] += 1
            
            return results
            
        except Exception as e:
            print(f"Batch cache get error for {model_name}: {e}")
            stats["cache_misses"] += len(frame_ids)
            return {frame_id: None for frame_id in frame_ids}
    
    async def set_frame_batch(self, model_name: str, frames: Dict[int, np.ndarray], ttl: Optional[int] = None) -> int:
        """Cache multiple frames in batch for specific model"""
        
        try:
            pipe = self.redis_client.pipeline()
            ttl = ttl or settings.FRAME_CACHE_TTL
            
            for frame_id, frame in frames.items():
                key = f"{self.FRAME_PREFIX}{model_name}:{frame_id}"
                serialized_frame = pickle.dumps(frame)
                pipe.setex(key, ttl, serialized_frame)
            
            pipe.execute()
            
            self._get_model_stats(model_name)["cache_sets"] += len(frames)
            return len(frames)
            
        except Exception as e:
            print(f"Batch cache set error for {model_name}: {e}")
            return 0
    
    async def get_audio_features(self, audio_hash: str) -> Optional[np.ndarray]:
        """Get cached audio features by hash (shared across models)"""
        
        try:
            key = f"{self.AUDIO_PREFIX}{audio_hash}"
            cached_data = self.redis_client.get(key)
            
            if cached_data is not None:
                audio_features = pickle.loads(cached_data)
                return audio_features
            
            return None
            
        except Exception as e:
            print(f"Audio cache get error: {e}")
            return None
    
    async def set_audio_features(self, audio_hash: str, audio_features: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache audio features with hash key (shared across models)"""
        
        try:
            key = f"{self.AUDIO_PREFIX}{audio_hash}"
            serialized_audio = pickle.dumps(audio_features)
            
            ttl = ttl or settings.AUDIO_CACHE_TTL
            self.redis_client.setex(key, ttl, serialized_audio)
            
            return True
            
        except Exception as e:
            print(f"Audio cache set error: {e}")
            return False
    
    async def preload_frames(self, model_name: str, frame_range: range, inference_engine) -> Dict[str, Any]:
        """Background preloading of frames for specific model"""
        
        start_time = time.time()
        preloaded_count = 0
        errors = []
        
        for frame_id in frame_range:
            try:
                # Check if already cached
                cached_frame = await self.get_frame(model_name, frame_id)
                if cached_frame is not None:
                    continue
                
                # Generate and cache frame
                frame, metadata = await inference_engine.generate_frame(model_name, frame_id)
                await self.set_frame(model_name, frame_id, frame)
                
                preloaded_count += 1
                
            except Exception as e:
                errors.append(f"Frame {frame_id}: {str(e)}")
        
        total_time = time.time() - start_time
        
        return {
            "model_name": model_name,
            "preloaded_count": preloaded_count,
            "total_frames": len(frame_range),
            "errors": errors,
            "processing_time_ms": int(total_time * 1000)
        }
    
    async def clear_model_cache(self, model_name: str) -> int:
        """Clear cache for specific model"""
        
        try:
            pattern = f"{self.FRAME_PREFIX}{model_name}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                cleared = self.redis_client.delete(*keys)
                # Reset stats for this model
                if model_name in self.model_stats:
                    self.model_stats[model_name] = {"cache_hits": 0, "cache_misses": 0, "cache_sets": 0}
                return cleared
            return 0
                
        except Exception as e:
            print(f"Cache clear error for {model_name}: {e}")
            return 0
    
    async def clear_all_cache(self) -> Dict[str, int]:
        """Clear all cache"""
        
        try:
            frame_keys = self.redis_client.keys(f"{self.FRAME_PREFIX}*")
            audio_keys = self.redis_client.keys(f"{self.AUDIO_PREFIX}*")
            meta_keys = self.redis_client.keys(f"{self.METADATA_PREFIX}*")
            
            frame_cleared = self.redis_client.delete(*frame_keys) if frame_keys else 0
            audio_cleared = self.redis_client.delete(*audio_keys) if audio_keys else 0
            meta_cleared = self.redis_client.delete(*meta_keys) if meta_keys else 0
            
            # Reset all stats
            self.model_stats = {}
            
            return {
                "frames_cleared": frame_cleared,
                "audio_cleared": audio_cleared,
                "metadata_cleared": meta_cleared,
                "total_cleared": frame_cleared + audio_cleared + meta_cleared
            }
                
        except Exception as e:
            print(f"Cache clear error: {e}")
            return {"error": str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        try:
            # Get Redis info
            redis_info = self.redis_client.info()
            
            # Count cached items per model
            model_frame_counts = {}
            all_frame_keys = self.redis_client.keys(f"{self.FRAME_PREFIX}*")
            
            for key in all_frame_keys:
                key_str = key.decode('utf-8')
                # Extract model name from key: "frame:model_name:frame_id"
                parts = key_str.split(':', 2)
                if len(parts) >= 2:
                    model_name = parts[1]
                    model_frame_counts[model_name] = model_frame_counts.get(model_name, 0) + 1
            
            # Get audio and metadata counts
            audio_count = len(self.redis_client.keys(f"{self.AUDIO_PREFIX}*"))
            metadata_count = len(self.redis_client.keys(f"{self.METADATA_PREFIX}*"))
            
            # Calculate overall statistics
            total_hits = sum(stats["cache_hits"] for stats in self.model_stats.values())
            total_misses = sum(stats["cache_misses"] for stats in self.model_stats.values())
            total_sets = sum(stats["cache_sets"] for stats in self.model_stats.values())
            total_requests = total_hits + total_misses
            overall_hit_ratio = total_hits / total_requests if total_requests > 0 else 0
            
            return {
                "overall_stats": {
                    "total_cache_hits": total_hits,
                    "total_cache_misses": total_misses,
                    "total_cache_sets": total_sets,
                    "overall_hit_ratio": overall_hit_ratio,
                    "total_cached_frames": sum(model_frame_counts.values()),
                    "cached_audio": audio_count,
                    "cached_metadata": metadata_count
                },
                "model_stats": self.model_stats,
                "model_frame_counts": model_frame_counts,
                "redis_info": {
                    "memory_mb": redis_info.get("used_memory", 0) / 1024 / 1024,
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "total_keys": redis_info.get("db0", {}).get("keys", 0) if "db0" in redis_info else 0
                }
            }
            
        except Exception as e:
            print(f"Cache stats error: {e}")
            return {
                "overall_stats": self.model_stats,
                "error": str(e)
            }
    
    def ping(self) -> bool:
        """Check Redis connection"""
        
        try:
            self.redis_client.ping()
            return True
        except:
            return False


# Global multi-model cache manager instance
multi_cache_manager = MultiModelCacheManager()
