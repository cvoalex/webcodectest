import redis
import json
import numpy as np
import cv2
from typing import Optional, Dict, Any, List
import pickle
import time
import hashlib
from config import settings


class CacheManager:
    """Redis-based cache manager for frames and metadata"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(
            settings.REDIS_URL,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            decode_responses=False  # We need binary data for images
        )
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        
        # Key prefixes
        self.FRAME_PREFIX = "frame:"
        self.AUDIO_PREFIX = "audio:"
        self.METADATA_PREFIX = "meta:"
        self.STATS_PREFIX = "stats:"
        
    async def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get cached frame by ID"""
        
        try:
            key = f"{self.FRAME_PREFIX}{frame_id}"
            cached_data = self.redis_client.get(key)
            
            if cached_data is not None:
                self.cache_hits += 1
                # Deserialize frame data
                frame = pickle.loads(cached_data)
                return frame
            else:
                self.cache_misses += 1
                return None
                
        except Exception as e:
            print(f"Cache get error: {e}")
            self.cache_misses += 1
            return None
    
    async def set_frame(self, frame_id: int, frame: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache frame with optional TTL"""
        
        try:
            key = f"{self.FRAME_PREFIX}{frame_id}"
            
            # Serialize frame data
            serialized_frame = pickle.dumps(frame)
            
            # Set with TTL
            ttl = ttl or settings.FRAME_CACHE_TTL
            self.redis_client.setex(key, ttl, serialized_frame)
            
            self.cache_sets += 1
            return True
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def get_frame_batch(self, frame_ids: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """Get multiple frames in batch"""
        
        keys = [f"{self.FRAME_PREFIX}{frame_id}" for frame_id in frame_ids]
        
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
                        self.cache_hits += 1
                    except:
                        results[frame_id] = None
                        self.cache_misses += 1
                else:
                    results[frame_id] = None
                    self.cache_misses += 1
            
            return results
            
        except Exception as e:
            print(f"Batch cache get error: {e}")
            self.cache_misses += len(frame_ids)
            return {frame_id: None for frame_id in frame_ids}
    
    async def set_frame_batch(self, frames: Dict[int, np.ndarray], ttl: Optional[int] = None) -> int:
        """Cache multiple frames in batch"""
        
        try:
            pipe = self.redis_client.pipeline()
            ttl = ttl or settings.FRAME_CACHE_TTL
            
            for frame_id, frame in frames.items():
                key = f"{self.FRAME_PREFIX}{frame_id}"
                serialized_frame = pickle.dumps(frame)
                pipe.setex(key, ttl, serialized_frame)
            
            pipe.execute()
            
            self.cache_sets += len(frames)
            return len(frames)
            
        except Exception as e:
            print(f"Batch cache set error: {e}")
            return 0
    
    async def get_audio_features(self, audio_hash: str) -> Optional[np.ndarray]:
        """Get cached audio features by hash"""
        
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
        """Cache audio features with hash key"""
        
        try:
            key = f"{self.AUDIO_PREFIX}{audio_hash}"
            serialized_audio = pickle.dumps(audio_features)
            
            ttl = ttl or settings.AUDIO_CACHE_TTL
            self.redis_client.setex(key, ttl, serialized_audio)
            
            return True
            
        except Exception as e:
            print(f"Audio cache set error: {e}")
            return False
    
    def calculate_audio_hash(self, audio_path: str) -> str:
        """Calculate hash for audio file"""
        
        # Use file path + modification time for hash
        try:
            import os
            stat = os.stat(audio_path)
            hash_input = f"{audio_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except:
            # Fallback to just path
            return hashlib.md5(audio_path.encode()).hexdigest()
    
    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached metadata"""
        
        try:
            redis_key = f"{self.METADATA_PREFIX}{key}"
            cached_data = self.redis_client.get(redis_key)
            
            if cached_data is not None:
                metadata = json.loads(cached_data.decode('utf-8'))
                return metadata
            
            return None
            
        except Exception as e:
            print(f"Metadata cache get error: {e}")
            return None
    
    async def set_metadata(self, key: str, metadata: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache metadata"""
        
        try:
            redis_key = f"{self.METADATA_PREFIX}{key}"
            serialized_metadata = json.dumps(metadata)
            
            ttl = ttl or settings.FRAME_CACHE_TTL
            self.redis_client.setex(redis_key, ttl, serialized_metadata)
            
            return True
            
        except Exception as e:
            print(f"Metadata cache set error: {e}")
            return False
    
    async def preload_frames(self, frame_range: range, inference_engine) -> Dict[str, Any]:
        """Background preloading of frames"""
        
        start_time = time.time()
        preloaded_count = 0
        errors = []
        
        for frame_id in frame_range:
            try:
                # Check if already cached
                cached_frame = await self.get_frame(frame_id)
                if cached_frame is not None:
                    continue
                
                # Generate and cache frame
                frame, metadata = await inference_engine.generate_frame(frame_id)
                await self.set_frame(frame_id, frame)
                
                preloaded_count += 1
                
            except Exception as e:
                errors.append(f"Frame {frame_id}: {str(e)}")
        
        total_time = time.time() - start_time
        
        return {
            "preloaded_count": preloaded_count,
            "total_frames": len(frame_range),
            "errors": errors,
            "processing_time_ms": int(total_time * 1000)
        }
    
    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache with optional pattern"""
        
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                # Clear all frame cache
                keys = self.redis_client.keys(f"{self.FRAME_PREFIX}*")
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
                
        except Exception as e:
            print(f"Cache clear error: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        try:
            # Get Redis info
            redis_info = self.redis_client.info()
            
            # Count cached items
            frame_count = len(self.redis_client.keys(f"{self.FRAME_PREFIX}*"))
            audio_count = len(self.redis_client.keys(f"{self.AUDIO_PREFIX}*"))
            metadata_count = len(self.redis_client.keys(f"{self.METADATA_PREFIX}*"))
            
            total_requests = self.cache_hits + self.cache_misses
            hit_ratio = self.cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_sets": self.cache_sets,
                "hit_ratio": hit_ratio,
                "cached_frames": frame_count,
                "cached_audio": audio_count,
                "cached_metadata": metadata_count,
                "redis_memory_mb": redis_info.get("used_memory", 0) / 1024 / 1024,
                "redis_connected_clients": redis_info.get("connected_clients", 0)
            }
            
        except Exception as e:
            print(f"Cache stats error: {e}")
            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_sets": self.cache_sets,
                "error": str(e)
            }
    
    def ping(self) -> bool:
        """Check Redis connection"""
        
        try:
            self.redis_client.ping()
            return True
        except:
            return False


# Global cache manager instance
cache_manager = CacheManager()
