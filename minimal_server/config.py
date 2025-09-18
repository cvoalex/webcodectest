import os
from typing import Optional

class Settings:
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # Service Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Performance Configuration
    GPU_MEMORY_LIMIT: int = int(os.getenv("GPU_MEMORY_LIMIT", "2048"))  # MB
    CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "1000"))  # frames
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))  # frames per batch
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Cache Configuration
    FRAME_CACHE_TTL: int = int(os.getenv("FRAME_CACHE_TTL", "3600"))  # seconds
    AUDIO_CACHE_TTL: int = int(os.getenv("AUDIO_CACHE_TTL", "7200"))  # seconds
    
    # Model Configuration
    MODEL_WARMUP_FRAMES: int = int(os.getenv("MODEL_WARMUP_FRAMES", "5"))
    ENABLE_HALF_PRECISION: bool = os.getenv("ENABLE_HALF_PRECISION", "true").lower() == "true"
    
    # API Configuration
    MAX_FRAME_REQUEST: int = int(os.getenv("MAX_FRAME_REQUEST", "100"))
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"

settings = Settings()
