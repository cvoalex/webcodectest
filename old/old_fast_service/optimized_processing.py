"""
Optimized image processing with memory pooling
"""
import cv2
import numpy as np
from typing import Dict, List

class ImageMemoryPool:
    """Pre-allocate and reuse image buffers"""
    
    def __init__(self, max_size: int = 100):
        self.pool_328 = []  # 328x328 buffers
        self.pool_320 = []  # 320x320 buffers  
        self.max_size = max_size
        
    def get_buffer_328(self) -> np.ndarray:
        if self.pool_328:
            return self.pool_328.pop()
        return np.zeros((328, 328, 3), dtype=np.uint8)
    
    def get_buffer_320(self) -> np.ndarray:
        if self.pool_320:
            return self.pool_320.pop()
        return np.zeros((320, 320, 3), dtype=np.uint8)
    
    def return_buffer_328(self, buffer: np.ndarray):
        if len(self.pool_328) < self.max_size:
            self.pool_328.append(buffer)
    
    def return_buffer_320(self, buffer: np.ndarray):
        if len(self.pool_320) < self.max_size:
            self.pool_320.append(buffer)

class OptimizedImageProcessor:
    """Optimized image processing pipeline"""
    
    def __init__(self):
        self.memory_pool = ImageMemoryPool()
        self.face_tracker = None  # For face tracking between frames
        
    def process_frame_optimized(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Optimized frame processing"""
        
        # Use pre-computed crop bounds
        xmin, ymin, xmax, ymax = self._get_crop_bounds(landmarks)
        
        # Get reusable buffer
        crop_buffer = self.memory_pool.get_buffer_328()
        
        # Single resize operation with faster interpolation
        crop_region = img[ymin:ymax, xmin:xmax]
        cv2.resize(crop_region, (328, 328), dst=crop_buffer, interpolation=cv2.INTER_LINEAR)
        
        # Optimized cropping (avoid copy)
        final_crop = crop_buffer[4:324, 4:324]
        
        return final_crop
        
    def _get_crop_bounds(self, landmarks: np.ndarray) -> tuple:
        """Pre-compute crop bounds from landmarks"""
        xmin = landmarks[1][0]
        ymin = landmarks[52][1] 
        xmax = landmarks[31][0]
        width = xmax - xmin
        ymax = ymin + width
        return xmin, ymin, xmax, ymax
