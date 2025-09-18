"""
GPU memory and inference optimizations
"""
import torch
import torch.nn as nn
from typing import Optional, List

class OptimizedInferenceEngine:
    """Optimized model inference with memory management"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Enable optimizations
        self.model.eval()
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)  # PyTorch 2.0 optimization
            
        # Pre-allocate common tensor shapes
        self._preallocate_tensors()
        
    def _preallocate_tensors(self):
        """Pre-allocate frequently used tensors"""
        self.image_tensor_pool = []
        self.audio_tensor_pool = []
        
        # Pre-allocate 10 image tensors
        for _ in range(10):
            img_tensor = torch.zeros((1, 6, 320, 320), device=self.device, dtype=torch.float32)
            self.image_tensor_pool.append(img_tensor)
            
    def get_image_tensor(self) -> torch.Tensor:
        """Get reusable image tensor"""
        if self.image_tensor_pool:
            return self.image_tensor_pool.pop()
        return torch.zeros((1, 6, 320, 320), device=self.device, dtype=torch.float32)
        
    def return_image_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        if len(self.image_tensor_pool) < 10:
            self.image_tensor_pool.append(tensor)
            
    @torch.inference_mode()  # Faster than torch.no_grad()
    def infer_optimized(self, image_data: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        """Optimized inference with memory reuse"""
        
        # Use pre-allocated tensor
        input_tensor = self.get_image_tensor()
        
        try:
            # Copy data to pre-allocated tensor (faster than creating new)
            input_tensor.copy_(image_data)
            
            # Mixed precision inference for speed
            with torch.cuda.amp.autocast():
                result = self.model(input_tensor, audio_feat)
                
            return result
            
        finally:
            # Return tensor to pool
            self.return_image_tensor(input_tensor)

class TensorOptimizer:
    """Optimize tensor operations"""
    
    @staticmethod
    def optimize_image_preprocessing(img: torch.Tensor) -> torch.Tensor:
        """Optimized image preprocessing"""
        # Use in-place operations where possible
        img.div_(255.0)  # In-place division
        return img
        
    @staticmethod
    def batch_tensor_ops(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Batch multiple tensor operations"""
        # Stack tensors for batch processing
        return torch.stack(tensors, dim=0)
