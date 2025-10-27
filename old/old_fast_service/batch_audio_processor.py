"""
Optimized audio processing with batching
"""
import torch
import numpy as np
from typing import List, Tuple

class BatchAudioProcessor:
    """Process audio in batches for better GPU utilization"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.audio_cache = {}  # Cache processed audio features
        
    def process_audio_batch(self, audio_path: str, batch_size: int = 32) -> torch.Tensor:
        """Process entire audio file in optimized batches"""
        
        # Check cache first
        if audio_path in self.audio_cache:
            return self.audio_cache[audio_path]
            
        # Process in larger batches for better GPU utilization
        from utils import AudDataset
        from torch.utils.data import DataLoader
        
        dataset = AudDataset(audio_path)
        # Increase batch size for better GPU utilization
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        outputs = []
        with torch.no_grad():
            for mel_batch in data_loader:
                mel_batch = mel_batch.to(self.device, non_blocking=True)
                # Process entire batch at once
                batch_output = self._process_mel_batch(mel_batch)
                outputs.append(batch_output.cpu())
                
        # Concatenate and cache result
        full_output = torch.cat(outputs, dim=0)
        self.audio_cache[audio_path] = full_output
        
        return full_output
        
    def _process_mel_batch(self, mel_batch: torch.Tensor) -> torch.Tensor:
        """Process mel spectrogram batch"""
        # This would use your AudioEncoder model
        # Placeholder for actual model inference
        return mel_batch  # Replace with actual model call
        
    def get_audio_features_optimized(self, audio_feats: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Optimized audio feature extraction"""
        # Pre-compute padding once
        if not hasattr(self, '_padded_features'):
            first_frame, last_frame = audio_feats[:1], audio_feats[-1:]
            self._padded_features = torch.cat([
                first_frame.repeat(1, 1), 
                audio_feats, 
                last_frame.repeat(1, 1)
            ], dim=0)
            
        # Return slice directly (no copy)
        return self._padded_features[frame_idx:frame_idx+3]
