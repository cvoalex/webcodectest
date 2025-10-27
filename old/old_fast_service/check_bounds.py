#!/usr/bin/env python3
"""Check bounds file format"""
import numpy as np

bounds = np.load("models/default_model/face_bounds/100.npy")
print(f"Bounds shape: {bounds.shape}")
print(f"Bounds: {bounds}")
print(f"Bounds dtype: {bounds.dtype}")
