#!/usr/bin/env python3
"""
Re-export UNet model with 512-dimensional audio encoder input (NEW format)
This will create a model compatible with the audio_encoder.onnx output
"""

import torch
import torch.nn as nn
import sys
import os

# Add path to unet_328.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unet_328 import Model

print("="*80)
print("🔧 Re-exporting UNet Model with Audio Encoder (512-dim input)")
print("="*80)

# Load the trained model
checkpoint_path = "old/old_minimal_server/models/sanders/checkpoint/checkpoint_epoch_335.pth.tar"
print(f"\n📦 Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cuda')
print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")

# Create model with 'ave' mode (the current trained model)
model = Model(n_channels=6, mode='ave')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.cuda()

print(f"✅ Model loaded successfully")

# CRITICAL: We need to modify the model to accept 512-dim audio input
# The current model has AudioConvAve which expects [batch, 32, 16, 16]
# We need to create a wrapper that reshapes [batch, 512] to work with the model

print(f"\n⚠️  PROBLEM: Current model expects audio shape [batch, 32, 16, 16] = 8,192")
print(f"   But audio encoder outputs [batch, 512]")
print(f"\n🔧 SOLUTION OPTIONS:")
print(f"   1. Retrain the model with 512-dim audio encoder")
print(f"   2. Create adapter layer to reshape 512 → 8,192")
print(f"   3. Modify audio encoder to output 8,192 features")

print(f"\n💡 RECOMMENDATION:")
print(f"   The model needs to be RETRAINED with the audio encoder")
print(f"   integrated into the training pipeline.")
print(f"\n   Current checkpoint was trained with PRE-COMPUTED 8,192-dim features")
print(f"   from 'aud_ave.npy' (16-frame window of old audio features)")

print(f"\n📊 Current Model Architecture:")
print(f"   - Visual input: [batch, 6, 320, 320]")
print(f"   - Audio input:  [batch, 32, 16, 16] ← OLD FORMAT")
print(f"   - Output:       [batch, 3, 320, 320]")

print(f"\n📊 Required Model Architecture (for audio encoder):")
print(f"   - Visual input: [batch, 6, 320, 320]")
print(f"   - Audio input:  [batch, 512, 1, 1] or [batch, 512] ← NEW FORMAT")
print(f"   - Output:       [batch, 3, 320, 320]")

print(f"\n" + "="*80)
print(f"CANNOT PROCEED WITHOUT RETRAINING")
print(f"="*80)
print(f"\nThe current model checkpoint (epoch 335) was trained with the OLD")
print(f"audio feature format. To use the audio encoder end-to-end, you need to:")
print(f"\n1. Modify training code to use audio_encoder.onnx during training")
print(f"2. Retrain the model from scratch (or fine-tune)")
print(f"3. Export the new checkpoint to ONNX")
print(f"\nOR")
print(f"\n4. Use the OLD audio feature format (32×16×16) for now")
print(f"   This means bypassing the audio encoder and using pre-computed")
print(f"   audio features like the current mock tests do.")
