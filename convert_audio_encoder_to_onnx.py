#!/usr/bin/env python3
"""
Convert audio_encoder.pth to ONNX format

This script:
1. Loads the AudioEncoder model architecture
2. Loads weights from audio_encoder.pth
3. Exports to ONNX with proper input shape
4. Validates the ONNX model
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os

# Audio encoder architecture (from test_single_frame_pth.py)
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.LeakyReLU(0.01, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)
        return out


def find_audio_encoder_checkpoint():
    """Find audio_encoder.pth in common locations"""
    search_paths = [
        "data_utils/ave/checkpoints/audio_encoder.pth",
        "checkpoints/audio_encoder.pth",
        "audio_encoder.pth",
        "data_utils/ave/audio_encoder.pth",
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        f"Could not find audio_encoder.pth. Searched:\n" + 
        "\n".join(f"  - {p}" for p in search_paths)
    )


def convert_to_onnx():
    print("=" * 70)
    print("Audio Encoder PyTorch → ONNX Conversion")
    print("=" * 70)
    
    # Find checkpoint
    print("\n1️⃣  Locating audio_encoder.pth...")
    pth_path = find_audio_encoder_checkpoint()
    print(f"   ✅ Found: {pth_path}")
    
    # Load model
    print("\n2️⃣  Loading PyTorch model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioEncoder().to(device)
    
    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    
    # The checkpoint has weights saved directly (not wrapped in audio_encoder)
    # So we need to load into model.audio_encoder instead of model
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.audio_encoder.load_state_dict(checkpoint['model'])
        print(f"   ✅ Loaded from checkpoint['model']")
    elif isinstance(checkpoint, dict):
        # Load directly into the audio_encoder submodule
        model.audio_encoder.load_state_dict(checkpoint)
        print(f"   ✅ Loaded state dict into audio_encoder")
    else:
        raise ValueError("Unexpected checkpoint format")
    
    model.eval()
    print(f"   ✅ Model loaded on {device}")
    
    # Create dummy input
    # Input shape: [batch_size, channels, mel_bins, time_steps]
    # Mel-spectrogram is transposed: [16, 80] -> [80, 16]
    # Then add batch and channel: [1, 1, 80, 16]
    print("\n3️⃣  Creating dummy input...")
    batch_size = 1
    channels = 1
    mel_bins = 80    # number of mel frequency bins
    time_steps = 16  # syncnet_mel_step_size
    
    dummy_input = torch.randn(batch_size, channels, mel_bins, time_steps).to(device)
    print(f"   Input shape: {list(dummy_input.shape)} [batch, channels, mel_bins, time_steps]")
    
    # Test forward pass
    print("\n4️⃣  Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {list(output.shape)}")
    print(f"   ✅ Forward pass successful")
    
    # Export to ONNX
    print("\n5️⃣  Exporting to ONNX...")
    onnx_path = "audio_encoder.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['mel_spectrogram'],
        output_names=['audio_features'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size'},
            'audio_features': {0: 'batch_size'}
        }
    )
    print(f"   ✅ Exported to: {onnx_path}")
    
    # Verify ONNX model
    print("\n6️⃣  Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ✅ ONNX model is valid")
    
    # Test ONNX Runtime
    print("\n7️⃣  Testing ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    print(f"   Input: {input_name} {ort_session.get_inputs()[0].shape}")
    print(f"   Output: {output_name} {ort_session.get_outputs()[0].shape}")
    
    # Test inference
    test_input = np.random.randn(1, 1, 80, 16).astype(np.float32)
    ort_output = ort_session.run([output_name], {input_name: test_input})[0]
    
    print(f"   ONNX output shape: {ort_output.shape}")
    print(f"   ✅ ONNX Runtime inference successful")
    
    # Compare PyTorch vs ONNX
    print("\n8️⃣  Comparing PyTorch vs ONNX outputs...")
    with torch.no_grad():
        torch_output = model(torch.from_numpy(test_input).to(device)).cpu().numpy()
    
    max_diff = np.max(np.abs(torch_output - ort_output))
    mean_diff = np.mean(np.abs(torch_output - ort_output))
    
    print(f"   Max difference: {max_diff:.6e}")
    print(f"   Mean difference: {mean_diff:.6e}")
    
    if max_diff < 1e-4:
        print(f"   ✅ Outputs match!")
    else:
        print(f"   ⚠️  Warning: Outputs differ by {max_diff}")
    
    # File size
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n📊 ONNX model size: {file_size_mb:.2f} MB")
    
    print("\n" + "=" * 70)
    print("✅ Conversion Complete!")
    print("=" * 70)
    print(f"\nONNX model saved to: {onnx_path}")
    print(f"\nInput shape:  [batch_size, 1, 80, 16]  (mel-spectrogram transposed)")
    print(f"Output shape: [batch_size, 512]         (audio features)")
    print("\nNote: The output will be reshaped to [batch_size, 32, 16] for lip-sync model")
    print("=" * 70)
    
    return onnx_path


if __name__ == "__main__":
    try:
        onnx_path = convert_to_onnx()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
