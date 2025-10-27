"""
Export UNet model to ONNX with correct audio input shape [batch, 16, 32, 32]
This matches our pipeline: 16 frames × 512 features = 8,192 reshaped to [16, 32, 32]
"""
import torch
import onnxruntime
import numpy as np
from unet_328 import Model

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model with 'ave' mode (AudioConvAve)
net = Model(n_channels=6, mode='ave').eval().to(device)

# Load checkpoint
checkpoint_path = 'data_utils/checkpoint_epoch_335.pth.tar'
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Extract only the Model state dict (exclude pfld_backbone and auxiliarynet)
state_dict = checkpoint if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint else checkpoint['state_dict']

# Filter to only include model weights (exclude pfld_backbone, auxiliarynet, epoch)
model_state_dict = {}
for key, value in state_dict.items():
    if not key.startswith('pfld_backbone') and not key.startswith('auxiliarynet') and key != 'epoch':
        model_state_dict[key] = value

net.load_state_dict(model_state_dict)
print("Model loaded successfully")

# Define input shapes
# Visual: [batch, 6, 320, 320] - 6 channels (reference frame BGR × 2)
# Audio: [batch, 16, 32, 32] - 16 frames × 512 features reshaped to [16, 32, 32]
batch_size = 1
visual_input = torch.zeros([batch_size, 6, 320, 320]).to(device)
audio_input = torch.zeros([batch_size, 16, 32, 32]).to(device)

print(f"\nInput shapes:")
print(f"  Visual: {list(visual_input.shape)}")
print(f"  Audio: {list(audio_input.shape)}")

# Test forward pass
with torch.no_grad():
    torch_out = net(visual_input, audio_input)
    print(f"  Output: {list(torch_out.shape)}")

# Export to ONNX
onnx_path = 'model_best_16x32x32.onnx'
print(f"\nExporting to ONNX: {onnx_path}")

with torch.no_grad():
    torch.onnx.export(
        net,
        (visual_input, audio_input),
        onnx_path,
        input_names=['input', 'audio'],
        output_names=['output'],
        opset_version=11,
        export_params=True,
        dynamic_axes={
            'input': {0: 'batch'},
            'audio': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )

print("ONNX export complete")

# Verify with ONNX Runtime
print("\nVerifying with ONNX Runtime...")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
print(f"Providers: {ort_session.get_providers()}")

# Print input/output info
print("\nONNX Model Info:")
for inp in ort_session.get_inputs():
    print(f"  Input '{inp.name}': {inp.shape}")
for out in ort_session.get_outputs():
    print(f"  Output '{out.name}': {out.shape}")

# Test inference
ort_inputs = {
    ort_session.get_inputs()[0].name: visual_input.cpu().numpy(),
    ort_session.get_inputs()[1].name: audio_input.cpu().numpy()
}
ort_outs = ort_session.run(None, ort_inputs)

# Compare outputs
np.testing.assert_allclose(
    torch_out.cpu().numpy(),
    ort_outs[0],
    rtol=1e-03,
    atol=1e-05
)
print("\n✅ ONNX model verified successfully!")
print(f"   Output shape: {ort_outs[0].shape}")
print(f"   Audio input: [batch, 16, 32, 32] = 16,384 elements")
print(f"\nNew model saved to: {onnx_path}")
