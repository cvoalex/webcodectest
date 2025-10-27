import onnx
import numpy as np

# Load the ONNX model
model_path = "d:/Projects/webcodecstest/old/old_minimal_server/models/sanders/checkpoint/model_best.onnx"
model = onnx.load(model_path)

print("=" * 80)
print("ONNX Model Input/Output Analysis")
print("=" * 80)

print("\n📥 INPUT TENSORS:")
for i, input_tensor in enumerate(model.graph.input):
    print(f"\n  Input {i}: {input_tensor.name}")
    print(f"    Type: {input_tensor.type.tensor_type.elem_type}")
    
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append(f"dynamic({dim.dim_param})")
    print(f"    Shape: {shape}")
    
    if shape:
        total_elements = 1
        for s in shape:
            if isinstance(s, int):
                total_elements *= s
        if isinstance(total_elements, int):
            print(f"    Total elements: {total_elements:,}")

print("\n📤 OUTPUT TENSORS:")
for i, output_tensor in enumerate(model.graph.output):
    print(f"\n  Output {i}: {output_tensor.name}")
    print(f"    Type: {output_tensor.type.tensor_type.elem_type}")
    
    shape = []
    for dim in output_tensor.type.tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append(f"dynamic({dim.dim_param})")
    print(f"    Shape: {shape}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)

# Get the audio input shape
audio_input = model.graph.input[1] if len(model.graph.input) > 1 else None
if audio_input:
    audio_shape = []
    for dim in audio_input.type.tensor_type.shape.dim:
        if dim.dim_value:
            audio_shape.append(dim.dim_value)
    
    if len(audio_shape) >= 1:
        print(f"\n🎵 Audio Input Shape: {audio_shape}")
        if len(audio_shape) > 1:
            print(f"   Format: batch × {' × '.join(map(str, audio_shape[1:]))}")
        
        total = 1
        for s in audio_shape:
            if isinstance(s, int):
                total *= s
        print(f"   Total audio features per sample: {total:,}")
        
        if total == 8192:
            print(f"\n   ✅ This is the OLD format (32 × 16 × 16 = 8,192)")
            print(f"   ❌ Does NOT match audio encoder output (512)")
            print(f"\n   🔧 SOLUTION: Need to re-export UNet with 512-dim audio input")
        elif total == 512:
            print(f"\n   ✅ This matches the audio encoder output!")
        else:
            print(f"\n   ⚠️  Unknown audio format")
