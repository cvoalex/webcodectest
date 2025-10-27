import onnxruntime as ort

model_path = "minimal_server/models/sanders/checkpoint/model_best.onnx"

session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

print("\n📝 Model Input Names:")
for input in session.get_inputs():
    print(f"   {input.name}: {input.shape}")

print("\n📤 Model Output Names:")
for output in session.get_outputs():
    print(f"   {output.name}: {output.shape}")
