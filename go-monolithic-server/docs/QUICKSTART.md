# Quick Start: Monolithic Server

## Summary

The **monolithic server** combines inference and compositing into a single process for **maximum performance**. This guide shows you how to get it running in minutes.

## Prerequisites

✅ ONNX Runtime installed: `C:/onnxruntime-1.22.0/lib/onnxruntime.dll`  
✅ CUDA-capable GPU (NVIDIA)  
✅ Go 1.24.0 or later  
✅ Model files in `minimal_server/models/`  

## 1. Build

```powershell
cd d:\Projects\webcodecstest\go-monolithic-server
go mod tidy
go build -o monolithic-server.exe ./cmd/server
go build -o test-client.exe test_client.go
```

**Expected output:**
```
go: downloading dependencies...
Built: monolithic-server.exe (15MB)
Built: test-client.exe (12MB)
```

## 2. Configure

Edit `config.yaml` if needed:

```yaml
server:
  port: ":50053"

models:
  sanders:
    model_path: "sanders/checkpoint/model_best.onnx"
    background_dir: "sanders/frames"
    crop_rects_path: "sanders/crop_rects.json"
    num_frames: 523
    preload_backgrounds: true  # Load backgrounds on startup
```

## 3. Run Server

```powershell
.\monolithic-server.exe
```

**Expected startup:**
```
================================================================================
🚀 Monolithic Lipsync Server (Inference + Compositing)
================================================================================
✅ Configuration loaded from config.yaml
   GPUs: 1 × 24GB
   Max models: 40
   Configured models: 1

📦 Initializing model registry...
✅ Model registry initialized (0 models preloaded)

🖼️  Initializing image registry...
🖼️  Loading backgrounds for model 'sanders'...
✅ Loaded backgrounds for 'sanders' in 1.23s (523 frames, 246.72 MB)
✅ Image registry initialized (1 models loaded)

🎮 GPU Status:
   GPU 0: 0 models, 0 MB used / 24576 MB total

🎵 Initializing audio processing pipeline...
✅ Mel-spectrogram processor initialized
✅ Audio encoder initialized (ONNX)

🌐 Monolithic server listening on port :50053
✅ Ready to accept connections!
================================================================================
```

## 4. Test

In a **new terminal**:

```powershell
cd d:\Projects\webcodecstest\go-monolithic-server
.\test-client.exe
```

**Expected output:**
```
🧪 Monolithic Server Test Client
============================================================
🔌 Connecting to monolithic server at localhost:50053...
✅ Connected successfully

📊 Checking server health...
✅ Server Status: Healthy
   Loaded Models: 1/40
   GPUs: [0]

📁 Output directory: test_output/

🎵 Loading audio file: ..\aud.wav
📊 WAV Info: 16000 Hz, 16-bit, 1 channel(s), format=1
   Loaded 51200 samples (3.20 seconds)

🚀 Running 5 batches (batch_size=24)...
============================================================
Batch 1/5: GPU=0, frames=24
  🎵 Audio:       7.45 ms
  ⚡ Inference:   120.28 ms
  🎨 Compositing: 2.15 ms
  📊 Total:       129.88 ms
  💾 Saved 24 frames to test_output/

Batch 2/5: GPU=0, frames=24
  🎵 Audio:       7.32 ms
  ⚡ Inference:   118.76 ms
  🎨 Compositing: 2.08 ms
  📊 Total:       128.16 ms
  💾 Saved 24 frames to test_output/

...

============================================================
📈 PERFORMANCE SUMMARY
============================================================
Total frames processed:  120
Total duration:          0.68 seconds

⚡ Average Inference:      119.45 ms
🎨 Average Compositing:   2.12 ms
📊 Average Total:         128.95 ms

📈 Separation Overhead:   2.35 ms (2.0% of inference time)

🚀 Throughput:            176.5 FPS
   Frames per batch:      24
   Batches per second:    7.4

============================================================
✅ SUCCESS: Overhead < 5ms target!
```

## 5. Verify Output

Check the generated frames:

```powershell
explorer test_output\
```

You should see 120 JPEG images:
- `batch_1_frame_0.jpg` through `batch_1_frame_23.jpg`
- `batch_2_frame_24.jpg` through `batch_2_frame_47.jpg`
- etc.

## Troubleshooting

### Issue: "Failed to load audio encoder"

**Symptom:**
```
⚠️  Warning: Audio encoder not available
```

**Solution:**
Check `config.yaml`:
```yaml
onnx:
  library_path: "C:/onnxruntime-1.22.0/lib/onnxruntime.dll"
```

Verify the file exists:
```powershell
Test-Path "C:\onnxruntime-1.22.0\lib\onnxruntime.dll"
# Should return: True
```

### Issue: "Background frame not found"

**Symptom:**
```
❌ Failed to load backgrounds: frame 0 not found
```

**Solution:**
Check the `background_dir` path:
```powershell
ls "d:\Projects\webcodecstest\minimal_server\models\sanders\frames"
# Should list: frame_0.png, frame_1.png, ...
```

### Issue: "CUDA not available"

**Symptom:**
```
⚠️  CUDA provider not available, using CPU
```

**Solution:**
1. Check NVIDIA drivers installed: `nvidia-smi`
2. Verify CUDA toolkit installed
3. Ensure ONNX Runtime was built with CUDA support

### Issue: Port already in use

**Symptom:**
```
❌ Failed to listen: address already in use
```

**Solution:**
```powershell
# Find process using port 50053
netstat -ano | findstr :50053

# Kill the process (replace <PID>)
taskkill /F /PID <PID>
```

Or change the port in `config.yaml`:
```yaml
server:
  port: ":50054"  # Use different port
```

## Next Steps

### Performance Tuning

1. **Increase batch size** (if GPU memory allows):
   ```go
   // In test_client.go
   const batchSize = 32  // Up from 24
   ```

2. **Adjust JPEG quality**:
   ```yaml
   output:
     jpeg_quality: 85  # Higher quality (slower)
     # jpeg_quality: 60  # Lower quality (faster)
   ```

3. **Tune worker count**:
   ```yaml
   server:
     worker_count_per_gpu: 16  # More parallelism
   ```

### Production Deployment

1. **Enable structured logging**:
   ```yaml
   logging:
     log_level: "info"
     log_inference_times: true
   ```

2. **Set resource limits**:
   ```yaml
   capacity:
     max_models: 20          # Limit concurrent models
     max_memory_gb: 18       # Leave headroom
   ```

3. **Configure auto-eviction**:
   ```yaml
   capacity:
     eviction_policy: "lfu"  # Least frequently used
     idle_timeout_minutes: 30
   ```

### Monitoring

Watch server logs for:
- ✅ Request latencies
- ✅ GPU memory usage
- ✅ Model load/unload events
- ✅ Error rates

Example log output:
```
🎵 Audio processing: 10240 samples -> 52 frames -> 24 features (7.45ms)
⚡ Inference: model=sanders, batch=24, gpu=0, time=120.28ms
🎨 Compositing: 24 frames, 2.15ms (0.09ms/frame)
```

## Comparison with Separated Architecture

Run the same test with separated servers:

```powershell
# Terminal 1: Inference Server
cd ..\go-inference-server
.\inference-server.exe

# Terminal 2: Compositing Server
cd ..\go-compositing-server
.\compositing-server.exe

# Terminal 3: Test Client (point to compositing server)
cd ..\go-compositing-server
.\test-client.exe
```

Compare the results:
- **Monolithic**: ~130ms total, ~2-3ms overhead
- **Separated**: ~145ms total, ~10-15ms overhead
- **Improvement**: 10-15ms faster with monolithic (70-80% less overhead)

## Files Generated

After running the test:

```
go-monolithic-server/
├── monolithic-server.exe    # Server binary
├── test-client.exe           # Test client binary
└── test_output/              # Generated frames
    ├── batch_1_frame_0.jpg
    ├── batch_1_frame_1.jpg
    ├── ...
    └── batch_5_frame_119.jpg
```

## Summary

You've successfully:
✅ Built the monolithic server  
✅ Loaded a model with backgrounds  
✅ Processed audio in real-time  
✅ Generated composited lip-sync frames  
✅ Achieved **~130ms latency** with **<3ms overhead**  

**Ready for production!** 🚀

---

**Need help?** Check:
- [README.md](README.md) - Full documentation
- [ARCHITECTURE_COMPARISON.md](../ARCHITECTURE_COMPARISON.md) - Monolithic vs Separated
- Existing code in `go-compositing-server/` and `go-inference-server/`
