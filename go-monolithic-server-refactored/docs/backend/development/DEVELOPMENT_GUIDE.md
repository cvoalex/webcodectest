# 🚀 Quick Start - Development Guide

> **Get the Go Monolithic Server running in under 30 minutes**

This guide will help you set up your development environment and start the server.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Go 1.21+** installed
  ```powershell
  go version  # Should show go1.21 or higher
  ```

- [ ] **ONNX Runtime** DLL available
  - Download from: https://github.com/microsoft/onnxruntime/releases
  - Extract to known location (e.g., `C:/onnxruntime-1.22.0/`)

- [ ] **CUDA/GPU** (optional but recommended)
  - NVIDIA GPU with CUDA 11.x or 12.x
  - CUDA toolkit installed

- [ ] **Git** for cloning the repository

- [ ] **Model files** available
  - ONNX model file (`.onnx`)
  - Background frames directory
  - Crop rectangles JSON

---

## Installation

### Step 1: Clone Repository

```powershell
cd d:\Projects\
git clone https://github.com/cvoalex/webcodectest.git
cd webcodectest\go-monolithic-server-refactored
```

### Step 2: Install Dependencies

```powershell
go mod download
```

Expected output:
```
go: downloading google.golang.org/grpc v1.xx.x
go: downloading github.com/yalue/onnxruntime_go v1.x.x
...
```

### Step 3: Configure ONNX Runtime Path

Edit `config.yaml`:

```yaml
onnx:
  library_path: "C:/onnxruntime-1.22.0/lib/onnxruntime.dll"  # ← Update this path
```

### Step 4: Configure Models

Edit `config.yaml`:

```yaml
models_root: "d:/Projects/webcodecstest/minimal_server/models"  # ← Update this

models:
  sanders:  # Your model ID
    model_path: "sanders/checkpoint/model_best.onnx"
    background_dir: "sanders/frames"
    crop_rects_path: "sanders/crop_rects.json"
    num_frames: 523
    preload_backgrounds: true
```

---

## Running the Server

### Step 1: Build

```powershell
go build -o server.exe ./cmd/server
```

### Step 2: Run

```powershell
.\server.exe
```

### Step 3: Verify Startup

Expected output:

```
================================================================================
🚀 Monolithic Lipsync Server (Inference + Compositing)
================================================================================
✅ Configuration loaded from config.yaml
   GPUs: 1 × 24GB
   Workers per GPU: 8 (total: 8 workers)
   Max models: 40
   Max memory: 20 GB
   Background cache: 600 frames per model
   Eviction policy: lfu
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
   Max message size: 100 MB
   Worker count: 8
   Queue size: 50

✅ Ready to accept connections!
================================================================================
```

✅ **Server is ready!**

---

## Running Tests

### All Functional Tests

```powershell
cd go-monolithic-server-refactored
go test ./functional-tests/... -v
```

Expected output:
```
ok  go-monolithic-server/functional-tests/audio-processing      0.709s
ok  go-monolithic-server/functional-tests/edgecases            1.153s
ok  go-monolithic-server/functional-tests/image-processing     0.748s
ok  go-monolithic-server/functional-tests/integration          1.363s
ok  go-monolithic-server/functional-tests/parallel-mel         0.765s
ok  go-monolithic-server/functional-tests/parallel-processing  0.778s
ok  go-monolithic-server/functional-tests/performance         27.605s
```

### Race Detector

```powershell
go test -race ./functional-tests/parallel-processing/...
```

### Benchmarks

```powershell
go test -bench=. ./functional-tests/performance/...
```

---

## Making Your First Request

### Using grpcurl

**1. Install grpcurl:**
```powershell
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
```

**2. List services:**
```powershell
grpcurl -plaintext localhost:50053 list
```

Output:
```
monolithic.MonolithicService
```

**3. Check health:**
```powershell
grpcurl -plaintext localhost:50053 monolithic.MonolithicService/Health
```

Output:
```json
{
  "status": "healthy",
  "loadedModels": 0,
  "maxModels": 40,
  "version": "1.0.0"
}
```

**4. List models:**
```powershell
grpcurl -plaintext localhost:50053 monolithic.MonolithicService/ListModels
```

---

## Development Workflow

### 1. Make Code Changes

Edit files in `internal/`, `cmd/`, etc.

### 2. Run Tests

```powershell
go test ./...
```

### 3. Rebuild

```powershell
go build -o server.exe ./cmd/server
```

### 4. Restart Server

```powershell
.\server.exe
```

---

## Troubleshooting

### Issue: "Cannot find ONNX Runtime DLL"

**Solution:** Update `config.yaml` with correct path:
```yaml
onnx:
  library_path: "C:/onnxruntime-1.22.0/lib/onnxruntime.dll"
```

### Issue: "Model not found"

**Solution:** Check `models_root` path in `config.yaml`:
```yaml
models_root: "d:/Projects/webcodecstest/minimal_server/models"  # Must exist
```

### Issue: "Background frames not loading"

**Solution:** Verify directory structure:
```
models/
└── sanders/
    ├── frames/
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   └── ...
    ├── checkpoint/
    │   └── model_best.onnx
    └── crop_rects.json
```

### Issue: "Tests failing"

**Solution:** Run with verbose output:
```powershell
go test -v ./functional-tests/...
```

Check [GOTCHAS.md](GOTCHAS.md) for common pitfalls.

---

## Next Steps

Once the server is running:

1. **Read [ARCHITECTURE.md](../ARCHITECTURE.md)** - Understand system design
2. **Review [API_REFERENCE.md](../API_REFERENCE.md)** - Learn all endpoints
3. **Check [TESTING.md](TESTING.md)** - Understand test coverage
4. **Read [GOTCHAS.md](GOTCHAS.md)** - Avoid common mistakes

---

## Contributing

### Before Submitting PR

- [ ] All tests pass (`go test ./...`)
- [ ] Race detector clean (`go test -race ./...`)
- [ ] Code formatted (`go fmt ./...`)
- [ ] Documentation updated
- [ ] ADR written (if architectural decision)

### Documentation Updates

If your changes affect:
- **Architecture** → Update `ARCHITECTURE.md`
- **API** → Update `API_REFERENCE.md`
- **Tests** → Update `TESTING.md`
- **Configuration** → Update `config.yaml` and docs

See [Documentation Maintenance Guidelines](../../README.md#documentation-maintenance)

---

**Questions? Issues?**  
- Check [GOTCHAS.md](GOTCHAS.md) for common problems
- Review [ADRs](../adr/) for decision context
- See [Session Notes](../session-notes/) for recent changes

---

**Last Updated:** November 6, 2025  
**Version:** 1.0.0
