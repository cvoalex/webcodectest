# Go + ONNX Batch Parallel with Full Compositing

## 🎯 What This Does

This benchmark implements the **FULL compositing pipeline** in Go, matching Python's best performance approach:

1. **Parallel Processing**: Uses Go goroutines for parallel workers (like Python's ThreadPoolExecutor)
2. **Batch Inference**: Processes frames in batches for efficiency
3. **RAM Caching**: Loads all background frames into RAM once
4. **Full Compositing**: Scales 320x320 mouth region to full 1280x720 output frames

## 🏆 Performance Goal

**Beat Python's 41.11 FPS** with native Go performance!

## 📊 Current Benchmarks (While Gaming - CPU Busy)

- **Python + Parallel Composite**: 41.11 FPS ← TARGET TO BEAT
- **Go Parallel (no composite)**: 26.77 FPS
- **Go Single-Frame**: 27.3 FPS
- **Go Batch Sequential**: 22.7 FPS

## 🚀 How to Run

### Quick Test (after gaming):
```bash
.\run-test.bat
```

### Build First:
```bash
.\build.bat
.\benchmark-sanders-batch-parallel-composite.exe
```

### Custom Settings:
```bash
# Try different worker counts
.\benchmark-sanders-batch-parallel-composite.exe -workers 2
.\benchmark-sanders-batch-parallel-composite.exe -workers 4
.\benchmark-sanders-batch-parallel-composite.exe -workers 8

# Try different batch sizes
.\benchmark-sanders-batch-parallel-composite.exe -batch 1
.\benchmark-sanders-batch-parallel-composite.exe -batch 4
.\benchmark-sanders-batch-parallel-composite.exe -batch 8

# Combine both
.\benchmark-sanders-batch-parallel-composite.exe -workers 4 -batch 8
```

## 🔧 Architecture

### Worker Structure:
```
Main Thread
├── Load Visual/Audio Data (RAM)
├── Load Background Frames (RAM Cache)
├── Initialize ONNX Inferencer
└── Spawn N Workers (Goroutines)
    ├── Worker 1: Frames 0-24
    │   ├── Batch Inference (4 frames at a time)
    │   └── Composite each frame (320x320 → 1280x720)
    ├── Worker 2: Frames 25-49
    ├── Worker 3: Frames 50-74
    └── Worker 4: Frames 75-99
```

### Data Flow per Worker:
```
Visual Data + Audio Data
    ↓
Batch of 4 frames
    ↓
ONNX Inference → 4x (320x320 outputs)
    ↓
For each output:
    Convert to OpenCV Mat
    ↓
    Composite with background frame
    ↓
    Resize & blend → 1280x720
    ↓
Save to disk
```

## 💡 Key Advantages

1. **Go Goroutines**: More efficient than Python threads
2. **Native Performance**: No GIL (Global Interpreter Lock)
3. **Memory Efficiency**: Shared RAM cache across workers
4. **Easy Scaling**: Just increase `-workers` flag

## 📈 Expected Improvements (After Gaming)

With full CPU/GPU resources:
- **Go Parallel Composite**: Potentially **45-50+ FPS**
- Better thread scheduling
- No game-related GPU/CPU contention
- True parallel execution on free cores

## 🔍 Performance Tips

### Optimal Settings:
- **Workers**: Match your CPU cores (4-8 typical)
- **Batch Size**: 4-8 for best GPU utilization
- **RAM**: Ensure ~300MB free for background cache

### What to Monitor:
- **Speedup**: Should be close to worker count (3.97x for 4 workers)
- **Inference Time**: Lower is better (Python: 8.18ms/frame)
- **Composite Time**: Should be minimal (Python: 1.68ms/frame)
- **Total FPS**: Wall time performance (goal: >41 FPS)

## 📝 Comparison with Python

| Feature | Python | Go | Winner |
|---------|--------|----|----|
| Parallel Workers | ThreadPoolExecutor | Goroutines | Go (lighter) |
| Inference | ONNX Runtime | ONNX Runtime | Tie |
| Compositing | PIL/OpenCV | gocv/OpenCV | Tie |
| GIL | Yes (limited) | No | Go |
| Native Speed | Interpreted | Compiled | Go |
| Current FPS | 41.11 | TBD | ? |

## 🎮 Next Steps

1. **After Gaming**: Run this benchmark to get real performance numbers
2. **Compare FPS**: See if Go beats Python's 41.11 FPS
3. **Optimize**: Tune workers/batch size based on results
4. **Production**: If faster, use Go for production pipeline!

## 📦 Dependencies

- Go 1.21+
- ONNX Runtime (with CUDA)
- gocv (OpenCV bindings)
- CUDA-capable GPU

## 🐛 Troubleshooting

**"cannot find package gocv"**
- Install gocv: Follow https://gocv.io/getting-started/windows/

**"cannot find OpenCV"**
- Install OpenCV with CUDA support
- Set environment variables

**"model not found"**
- Check model path: `d:/Projects/webcodecstest/minimal_server/models/sanders/checkpoint/model_best.onnx`

**"background frames not found"**
- Check frames path: `d:/Projects/webcodecstest/realtime_lipsync/data/sanders/frames/`

## 🏁 Ready to Test!

When gaming is done, just run:
```bash
.\run-test.bat
```

And see if Go beats Python's 41.11 FPS! 🚀
