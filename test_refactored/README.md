# Refactoring Test Suite

This directory contains performance comparison tests for the **original** vs **refactored** monolithic server.

## 📁 Files

- **`test_original.go`** - Tests the original monolithic server (localhost:50053)
- **`test_refactored.go`** - Tests the refactored server (localhost:50054)
- **`run_comparison.ps1`** - PowerShell script to run both tests and compare results
- **`output_original/`** - Output directory for original server test results
- **`output_refactored/`** - Output directory for refactored server test results

## 🎯 Purpose

Validate that the refactoring from a single `main.go` (894 lines) to a modular structure (7 files) maintains:

- ✅ **Identical performance** (FPS, latency)
- ✅ **Identical output quality** (JPEG frames)
- ✅ **Identical memory usage** (~0.94MB per request)
- ✅ **No algorithmic changes** (exact code extraction)

## 🚀 Quick Start

### Prerequisites

1. **Go 1.24+** installed
2. **Python 3.x** with NumPy installed
3. **Test data files** in `../go-monolithic-server/testing/`:
   - `aud.wav` (real audio file)
   - `visual_frames_6.npy` (real visual frames)

### Running Tests

#### Option 1: Automated Comparison (Recommended)

```powershell
cd d:\Projects\webcodecstest\test_refactored
.\run_comparison.ps1
```

Select option 3 to run both tests and see a detailed comparison table.

#### Option 2: Test Original Server Only

1. Start the original server:
   ```powershell
   cd ..\go-monolithic-server
   go run cmd/server/main.go
   ```

2. In a new terminal:
   ```powershell
   cd ..\test_refactored
   go run test_original.go
   ```

#### Option 3: Test Refactored Server Only

1. Start the refactored server (on different port):
   ```powershell
   cd ..\go-monolithic-server-refactored
   go run cmd/server/main.go
   ```
   
   **NOTE:** Edit `config.yaml` to use port 50054:
   ```yaml
   server:
     port: 50054  # Changed from 50053
   ```

2. In a new terminal:
   ```powershell
   cd ..\test_refactored
   go run test_refactored.go
   ```

## 📊 Test Configuration

Both tests use identical parameters:

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 frames |
| Warmup Runs | 3 iterations |
| Timed Runs | 3 iterations (averaged) |
| Model | sanders |
| Audio | Real audio from `aud.wav` |
| Visual Frames | Real frames from `visual_frames_6.npy` |
| Server (Original) | localhost:50053 |
| Server (Refactored) | localhost:50054 |

## ✅ Success Criteria

The refactored server must achieve:

- **≥42 FPS** (batch size 8) - Known baseline from original
- **Audio Processing:** ~23ms (similar to original)
- **Inference:** ~165ms (similar to original)
- **Compositing:** ~4ms (similar to original)
- **Total Time:** ~192ms (similar to original)
- **Output Quality:** Visually identical JPEG frames

## 📈 Expected Results

Based on the original server's performance (batch 8):

```
⚡ Average FPS: 42.00
🎵 Average Audio Processing: 23.00 ms
🧠 Average Inference: 165.00 ms
🎨 Average Compositing: 4.00 ms
⏱️  Average Total: 192.00 ms
```

The refactored server should match these numbers within ±1% tolerance.

## 🔍 What Gets Tested

### 1. Audio Processing Pipeline
- Raw audio → STFT → Mel-spectrogram → Audio Encoder
- 25fps windowing with overlap
- Zero-padding for frame alignment

### 2. Visual Processing
- Load real visual frames (6 channels: crop + ROI)
- Flatten into single byte array

### 3. Inference
- Model loading from registry
- Batch inference (8 frames)
- Audio feature encoding (pooled encoders)

### 4. Compositing
- Parallel frame compositing (goroutines)
- Bilinear interpolation (pooled buffers)
- JPEG encoding (pooled images)

### 5. Memory Management
- Buffer pools (5 types)
- Image pools (3 sizes)
- Encoder pool (4 instances)

## 🐛 Troubleshooting

### Error: "cannot find package go-monolithic-server/proto"

The test files need to import the proto package. Make sure your `go.mod` is set up correctly:

```bash
cd test_refactored
go mod init test_refactored
go mod edit -replace go-monolithic-server=../go-monolithic-server
go mod tidy
```

### Error: "Failed to connect: connection refused"

Make sure the server is running on the correct port before running tests:
- Original: `localhost:50053`
- Refactored: `localhost:50054`

### Error: "Failed to load audio/visual frames"

Ensure test data exists in `../go-monolithic-server/testing/`:
```bash
ls ../go-monolithic-server/testing/aud.wav
ls ../go-monolithic-server/testing/visual_frames_6.npy
```

### Performance Regression

If the refactored server is slower:

1. **Check for missing optimizations**:
   - Verify all 5 memory pools are created
   - Verify encoder pool has 4 instances
   - Verify STFT/mel parallelization (8 workers each)

2. **Check constants**:
   - `visualFrameSize = 6 * 320 * 320` (not 3*320*320)
   - `audioFrameSize = 32 * 16 * 16`
   - `outputFrameSize = 3 * 320 * 320`

3. **Check algorithm extraction**:
   - Compare `InferBatchComposite()` line-by-line
   - Verify no shortcuts were taken

4. **Rollback if needed**:
   - Backup available: `go-monolithic-server-backup-oct28-2025/`

## 📝 Output Files

After running tests, you'll find:

```
test_refactored/
├── output_original/
│   └── frame_0_original.jpg      # Sample output from original server
├── output_refactored/
│   └── frame_0_refactored.jpg    # Sample output from refactored server
```

Compare these frames visually to ensure identical quality.

## 🎓 What This Proves

A successful test run (≥42 FPS, identical output) proves:

1. ✅ **Code extraction was exact** - No algorithm changes
2. ✅ **Refactoring preserved performance** - No speed regression
3. ✅ **Memory pools work correctly** - No memory leaks
4. ✅ **All optimizations intact** - Parallelization still effective
5. ✅ **Output quality maintained** - Identical JPEG generation

This validates that the refactoring achieved its goal: **better code organization without sacrificing performance**.

## 🔗 Related Documentation

- `../go-monolithic-server/` - Original monolithic server
- `../go-monolithic-server-refactored/` - Refactored modular server
- `../go-monolithic-server-backup-oct28-2025/` - Safety backup

## 📊 Comparison Table Format

The `run_comparison.ps1` script generates a table like this:

```
┌─────────────────────────┬──────────────┬──────────────┬──────────────┐
│ Metric                  │ Original     │ Refactored   │ Difference   │
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ Throughput (FPS)        │        42.00 │        42.00 │      +0.00 │
│ Audio Processing (ms)   │        23.00 │        23.00 │      +0.00 │
│ Inference (ms)          │       165.00 │       165.00 │      +0.00 │
│ Compositing (ms)        │         4.00 │         4.00 │      +0.00 │
│ Total Time (ms)         │       192.00 │       192.00 │      +0.00 │
└─────────────────────────┴──────────────┴──────────────┴──────────────┘
```

Green values = good (faster/equal), Red values = regression (slower).

---

**Last Updated:** January 28, 2025
**Author:** GitHub Copilot
**Purpose:** Validate monolithic server refactoring
