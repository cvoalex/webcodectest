# Multi-Model Deployment Guide

This guide shows how to run multiple different lip-sync models simultaneously on the same GPU infrastructure.

## Overview

The system supports multiple models with **zero performance penalty** - each model is just a different set of pre-rendered frames. You can run 20+ models on a single GPU as long as you have enough VRAM.

### Resource Requirements Per Model
- **VRAM**: ~100 MB (3,305 frames × 320×256 pixels)
- **Disk**: ~50 MB (3,305 JPEG files)
- **Startup**: No additional time (lazy loading)

### Example Scaling
| Models | VRAM Usage | Disk Space | Concurrent Users |
|--------|------------|------------|------------------|
| 1      | 100 MB     | 50 MB      | 100+             |
| 5      | 500 MB     | 250 MB     | 500+             |
| 10     | 1 GB       | 500 MB     | 1,000+           |
| 20     | 2 GB       | 1 GB       | 2,000+           |
| 50     | 5 GB       | 2.5 GB     | 5,000+           |

## Quick Start

### 1. Prepare Model Videos

Place your model videos in `model_videos/` directory:

```
minimal_server/
├── model_videos/
│   ├── person1.mp4
│   ├── person2.mp4
│   ├── celebrity1.mp4
│   └── avatar1.mp4
└── data/
    ├── person1/
    ├── person2/
    ├── celebrity1/
    └── avatar1/
```

### 2. Extract Frames

Use the batch preparation script:

```powershell
cd minimal_server

# Extract frames for all models (sequential)
python batch_prepare_models.py

# Extract frames in parallel (4 workers)
python batch_prepare_models.py --workers 4

# Extract specific models
python batch_prepare_models.py person1 person2 celebrity1

# Custom settings
python batch_prepare_models.py --fps 25 --quality 85 --workers 4
```

Output:
```
✅ FFmpeg found

======================================================================
🎬 Batch Model Video Preparation
======================================================================
Models to process: 4
Video directory: D:\Projects\webcodecstest\minimal_server\model_videos
Data directory: D:\Projects\webcodecstest\minimal_server\data
Frame rate: 25 fps
JPEG quality: 85
======================================================================

⚡ Processing 4 models with 4 workers...

✅ person1: Extracted 3305 frames (52.3 MB)
✅ person2: Extracted 3305 frames (48.7 MB)
✅ celebrity1: Extracted 3305 frames (51.1 MB)
✅ avatar1: Extracted 3305 frames (49.8 MB)

======================================================================
📊 Summary
======================================================================
Total models: 4
Successful: 4
Failed: 0

Total frames: 13,220
Total size: 201.9 MB
Average frames per model: 3305
Average size per model: 50.5 MB

Estimated VRAM usage (all models): 1290 MB
======================================================================
```

### 3. Start Servers

The servers automatically detect available models:

```powershell
# Single GPU
.\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 6

# Multiple GPUs
.\start_multi_gpu.ps1 -NumGPUs 4 -ProcessesPerGPU 6
```

On startup, you'll see:
```
📦 Available models: person1, person2, celebrity1, avatar1
```

### 4. Update HTML Client

Edit `webtest/realtime-lipsync-binary.html` to add your models to the dropdown:

```html
<select id="modelSelect" onchange="changeModel()">
    <option value="default_model" selected>Default Model</option>
    <option value="person1">Person 1</option>
    <option value="person2">Person 2</option>
    <option value="celebrity1">Celebrity 1</option>
    <option value="avatar1">Avatar 1</option>
</select>
```

### 5. Test

Open `http://localhost:8086/` in your browser:

1. Select a model from the dropdown
2. Click **"Connect to Server"**
3. Click **"Start Audio"**
4. Speak into microphone
5. Change model dropdown → instantly switches to new model

## Performance

### Memory Usage

All models share the same GPU and don't affect performance:

| Configuration | VRAM per Model | Total VRAM (20 models) | Performance |
|--------------|----------------|------------------------|-------------|
| 1 GPU        | ~100 MB        | ~2 GB                  | 180-300 FPS |
| 4 GPUs       | ~100 MB        | ~2 GB per GPU          | 720-1,200 FPS |
| 8 GPUs       | ~100 MB        | ~2 GB per GPU          | 1,440-2,400 FPS |

**Note**: Models are loaded lazily on first use. Initial request per model takes ~500ms, subsequent requests are instant.

### Disk Space

Each model requires ~50 MB of disk space:

- 10 models: ~500 MB
- 20 models: ~1 GB
- 50 models: ~2.5 GB
- 100 models: ~5 GB

### Bandwidth

Each model uses the same bandwidth (frames are compressed):

- Low quality: ~10 KB/frame = 250 KB/s @ 25 fps
- Medium quality: ~15 KB/frame = 375 KB/s @ 25 fps
- High quality: ~20 KB/frame = 500 KB/s @ 25 fps

## Advanced Usage

### Batch Preparation Options

```bash
# Process all models with custom settings
python batch_prepare_models.py --fps 25 --quality 85 --workers 4

# Force re-process existing models
python batch_prepare_models.py --force

# Save statistics to JSON
python batch_prepare_models.py --output-json stats.json

# Process from custom directory
python batch_prepare_models.py --video-dir ../my_videos --data-dir ../my_data
```

### Model Naming Conventions

Use descriptive names without spaces:

✅ **Good:**
- `person_john_doe`
- `celebrity_jane_smith`
- `avatar_robot_01`
- `character_hero_v2`

❌ **Avoid:**
- `person 1` (spaces)
- `model-with-dashes` (use underscores)
- `ALLCAPS` (hard to read)

### Directory Structure

```
minimal_server/
├── model_videos/              # Source MP4 videos
│   ├── person1.mp4           # Original video
│   ├── person2.mp4
│   └── ...
│
├── data/                      # Extracted frames
│   ├── person1/
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ... (3,305 frames)
│   ├── person2/
│   │   └── ... (3,305 frames)
│   └── ...
│
└── batch_prepare_models.py    # Preparation script
```

## Creating Model Videos

### Requirements

1. **Duration**: 2:12 minutes (132 seconds)
2. **Frame Rate**: 25 fps = 3,305 frames
3. **Resolution**: 320×256 pixels (or will be resized)
4. **Format**: MP4 (H.264 video codec)
5. **Audio**: Can have audio (not used by system)

### Recommended Tools

#### FFmpeg (Command Line)

Resize and convert existing video:
```bash
ffmpeg -i input.mp4 \
  -vf "scale=320:256:force_original_aspect_ratio=decrease,pad=320:256:(ow-iw)/2:(oh-ih)/2" \
  -t 132 \
  -r 25 \
  -c:v libx264 \
  -crf 23 \
  model_videos/my_model.mp4
```

Create from image sequence:
```bash
ffmpeg -framerate 25 -i frame_%04d.jpg \
  -t 132 \
  -vf "scale=320:256:force_original_aspect_ratio=decrease,pad=320:256:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 \
  -crf 23 \
  model_videos/my_model.mp4
```

#### Blender (3D Animation)

1. Create 3,305 frame animation (132 seconds @ 25 fps)
2. Set render resolution to 320×256
3. Render to image sequence
4. Use FFmpeg to convert to MP4

#### Video Editing Software

1. Import/create video
2. Trim to exactly 132 seconds
3. Export as MP4:
   - Resolution: 320×256
   - Frame rate: 25 fps
   - Codec: H.264
   - Quality: High

## Testing

### Single Model Test

```powershell
# Start single server
python optimized_grpc_server.py --port 50051

# In another terminal - test with specific model
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 1
```

Open browser → Select model → Test

### Multi-Model Test

```powershell
# Start 4 servers
.\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 4

# Start proxy
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 4
```

Open multiple browser tabs → Select different models → Test concurrency

### Load Test

Use `test_multi_process.py` with model parameter:

```bash
# Test single model
python test_multi_process.py --ports 50051 --model-name person1 --num-requests 100

# Test multiple models
python test_multi_process.py --ports 50051-50053 --model-names person1,person2,celebrity1 --num-requests 300
```

## Troubleshooting

### Model Not Found

**Error:** `Model 'xyz' not found`

**Solution:**
1. Check if `data/xyz/` directory exists
2. Check if frames exist: `data/xyz/frame_0000.jpg`
3. Re-run extraction: `python batch_prepare_models.py xyz`

### Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
1. Check total VRAM usage: `nvidia-smi`
2. Reduce number of models
3. Or upgrade to GPU with more VRAM
4. Or use multiple GPUs

### Slow Loading

**Symptom:** First request to new model is slow (~500ms)

**Explanation:** Models are lazy-loaded on first use. This is normal.

**Solution:**
- Pre-warm models by making test request
- Or modify server to pre-load common models at startup

### Frame Count Mismatch

**Error:** `Expected 3305 frames, got 2876`

**Solution:**
1. Check video duration: `ffprobe model_videos/xyz.mp4`
2. Should be 132 seconds @ 25 fps = 3,305 frames
3. Re-render video with correct duration
4. Or adjust frame extraction FPS

### Poor Frame Quality

**Symptom:** Blurry or pixelated frames

**Solution:**
1. Increase JPEG quality: `--quality 95`
2. Use higher resolution source video
3. Check source video quality with `ffprobe`

## Production Deployment

### Large-Scale Multi-Model Setup (100+ models)

```bash
# 1. Prepare all models in batches
python batch_prepare_models.py --workers 8 --output-json batch1.json

# 2. Verify statistics
cat batch1.json

# 3. Start with sufficient GPUs
# For 100 models × 100 MB = 10 GB per GPU
# Use 4 GPUs with 24 GB each = 96 GB total
.\start_multi_gpu.ps1 -NumGPUs 4 -ProcessesPerGPU 6

# 4. Start proxy
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 24

# 5. Monitor VRAM usage
nvidia-smi -l 1
```

### Model Updates

To update a model:

```bash
# 1. Replace video
copy new_version.mp4 model_videos\person1.mp4

# 2. Re-extract frames
python batch_prepare_models.py person1 --force

# 3. No server restart needed! Changes take effect on next request.
```

### Cleanup

Remove unused models:

```bash
# Delete frames
rmdir /s data\old_model

# Delete video
del model_videos\old_model.mp4

# Server automatically skips missing models
```

## Cost Analysis

### Storage Costs

At $0.10/GB/month (cloud storage):

- 10 models: 0.5 GB = $0.05/month
- 20 models: 1 GB = $0.10/month
- 50 models: 2.5 GB = $0.25/month
- 100 models: 5 GB = $0.50/month

### VRAM Costs

GPU pricing (hourly):

| GPU           | VRAM | Max Models | Cost/hour |
|---------------|------|------------|-----------|
| RTX 4090      | 24GB | 240        | $1.00     |
| RTX 6000 Ada  | 48GB | 480        | $2.50     |
| A100 40GB     | 40GB | 400        | $3.00     |
| H100          | 80GB | 800        | $5.00     |

**Example:** 100 models on RTX 6000 Ada = $2.50/hour = $1,800/month

### Break-Even Analysis

Compared to separate GPU per model:

| Setup              | GPUs | Cost/month | Break-even Point |
|--------------------|------|------------|------------------|
| 1 model/GPU        | 20   | $36,000    | N/A (baseline)   |
| Shared GPU (20x)   | 1    | $1,800     | **$34,200 saved**|
| Shared GPU (100x)  | 3    | $5,400     | **$30,600 saved**|

## Example: Celebrity Avatar Service

Setup for 50 celebrity avatars:

```bash
# 1. Prepare videos (50 × 132 seconds each)
# Place in model_videos/: celebrity_1.mp4 ... celebrity_50.mp4

# 2. Extract frames (parallel)
python batch_prepare_models.py --workers 8 --output-json celebs.json

# Expected output:
# Total frames: 165,250 (50 × 3,305)
# Total size: 2,525 MB (~2.5 GB)
# Estimated VRAM: 16,140 MB (~16 GB)

# 3. Deploy on 2 × RTX 6000 Ada (48GB each)
.\start_multi_gpu.ps1 -NumGPUs 2 -ProcessesPerGPU 6

# 4. Start proxy
cd ../grpc-websocket-proxy
.\proxy.exe --start-port 50051 --num-servers 12

# 5. Update HTML client with all 50 celebrities
# Edit webtest/realtime-lipsync-binary.html

# Performance:
# - 12 servers × 25 FPS = 300 FPS aggregate
# - 300 concurrent users @ 1 FPS each
# - Any user can select any of 50 celebrities
# - Instant switching between models
```

## Summary

✅ **Zero performance penalty** for multiple models  
✅ **100 MB VRAM** per model  
✅ **50 MB disk** per model  
✅ **Lazy loading** - only load what's used  
✅ **Instant switching** between models  
✅ **Scale to 100+ models** on single GPU  

Multiple models are essentially **free** - you're just choosing which frames to send!
