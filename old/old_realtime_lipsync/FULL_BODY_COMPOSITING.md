# Full-Body Compositing Implementation

## 🎯 Overview

The real-time lip sync system now supports **full-body compositing** where generated mouth regions are overlaid onto complete reference videos, providing a much more natural and immersive experience.

## 🏗️ Architecture

```
Browser Client:
┌─────────────────────────────────────────────────────────────┐
│ ModelVideoManager    │ FaceCompositor   │ RealtimeLipSyncApp │
│ - Downloads videos   │ - Composites     │ - Coordinates      │
│ - Extracts frames    │   mouth regions  │   components       │
│ - Caches locally     │ - Renders final  │ - Handles UI       │
│                      │   composite      │                    │
└─────────────────────────────────────────────────────────────┘
          ↓                      ↓                      ↓
     Video Download         Frame Composite         Audio Processing
          ↓                      ↓                      ↓
┌─────────────────────────────────────────────────────────────┐
│                    Go Token Server                          │
│ - Serves model videos via /api/model-video/{model_name}     │
│ - Static file serving                                       │
│ - OpenAI token generation                                   │
└─────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────┐
│                 Python Frame Generator                      │
│ - gRPC integration                                          │
│ - Audio processing                                          │
│ - Returns mouth regions + bounds data                       │
└─────────────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────────────┐
│                    gRPC Lip Sync Service                    │
│ - Generates 320x320 mouth regions                          │
│ - Returns bounds for compositing                           │
│ - High-performance inference                               │
└─────────────────────────────────────────────────────────────┘
```

## 🎬 Model Video Setup

### 1. Video Requirements
- **Format**: MP4 (H.264 recommended)
- **Resolution**: Any (system auto-resizes)
- **Duration**: 10-30 seconds recommended
- **Content**: Full-body shots with clear face visibility
- **Expression**: Neutral or talking expressions

### 2. Video Placement
Place model videos in one of these locations:
```
realtime_lipsync/model_videos/{model_name}.mp4
model_videos/{model_name}.mp4
datasets_test/{model_name}.mp4
```

### 3. Video Naming
Videos should match model names exactly:
```
test_optimized_package_fixed_1.mp4
test_optimized_package_fixed_2.mp4
test_optimized_package_fixed_3.mp4
...
```

## 🔧 How It Works

### 1. **Model Preparation**
When a model is selected:
1. `ModelVideoManager` downloads the video via `/api/model-video/{model_name}`
2. Video is extracted frame-by-frame to `ImageData` objects
3. Frames are cached in memory for instant access

### 2. **Frame Generation**
During conversation:
1. OpenAI Realtime API provides audio
2. Python frame generator processes audio via gRPC
3. gRPC service returns:
   - `prediction_data`: 320x320 mouth region (JPEG)
   - `bounds`: [xmin, ymin, xmax, ymax] face coordinates

### 3. **Compositing**
For each frame:
1. `FaceCompositor` gets appropriate full-body frame
2. Mouth region is overlaid at `bounds` position
3. Composite result is displayed on canvas

## 🎮 Client Components

### ModelVideoManager
```javascript
// Prepare a model
await modelVideoManager.prepareModel('model_name', (progress, status) => {
    console.log(`${progress}% - ${status}`);
});

// Get a frame
const frame = modelVideoManager.getModelFrame('model_name', frameIndex);

// Check if ready
const ready = modelVideoManager.isModelReady('model_name');
```

### FaceCompositor
```javascript
// Composite mouth onto full-body frame
const composite = await faceCompositor.compositeFrame(
    fullBodyFrame,    // ImageData from model video
    mouthRegionData,  // Base64 mouth region from gRPC
    bounds           // [xmin, ymin, xmax, ymax]
);

// Display result
image.src = composite; // Base64 data URL
```

## 🚀 Performance Features

### Memory Management
- **Frame Caching**: Extracted frames stored as `ImageData`
- **Model Limits**: Maximum 5 cached models (configurable)
- **Auto Cleanup**: LRU eviction when cache is full

### Progressive Loading
- **Background Downloads**: Videos download while app is running
- **Progress Tracking**: Real-time download/extraction progress
- **Fallback Mode**: Shows mouth-only if model not ready

### Optimizations
- **Canvas Reuse**: Single canvas instances for compositing
- **Blend Modes**: Optional blend modes for better integration
- **Frame Selection**: Smart frame indexing (future: based on timing)

## 🎯 Benefits

### Visual Quality
- **Full Context**: Complete body/background instead of mouth-only
- **Natural Integration**: Mouth regions properly positioned
- **Smooth Playback**: Pre-cached frames for instant access

### User Experience
- **Immersive**: Complete avatar instead of floating mouth
- **Flexible**: Support for any video content
- **Responsive**: Automatic model switching and preparation

### Performance
- **Client-Side**: No server compositing overhead
- **Cached**: Frames extracted once, reused indefinitely
- **Scalable**: Handles multiple models efficiently

## 🔧 Configuration

### Buffer Settings
```javascript
// In ModelVideoManager
this.maxCachedModels = 5;  // Maximum cached models

// Frame extraction settings
const fps = 25;            // Assumed video FPS
```

### Video Paths
```go
// In Go server - add more locations as needed
possiblePaths := []string{
    filepath.Join("..", "realtime_lipsync", "model_videos", modelName+".mp4"),
    filepath.Join("..", "model_videos", modelName+".mp4"),
    // ... more paths
}
```

## 🐛 Troubleshooting

### Common Issues

**1. Model Video Not Found**
```
❌ Model video not found for: model_name
```
- Check video file exists in expected location
- Verify file naming matches model name exactly
- Check Go server logs for searched paths

**2. Video Loading Failed**
```
❌ Error downloading video for model_name
```
- Verify Go server is running on port 3000
- Check browser network tab for HTTP errors
- Ensure video file is valid MP4

**3. Compositing Errors**
```
❌ Error displaying composite frame
```
- Check that bounds data is valid: [xmin, ymin, xmax, ymax]
- Verify mouth region data is valid base64 JPEG
- Fallback to mouth-only mode automatically triggered

### Debug Information
```javascript
// Check model status
console.log(app.modelVideoManager.getMemoryStats());

// Test compositor
app.faceCompositor.testComposite(frame, mouth, bounds);

// Check preparation progress
const progress = app.modelPreparationProgress.get('model_name');
```

## 🚀 Future Enhancements

### Smart Frame Selection
- **Audio Timing**: Select frames based on audio timing/duration
- **Expression Matching**: Choose frames that match speech patterns
- **Looping Logic**: Intelligent video looping for long conversations

### Advanced Compositing
- **Blend Modes**: Better integration with lighting/color matching
- **Edge Feathering**: Softer mouth region boundaries
- **Dynamic Positioning**: Real-time face tracking adjustments

### Performance Optimizations
- **WebGL Compositing**: GPU-accelerated frame compositing
- **Compressed Storage**: More efficient frame storage formats
- **Streaming Extraction**: Extract frames on-demand vs. bulk processing

This implementation provides a solid foundation for full-body lip sync that can be enhanced with additional features as needed!
