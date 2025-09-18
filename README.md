# WebCodecs Audio/Video Player Test

This project demonstrates advanced video processing capabilities using modern web APIs:

1. **Download MP4 videos** to the client browser
2. **Extract video frames** using WebCodecs API with Canvas fallback
3. **Extract synchronized audio chunks** using Web Audio API
4. **Cache both audio and video data** locally using IndexedDB
5. **Play synchronized audio/video** with frame-perfect timing

## Features

### Video Processing
- Download videos from URLs or load local files
- Extract frames at configurable intervals (0.01s to any duration)
- Configurable frame quality (0.1 to 1.0) and maximum frame count
- Canvas-based frame capture with WebCodecs integration
- Real-time progress tracking during extraction

### Audio Processing
- Extract 40ms audio chunks synchronized with video frames
- Web Audio API decoding and processing
- Base64 encoding for efficient caching
- Synchronized audio playback with video frames

### Caching System
- Dual IndexedDB stores for frames and audio chunks
- SHA-256 hash-based cache keys for efficient lookups
- Automatic cache hit detection and loading
- Cache size monitoring and management tools

### Playback Engine
- Synchronized audio/video playback at various speeds (12-60 fps)
- Frame-perfect timing using performance.now()
- Loop playback option
- Individual frame navigation and preview
- Real-time playback controls (play/pause/stop)

### User Interface
- Clean, responsive design with progress indicators
- Frame grid display with click-to-preview
- Dedicated playback canvas with zoom controls
- Cache information display with size metrics
- Error handling with user-friendly status messages

## Browser Support

This demo requires a modern browser with WebCodecs and Web Audio API support:
- **Chrome 94+** (Full support)
- **Edge 94+** (Full support)  
- **Opera 80+** (Full support)
- **Firefox** (Limited - WebCodecs behind flag)
- **Safari** (Limited - WebCodecs not supported)

**Note**: The application includes Canvas-based fallbacks for browsers without WebCodecs support.

## Usage

### Basic Operation
1. Open `index.html` in a supported browser
2. Enter a video URL or select a local MP4 file
3. Configure extraction settings:
   - **Frame interval**: 0.01-1.0 seconds (default: 0.04s = 25fps)
   - **Max frames**: Limit total frames extracted (default: 250)
   - **Frame quality**: 0.1-1.0 JPEG quality (default: 0.8)
4. Click "Download & Process Video" or "Process Local File"
5. Wait for extraction to complete (progress bar shows status)

### Playback Controls
- **Play**: Start synchronized audio/video playback
- **Pause**: Temporarily stop playback (can resume)
- **Stop**: Stop and reset to beginning
- **Speed**: Adjust playback rate (12-60 fps)
- **Loop**: Enable/disable continuous playback
- **Frame Navigation**: Click any frame in grid to preview

### Cache Management
- **Show Cache Info**: View storage usage and cached content
- **Clear Cache**: Remove all cached frames and audio data
- Cache persists between browser sessions automatically

## Technical Details

### WebCodecs Integration
- Primary method: Browser's native VideoDecoder for optimal performance
- Fallback method: Canvas-based frame capture for compatibility
- Support for MP4, WebM, and other modern video formats
- Efficient frame-by-frame processing with minimal memory overhead

### Audio Processing Engine
- **Web Audio API**: Decodes entire audio track into memory
- **40ms chunks**: Synchronized with 25fps video timing
- **Base64 encoding**: Efficient storage in IndexedDB
- **Scheduled playback**: Precise timing using AudioContext

### Advanced Caching Strategy
- **Dual storage**: Separate IndexedDB stores for video frames and audio chunks
- **SHA-256 hashing**: Video content-based cache keys prevent duplicates
- **Automatic versioning**: Database schema upgrades handled gracefully
- **Size monitoring**: Real-time cache size calculation and display
- **Graceful degradation**: Functions without cache if IndexedDB unavailable

### Synchronization System
- **Performance.now()**: High-precision timing for frame accuracy
- **RequestAnimationFrame**: Smooth video playback loop
- **Audio scheduling**: Web Audio API scheduling for sample-accurate timing
- **Frame alignment**: 40ms audio chunks perfectly match 25fps video frames

### Memory Management
- **Progressive loading**: Non-blocking frame extraction with delays
- **Resource cleanup**: Automatic URL revocation and buffer management
- **Error recovery**: Robust error handling prevents memory leaks
- **Optimized encoding**: JPEG compression reduces storage requirements

## Sample URLs for Testing

### Short Test Videos
- **Sample 1MB**: `https://sample-videos.com/zip/10/mp4/SampleVideo_640x360_1mb.mp4`
- **Sample 5MB**: `https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_5mb.mp4`

### Standard Test Content
- **Big Buck Bunny (10s)**: `https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4`
- **Elephant Dream**: `https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4`

### Audio-Rich Content
- **Sintel (with complex audio)**: `https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4`
- **We Are Going On Bullrun**: `https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4`

**Tip**: Start with smaller files for faster testing, then try larger ones to test caching performance.

## Files

- **`index.html`** - Complete synchronized audio/video player application
- **`README.md`** - This comprehensive documentation

## Performance Notes

### Optimization Tips
- **Frame intervals**: Use 0.04s (25fps) for smooth playback, higher for faster processing
- **Quality settings**: 0.8 provides good balance of quality vs. storage size
- **Frame limits**: Set max frames based on available memory (250 frames ≈ 10 seconds @ 25fps)

### Expected Performance
- **Extraction speed**: ~10-50 frames/second depending on video resolution
- **Cache loading**: Nearly instant for previously processed videos
- **Memory usage**: ~2-5MB per 100 frames (depends on resolution and quality)
- **Playback**: Smooth at up to 60fps on modern hardware

## Advanced Features

### Developer Tools
- **Console logging**: Detailed extraction and playback information
- **Cache inspection**: View exact storage usage and content details
- **Error reporting**: Comprehensive error messages for debugging

### Extensibility
- **Modular design**: Easy to extend with additional video formats
- **Plugin architecture**: Audio/video processing can be customized
- **API integration**: Ready for integration with external video services

## Notes

- **Large videos**: May take time to download and process (progress bars show status)
- **Memory considerations**: Very long videos may require chunked processing
- **Cache persistence**: Data persists between browser sessions until manually cleared
- **WebCodecs status**: Still experimental but rapidly stabilizing across browsers
- **Audio sync**: Perfectly synchronized playback requires stable frame timing

## Troubleshooting

### Common Issues
- **No frames extracted**: Check video format compatibility (MP4 recommended)
- **Slow performance**: Reduce frame quality or increase interval
- **Audio not playing**: Ensure browser allows audio playback (user gesture required)
- **Cache errors**: Clear cache and try again, or use browser without IndexedDB restrictions
