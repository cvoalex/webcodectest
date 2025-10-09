# 🚀 Client Guide: Ultra-Optimized Lip Sync

## Quick Start

### 1. Start the Optimized Server

```bash
cd minimal_server
python optimized_server.py
```

**Expected output:**
```
🚀 STARTING ULTRA-OPTIMIZED BINARY WEBSOCKET SERVER
📦 Loading optimized sanders model...
✅ Sanders model loaded successfully!
⚡ OPTIMIZED WebSocket Server running on ws://localhost:8085
```

### 2. Open the Client

Open in your browser:
```
file:///D:/Projects/webcodecstest/minimal_server/optimized-lipsync-client.html
```

Or simply double-click: `optimized-lipsync-client.html`

### 3. Connect & Play

1. Click **"🔌 Connect to Server"**
2. Wait for connection confirmation
3. Click **"▶️ Auto Play"** to start playback
4. Watch real-time lip sync at 50+ FPS! 🚀

---

## Client Features

### 🎛️ Connection & Controls

**Buttons:**
- **Connect to Server** - Establish WebSocket connection to port 8085
- **Disconnect** - Close connection
- **Request Frame** - Request a single frame
- **Auto Play** - Automatically play through all frames
- **Stop** - Stop auto-play

**Frame Slider:**
- Manually select frame ID (0-522)
- Disabled during auto-play
- Instant preview on drag

### 📊 Performance Stats

Real-time monitoring:
- **Requests** - Total frames requested
- **Avg Time** - Average server response time
- **FPS** - Current frames per second
- **Last Frame** - Most recent frame time
- **Min/Max Time** - Performance range

**Performance Chart:**
- Live visualization of last 50 frames
- Color-coded by performance:
  - 🟢 Green: <25ms (excellent)
  - 🟡 Yellow: 25-40ms (good)
  - 🔴 Red: >40ms (slow)

### 🎬 Frame Display

- **Resolution:** 1280×720
- **Format:** JPEG
- **Updates:** Real-time as frames arrive
- **Info Bar:** Shows current frame and format

### 📝 Console Log

- Timestamped events
- Color-coded messages:
  - 🔵 Info (blue)
  - 🟢 Success (green)
  - 🟡 Warning (yellow)
  - 🔴 Error (red)
- **Clear Log** button to reset

---

## Optimizations Display

The client shows active optimizations:
- ✓ **Pre-loaded Videos** - All frames in RAM
- ✓ **Memory-mapped Audio** - Zero-copy access
- ✓ **Cached Metadata** - No file reads
- ✓ **Zero I/O** - Instant frame access

---

## Performance Expectations

### Expected Performance

| Metric | Value |
|--------|-------|
| **Server Response** | 18-28ms |
| **Total Round-trip** | 20-35ms |
| **Sustainable FPS** | 40-50 FPS |
| **Peak FPS** | 55+ FPS |

### Auto-Play Settings

**Default:** 20 FPS (50ms interval)
- Conservative for smooth playback
- Well below server capability

**Can be increased to:**
- 30 FPS (33ms) - Very smooth
- 40 FPS (25ms) - Maximum smooth
- 50 FPS (20ms) - At server limit

**To modify auto-play speed:**
```javascript
// In optimized-lipsync-client.html, line ~650
autoPlayInterval = setInterval(() => {
    requestFrame();
    // ...
}, 50); // Change 50ms to your desired interval
```

---

## Binary Protocol Details

### Request Format

```
[4 bytes] Model name length
[N bytes] Model name (UTF-8)
[4 bytes] Frame ID
[4 bytes] Audio length (always 0 for pre-processed)
```

### Response Format

```
[1 byte]  Success (1 = success, 0 = error)
[4 bytes] Frame ID
[4 bytes] Processing time (ms)
[4 bytes] Image data length
[N bytes] JPEG image data
[4 bytes] Bounds data length
[M bytes] Bounds (Float32Array)
```

---

## Troubleshooting

### Connection Issues

**Problem:** Cannot connect to server

**Solutions:**
1. Verify server is running: `python optimized_server.py`
2. Check server logs for errors
3. Ensure port 8085 is not blocked
4. Try refreshing the page

---

### Performance Issues

**Problem:** FPS lower than expected (<30 FPS)

**Possible causes:**
1. **CPU bottleneck** - Close other applications
2. **Network latency** - Use localhost (should be <1ms)
3. **Browser rendering** - Try different browser (Chrome recommended)
4. **Auto-play too fast** - Increase interval time

**Check:**
```javascript
// In browser console
console.log('Last 10 times:', responseTimes.slice(-10));
```

---

### Display Issues

**Problem:** Frames not displaying

**Solutions:**
1. Check browser console for errors (F12)
2. Verify canvas element exists
3. Check image blob creation
4. Verify server response has image data

---

## Advanced Usage

### Benchmark Mode

Test maximum server performance:

```javascript
// In browser console
async function benchmark() {
    const times = [];
    for (let i = 0; i < 100; i++) {
        const start = performance.now();
        requestFrame();
        await new Promise(resolve => setTimeout(resolve, 20));
        times.push(performance.now() - start);
    }
    console.log('Avg:', times.reduce((a,b)=>a+b)/times.length);
    console.log('Min:', Math.min(...times));
    console.log('Max:', Math.max(...times));
}
benchmark();
```

### Custom Frame Sequence

Play specific frame range:

```javascript
// In browser console
function playRange(start, end, fps = 24) {
    let frame = start;
    const interval = setInterval(() => {
        currentFrameId = frame;
        requestFrame();
        frame++;
        if (frame > end) {
            clearInterval(interval);
        }
    }, 1000 / fps);
}

// Example: Play frames 100-200 at 30 FPS
playRange(100, 200, 30);
```

### Export Performance Data

Save performance data to CSV:

```javascript
// In browser console
function exportStats() {
    const csv = responseTimes.map((t, i) => `${i},${t}`).join('\n');
    const blob = new Blob(['Frame,Time(ms)\n' + csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'performance_stats.csv';
    a.click();
}
exportStats();
```

---

## Comparison with Original Client

| Feature | Original (`client.html`) | Optimized (`optimized-lipsync-client.html`) |
|---------|-------------------------|---------------------------------------------|
| **Server Port** | 8084 | 8085 |
| **Target Server** | Original | Ultra-Optimized |
| **Expected FPS** | 12-16 | 40-50+ |
| **Performance Chart** | Basic | Advanced with color coding |
| **Auto-Play** | Yes | Yes (faster) |
| **Real-time Stats** | Basic | Comprehensive |
| **Visual Design** | Standard | Modern gradient |
| **Optimization Badges** | No | Yes |

---

## Tips for Best Performance

### 1. Browser Selection
- ✅ **Chrome/Edge** - Best WebSocket performance
- ✅ Firefox - Good performance
- ⚠️ Safari - May have limitations

### 2. System Resources
- Close unnecessary applications
- Ensure GPU acceleration enabled
- Use wired connection (if remote)

### 3. Server Configuration
- Ensure server has adequate RAM (8+ GB)
- Verify CUDA is available for GPU inference
- Monitor server logs for bottlenecks

### 4. Network
- Use localhost for best performance
- Wired connection if remote
- Low latency network (<10ms)

---

## Keyboard Shortcuts (Future Enhancement)

Planned shortcuts:
- **Space** - Play/Pause
- **Left/Right Arrow** - Previous/Next frame
- **Home** - Go to first frame
- **End** - Go to last frame
- **R** - Request current frame

---

## FAQ

**Q: Why is auto-play at 20 FPS when server can do 50+?**  
A: Conservative default for smooth playback. You can increase it!

**Q: Can I use this with the original server (port 8084)?**  
A: No, this client is optimized for port 8085. Use `client.html` for port 8084.

**Q: What's the maximum FPS I can achieve?**  
A: With proper hardware and local connection, 50-60 FPS is achievable.

**Q: Can I modify the client for my needs?**  
A: Absolutely! The code is well-commented and modular.

**Q: Does this support audio input like the original?**  
A: No, this client uses pre-processed models with pre-extracted audio features.

---

## Next Steps

1. **Try it out** - Connect and play!
2. **Monitor performance** - Watch the stats
3. **Experiment with speed** - Adjust auto-play FPS
4. **Test different frames** - Use the slider
5. **Benchmark** - Test maximum throughput

---

## Support

- **Documentation:** [OPTIMIZED_README.md](OPTIMIZED_README.md)
- **Server Comparison:** [PERFORMANCE_COMPARISON.md](PERFORMANCE_COMPARISON.md)
- **Server Overview:** [SERVER_IMPLEMENTATIONS.md](SERVER_IMPLEMENTATIONS.md)

---

**Enjoy real-time lip sync at 50+ FPS!** 🚀
