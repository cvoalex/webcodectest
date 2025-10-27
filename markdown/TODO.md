# Performance Optimization TODO

## High Priority - Binary Protocol Optimizations

### 🚀 Remove Base64 Conversion in Inference Engine
**Location**: `fast_service/direct_websocket_server_binary.py` line ~110
**Issue**: Server converts raw binary audio → base64 → back to binary in inference engine
**Current Code**:
```python
audio_override = base64.b64encode(audio_data).decode('utf-8')
```
**Goal**: Pass raw binary audio directly to inference engine
**Impact**: Eliminate ~30-40% encoding/decoding overhead
**Status**: Working system first, optimize later

### Technical Notes:
- Binary client already sends raw Uint8Array audio data
- Server receives raw binary audio correctly 
- Inference engine currently expects base64 strings
- Need to modify inference engine to accept raw binary audio
- Could save significant CPU time on audio processing

### Files to modify:
1. `multi_model_engine.py` - Accept raw binary audio parameter
2. `direct_websocket_server_binary.py` - Pass raw audio directly
3. Test with binary protocol to ensure no regressions

---

## Other Performance Items

### 🎯 Audio Buffer Optimizations
- Consider using shared memory for audio buffers
- Reduce memory allocations in hot paths

### 📊 Protocol Benchmarking  
- Add timing metrics for base64 vs binary performance
- Compare binary vs JSON protocol end-to-end latency

---

*Created: September 10, 2025*
*Priority: High - after current system is stable*
