# Test Full System: gRPC Server → Go Proxy → Web Client

## Quick Test (3 steps)

### Terminal 1: Start gRPC Server
```powershell
cd D:\Projects\webcodecstest\minimal_server
.\start_grpc_single.bat
```

Wait for: `✅ All models loaded and ready!`

### Terminal 2: Start Go Proxy
```powershell
cd D:\Projects\webcodecstest\grpc-websocket-proxy
.\lipsync-proxy.exe --ws-port 8086 --num-servers 1 --start-port 50051
```

Wait for: `WebSocket server listening on :8086`

### Terminal 3: Open Web Client
```powershell
cd D:\Projects\webcodecstest\webtest
# Open in browser:
start chrome "file:///D:/Projects/webcodecstest/webtest/realtime-lipsync-binary.html"
```

Or manually open: `D:\Projects\webcodecstest\webtest\realtime-lipsync-binary.html`

### In Browser:
1. Click **"Connect to Server"**
2. Click **"🎤 Start Audio"**
3. Speak into microphone
4. See real-time lip-sync!

---

## Multi-Process Test (Higher Performance)

### Terminal 1: Start 4 gRPC Servers
```powershell
cd D:\Projects\webcodecstest\minimal_server
.\start_multi_gpu.ps1 -NumGPUs 1 -ProcessesPerGPU 4
```

Wait for all 4 servers to show: `✅ All models loaded and ready!`

### Terminal 2: Start Proxy (Load Balancing)
```powershell
cd D:\Projects\webcodecstest\grpc-websocket-proxy
.\lipsync-proxy.exe --ws-port 8086 --num-servers 4 --start-port 50051
```

You'll see:
```
📍 Backend configuration:
   [1] localhost:50051
   [2] localhost:50052
   [3] localhost:50053
   [4] localhost:50054

✅ Connected to 4/4 gRPC servers
⚖️  Load balancing: Round-robin across 4 backends
```

### Terminal 3: Open Web Client
Same as above - open `realtime-lipsync-binary.html` in browser

**Expected Performance:**
- Latency: 17-22ms
- Throughput: 100-150 FPS aggregate
- Load balanced across 4 servers

---

## Troubleshooting

### gRPC Server Won't Start

**Use the batch file (handles environment automatically):**
```powershell
cd D:\Projects\webcodecstest\minimal_server
.\start_grpc_single.bat
```

**Or manually:**
```powershell
cd D:\Projects\webcodecstest
.\.venv312\Scripts\activate.bat
cd minimal_server
python optimized_grpc_server.py --port 50051
```

**Check dependencies:**
```powershell
pip install -r requirements.txt
```

**Check model files:**
```powershell
ls data\default_model\frame_0000.jpg
# Should exist
```

### Go Proxy Won't Start

**Check if built:**
```powershell
ls grpc-websocket-proxy\lipsync-proxy.exe
# Should exist (18MB)
```

**Rebuild if needed:**
```powershell
cd grpc-websocket-proxy
go build -o lipsync-proxy.exe .
```

**Check ports:**
```powershell
# Make sure 8086 is free
netstat -an | findstr :8086
```

### Web Client Won't Connect

**Check WebSocket URL:**
- Should be: `ws://localhost:8086`
- In browser console, check for connection errors

**Check proxy is running:**
```powershell
curl http://localhost:8086/health
# Should return: OK
```

**Check firewall:**
```powershell
# Allow through Windows Firewall if needed
New-NetFirewallRule -DisplayName "LipSync Proxy" -Direction Inbound -LocalPort 8086 -Protocol TCP -Action Allow
```

### No Video Output

**Check model selection:**
- In browser, dropdown should show `default_model`
- Try selecting different model if available

**Check audio input:**
- Click "🎤 Test Microphone" first
- Grant microphone permissions
- Check browser console for errors

**Check gRPC connection:**
```powershell
# In Python
grpcurl -plaintext localhost:50051 list
# Should show: OptimizedLipSyncService
```

---

## Performance Monitoring

### Check GPU Usage
```powershell
nvidia-smi -l 1
# Should show 90-100% GPU utilization
```

### Check Process Count
```powershell
Get-Process python | Where-Object {$_.MainWindowTitle -like "*grpc*"}
# Should show number of servers running
```

### Check Proxy Stats
```powershell
# View backend statistics
curl http://localhost:8086/stats

# Returns JSON:
# {
#   "backends": [
#     {"address": "localhost:50051", "requests": 123, "errors": 0},
#     ...
#   ]
# }
```

### Browser Console
Open Developer Tools (F12) → Console

Look for:
```
✅ WebSocket connected
✅ Received frame: 320x256
⚡ Inference: 18ms
```

---

## Next Steps

### Test Multi-Model
1. Prepare another model:
   ```powershell
   python batch_prepare_models.py model2
   ```

2. Update HTML dropdown:
   ```html
   <option value="model2">Model 2</option>
   ```

3. Restart servers and test model switching

### Test Multi-GPU
1. Start with 8 GPUs:
   ```powershell
   .\start_multi_gpu.ps1 -NumGPUs 8 -ProcessesPerGPU 6
   ```

2. Start proxy with 48 servers:
   ```powershell
   .\lipsync-proxy.exe --ws-port 8086 --num-servers 48 --start-port 50051
   ```

3. Open multiple browser tabs and test concurrent users

### Benchmark Performance
```powershell
cd minimal_server
python test_multi_process.py --ports 50051-50054 --num-requests 1000
```

---

## Architecture Diagram

```
┌─────────────┐
│   Browser   │ WebSocket (binary)
│   Client    │
└──────┬──────┘
       │ ws://localhost:8086
       ↓
┌─────────────────────┐
│   Go Proxy Server   │ Load Balancer
│   (lipsync-proxy)   │
└──────┬──────────────┘
       │ gRPC
       ├─→ localhost:50051 (GPU 0, Process 0)
       ├─→ localhost:50052 (GPU 0, Process 1)
       ├─→ localhost:50053 (GPU 0, Process 2)
       └─→ localhost:50054 (GPU 0, Process 3)
           ↓
      ┌──────────────┐
      │ PyTorch GPU  │
      │   Inference  │
      └──────────────┘
```

---

## Success Criteria

✅ gRPC server starts in <10 seconds  
✅ Proxy connects to all backends  
✅ Browser connects via WebSocket  
✅ Real-time video generation (17-25ms latency)  
✅ Smooth playback with no stuttering  
✅ GPU utilization 90-100%  
✅ Load balanced across multiple processes  

If all checks pass → **System is working perfectly!** 🎉
