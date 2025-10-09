# 🚀 Quick Start - Browser to gRPC in 5 Minutes

## Why This Exists

**Problem:** Browsers cannot directly connect to gRPC servers.

**Solution:** This Go proxy translates WebSocket ↔ gRPC.

```
Browser (WebSocket) → Go Proxy (8086) → Python gRPC Server (50051)
```

---

## 📋 Prerequisites

1. ✅ **Python gRPC server running** (see below)
2. ✅ **Go 1.21+** installed
3. ✅ **protoc** (Protocol Buffers compiler)
4. ✅ **Go protobuf plugins**:
   ```bash
   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
   ```

---

## 🎯 5-Minute Setup

### Step 1: Start Python gRPC Server

```bash
cd ..\minimal_server
D:\Projects\webcodecstest\.venv312\Scripts\python.exe optimized_grpc_server.py
```

You should see:
```
================================================================================
🚀 ULTRA-OPTIMIZED GRPC SERVER
================================================================================
🚀 gRPC server started on [::]:50051
```

**Keep this running!**

---

### Step 2: Generate Go Code

```bash
cd grpc-websocket-proxy
generate_proto.bat
```

Output:
```
✅ Proto generation successful!
Generated files:
  - pb/optimized_lipsyncsrv.pb.go
  - pb/optimized_lipsyncsrv_grpc.pb.go
```

---

### Step 3: Build Proxy

```bash
build.bat
```

Output:
```
✅ Build successful!
Executable: lipsync-proxy.exe
```

---

### Step 4: Run Proxy

```bash
run.bat
```

Output:
```
================================================================================
🌉 GRPC-TO-WEBSOCKET PROXY SERVER
================================================================================
🔌 Connecting to gRPC server at localhost:50051...
✅ Connected to gRPC server!
   Status: SERVING
   Loaded models: 1

🚀 WebSocket proxy started on ws://localhost:8086/ws
📁 Static files served from ./static/
```

**Keep this running too!**

---

### Step 5: Open Browser

Open in your browser:
```
http://localhost:8086/grpc-lipsync-client.html
```

Click "Connect" and enjoy **50+ FPS lip sync** via gRPC! 🎉

---

## ✅ Verification

### Test Health Check

```bash
curl http://localhost:8086/health
```

Should return:
```json
{
  "status": "SERVING",
  "healthy": true,
  "loaded_models": 1,
  "uptime": 12.34
}
```

### Check Logs

**Terminal 1 (Python gRPC):**
```
✅ Generated frame 0 in 17.8ms
✅ Generated frame 1 in 16.9ms
```

**Terminal 2 (Go Proxy):**
```
✅ sanders frame 0: gRPC=17ms total=19ms size=98,432 bytes
✅ sanders frame 1: gRPC=16ms total=18ms size=97,821 bytes
```

---

## 🎮 Using the Web Client

1. **Click "Connect"** - Connects to proxy via WebSocket
2. **Move slider** - Navigate frames (0-522)
3. **Click "Play"** - Start real-time playback
4. **Adjust FPS** - Change playback speed (12-60 FPS)
5. **Monitor stats** - Watch latency and throughput

**Performance:**
- Proxy latency: 17-22ms total
- gRPC time: 15-18ms
- FPS: 45-55 sustained

---

## 🐛 Troubleshooting

### "failed to connect to gRPC"

**Problem:** Python server not running

**Fix:**
```bash
cd ..\minimal_server
python optimized_grpc_server.py
```

---

### "bind: address already in use"

**Problem:** Port 8086 in use

**Fix:**
```bash
# Use different port
lipsync-proxy.exe -ws-port 8087

# Update client to connect to ws://localhost:8087/ws
```

---

### "protoc-gen-go: program not found"

**Problem:** Go plugins not installed

**Fix:**
```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Add to PATH (Windows)
set PATH=%PATH%;%USERPROFILE%\go\bin

# Or add permanently via System Properties → Environment Variables
```

---

### Browser Can't Connect

**Problem:** CORS or proxy not running

**Fix:**
1. Check proxy is running: `http://localhost:8086/health`
2. Look for errors in proxy terminal
3. Try Chrome/Edge (better WebSocket support)
4. Check firewall isn't blocking port 8086

---

## 📊 Architecture Overview

### Full Stack

```
┌──────────────────┐
│  Browser Client  │
│  (HTML/JS)       │
└────────┬─────────┘
         │ WebSocket (Binary Protocol)
         │ Port 8086
         │
┌────────▼─────────┐
│   Go Proxy       │
│  (main.go)       │
│  - WebSocket→gRPC│
│  - Protocol Trans│
└────────┬─────────┘
         │ gRPC (HTTP/2 + Protobuf)
         │ Port 50051
         │
┌────────▼─────────┐
│ Python gRPC      │
│ (optimized_grpc_│
│  server.py)      │
│  - Inference     │
│  - Pre-loaded    │
└──────────────────┘
```

### Protocol Translation

**Browser sends:**
```
Binary: [model_len][model_name][frame_id]
```

**Proxy translates to gRPC:**
```protobuf
OptimizedInferenceRequest {
  model_name: "sanders"
  frame_id: 50
}
```

**Server responds:**
```protobuf
OptimizedInferenceResponse {
  success: true
  frame_id: 50
  prediction_data: [JPEG bytes]
  processing_time_ms: 17.8
}
```

**Proxy sends to browser:**
```
Binary: [JPEG bytes]
```

---

## 📁 What You Built

```
grpc-websocket-proxy/
├── lipsync-proxy.exe        # Compiled Go binary
├── pb/                      # Generated Go code
│   ├── optimized_lipsyncsrv.pb.go
│   └── optimized_lipsyncsrv_grpc.pb.go
└── static/
    └── grpc-lipsync-client.html  # Web client
```

**Size:** ~15 MB (includes all dependencies)

**Performance:** Adds only 1-4ms latency

---

## 🎯 Next Steps

### Production Deployment

1. **Build for production:**
   ```bash
   go build -ldflags="-s -w" -o lipsync-proxy.exe .
   ```

2. **Run as Windows service:**
   - Use NSSM (Non-Sucking Service Manager)
   - Or sc.exe to create service

3. **Add TLS:**
   - Use Let's Encrypt certificates
   - Modify proxy to use `wss://` (secure WebSocket)

### Scaling

1. **Multiple proxy instances:**
   ```bash
   lipsync-proxy.exe -ws-port 8086 &
   lipsync-proxy.exe -ws-port 8087 &
   lipsync-proxy.exe -ws-port 8088 &
   ```

2. **Load balancer:**
   - nginx reverse proxy
   - HAProxy
   - Cloud load balancer (AWS ALB, Azure Load Balancer)

3. **Multiple gRPC backends:**
   - Modify proxy to use gRPC client-side load balancing
   - Or use service mesh (Istio, Linkerd)

---

## 📚 Full Documentation

- **Proxy Details:** [README.md](README.md)
- **gRPC Server:** [../minimal_server/GRPC_SERVER_README.md](../minimal_server/GRPC_SERVER_README.md)
- **All Servers:** [../minimal_server/SERVER_IMPLEMENTATIONS.md](../minimal_server/SERVER_IMPLEMENTATIONS.md)
- **Project Overview:** [../PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)

---

## 🎉 Success!

You now have:

✅ Python gRPC server (50051)  
✅ Go WebSocket proxy (8086)  
✅ Browser client with real-time video  
✅ **17-22ms total latency**  
✅ **45-55 FPS** sustained throughput  

**Enjoy your high-performance browser-to-gRPC connection!** 🚀
