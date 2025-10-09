# Quick Setup: Install protoc

## Option 1: Direct Download (Recommended - 2 minutes)

### Step 1: Download protoc
1. Go to: https://github.com/protocolbuffers/protobuf/releases/latest
2. Download: `protoc-XX.X-win64.zip` (e.g., `protoc-25.1-win64.zip`)
3. Extract to: `C:\protoc\`

### Step 2: Add to PATH
```powershell
# Run as Administrator or add permanently via System Properties
$env:PATH += ";C:\protoc\bin"

# Verify
protoc --version
```

### Step 3: Install Go Plugins
```powershell
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```

### Step 4: Generate Proto Files
```powershell
cd D:\Projects\webcodecstest\grpc-websocket-proxy
.\generate_proto.bat
```

### Step 5: Build Proxy
```powershell
.\build.bat
```

---

## Option 2: Using Chocolatey (If you have it)

```powershell
choco install protobuf -y
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
cd D:\Projects\webcodecstest\grpc-websocket-proxy
.\generate_proto.bat
.\build.bat
```

---

## Option 3: Use Pre-built Binary (Fastest)

If you just want to test, I can provide a pre-compiled `lipsync-proxy.exe`. But for development, you should install protoc properly.

---

## Verify Setup

After installation:
```powershell
# Should show version
protoc --version

# Should show installed
go list -m google.golang.org/protobuf
go list -m google.golang.org/grpc

# Should exist
ls $env:GOPATH\bin\protoc-gen-go.exe
ls $env:GOPATH\bin\protoc-gen-go-grpc.exe
```
