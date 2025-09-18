# 🚨 CRITICAL REMINDER: NO MOCK DATA OR FASTAPI EVER 🚨

## ABSOLUTE RULES FOR THIS PROJECT:

### ❌ NEVER USE:
- **MOCK DATA** - No dummy frames, no test patterns, no fake responses
- **FASTAPI** - The Python service uses gRPC ONLY on port 50051
- **Dummy image generation** - No generateTestFrame() or similar functions
- **HTTP calls to port 8000** - FastAPI is NOT used in this project

### ✅ ALWAYS USE:
- **Real Python gRPC service** running on port 50051
- **Actual lip-sync frame generation** from the Python service
- **Real audio data** triggering real frame generation
- **Direct gRPC calls** to the Python service

## THE ARCHITECTURE:
```
JavaScript Audio Buffer (40ms chunks) 
    ↓
Go Server gRPC Proxy (/api/grpc-proxy)
    ↓
Python gRPC Client Script (grpc_client.py)
    ↓ 
Python gRPC Service (port 50051) 
    ↓
REAL LIP-SYNC FRAMES
```

## CURRENT WORKING COMPONENTS:
- ✅ Python gRPC service running on port 50051
- ✅ Python gRPC client script (grpc_client.py) tested and working
- ✅ Go server with gRPC proxy endpoint
- ✅ JavaScript calling Go server successfully (300+ frame requests)

## THE PROBLEM TO FIX:
The Go server currently has compilation errors due to duplicate declarations. 
Once fixed, it MUST call the real Python gRPC client script to get actual lip-sync frames.

## NEVER FORGET:
**The user has made it explicitly clear MULTIPLE TIMES:**
- NO DUMMY DATA
- NO FASTAPI  
- ONLY REAL gRPC DATA

**Any violation of these rules is unacceptable.**
