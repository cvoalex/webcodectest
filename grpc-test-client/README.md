# gRPC Test Client

A simple Go program to test the gRPC lip-sync inference server.

## Features

- Calls `GenerateInference` RPC method
- Sends empty audio feature (to test frame-only inference)
- Displays timing and response details
- Saves the resulting JPEG image to disk

## Building

```bash
# Build the executable
.\build.bat

# Or manually:
go build -o grpc-test-client.exe .
```

## Usage

```bash
# Test frame 100 (default)
.\run.bat

# Or with custom options:
.\grpc-test-client.exe -frame 100
.\grpc-test-client.exe -model sanders -frame 50
.\grpc-test-client.exe -server localhost:50051 -frame 0
```

## Command-Line Options

- `-server` - gRPC server address (default: `localhost:50051`)
- `-model` - Model name to use (default: `sanders`)
- `-frame` - Frame ID to request (default: `100`)

## Example Output

```
🔌 Connecting to gRPC server at localhost:50051...
✅ Connected!

📤 Sending inference request:
   Model: sanders
   Frame: 100
   Audio: empty (0 bytes)

✅ Success!

📊 Response Details:
   Success: true
   Frame ID: 100
   Processing Time: 15.30ms
   Prepare Time: 0.50ms
   Inference Time: 12.20ms
   Composite Time: 2.10ms
   Total Time: 18.45ms
   Image Size: 18234 bytes (17.81 KB)
   Bounds: [0.1 0.2 0.8 0.9]

💾 Image saved to: frame_100.jpg
```

## Output

- Displays timing metrics and response details
- Saves the result as `frame_<N>.jpg` in the current directory

## Requirements

- Go 1.21+
- gRPC server running on `localhost:50051`
- Model loaded (default: `sanders`)
