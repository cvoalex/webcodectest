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
# Test frames 95-100 (default - 5 frames)
.\run.bat

# Or with custom options:
.\grpc-test-client.exe -start 95 -count 5
.\grpc-test-client.exe -start 0 -count 10
.\grpc-test-client.exe -model sanders -start 100 -count 20
.\grpc-test-client.exe -server localhost:50051 -start 50 -count 3
```

## Command-Line Options

- `-server` - gRPC server address (default: `localhost:50051`)
- `-model` - Model name to use (default: `sanders`)
- `-start` - Starting frame ID (default: `95`)
- `-count` - Number of frames to generate (default: `5`)

## Example Output

```
🔌 Connecting to gRPC server at localhost:50051...
✅ Connected!

� Generating 5 frames (frames 95-100)...
   Model: sanders
   Note: Uses pre-extracted audio features

✅ Frame 95: 268.45ms (267.39ms inference) - 17224 bytes - saved to frame_95.jpg
✅ Frame 96: 45.23ms (44.12ms inference) - 17156 bytes - saved to frame_96.jpg
✅ Frame 97: 42.89ms (41.87ms inference) - 17089 bytes - saved to frame_97.jpg
✅ Frame 98: 43.12ms (42.01ms inference) - 17203 bytes - saved to frame_98.jpg
✅ Frame 99: 44.67ms (43.54ms inference) - 17178 bytes - saved to frame_99.jpg

======================================================================
📈 PERFORMANCE STATISTICS
======================================================================

📊 Summary:
   Total Frames: 5
   Successful: 5
   Failed: 0
   Total Time: 444.36ms (0.44s)
   Total Data: 85850 bytes (0.08 MB)

⚡ Processing Time:
   Average: 88.87ms
   Min: 42.89ms
   Max: 268.45ms

🧠 Inference Time:
   Average: 87.79ms
   Min: 41.87ms
   Max: 267.39ms

🌐 Total Time (including network):
   Average: 88.87ms

💾 Image Size:
   Average: 17170 bytes (16.77 KB)

🚀 Throughput:
   Frames per second: 11.26 FPS
   Data rate: 0.18 MB/s

======================================================================

✅ All frames generated successfully!
```

## Output

- Generates multiple JPEG images: `frame_95.jpg`, `frame_96.jpg`, etc.
- Displays comprehensive performance statistics
- Shows average, min, and max timing metrics
- Calculates throughput (FPS and data rate)

## Requirements

- Go 1.21+
- gRPC server running on `localhost:50051`
- Model loaded (default: `sanders`)
