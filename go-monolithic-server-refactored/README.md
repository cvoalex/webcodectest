# Go Monolithic Lip-Sync Server

> **High-performance real-time lip-sync video generation server with gRPC API**

[![Go Version](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![gRPC](https://img.shields.io/badge/gRPC-Protocol_Buffers-244c5a?style=flat)](https://grpc.io/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-005CED?style=flat)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Performance:** 60 FPS | **Latency:** <500ms | **Batch Size:** 1-32 frames

---

## 🎯 Project Vision

This server provides **production-ready lip-sync video generation** combining:
- **Neural network inference** (ONNX Runtime with GPU acceleration)
- **Real-time audio processing** (mel spectrogram extraction)
- **High-performance image processing** (parallel BGR conversion + resizing)
- **Frame compositing** (overlay generated mouth on background video)

**Goal:** Enable real-time lip-sync video generation for conversational AI, virtual avatars, and content creation.

---

## ✨ Key Features

### Performance
- ✅ **60 FPS throughput** (5x faster than baseline)
- ✅ **< 500ms latency** for 25-frame batches
- ✅ **Parallel processing** (8-16 workers, 4-5x speedup)
- ✅ **Memory pooling** (99.9% allocation reduction)
- ✅ **GPU acceleration** (CUDA + cuDNN support)

### Architecture
- ✅ **Monolithic design** (inference + compositing in one service)
- ✅ **gRPC API** (efficient binary protocol)
- ✅ **Multi-model support** (load/unload models dynamically)
- ✅ **Production-ready** (comprehensive testing, monitoring, logging)

### Developer Experience
- ✅ **Comprehensive documentation** (12+ guides, ADRs, session notes)
- ✅ **47 functional tests** (85% code coverage)
- ✅ **Easy setup** (10-minute quick start)
- ✅ **Debugging tools** (profiling, race detection, logging)

---

## 🚀 Quick Start

### Prerequisites

- **Go 1.21+** (download from [go.dev](https://go.dev/dl/))
- **CUDA 11.8+** + cuDNN (for GPU acceleration)
- **NVIDIA GPU** (8GB+ VRAM recommended)
- **Windows 10/11** or Linux

### Installation (5 minutes)

```powershell
# 1. Clone repository
git clone https://github.com/your-org/go-monolithic-server-refactored.git
cd go-monolithic-server-refactored

# 2. Install dependencies
go mod download

# 3. Configure models
cp config.yaml.example config.yaml
# Edit config.yaml - set absolute paths for models_root and background_root

# 4. Run server
go run cmd/server/main.go
```

**Server starts on:** `localhost:50053`

**Next Steps:** See **[📖 Development Guide](docs/backend/development/DEVELOPMENT_GUIDE.md)** for detailed setup.

---

## 📖 Documentation

### 🎯 Start Here

| Document | Purpose | Time |
|----------|---------|------|
| **[📚 Documentation Hub](docs/README.md)** | Central navigation for all docs | 2 min |
| **[🚀 Quick Start Guide](docs/backend/development/DEVELOPMENT_GUIDE.md)** | Get running in 10 minutes | 10 min |
| **[🏗️ Architecture Overview](docs/backend/ARCHITECTURE.md)** | System design and components | 30 min |
| **[📡 API Reference](docs/backend/API_REFERENCE.md)** | gRPC endpoints with examples | 20 min |

### 📚 Comprehensive Guides

**Development:**
- [Development Guide](docs/backend/development/DEVELOPMENT_GUIDE.md) - Setup, running, first request
- [Testing Guide](docs/backend/development/TESTING.md) - 47 tests, coverage, how to run
- [Common Gotchas](docs/backend/development/GOTCHAS.md) - 40+ pitfalls to avoid

**Architecture:**
- [System Architecture](docs/backend/ARCHITECTURE.md) - Components, patterns, optimizations
- [API Reference](docs/backend/API_REFERENCE.md) - gRPC endpoints, request/response formats
- [Performance Analysis](docs/PERFORMANCE_ANALYSIS.md) - Benchmarks, profiling, optimization results

**Decision Records (ADRs):**
- [ADR-001: Parallel Image Processing](docs/backend/adr/ADR-001-parallel-image-processing.md) - 4-5x speedup
- [ADR-002: Memory Pooling](docs/backend/adr/ADR-002-memory-pooling.md) - 99.9% allocation reduction
- [ADR-003: Parallel Mel Extraction](docs/backend/adr/ADR-003-parallel-mel-extraction.md) - 1.5x speedup

**Session Notes:**
- [Phase 1 Optimization](docs/backend/session-notes/2025-10-29-phase1-optimization.md) - Parallel processing + memory pooling
- [Phase 2 Optimization](docs/backend/session-notes/2025-10-29-phase2-mel-extraction.md) - Parallel mel extraction

---

## 🏗️ Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Go Monolithic Server                       │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐      │
│  │   gRPC API  │  │ Model        │  │ Image         │      │
│  │   (50053)   │──│ Registry     │──│ Registry      │      │
│  └─────────────┘  └──────────────┘  └───────────────┘      │
│         │                                                     │
│         ├─── Audio Processing (Mel Spectrogram)             │
│         │         ├─ STFT computation                        │
│         │         └─ Mel filterbank (80 bands)              │
│         │                                                     │
│         ├─── Image Processing (Parallel, 8 workers)         │
│         │         ├─ BGR → RGBA conversion                   │
│         │         └─ Resize 640×360 → 320×320               │
│         │                                                     │
│         ├─── ONNX Inference (GPU)                           │
│         │         └─ UNet-based lip-sync model              │
│         │                                                     │
│         └─── Compositing (Overlay + JPEG encoding)          │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Memory Pools (sync.Pool)                             │   │
│  │  - RGBA buffers   - BGR buffers   - JPEG buffers     │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
         │
         └─── GPU (CUDA + cuDNN)
```

**Key Design Patterns:**
- **Parallel Processing** - 8-worker pool for image processing (4-5x speedup)
- **Memory Pooling** - sync.Pool eliminates 99.9% of allocations
- **Registry Pattern** - Centralized model and image management
- **Dependency Injection** - Clean component boundaries

**See:** [Architecture Documentation](docs/backend/ARCHITECTURE.md) for full details.

---

## 🎨 Tech Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Go | 1.21+ | High-performance server |
| **API** | gRPC | Latest | Efficient RPC communication |
| **ML Runtime** | ONNX Runtime | 1.16+ | GPU-accelerated inference |
| **GPU** | CUDA + cuDNN | 11.8+ | Neural network acceleration |
| **Audio** | librosa-go | Custom | Mel spectrogram extraction |
| **Image** | Custom | - | Parallel BGR/RGBA processing |

### Development Tools

- **Testing:** Go standard library (`testing`, 47 tests, 85% coverage)
- **Profiling:** pprof (CPU, memory, goroutine profiling)
- **Race Detection:** `go test -race` (zero data races)
- **Logging:** Structured logging with configurable levels
- **Build:** Standard Go toolchain

---

## 📊 Performance Metrics

### Benchmark Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **FPS Throughput** | 12 | 60 | **5.0x** ✅ |
| **Batch Latency** | 2,083ms | 417ms | **5.0x** |
| **Image Processing** | 1,600ms | 200ms | **8.0x** |
| **Memory Allocations** | 10,000/sec | 10/sec | **1000x** |
| **GC Pause Time** | 50-100ms | <5ms | **20x** |

### Optimization Phases

**Phase 1: Parallel Image Processing + Memory Pooling**
- Parallel row processing: 4-5x speedup
- sync.Pool: 99.9% allocation reduction
- Result: 12 FPS → 48 FPS (4x improvement)

**Phase 2: Parallel Mel Extraction**
- Goroutine-per-window parallelization
- Result: 48 FPS → 60 FPS (1.25x improvement)

**Total:** 12 FPS → 60 FPS (5x improvement)

**See:** [Performance Analysis](docs/PERFORMANCE_ANALYSIS.md) for detailed benchmarks.

---

## 🧪 Testing

### Test Coverage

- **47 Functional Tests** across 7 categories
- **85% Code Coverage** (critical paths well-tested)
- **Zero Data Races** (validated with `-race` flag)
- **100% Pixel-Perfect** (parallel output identical to sequential)

### Test Categories

| Category | Tests | Focus |
|----------|-------|-------|
| Audio Processing | 4 | Mel spectrogram accuracy |
| Image Processing | 5 | BGR conversion, resizing |
| Parallel Processing | 5 | Concurrency correctness |
| Parallel Mel | 6 | Audio parallelization |
| Integration | 4 | End-to-end flows |
| Performance | 5 | Optimization validation |
| Edge Cases | 18 | Boundary conditions |

### Running Tests

```powershell
# All tests
go test ./... -v

# With coverage
go test ./... -v -cover

# With race detection
go test ./... -v -race

# Specific suite
go test ./functional-tests/performance -v
```

**See:** [Testing Guide](docs/backend/development/TESTING.md) for comprehensive test documentation.

---

## 🔧 Configuration

### Sample config.yaml

```yaml
server:
  port: 50053
  log_level: "info"
  max_message_size_mb: 100

models_root: "D:/Projects/webcodecstest/model"
background_root: "D:/Projects/webcodecstest/backgrounds"

models:
  - model_id: "sanders"
    model_path: "sanders_unet_328.onnx"
    background_dir: "sanders_backgrounds"
    max_cached_models: 40

gpu:
  device_id: 0
  memory_limit_gb: 8
```

**Important:** Use absolute paths for `models_root` and `background_root`.

---

## 📡 API Usage

### gRPC Endpoints

**Main Endpoint:**
```protobuf
rpc InferBatchComposite(CompositeBatchRequest) returns (CompositeBatchResponse);
```

**Management:**
```protobuf
rpc Health(HealthRequest) returns (HealthResponse);
rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
rpc GetModelStats(GetModelStatsRequest) returns (GetModelStatsResponse);
```

### Example Request (Go Client)

```go
req := &pb.CompositeBatchRequest{
    ModelId:       "sanders",
    VisualFrames:  visualData,  // float32 [25][6][320][320]
    RawAudio:      audioData,    // float32 [10240] (640ms @ 16kHz)
    BatchSize:     25,
    StartFrameIdx: 0,
    SampleRate:    16000,
}

resp, err := client.InferBatchComposite(ctx, req)
if err != nil {
    log.Fatal(err)
}

// resp.CompositedFrames: [][]byte (JPEG-encoded frames)
// resp.TotalTimeMs: float32 (latency)
```

**See:** [API Reference](docs/backend/API_REFERENCE.md) for all endpoints with examples.

---

## 🏃 Development Workflow

### Running the Server

```powershell
# Development mode (with logging)
go run cmd/server/main.go

# Production build
go build -o server.exe cmd/server/main.go
./server.exe
```

### Making Changes

1. **Read relevant docs** (Architecture, Gotchas)
2. **Write tests first** (TDD approach recommended)
3. **Implement feature** (follow existing patterns)
4. **Run tests with -race** (`go test -race ./...`)
5. **Profile if performance-critical** (`go test -cpuprofile=cpu.prof`)
6. **Update documentation** (ADRs for architectural changes)

### Common Commands

```powershell
# Build
go build ./...

# Test
go test ./... -v

# Test with coverage
go test ./... -cover

# Race detection
go test ./... -race

# Profile (CPU)
go test -cpuprofile=cpu.prof -bench=.
go tool pprof cpu.prof

# Profile (Memory)
go test -memprofile=mem.prof -bench=.
go tool pprof mem.prof
```

---

## 🐛 Troubleshooting

### Common Issues

**Server won't start:**
```
Error: model file not found
→ Check config.yaml paths are absolute
→ Verify models exist in models_root
```

**Out of memory:**
```
Error: CUDA out of memory
→ Reduce batch size (25 → 8)
→ Check GPU memory with nvidia-smi
→ Unload unused models
```

**Low performance:**
```
FPS < 40
→ Check GPU is being used (nvidia-smi)
→ Verify CUDA 11.8+ installed
→ Run performance tests to identify bottleneck
```

**See:** [Common Gotchas](docs/backend/development/GOTCHAS.md) for 40+ pitfalls and solutions.

---

## 📈 Development Philosophy

### Core Principles

1. **Performance First** - Profile before optimizing, measure everything
2. **Test Thoroughly** - 85% coverage, zero data races
3. **Document Decisions** - ADRs for architectural changes
4. **Keep It Simple** - Avoid premature optimization
5. **Production Ready** - Comprehensive error handling, logging, monitoring

### Best Practices

**Code Quality:**
- ✅ Write tests first (TDD)
- ✅ Use table-driven tests
- ✅ Always run with `-race` flag
- ✅ Profile before optimizing
- ✅ Document non-obvious code

**Architecture:**
- ✅ Follow established patterns (registry, DI, parallel processing)
- ✅ Use sync.Pool for temporary buffers
- ✅ Keep goroutine count predictable
- ✅ Avoid premature abstraction

**Documentation:**
- ✅ Update docs with code changes
- ✅ Write ADRs for architectural decisions
- ✅ Maintain session notes for optimization work
- ✅ Keep examples up-to-date

---

## 🤝 Contributing

We welcome contributions! Please:

1. **Read the docs** - Especially [Architecture](docs/backend/ARCHITECTURE.md) and [Gotchas](docs/backend/development/GOTCHAS.md)
2. **Open an issue** - Discuss your proposal first
3. **Write tests** - Maintain 85%+ coverage
4. **Follow patterns** - Use existing code as reference
5. **Document changes** - Update relevant docs

### Contribution Checklist

- [ ] Tests written (with `-race` flag)
- [ ] Documentation updated
- [ ] ADR written (if architectural change)
- [ ] Performance validated (if optimization)
- [ ] Code reviewed

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 📞 Support

- **Documentation:** [docs/README.md](docs/README.md)
- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions

---

## 🙏 Acknowledgments

Built with:
- [Go](https://go.dev/) - Excellent concurrency primitives
- [gRPC](https://grpc.io/) - Efficient RPC framework
- [ONNX Runtime](https://onnxruntime.ai/) - Fast ML inference
- [CUDA + cuDNN](https://developer.nvidia.com/cuda-toolkit) - GPU acceleration

---

## 📊 Project Stats

- **Lines of Code:** ~15,000 (server) + ~15,000 (docs)
- **Documentation:** 12+ comprehensive guides
- **Tests:** 47 functional tests, 85% coverage
- **Performance:** 60 FPS, <500ms latency
- **Optimizations:** 5x speedup over baseline

---

**Status:** ✅ Production Ready  
**Version:** 1.0.0  
**Last Updated:** November 6, 2025

---

**[📚 Start with the Documentation Hub →](docs/README.md)**
