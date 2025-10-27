# Documentation - Real-Time Lip Sync System

This directory contains all documentation for the separated architecture lip sync system (Inference Server + Compositing Server).

## 📚 Documentation Index

### Getting Started

**[SETUP_AND_EXECUTION_GUIDE.md](SETUP_AND_EXECUTION_GUIDE.md)** - **START HERE!**
- Complete installation and setup instructions
- How to build and run both servers
- Configuration reference
- Testing and validation
- Troubleshooting guide
- **Who should read**: Everyone (developers, operators, new users)

**[QUICK_START.md](QUICK_START.md)**
- Quick reference for common tasks
- Essential commands
- Configuration quick reference
- **Who should read**: Users already familiar with the system

---

### Production Deployment

**[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)**
- Production deployment architecture
- Scaling strategies
- Monitoring and metrics
- Load balancing
- Security considerations
- Backup and recovery
- **Who should read**: DevOps, system administrators, production engineers

---

### Performance & Optimization

**[PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)**
- Detailed performance metrics and benchmarks
- Bottleneck analysis
- Optimization history
- Disk I/O analysis
- gRPC overhead breakdown
- **Who should read**: Performance engineers, developers optimizing the system

---

### Features & Configuration

**[MODELS_ROOT_FEATURE.md](MODELS_ROOT_FEATURE.md)**
- `models_root` configuration feature
- Simplifying model path management
- Before/after examples
- Path resolution logic
- **Who should read**: Developers configuring models, multi-tenant deployments

---

## 🏗️ System Architecture

### Server Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Client Applications                  │
│              (WebRTC, HTTP, gRPC clients)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Compositing Server (Port 50052)                │
│  • CPU-based compositing                                 │
│  • Background frame management (LRU cache)               │
│  • JPEG encoding                                         │
│  • Multi-model support (11,000+ models)                  │
│  • WebRTC integration ready                              │
└────────────────────┬────────────────────────────────────┘
                     │ gRPC
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Inference Server (Port 50051)                  │
│  • GPU-based inference (NVIDIA RTX 4090)                 │
│  • ONNX Runtime                                          │
│  • Multi-model support (40+ models per GPU)              │
│  • Worker pool (8 concurrent workers)                    │
│  • Round-robin GPU assignment                            │
└─────────────────────────────────────────────────────────┘
```

### Key Technologies

- **Language**: Go 1.24.0
- **RPC Framework**: gRPC with Protocol Buffers
- **Inference Engine**: ONNX Runtime 1.22.0 GPU
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Image Encoding**: JPEG (quality 75)
- **Caching**: LRU cache for background frames

---

## 📊 Performance Summary

**Production Performance (Warm Servers):**
- **Throughput**: 31.4 FPS per client
- **System Capacity**: ~251 FPS (8 workers)
- **Compositing Time**: 49ms per batch (24 frames)
- **Inference Time**: 416ms per batch (24 frames)
- **Latency**: ~465ms total per batch
- **Overhead**: 11-13% (gRPC separation cost)

**Resource Usage:**
- **GPU Memory**: ~500MB per model
- **RAM**: ~2.1GB per model (with 523 frames preloaded)
- **Disk I/O**: Zero during processing (100% cache hit)

---

## 🚀 Quick Links by Role

### Developers
1. [SETUP_AND_EXECUTION_GUIDE.md](SETUP_AND_EXECUTION_GUIDE.md) - Installation and setup
2. [MODELS_ROOT_FEATURE.md](MODELS_ROOT_FEATURE.md) - Configuration features
3. [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) - Understanding performance

### DevOps / Operators
1. [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) - Deployment guide
2. [QUICK_START.md](QUICK_START.md) - Quick reference
3. [SETUP_AND_EXECUTION_GUIDE.md](SETUP_AND_EXECUTION_GUIDE.md#troubleshooting) - Troubleshooting

### Performance Engineers
1. [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) - Detailed analysis
2. [SETUP_AND_EXECUTION_GUIDE.md](SETUP_AND_EXECUTION_GUIDE.md#performance-tuning) - Tuning guide

---

## 📁 Related Files in Project

### Server Code
- `go-inference-server/` - GPU inference server source code
- `go-compositing-server/` - CPU compositing server source code

### Configuration
- `go-inference-server/config.yaml` - Inference server configuration
- `go-compositing-server/config.yaml` - Compositing server configuration

### Model Data
- `minimal_server/models/` - Model files and background frames

### Scripts
- `run-separated-test.ps1` - Test script (starts servers + runs test)

---

## 🔄 Document Maintenance

**Last Updated**: October 24, 2025  
**System Version**: 1.0.0  
**Status**: Production Ready ✅

**Update Schedule**:
- Update after significant feature additions
- Update after performance optimizations
- Update after production deployment changes
- Review quarterly for accuracy

---

## 💡 Contributing to Documentation

When adding new documentation:
1. Place in `docs/` directory
2. Update this README.md with a link and description
3. Use clear, descriptive filenames
4. Include "Who should read" section
5. Add to appropriate quick links section

---

## 📞 Support

For questions about:
- **Installation/Setup**: See [SETUP_AND_EXECUTION_GUIDE.md](SETUP_AND_EXECUTION_GUIDE.md)
- **Production Issues**: See [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)
- **Performance**: See [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)
- **Configuration**: See [SETUP_AND_EXECUTION_GUIDE.md](SETUP_AND_EXECUTION_GUIDE.md#configuration) or [MODELS_ROOT_FEATURE.md](MODELS_ROOT_FEATURE.md)

---

**System Ready**: Both servers operational and tested at 31.4 FPS! 🚀
