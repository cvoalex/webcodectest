# Refactoring Plan - Monolithic Lip-Sync Server

**Date**: October 28, 2025  
**Purpose**: Improve code organization, maintainability, and testability while preserving performance  
**Status**: Planning phase

---

## 🎯 Goals

1. **Separation of Concerns**: Split 1064-line main.go into focused, single-responsibility files
2. **Testability**: Make each component independently testable
3. **Maintainability**: Clear structure that's easy to navigate and modify
4. **Performance**: Maintain all optimizations (memory pooling, parallelization)
5. **Backward Compatibility**: Same API, same performance, cleaner code

---

## 📁 Proposed Directory Structure

```
go-monolithic-server-refactored/
├── cmd/
│   └── server/
│       └── main.go                    (100 lines - just wiring/initialization)
├── internal/
│   ├── server/
│   │   ├── server.go                  (Server struct + constructor)
│   │   ├── inference.go               (InferBatchComposite handler - core logic)
│   │   ├── health.go                  (Health, GetModelStats endpoints)
│   │   └── model_management.go        (LoadModel, UnloadModel, ListModels)
│   ├── compositing/
│   │   ├── compositor.go              (Compositor struct + CompositeFrame)
│   │   ├── image_ops.go               (OutputToImage, ResizeImage, etc)
│   │   └── pools.go                   (All memory pools)
│   └── utils/
│       └── conversion.go              (BytesToFloat32, etc)
├── audio/                             (✅ Already well-structured)
│   ├── processor.go                   (STFT, mel-spectrogram)
│   ├── encoder.go                     (Audio encoder + pool)
│   └── config.go
├── config/                            (✅ Already well-structured)
│   └── config.go
├── logger/                            (✅ Already well-structured)
│   └── buffered_logger.go
├── proto/                             (✅ Already well-structured)
│   ├── monolithic.proto
│   └── monolithic.pb.go
├── registry/                          (✅ Already well-structured)
│   ├── model_registry.go
│   └── image_registry.go
└── testing/                           (Keep as is)
    └── *.go
```

---

## 📊 Current vs Refactored Breakdown

### Current State (go-monolithic-server/)

| File | Lines | Responsibilities |
|------|-------|------------------|
| `cmd/server/main.go` | **1064** | Everything! 😱 |
| - Initialization | ~150 | Server setup, config loading |
| - InferBatchComposite | ~500 | Audio processing + inference + compositing |
| - Compositing helpers | ~200 | Image ops, pools, conversions |
| - Health/Management | ~200 | Health, stats, model management |

### Refactored State (go-monolithic-server-refactored/)

| File | Lines | Responsibilities |
|------|-------|------------------|
| `cmd/server/main.go` | ~100 | Just initialization + wiring |
| `internal/server/server.go` | ~50 | Server struct + constructor |
| `internal/server/inference.go` | ~300 | Core inference logic |
| `internal/server/health.go` | ~80 | Health + stats endpoints |
| `internal/server/model_management.go` | ~150 | Load/unload/list models |
| `internal/compositing/compositor.go` | ~60 | Compositor struct + main method |
| `internal/compositing/image_ops.go` | ~150 | Image operations |
| `internal/compositing/pools.go` | ~80 | Memory pools |
| `internal/utils/conversion.go` | ~40 | Utility functions |

**Total**: Same functionality, split into 9 focused files instead of 1 monolith!

---

## 🔧 Refactoring Strategy

### Phase 1: Create Structure ✅ DONE
- [x] Create backup: `go-monolithic-server-backup-oct28-2025/`
- [x] Create new directory: `go-monolithic-server-refactored/`
- [x] Copy well-structured modules (audio, config, logger, proto, registry)
- [x] Create internal package structure

### Phase 2: Extract Compositing Logic ✅ DONE
- [x] `internal/compositing/pools.go` - All memory pools
- [x] `internal/compositing/image_ops.go` - Image operations
- [x] `internal/compositing/compositor.go` - Main compositor

### Phase 3: Extract Utilities ✅ DONE
- [x] `internal/utils/conversion.go` - Byte/float32 conversions

### Phase 4: Extract Server Logic (TODO)
- [ ] `internal/server/server.go` - Server struct
- [ ] `internal/server/inference.go` - InferBatchComposite
- [ ] `internal/server/health.go` - Health/stats
- [ ] `internal/server/model_management.go` - Model CRUD

### Phase 5: Simplify Main (TODO)
- [ ] `cmd/server/main.go` - Clean initialization only

### Phase 6: Testing (TODO)
- [ ] Unit tests for compositing package
- [ ] Unit tests for server package
- [ ] Integration tests
- [ ] Performance benchmarks (must match current!)

### Phase 7: Migration (TODO)
- [ ] Verify all tests pass
- [ ] Performance validation (must be ≥ current)
- [ ] Replace original with refactored
- [ ] Update documentation

---

## 🎨 Key Design Decisions

### 1. Memory Pools Stay Central
**Decision**: Keep all pools in `internal/compositing/pools.go`  
**Reason**: Centralized management, easy to monitor, prevents duplication

### 2. Compositor is Stateless (Mostly)
**Decision**: Compositor only holds JPEG quality setting  
**Reason**: Thread-safe, easy to test, can be reused

### 3. Server Package is Internal
**Decision**: Use `internal/server` instead of top-level `server`  
**Reason**: Enforce encapsulation, API is only through gRPC

### 4. Keep Audio Package Unchanged
**Decision**: Don't refactor `audio/` package  
**Reason**: Already well-structured, heavily optimized, works perfectly

### 5. Separate Concerns Clearly
**Decision**: Each file has ONE clear responsibility  
**Reason**: Easy to find code, easy to test, easy to modify

---

## 📝 Migration Checklist

Before replacing current server with refactored version:

### Performance Validation
- [ ] Batch 25 synthetic: Must achieve ≥125 FPS
- [ ] Batch 8 real audio: Must achieve ≥42 FPS
- [ ] Memory per request: Must be ≤1MB
- [ ] Audio processing: Must be ≤25ms (batch 8)
- [ ] No memory leaks over 1000 requests

### Functional Validation
- [ ] All gRPC endpoints work
- [ ] Health check returns correct data
- [ ] Model loading/unloading works
- [ ] Inference produces identical output
- [ ] Compositing produces identical JPEGs

### Code Quality
- [ ] All files compile without warnings
- [ ] All imports are used
- [ ] No dead code
- [ ] Consistent naming conventions
- [ ] All public functions have comments

---

## 🚀 Benefits of Refactoring

### Developer Experience
- **Find code 10x faster**: "Where's the resize logic?" → `compositing/image_ops.go`
- **Test in isolation**: Test compositor without full server
- **Modify safely**: Changes to one area don't break others
- **Onboard new devs**: Clear structure, easy to understand

### Maintainability
- **Single Responsibility**: Each file does ONE thing well
- **Clear Dependencies**: Easy to see what depends on what
- **Easy Debugging**: Smaller files, focused logic
- **Better IDE Support**: Jump-to-definition actually works

### Testing
- **Unit Tests**: Test each component independently
- **Mocking**: Easy to mock dependencies
- **Benchmarks**: Benchmark specific operations
- **Coverage**: Measure coverage per component

---

## ⚠️ Risks & Mitigation

### Risk 1: Performance Regression
**Mitigation**:
- Keep all memory pools
- Keep all parallelization
- Benchmark before/after
- No premature abstraction

### Risk 2: Breaking Changes
**Mitigation**:
- gRPC API stays identical
- Thorough integration testing
- Keep backup (go-monolithic-server-backup-oct28-2025/)
- Can revert instantly

### Risk 3: Incomplete Refactoring
**Mitigation**:
- Do it in phases
- Each phase is shippable
- Test after each phase
- Don't rush

---

## 📈 Success Criteria

Refactoring is successful if:

1. ✅ **Performance**: Same or better (≥125 FPS batch 25, ≥42 FPS batch 8)
2. ✅ **Functionality**: All tests pass, identical output
3. ✅ **Code Quality**: <200 lines per file, clear separation
4. ✅ **Maintainability**: New features take less time to implement
5. ✅ **Testing**: Can unit test each component

---

## 🎯 Next Steps

### Option A: Complete the Refactoring Now
Continue implementing phases 4-7, test thoroughly, migrate.

**Time estimate**: 4-6 hours  
**Risk**: Medium (need thorough testing)  
**Benefit**: Clean codebase ready for future work

### Option B: Incremental Refactoring
Keep current server, refactor one component at a time, migrate gradually.

**Time estimate**: 1-2 hours per component  
**Risk**: Low (smaller changes, easier to test)  
**Benefit**: Can ship improvements incrementally

### Option C: Keep Current, Document Better
Leave code as-is, just add better comments and documentation.

**Time estimate**: 1 hour  
**Risk**: Very low  
**Benefit**: Quick win, code works now

---

## 💡 Recommendation

**I recommend Option A** (Complete refactoring now) because:

1. **Code is already backed up** - We can revert instantly if needed
2. **We have excellent tests** - test_batch_8_real.go validates everything
3. **Performance is measured** - We know exactly what to target
4. **Current momentum** - We're already in refactoring mode
5. **Future benefits** - Makes all future work easier

The refactored code will be much easier to:
- Add new features
- Fix bugs
- Optimize further
- Hand off to other developers
- Scale to Blackwell deployment

---

## 📚 Files Created So Far

### ✅ Completed
- `internal/compositing/pools.go` - Memory pools
- `internal/compositing/image_ops.go` - Image operations
- `internal/compositing/compositor.go` - Main compositor
- `internal/utils/conversion.go` - Utility functions
- `internal/server/server.go` - Server struct (partial)

### 🔨 In Progress
- `internal/server/inference.go` - Extract core inference logic
- `internal/server/health.go` - Extract health/stats endpoints
- `internal/server/model_management.go` - Extract model management
- `cmd/server/main.go` - Simplify to just initialization

---

**Status**: Ready to proceed with complete refactoring  
**Next**: Extract inference.go, health.go, model_management.go, then create clean main.go  
**ETA**: 2-3 hours to complete + 1 hour testing = 3-4 hours total

