# Functional Test Suite

Comprehensive functional tests for all optimizations in go-monolithic-server-refactored.

## Test Categories

### 1. Image Processing (`image-processing/`)
- BGR to RGBA conversion accuracy (5 test cases)
- Bilinear interpolation correctness (4 scenarios)  
- Image resizing with scale factors
- Color clamping boundaries (7 edge cases)
- Image dimension handling (7 test cases)
- **Coverage**: 10 test functions, 2 benchmarks

### 2. Audio Processing (`audio-processing/`)
- Zero-padding operations (4 test cases)
- Feature copying accuracy (3 scenarios)
- Mel-spectrogram window extraction (3 frames)
- Audio buffer management
- Full batch integrity validation
- **Coverage**: 5 test functions, 3 benchmarks

### 3. Parallel Processing (`parallel-processing/`)
- Worker row distribution (4 test cases)
- Goroutine coordination (2 tests)
- Race condition detection (100 iterations)
- Parallel image processing validation
- Load balancing verification
- **Coverage**: 5 test functions, 3 benchmarks

### 4. Integration (`integration/`)
- Full pipeline flow (batch 1, 8, 25)
- Memory pooling behavior (100 iterations)
- Concurrent requests (10 simultaneous)
- Error recovery (3 scenarios)
- Cross-component interaction
- **Coverage**: 4 test suites, 1 benchmark

### 5. Performance (`performance/`)
- FPS throughput (batch 1: 60fps, 8: 25fps, 25: 47fps)
- Memory allocation patterns (3 operations)
- Parallel scaling (1-16 workers)
- Cache effectiveness (pooling vs non-pooling)
- Concurrent throughput (1-16 concurrency)
- **Coverage**: 5 test suites, 2 benchmarks

### 6. Edge Cases (`edgecases/`)
- Boundary conditions (5 tests)
- Numerical stability (7 float edge cases)
- Audio edge cases (4 scenarios)
- Bilinear interpolation edges (9 cases)
- Memory overflow conditions (2 tests)
- Concurrent access safety (2 tests)
- **Coverage**: 6 test suites

## Running Tests

### Quick Start
```powershell
# Run all tests with summary
cd functional-tests
.\run_all_tests.ps1
```

### Run All Tests
```powershell
# From project root
cd go-monolithic-server-refactored
go test -v ./functional-tests/...
```

### Run Specific Category
```powershell
go test -v ./functional-tests/image-processing
go test -v ./functional-tests/audio-processing
go test -v ./functional-tests/parallel-processing
go test -v ./functional-tests/integration
go test -v ./functional-tests/performance
go test -v ./functional-tests/edgecases
```

### Run Benchmarks
```powershell
# Automated benchmark runner
.\run_benchmarks.ps1

# Specific category
go test -bench=. -benchmem ./functional-tests/image-processing
go test -bench=. -benchmem ./functional-tests/performance
```

### Run with Race Detector
```powershell
# Critical for parallel tests
go test -race ./functional-tests/parallel-processing
go test -race ./functional-tests/integration
```

### Test Coverage
```powershell
go test -coverprofile=coverage.out ./functional-tests/...
go tool cover -html=coverage.out
```

## Expected Results

### Performance Targets
- 🎯 Batch 1: ≥60 FPS
- 🎯 Batch 8: ≥25 FPS (up from 23.1 baseline)
- 🎯 Batch 25: ≥47 FPS (up from 43.9 baseline)

### Optimization Metrics
- ⚡ BGR→RGBA: ~160μs (4.2x speedup)
- ⚡ Image resize: ~323μs (4.9x speedup)
- ⚡ Zero-padding: ~2μs (0 allocations)
- 📊 8-worker speedup: 4-6x sequential
- 💾 Memory pooling: 10-100x fewer allocations

### Test Success Criteria
- ✅ All image conversions pixel-perfect
- ✅ No data races with `-race`
- ✅ All 10 concurrent requests handled
- ✅ Error recovery for invalid inputs

## Test Structure

```
functional-tests/
├── image-processing/      # 10 tests, 2 benchmarks
├── audio-processing/      # 5 tests, 3 benchmarks  
├── parallel-processing/   # 5 tests, 3 benchmarks
├── integration/          # 4 suites, 1 benchmark
├── performance/          # 5 suites, 2 benchmarks
├── edgecases/            # 6 test suites
├── run_all_tests.ps1     # Automated test runner
└── run_benchmarks.ps1    # Automated benchmark runner
```
