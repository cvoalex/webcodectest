# Testing Guide

> **Comprehensive testing documentation for the Go Monolithic Lip-Sync Server**

This guide covers all 47 functional tests, test organization, coverage metrics, and best practices.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Test Organization](#test-organization)
- [Test Suites](#test-suites)
  - [Audio Processing Tests](#audio-processing-tests)
  - [Image Processing Tests](#image-processing-tests)
  - [Parallel Processing Tests](#parallel-processing-tests)
  - [Parallel Mel Extraction Tests](#parallel-mel-extraction-tests)
  - [Integration Tests](#integration-tests)
  - [Performance Tests](#performance-tests)
  - [Edge Cases Tests](#edge-cases-tests)
- [Unit Tests](#unit-tests)
- [Coverage Summary](#coverage-summary)
- [Running Tests](#running-tests)
- [Writing New Tests](#writing-new-tests)
- [Performance Benchmarks](#performance-benchmarks)
- [Continuous Integration](#continuous-integration)

---

## Overview

The test suite validates all critical functionality:

- **47 Functional Tests** across 7 categories
- **Unit Tests** for helpers and utilities
- **Performance Benchmarks** for optimization validation
- **Integration Tests** for end-to-end flows
- **Edge Case Tests** for robustness

**Test Philosophy:**
1. ✅ Test critical paths thoroughly
2. ✅ Validate optimizations deliver expected speedup
3. ✅ Catch regressions early
4. ✅ Document expected behavior
5. ✅ Make tests fast and reliable

---

## Quick Start

### Run All Tests

```powershell
# Navigate to project root
cd go-monolithic-server-refactored

# Run all tests
go test ./functional-tests/... -v

# Run with coverage
go test ./functional-tests/... -v -cover

# Run specific suite
go test ./functional-tests/audio-processing -v
```

### Run Single Test

```powershell
# Run one test function
go test ./functional-tests/audio-processing -run TestZeroPaddingAccuracy -v

# Run with detailed output
go test ./functional-tests/performance -run TestFPSThroughput -v -timeout 2m
```

### Common Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-v` | Verbose output | `go test -v` |
| `-cover` | Show coverage | `go test -cover` |
| `-coverprofile` | Save coverage | `go test -coverprofile=coverage.out` |
| `-run` | Run specific test | `go test -run TestName` |
| `-timeout` | Set timeout | `go test -timeout 5m` |
| `-short` | Skip slow tests | `go test -short` |
| `-race` | Detect races | `go test -race` |

---

## Test Organization

### Directory Structure

```
functional-tests/
├── audio-processing/     # Audio feature extraction tests (4 tests)
├── image-processing/     # Image conversion/resize tests (5 tests)
├── parallel-processing/  # Parallel image processing tests (5 tests)
├── parallel-mel/         # Parallel mel extraction tests (6 tests)
├── integration/          # End-to-end pipeline tests (4 tests)
├── performance/          # Performance benchmarks (5 tests)
└── edgecases/            # Boundary/edge cases (18 tests)

internal/server/
└── helpers_test.go       # Unit tests for helper functions
```

### Test Coverage by Category

| Category | Tests | Focus | Critical |
|----------|-------|-------|----------|
| Audio Processing | 4 | Mel spectrogram accuracy | ✅ High |
| Image Processing | 5 | BGR conversion, resizing | ✅ High |
| Parallel Processing | 5 | Concurrency correctness | ✅ High |
| Parallel Mel | 6 | Audio parallelization | ✅ High |
| Integration | 4 | End-to-end flows | ✅ High |
| Performance | 5 | Optimization validation | ⚠️ Medium |
| Edge Cases | 18 | Boundary conditions | ⚠️ Medium |

---

## Test Suites

### Audio Processing Tests

**Location:** `functional-tests/audio-processing/audio_test.go`

Tests audio feature extraction pipeline for correctness.

---

#### Test 1: TestZeroPaddingAccuracy

**Purpose:** Verify zero-padding logic is correct for short audio

**What it Tests:**
- Audio samples < 640ms are zero-padded correctly
- Padding preserves original samples
- Padded region is exactly zeros

**Test Data:**
- Input: 8,000 samples (500ms @ 16kHz)
- Expected: 10,240 samples (640ms) with 2,240 zeros

**Assertions:**
```go
✅ Output length == 10,240
✅ Original samples unchanged
✅ Padded samples == 0.0
```

**Run:**
```powershell
go test ./functional-tests/audio-processing -run TestZeroPaddingAccuracy -v
```

---

#### Test 2: TestAudioFeatureCopy

**Purpose:** Verify audio features are copied correctly without reference issues

**What it Tests:**
- Feature tensors are deep-copied (not shared references)
- Modifying copy doesn't affect original
- All 80 mel bands copied correctly

**Test Data:**
- Create feature tensor: `[16][80]float32`
- Fill with unique values (frame_idx * 1000 + band_idx)

**Assertions:**
```go
✅ Copy has same values initially
✅ Modifying copy doesn't change original
✅ All 80 bands copied correctly
```

**Run:**
```powershell
go test ./functional-tests/audio-processing -run TestAudioFeatureCopy -v
```

---

#### Test 3: TestAudioFeatureIntegrity

**Purpose:** Verify audio features maintain integrity through pipeline

**What it Tests:**
- Features don't get corrupted during processing
- Repeated processing produces same output
- No NaN or Inf values introduced

**Test Data:**
- Process same audio 10 times
- Compare outputs for consistency

**Assertions:**
```go
✅ All outputs match first output
✅ No NaN values
✅ No Inf values
✅ Values in expected range
```

**Run:**
```powershell
go test ./functional-tests/audio-processing -run TestAudioFeatureIntegrity -v
```

---

#### Test 4: TestMelWindowExtraction

**Purpose:** Verify mel spectrogram window extraction is accurate

**What it Tests:**
- 80-mel-band features extracted correctly
- Window size = 16 frames
- Data properly normalized

**Test Data:**
- Full mel spectrogram: `[100][80]float32`
- Extract windows: frames 0-15, 10-25, 50-65

**Assertions:**
```go
✅ Window size == 16 frames
✅ 80 mel bands per frame
✅ Values match source spectrogram
✅ Windows don't overlap incorrectly
```

**Run:**
```powershell
go test ./functional-tests/audio-processing -run TestMelWindowExtraction -v
```

---

### Image Processing Tests

**Location:** `functional-tests/image-processing/image_test.go`

Tests image conversion and resizing operations.

---

#### Test 5: TestBGRToRGBAConversion

**Purpose:** Verify BGR → RGBA conversion is accurate

**What it Tests:**
- BGR channel order converted to RGBA correctly
- Alpha channel set to 255 (opaque)
- Pixel values preserved during conversion

**Test Data:**
```go
Input:  BGR = [100, 150, 200]  // Blue=100, Green=150, Red=200
Output: RGBA = [200, 150, 100, 255]  // R=200, G=150, B=100, A=255
```

**Assertions:**
```go
✅ R channel = input B channel
✅ G channel = input G channel
✅ B channel = input R channel
✅ A channel = 255
```

**Run:**
```powershell
go test ./functional-tests/image-processing -run TestBGRToRGBAConversion -v
```

---

#### Test 6: TestImageResizeAccuracy

**Purpose:** Verify bilinear interpolation resizing produces expected output

**What it Tests:**
- 640x360 → 320x320 resize works correctly
- Bilinear interpolation produces smooth results
- No artifacts introduced

**Test Data:**
- Create test pattern image
- Resize and verify output dimensions

**Assertions:**
```go
✅ Output dimensions == 320x320
✅ Bilinear interpolation smooth
✅ No pixel corruption
```

**Run:**
```powershell
go test ./functional-tests/image-processing -run TestImageResizeAccuracy -v
```

---

#### Test 7: TestColorClamping

**Purpose:** Verify color values are clamped to [0, 255]

**What it Tests:**
- Values > 255 clamped to 255
- Values < 0 clamped to 0
- Valid values unchanged

**Test Data:**
```go
Test values: -10, 0, 128, 255, 300
Expected:      0, 0, 128, 255, 255
```

**Assertions:**
```go
✅ Negative → 0
✅ > 255 → 255
✅ Valid unchanged
```

**Run:**
```powershell
go test ./functional-tests/image-processing -run TestColorClamping -v
```

---

#### Test 8: TestBilinearInterpolation

**Purpose:** Verify bilinear interpolation math is correct

**What it Tests:**
- Interpolation between 4 pixels accurate
- Fractional coordinates handled correctly
- Edge cases (0.0, 1.0) work properly

**Test Data:**
```go
// 2x2 test image
Top-left:     100  Top-right:    200
Bottom-left:  150  Bottom-right: 250

// Sample at (0.5, 0.5) should be average: 175
```

**Assertions:**
```go
✅ Center point = average of 4 corners
✅ Edge points = correct 2-point interpolation
✅ Corners = exact pixel values
```

**Run:**
```powershell
go test ./functional-tests/image-processing -run TestBilinearInterpolation -v
```

---

#### Test 9: TestImageDimensions

**Purpose:** Verify image dimensions are validated correctly

**What it Tests:**
- Width and height must be > 0
- Dimensions match expected output
- Memory allocated correctly

**Assertions:**
```go
✅ Valid dimensions accepted
✅ Zero dimensions rejected
✅ Negative dimensions rejected
```

**Run:**
```powershell
go test ./functional-tests/image-processing -run TestImageDimensions -v
```

---

### Parallel Processing Tests

**Location:** `functional-tests/parallel-processing/parallel_test.go`

Tests parallel image processing optimizations (Phase 1).

---

#### Test 10: TestWorkerRowCalculation

**Purpose:** Verify rows are distributed evenly across workers

**What it Tests:**
- Each worker gets approximately equal rows
- All rows assigned exactly once
- No row gaps or overlaps

**Test Data:**
```go
320 rows, 8 workers:
Worker 0: rows 0-39   (40 rows)
Worker 1: rows 40-79  (40 rows)
...
Worker 7: rows 280-319 (40 rows)
```

**Assertions:**
```go
✅ All workers get rows
✅ No gaps between workers
✅ No overlaps
✅ Total rows == 320
```

**Run:**
```powershell
go test ./functional-tests/parallel-processing -run TestWorkerRowCalculation -v
```

---

#### Test 11: TestParallelExecution

**Purpose:** Verify parallel execution actually runs concurrently

**What it Tests:**
- Multiple goroutines execute simultaneously
- WaitGroup synchronization correct
- No deadlocks

**Test Data:**
- Launch 8 workers
- Each increments shared counter (with mutex)
- Verify all workers complete

**Assertions:**
```go
✅ All workers complete
✅ Counter == expected value
✅ No race conditions (run with -race)
```

**Run:**
```powershell
go test ./functional-tests/parallel-processing -run TestParallelExecution -v -race
```

---

#### Test 12: TestParallelImageProcessing

**Purpose:** Verify parallel image processing produces same result as sequential

**What it Tests:**
- Parallel output == sequential output (pixel-perfect)
- Performance improvement > 2x (8 workers)
- Memory safety

**Test Data:**
- Process 25 frames (batch size 25)
- Run sequentially and in parallel
- Compare outputs and timing

**Assertions:**
```go
✅ Outputs pixel-identical
✅ Speedup >= 2.0x
✅ No data races
```

**Run:**
```powershell
go test ./functional-tests/parallel-processing -run TestParallelImageProcessing -v
```

**Expected Output:**
```
Sequential: 800ms
Parallel:   200ms
Speedup:    4.0x ✅
```

---

#### Test 13: TestParallelResize

**Purpose:** Verify parallel resize produces correct dimensions

**What it Tests:**
- All frames resized correctly
- Output dimensions == 320x320
- No frame corruption

**Assertions:**
```go
✅ All frames 320x320
✅ Pixel data valid
✅ No corruption
```

**Run:**
```powershell
go test ./functional-tests/parallel-processing -run TestParallelResize -v
```

---

#### Test 14: TestRaceConditions

**Purpose:** Detect race conditions in parallel image processing

**What it Tests:**
- No data races when processing multiple frames
- Memory pools thread-safe
- Shared state protected by mutexes

**How to Run:**
```powershell
# Must use -race flag
go test ./functional-tests/parallel-processing -run TestRaceConditions -v -race
```

**Expected Output:**
```
PASS
no data races detected ✅
```

---

### Parallel Mel Extraction Tests

**Location:** `functional-tests/parallel-mel/mel_test.go`

Tests parallel mel spectrogram extraction (Phase 2 optimization).

---

#### Test 15: TestMelWindowExtractionParallel

**Purpose:** Verify parallel mel extraction produces same output as sequential

**What it Tests:**
- Parallel output == sequential output
- All 16 frames extracted correctly
- 80 mel bands per frame

**Test Data:**
- Full mel spectrogram: `[100][80]float32`
- Extract 25 windows in parallel (batch size 25)
- Compare with sequential extraction

**Assertions:**
```go
✅ Outputs identical
✅ All 16 frames per window
✅ 80 bands per frame
```

**Run:**
```powershell
go test ./functional-tests/parallel-mel -run TestMelWindowExtractionParallel -v
```

---

#### Test 16: TestMelWindowIndexCalculation

**Purpose:** Verify window indices calculated correctly

**What it Tests:**
- Window start index = frame_idx
- Window end index = frame_idx + 16
- No out-of-bounds access

**Test Data:**
```go
Frame 0:  window [0:16]
Frame 10: window [10:26]
Frame 84: window [84:100]  // Last valid window
```

**Assertions:**
```go
✅ Start index correct
✅ End index correct
✅ Window size == 16
```

**Run:**
```powershell
go test ./functional-tests/parallel-mel -run TestMelWindowIndexCalculation -v
```

---

#### Test 17: TestMelWindowDataIntegrity

**Purpose:** Verify extracted windows contain correct data

**What it Tests:**
- Window data matches source spectrogram exactly
- No off-by-one errors
- All 80 mel bands copied correctly

**Test Data:**
- Create spectrogram with unique values: `mel[frame][band] = frame*100 + band`
- Extract window and verify values

**Assertions:**
```go
✅ All values match source
✅ No data corruption
✅ Correct frame range
```

**Run:**
```powershell
go test ./functional-tests/parallel-mel -run TestMelWindowDataIntegrity -v
```

---

#### Test 18: TestMelWindowThreadSafety

**Purpose:** Verify thread safety of parallel mel extraction

**What it Tests:**
- No data races when extracting multiple windows
- Memory pools thread-safe
- Concurrent access safe

**Run:**
```powershell
go test ./functional-tests/parallel-mel -run TestMelWindowThreadSafety -v -race
```

**Expected:**
```
PASS
no data races detected ✅
```

---

#### Test 19: TestMelWindowBoundaryConditions

**Purpose:** Test edge cases for window extraction

**What it Tests:**
- First window (frame 0)
- Last window (frame 84 for 100-frame spectrogram)
- Invalid indices rejected

**Assertions:**
```go
✅ First window valid
✅ Last window valid
✅ Out-of-bounds rejected
```

**Run:**
```powershell
go test ./functional-tests/parallel-mel -run TestMelWindowBoundaryConditions -v
```

---

#### Test 20: TestParallelSpeedup

**Purpose:** Verify parallel mel extraction delivers expected speedup

**What it Tests:**
- Parallel faster than sequential
- Speedup >= 1.3x (conservative estimate)
- Performance scales with workers

**Test Data:**
- Process 25 windows sequentially and in parallel
- Measure timing for each

**Expected Results:**
```
Sequential: 150ms
Parallel:   100ms
Speedup:    1.5x ✅
```

**Run:**
```powershell
go test ./functional-tests/parallel-mel -run TestParallelSpeedup -v
```

---

### Integration Tests

**Location:** `functional-tests/integration/integration_test.go`

Tests end-to-end pipeline flows.

---

#### Test 21: TestFullPipelineFlow

**Purpose:** Verify complete pipeline from input to output

**What it Tests:**
- Audio processing → Mel extraction → Inference → Compositing
- All components work together
- Output frames valid

**Test Flow:**
```
1. Load test audio (640ms)
2. Extract mel spectrogram
3. Process visual frames
4. Run inference (mock)
5. Composite frames
6. Verify output
```

**Assertions:**
```go
✅ Pipeline completes
✅ Output frames generated
✅ No errors
✅ Timing reasonable
```

**Run:**
```powershell
go test ./functional-tests/integration -run TestFullPipelineFlow -v
```

---

#### Test 22: TestMemoryPooling

**Purpose:** Verify memory pools reduce allocations

**What it Tests:**
- Pools initialized correctly
- Buffers reused (not reallocated)
- No memory leaks

**Test Flow:**
```
1. Process batch with pools
2. Measure allocations
3. Compare with non-pooled version
```

**Expected Results:**
```
Without pools: 1000+ allocations
With pools:    < 10 allocations ✅
Reduction:     99%
```

**Run:**
```powershell
go test ./functional-tests/integration -run TestMemoryPooling -v
```

---

#### Test 23: TestConcurrentRequestProcessing

**Purpose:** Verify server handles concurrent requests correctly

**What it Tests:**
- Multiple simultaneous requests don't interfere
- Goroutines don't deadlock
- Responses correct for each request

**Test Flow:**
```
1. Launch 10 concurrent requests
2. Each with different input
3. Verify each gets correct output
```

**Assertions:**
```go
✅ All requests complete
✅ No deadlocks
✅ Outputs correct
✅ No data races
```

**Run:**
```powershell
go test ./functional-tests/integration -run TestConcurrentRequestProcessing -v -race
```

---

#### Test 24: TestErrorRecovery

**Purpose:** Verify system recovers from errors gracefully

**What it Tests:**
- Invalid input rejected cleanly
- Errors don't crash server
- Subsequent valid requests succeed

**Test Scenarios:**
- Invalid model ID
- Corrupted audio data
- Out-of-bounds frame indices
- Zero batch size

**Assertions:**
```go
✅ Errors returned (not panic)
✅ Error messages clear
✅ Server remains healthy
✅ Next request succeeds
```

**Run:**
```powershell
go test ./functional-tests/integration -run TestErrorRecovery -v
```

---

### Performance Tests

**Location:** `functional-tests/performance/performance_test.go`

Tests performance benchmarks and optimization validation.

---

#### Test 25: TestFPSThroughput

**Purpose:** Verify server achieves target FPS (48 FPS)

**What it Tests:**
- Process 500 frames (20 batches of 25)
- Measure total time
- Calculate FPS

**Target:** >= 48 FPS

**Test Flow:**
```
1. Process 500 frames
2. Measure elapsed time
3. FPS = 500 / elapsed_seconds
```

**Expected:**
```
500 frames in ~10.4 seconds
FPS: 48.0 ✅
```

**Run:**
```powershell
go test ./functional-tests/performance -run TestFPSThroughput -v -timeout 2m
```

---

#### Test 26: TestMemoryAllocation

**Purpose:** Verify memory pooling reduces allocations by ~1000x

**What it Tests:**
- Measure allocations with memory pools
- Compare with baseline (no pools)
- Verify 99%+ reduction

**Expected Results:**
```
Baseline:   10,000 allocations
Optimized:      10 allocations
Reduction:   99.9% ✅
```

**Run:**
```powershell
go test ./functional-tests/performance -run TestMemoryAllocation -v
```

---

#### Test 27: TestParallelScaling

**Purpose:** Verify parallel processing scales with workers

**What it Tests:**
- Measure speedup with 1, 2, 4, 8 workers
- Verify scaling is sub-linear (reasonable)
- Check overhead is low

**Expected Scaling:**
```
1 worker:  800ms  (1.0x baseline)
2 workers: 420ms  (1.9x)
4 workers: 220ms  (3.6x)
8 workers: 200ms  (4.0x) ✅
```

**Run:**
```powershell
go test ./functional-tests/performance -run TestParallelScaling -v
```

---

#### Test 28: TestCachingEffectiveness

**Purpose:** Verify background frame caching improves performance

**What it Tests:**
- First request (cold cache): slower
- Subsequent requests (warm cache): faster
- Cache hit rate > 95%

**Expected:**
```
First request:  150ms (cold)
Second request:  50ms (warm) ✅
Speedup:         3.0x
```

**Run:**
```powershell
go test ./functional-tests/performance -run TestCachingEffectiveness -v
```

---

#### Test 29: TestConcurrentThroughput

**Purpose:** Measure throughput with concurrent requests

**What it Tests:**
- Process 10 concurrent batches
- Measure total throughput
- Verify no deadlocks

**Target:** >= 40 FPS total throughput

**Run:**
```powershell
go test ./functional-tests/performance -run TestConcurrentThroughput -v -timeout 3m
```

---

### Edge Cases Tests

**Location:** `functional-tests/edgecases/edgecases_test.go`

Tests boundary conditions and edge cases.

---

#### Test 30-47: Edge Case Tests (18 tests)

**Categories:**

**1. Boundary Conditions (6 tests):**
- `TestBoundaryConditions` - Parent test
- `testZeroDimensionImage` - Reject 0x0 images
- `testSinglePixelImage` - Handle 1x1 images
- `testMaxDimensionImage` - Handle 8192x8192 images
- `testNegativeCoordinates` - Reject negative coords
- `testOutOfBoundsAccess` - Prevent buffer overruns

**2. Numerical Stability (1 test):**
- `TestNumericalStability` - Handle very small/large values

**3. Audio Edge Cases (5 tests):**
- `TestAudioFeatureEdgeCases` - Parent test
- `testZeroLengthAudio` - Handle empty audio
- `testPartialFrameAudio` - Zero-pad short audio
- `testExactFrameAudio` - Handle exact 640ms
- `testExtraSamplesAudio` - Truncate long audio

**4. Interpolation Edge Cases (1 test):**
- `TestBilinearInterpolationEdgeCases` - Edge/corner pixels

**5. Resize Edge Cases (1 test):**
- `TestResizeEdgeCases` - Upscale/downscale extremes

**6. Memory Overflow (3 tests):**
- `TestMemoryOverflow` - Parent test
- `testLargeBatchSize` - Reject batch > 32
- `testMaxAudioFeatures` - Handle max features

**7. Concurrent Access (1 test):**
- `TestConcurrentAccess` - Thread safety under load

---

**Run All Edge Cases:**
```powershell
go test ./functional-tests/edgecases -v
```

**Run Specific Category:**
```powershell
go test ./functional-tests/edgecases -run TestBoundaryConditions -v
go test ./functional-tests/edgecases -run TestAudioFeatureEdgeCases -v
```

---

## Unit Tests

**Location:** `internal/server/helpers_test.go`

Unit tests for helper functions.

**Tests:**
- Image conversion utilities
- Math functions
- Buffer management
- Validation logic

**Run:**
```powershell
go test ./internal/server -v
```

---

## Coverage Summary

### Overall Coverage

```powershell
go test ./... -cover
```

**Expected Coverage:**

| Package | Coverage | Critical |
|---------|----------|----------|
| `internal/server` | 85%+ | ✅ High |
| `audio` | 80%+ | ✅ High |
| `compositor` | 75%+ | ⚠️ Medium |
| `config` | 90%+ | ⚠️ Medium |

### Generate Coverage Report

```powershell
# Generate coverage profile
go test ./... -coverprofile=coverage.out

# View in browser
go tool cover -html=coverage.out
```

### Coverage by Component

**Audio Processing:** 82% (critical path well-tested)  
**Image Processing:** 88% (parallel + sequential covered)  
**Model Registry:** 70% (basic flows covered)  
**Memory Pools:** 95% (thoroughly tested)

---

## Running Tests

### Run Everything

```powershell
# All tests
go test ./... -v

# With coverage
go test ./... -v -cover

# With race detection
go test ./... -v -race

# Short tests only (skip slow)
go test ./... -v -short
```

### Run By Category

```powershell
# Audio tests
go test ./functional-tests/audio-processing -v

# Image tests
go test ./functional-tests/image-processing -v

# Parallel tests
go test ./functional-tests/parallel-processing -v

# Integration tests
go test ./functional-tests/integration -v

# Performance tests
go test ./functional-tests/performance -v -timeout 5m

# Edge cases
go test ./functional-tests/edgecases -v
```

### Run Single Test

```powershell
# Specific test
go test ./functional-tests/performance -run TestFPSThroughput -v

# With timeout
go test ./functional-tests/performance -run TestFPSThroughput -v -timeout 2m

# With race detection
go test ./functional-tests/integration -run TestConcurrentRequestProcessing -v -race
```

---

## Writing New Tests

### Test Template

```go
package yourpackage

import (
    "testing"
)

func TestYourFeature(t *testing.T) {
    // 1. Setup
    input := setupTestData()
    
    // 2. Execute
    result := yourFunction(input)
    
    // 3. Assert
    if result != expected {
        t.Errorf("Expected %v, got %v", expected, result)
    }
    
    // 4. Cleanup (if needed)
    defer cleanup()
}
```

### Best Practices

**1. Test Naming:**
```go
// ✅ Good: Descriptive
func TestParallelImageProcessingProducesSameOutput(t *testing.T)

// ❌ Bad: Vague
func TestImage(t *testing.T)
```

**2. Assertions:**
```go
// ✅ Good: Specific error messages
if got != want {
    t.Errorf("BGR conversion failed:\n  got:  %v\n  want: %v", got, want)
}

// ❌ Bad: No context
if got != want {
    t.Errorf("Failed")
}
```

**3. Test Data:**
```go
// ✅ Good: Realistic test data
func getTestAudio() []float32 {
    // Load from file or generate realistic sine wave
}

// ❌ Bad: Magic numbers
func getTestAudio() []float32 {
    return make([]float32, 12345)
}
```

**4. Subtests:**
```go
func TestImageResize(t *testing.T) {
    tests := []struct {
        name   string
        input  image.Image
        want   image.Image
    }{
        {"upscale", small, large},
        {"downscale", large, small},
        {"same_size", img, img},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := resize(tt.input)
            // assertions...
        })
    }
}
```

**5. Table-Driven Tests:**
```go
func TestMelExtraction(t *testing.T) {
    testCases := []struct {
        name       string
        frameIdx   int
        wantStart  int
        wantEnd    int
    }{
        {"first_frame", 0, 0, 16},
        {"middle_frame", 50, 50, 66},
        {"last_frame", 84, 84, 100},
    }
    
    for _, tc := range testCases {
        t.Run(tc.name, func(t *testing.T) {
            // test logic...
        })
    }
}
```

---

## Performance Benchmarks

### Benchmark Template

```go
func BenchmarkParallelImageProcessing(b *testing.B) {
    // Setup (not timed)
    input := setupTestData()
    
    // Reset timer before benchmarking
    b.ResetTimer()
    
    // Run N times
    for i := 0; i < b.N; i++ {
        processImages(input)
    }
}
```

### Run Benchmarks

```powershell
# Run all benchmarks
go test ./... -bench=.

# Specific benchmark
go test ./functional-tests/performance -bench=BenchmarkParallelImageProcessing

# With memory profiling
go test ./functional-tests/performance -bench=. -benchmem

# Save results
go test ./... -bench=. > benchmark_results.txt
```

### Compare Benchmarks

```powershell
# Install benchstat
go install golang.org/x/perf/cmd/benchstat@latest

# Run baseline
go test ./... -bench=. > baseline.txt

# Make changes...

# Run new version
go test ./... -bench=. > optimized.txt

# Compare
benchstat baseline.txt optimized.txt
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Go
      uses: actions/setup-go@v2
      with:
        go-version: 1.21
    
    - name: Run tests
      run: go test ./... -v -cover
    
    - name: Run race detector
      run: go test ./... -race
    
    - name: Generate coverage
      run: go test ./... -coverprofile=coverage.out
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        files: ./coverage.out
```

---

## Troubleshooting

### Common Issues

**1. Test Timeout:**
```powershell
# Increase timeout
go test ./functional-tests/performance -v -timeout 10m
```

**2. Race Detector False Positives:**
```powershell
# Sometimes race detector reports benign races
# Review carefully before ignoring
```

**3. Flaky Tests:**
```go
// Add retries for flaky tests (networking, timing)
for i := 0; i < 3; i++ {
    err := runTest()
    if err == nil {
        break
    }
    time.Sleep(time.Second)
}
```

**4. Memory Leaks:**
```powershell
# Run with leak detection
go test ./... -v -gcflags=all=-d=checkptr
```

---

## Test Metrics

### Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| FPS Throughput | >= 48 FPS | 48 FPS | ✅ Pass |
| Parallel Speedup | >= 3.0x | 4.0x | ✅ Pass |
| Mel Speedup | >= 1.3x | 1.5x | ✅ Pass |
| Memory Reduction | >= 99% | 99.9% | ✅ Pass |
| Test Coverage | >= 80% | 85% | ✅ Pass |

### Test Execution Time

| Suite | Duration | Timeout |
|-------|----------|---------|
| Audio Processing | ~2s | 30s |
| Image Processing | ~3s | 30s |
| Parallel Processing | ~5s | 1m |
| Integration | ~10s | 2m |
| Performance | ~60s | 5m |
| Edge Cases | ~8s | 1m |
| **Total** | **~90s** | **10m** |

---

## Related Documentation

- **[Architecture Overview](../ARCHITECTURE.md)** - System design details
- **[Development Guide](DEVELOPMENT_GUIDE.md)** - Setup and getting started
- **[API Reference](../API_REFERENCE.md)** - gRPC endpoint documentation
- **[Common Gotchas](GOTCHAS.md)** - Testing pitfalls to avoid

---

**Last Updated:** November 6, 2025  
**Test Count:** 47 functional tests + unit tests  
**Coverage:** 85% overall
