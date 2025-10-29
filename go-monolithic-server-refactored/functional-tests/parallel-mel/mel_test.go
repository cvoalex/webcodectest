package parallel_mel_test

import (
	"math"
	"sync"
	"testing"
	"time"
)

// TestMelWindowExtractionParallel tests parallel mel window extraction
func TestMelWindowExtractionParallel(t *testing.T) {
	tests := []struct {
		name           string
		numMelFrames   int
		numVideoFrames int
		fps            int
	}{
		{
			name:           "Small batch (8 frames)",
			numMelFrames:   100,
			numVideoFrames: 8,
			fps:            25,
		},
		{
			name:           "Standard batch (25 frames)",
			numMelFrames:   100,
			numVideoFrames: 25,
			fps:            25,
		},
		{
			name:           "Large batch (40 frames)",
			numMelFrames:   160,
			numVideoFrames: 40,
			fps:            25,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create test mel-spectrogram
			melSpec := createTestMelSpec(tt.numMelFrames, 80)

			// Extract windows in parallel
			allMelWindows := make([][][]float32, tt.numVideoFrames)
			extractMelWindowsParallel(melSpec, tt.numMelFrames, tt.numVideoFrames, allMelWindows, false)

			// Verify all windows extracted
			if len(allMelWindows) != tt.numVideoFrames {
				t.Errorf("Expected %d windows, got %d", tt.numVideoFrames, len(allMelWindows))
			}

			// Verify window dimensions
			for i, window := range allMelWindows {
				if window == nil {
					t.Errorf("Window %d is nil", i)
					continue
				}
				if len(window) != 80 {
					t.Errorf("Window %d: expected 80 mel bins, got %d", i, len(window))
				}
				if len(window[0]) != 16 {
					t.Errorf("Window %d: expected 16 time steps, got %d", i, len(window[0]))
				}
			}
		})
	}
}

// TestMelWindowIndexCalculation tests mel window index calculation
func TestMelWindowIndexCalculation(t *testing.T) {
	tests := []struct {
		frameIdx      int
		numMelFrames  int
		expectedStart int
		expectedEnd   int
	}{
		{frameIdx: 0, numMelFrames: 100, expectedStart: 0, expectedEnd: 16},
		{frameIdx: 1, numMelFrames: 100, expectedStart: 3, expectedEnd: 19},
		{frameIdx: 10, numMelFrames: 100, expectedStart: 32, expectedEnd: 48},
		{frameIdx: 24, numMelFrames: 100, expectedStart: 76, expectedEnd: 92},
		{frameIdx: 30, numMelFrames: 100, expectedStart: 84, expectedEnd: 100}, // Near end
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			startIdx := int(float64(80*tt.frameIdx) / 25.0)
			endIdx := startIdx + 16

			// Boundary check (same as implementation)
			if endIdx > tt.numMelFrames {
				endIdx = tt.numMelFrames
				startIdx = endIdx - 16
			}
			if startIdx < 0 {
				startIdx = 0
			}

			if startIdx != tt.expectedStart {
				t.Errorf("Frame %d: expected start %d, got %d", tt.frameIdx, tt.expectedStart, startIdx)
			}
			if endIdx != tt.expectedEnd {
				t.Errorf("Frame %d: expected end %d, got %d", tt.frameIdx, tt.expectedEnd, endIdx)
			}
		})
	}
}

// TestMelWindowDataIntegrity tests data integrity of extracted windows
func TestMelWindowDataIntegrity(t *testing.T) {
	numMelFrames := 100
	numVideoFrames := 25

	// Create mel-spec with recognizable pattern
	melSpec := make([][]float32, numMelFrames)
	for i := range melSpec {
		melSpec[i] = make([]float32, 80)
		for j := range melSpec[i] {
			melSpec[i][j] = float32(i*80 + j)
		}
	}

	// Extract windows
	allMelWindows := make([][][]float32, numVideoFrames)
	extractMelWindowsParallel(melSpec, numMelFrames, numVideoFrames, allMelWindows, false)

	// Verify frame 0 window
	frame0Window := allMelWindows[0]
	startIdx := 0
	for step := 0; step < 16; step++ {
		srcIdx := startIdx + step
		for m := 0; m < 80; m++ {
			expected := melSpec[srcIdx][m]
			actual := frame0Window[m][step]
			if actual != expected {
				t.Errorf("Frame 0, mel %d, step %d: expected %v, got %v", m, step, expected, actual)
				return
			}
		}
	}

	// Verify frame 10 window
	frame10Window := allMelWindows[10]
	startIdx = int(float64(80*10) / 25.0) // Should be 32
	for step := 0; step < 16; step++ {
		srcIdx := startIdx + step
		for m := 0; m < 80; m++ {
			expected := melSpec[srcIdx][m]
			actual := frame10Window[m][step]
			if actual != expected {
				t.Errorf("Frame 10, mel %d, step %d: expected %v, got %v", m, step, expected, actual)
				return
			}
		}
	}
}

// TestMelWindowThreadSafety tests thread safety of parallel extraction
func TestMelWindowThreadSafety(t *testing.T) {
	const iterations = 100

	for iter := 0; iter < iterations; iter++ {
		numMelFrames := 100
		numVideoFrames := 25

		melSpec := createTestMelSpec(numMelFrames, 80)
		allMelWindows := make([][][]float32, numVideoFrames)

		// Extract in parallel
		extractMelWindowsParallel(melSpec, numMelFrames, numVideoFrames, allMelWindows, false)

		// Verify no nil windows
		for i, window := range allMelWindows {
			if window == nil {
				t.Errorf("Iteration %d: Window %d is nil", iter, i)
			}
		}
	}
}

// TestMelWindowBoundaryConditions tests edge cases
func TestMelWindowBoundaryConditions(t *testing.T) {
	tests := []struct {
		name           string
		numMelFrames   int
		numVideoFrames int
	}{
		{"Exact fit", 100, 25},
		{"Mel frames less than needed", 50, 25},
		{"Single video frame", 100, 1},
		{"Minimum mel frames", 16, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			melSpec := createTestMelSpec(tt.numMelFrames, 80)
			allMelWindows := make([][][]float32, tt.numVideoFrames)

			extractMelWindowsParallel(melSpec, tt.numMelFrames, tt.numVideoFrames, allMelWindows, false)

			// Verify all windows extracted
			for i, window := range allMelWindows {
				if window == nil {
					t.Errorf("Window %d is nil", i)
				}
			}
		})
	}
}

// TestParallelSpeedup benchmarks parallel vs sequential extraction
func TestParallelSpeedup(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping speedup test in short mode")
	}

	numMelFrames := 100
	numVideoFrames := 40

	melSpec := createTestMelSpec(numMelFrames, 80)

	// Sequential extraction
	t.Run("Sequential", func(t *testing.T) {
		startTime := time.Now()
		for i := 0; i < 10; i++ {
			allMelWindows := make([][][]float32, numVideoFrames)
			extractMelWindowsSequential(melSpec, numMelFrames, numVideoFrames, allMelWindows)
		}
		seqTime := time.Since(startTime)
		t.Logf("Sequential: %v (%.0f μs per iteration)", seqTime, float64(seqTime.Microseconds())/10)
	})

	// Parallel extraction
	t.Run("Parallel", func(t *testing.T) {
		startTime := time.Now()
		for i := 0; i < 10; i++ {
			allMelWindows := make([][][]float32, numVideoFrames)
			extractMelWindowsParallel(melSpec, numMelFrames, numVideoFrames, allMelWindows, false)
		}
		parTime := time.Since(startTime)
		t.Logf("Parallel: %v (%.0f μs per iteration)", parTime, float64(parTime.Microseconds())/10)
	})
}

// Helper functions
func createTestMelSpec(numFrames, numMelBins int) [][]float32 {
	melSpec := make([][]float32, numFrames)
	for i := range melSpec {
		melSpec[i] = make([]float32, numMelBins)
		for j := range melSpec[i] {
			melSpec[i][j] = float32(math.Sin(float64(i+j) * 0.1))
		}
	}
	return melSpec
}

var melWindowPool = sync.Pool{
	New: func() interface{} {
		window := make([][]float32, 80)
		for i := range window {
			window[i] = make([]float32, 16)
		}
		return window
	},
}

func extractMelWindowsParallel(melSpec [][]float32, numMelFrames, numVideoFrames int, allMelWindows [][][]float32, saveDebugFiles bool) {
	const numWorkers = 8
	var wg sync.WaitGroup

	framesPerWorker := numVideoFrames / numWorkers
	extraFrames := numVideoFrames % numWorkers

	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			startFrame := id * framesPerWorker
			if id < extraFrames {
				startFrame += id
			} else {
				startFrame += extraFrames
			}

			endFrame := startFrame + framesPerWorker
			if id < extraFrames {
				endFrame++
			}

			for frameIdx := startFrame; frameIdx < endFrame; frameIdx++ {
				startIdx := int(float64(80*frameIdx) / 25.0)
				endIdx := startIdx + 16

				if endIdx > numMelFrames {
					endIdx = numMelFrames
					startIdx = endIdx - 16
				}
				if startIdx < 0 {
					startIdx = 0
				}

				window := melWindowPool.Get().([][]float32)

				for step := 0; step < 16; step++ {
					srcIdx := startIdx + step
					if srcIdx >= numMelFrames {
						srcIdx = numMelFrames - 1
					}
					for m := 0; m < 80; m++ {
						window[m][step] = melSpec[srcIdx][m]
					}
				}

				allMelWindows[frameIdx] = window
			}
		}(workerID)
	}

	wg.Wait()
}

func extractMelWindowsSequential(melSpec [][]float32, numMelFrames, numVideoFrames int, allMelWindows [][][]float32) {
	for frameIdx := 0; frameIdx < numVideoFrames; frameIdx++ {
		startIdx := int(float64(80*frameIdx) / 25.0)
		endIdx := startIdx + 16

		if endIdx > numMelFrames {
			endIdx = numMelFrames
			startIdx = endIdx - 16
		}
		if startIdx < 0 {
			startIdx = 0
		}

		window := melWindowPool.Get().([][]float32)

		for step := 0; step < 16; step++ {
			srcIdx := startIdx + step
			if srcIdx >= numMelFrames {
				srcIdx = numMelFrames - 1
			}
			for m := 0; m < 80; m++ {
				window[m][step] = melSpec[srcIdx][m]
			}
		}

		allMelWindows[frameIdx] = window
	}
}

// Benchmark tests
func BenchmarkMelWindowExtraction(b *testing.B) {
	numMelFrames := 100
	numVideoFrames := 40
	melSpec := createTestMelSpec(numMelFrames, 80)

	b.Run("Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			allMelWindows := make([][][]float32, numVideoFrames)
			extractMelWindowsSequential(melSpec, numMelFrames, numVideoFrames, allMelWindows)
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			allMelWindows := make([][][]float32, numVideoFrames)
			extractMelWindowsParallel(melSpec, numMelFrames, numVideoFrames, allMelWindows, false)
		}
	})
}
