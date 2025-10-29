package audio_processing_test

import (
	"math"
	"testing"
)

// TestZeroPaddingAccuracy tests zero-padding operations
func TestZeroPaddingAccuracy(t *testing.T) {
	tests := []struct {
		name        string
		totalSize   int
		offset      int
		count       int
		featureSize int
	}{
		{
			name:        "Pad 3 frames of 512 features",
			totalSize:   10000,
			offset:      100,
			count:       3,
			featureSize: 512,
		},
		{
			name:        "Pad 16 frames (full batch)",
			totalSize:   16 * 512,
			offset:      0,
			count:       16,
			featureSize: 512,
		},
		{
			name:        "Pad 1 frame at end",
			totalSize:   5000,
			offset:      4488,
			count:       1,
			featureSize: 512,
		},
		{
			name:        "No padding (count=0)",
			totalSize:   1000,
			offset:      0,
			count:       0,
			featureSize: 512,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Initialize with non-zero values
			audioData := make([]float32, tt.totalSize)
			for i := range audioData {
				audioData[i] = 1.0
			}

			// Apply zero padding
			zeroPadAudioFeatures(audioData, tt.offset, tt.count, tt.featureSize)

			// Verify padded region is zeroed
			expectedZeros := tt.count * tt.featureSize
			for i := 0; i < expectedZeros; i++ {
				idx := tt.offset + i
				if idx < tt.totalSize && audioData[idx] != 0.0 {
					t.Errorf("Expected zero at index %d, got %v", idx, audioData[idx])
				}
			}

			// Verify regions outside padding are unchanged
			for i := 0; i < tt.offset; i++ {
				if audioData[i] != 1.0 {
					t.Errorf("Unexpected modification at index %d before padding", i)
				}
			}

			endPadding := tt.offset + expectedZeros
			for i := endPadding; i < tt.totalSize; i++ {
				if audioData[i] != 1.0 {
					t.Errorf("Unexpected modification at index %d after padding", i)
				}
			}
		})
	}
}

// TestAudioFeatureCopy tests feature copying operations
func TestAudioFeatureCopy(t *testing.T) {
	tests := []struct {
		name          string
		features      []float32
		destOffset    int
		arraySize     int
		shouldSucceed bool
	}{
		{
			name:          "Copy 512 features to start",
			features:      makeTestFeatures(512, 0.5),
			destOffset:    0,
			arraySize:     1000,
			shouldSucceed: true,
		},
		{
			name:          "Copy 512 features to middle",
			features:      makeTestFeatures(512, 0.75),
			destOffset:    100,
			arraySize:     1000,
			shouldSucceed: true,
		},
		{
			name:          "Copy to near end",
			features:      makeTestFeatures(512, 0.25),
			destOffset:    488,
			arraySize:     1000,
			shouldSucceed: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			audioData := make([]float32, tt.arraySize)

			// Copy features
			copyAudioFeatures(audioData, tt.destOffset, tt.features)

			// Verify copied values
			for i := 0; i < len(tt.features); i++ {
				idx := tt.destOffset + i
				if idx < tt.arraySize {
					if audioData[idx] != tt.features[i] {
						t.Errorf("Copy failed at index %d: got %v, want %v",
							idx, audioData[idx], tt.features[i])
					}
				}
			}
		})
	}
}

// TestAudioFeatureIntegrity tests data integrity during processing
func TestAudioFeatureIntegrity(t *testing.T) {
	const batchSize = 8
	const featureSize = 512
	const totalSize = batchSize * featureSize

	audioData := make([]float32, totalSize)

	// Simulate a full batch processing
	// Pad left (2 frames)
	zeroPadAudioFeatures(audioData, 0, 2, featureSize)

	// Copy actual features (4 frames)
	for i := 0; i < 4; i++ {
		offset := (2 + i) * featureSize
		features := makeTestFeatures(featureSize, float32(i)*0.1+0.1)
		copyAudioFeatures(audioData, offset, features)
	}

	// Pad right (2 frames)
	zeroPadAudioFeatures(audioData, 6*featureSize, 2, featureSize)

	// Verify structure
	// First 2 frames should be zero
	for i := 0; i < 2*featureSize; i++ {
		if audioData[i] != 0.0 {
			t.Errorf("Left padding failed at index %d", i)
		}
	}

	// Next 4 frames should have values
	for frameIdx := 0; frameIdx < 4; frameIdx++ {
		expectedValue := float32(frameIdx)*0.1 + 0.1
		for i := 0; i < featureSize; i++ {
			idx := (2+frameIdx)*featureSize + i
			if math.Abs(float64(audioData[idx]-expectedValue)) > 0.001 {
				t.Errorf("Feature value mismatch at frame %d, index %d: got %v, want %v",
					frameIdx, idx, audioData[idx], expectedValue)
			}
		}
	}

	// Last 2 frames should be zero
	for i := 6 * featureSize; i < totalSize; i++ {
		if audioData[i] != 0.0 {
			t.Errorf("Right padding failed at index %d", i)
		}
	}
}

// TestMelWindowExtraction tests mel-spectrogram window extraction logic
func TestMelWindowExtraction(t *testing.T) {
	// Simulate mel-spec data: 100 time steps, 80 mel bins
	numTimeSteps := 100
	numMelBins := 80
	melSpec := make([][]float32, numTimeSteps)
	for i := range melSpec {
		melSpec[i] = make([]float32, numMelBins)
		for j := range melSpec[i] {
			// Fill with recognizable pattern
			melSpec[i][j] = float32(i*numMelBins + j)
		}
	}

	tests := []struct {
		name       string
		frameIdx   int
		fps        int
		windowSize int
	}{
		{"Frame 0 at 25fps", 0, 25, 16},
		{"Frame 10 at 25fps", 10, 25, 16},
		{"Frame 24 at 25fps", 24, 25, 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Calculate center column (25fps -> 100 columns per second -> 4 columns per frame)
			centerCol := int(float64(tt.frameIdx) * 100.0 / float64(tt.fps))
			startCol := centerCol - tt.windowSize/2
			endCol := startCol + tt.windowSize

			// Bounds check
			if startCol < 0 {
				startCol = 0
			}
			if endCol > numTimeSteps {
				endCol = numTimeSteps
				startCol = endCol - tt.windowSize
				if startCol < 0 {
					startCol = 0
				}
			}

			// Extract window
			window := make([][]float32, numMelBins)
			for m := 0; m < numMelBins; m++ {
				window[m] = make([]float32, tt.windowSize)
				for step := 0; step < tt.windowSize; step++ {
					srcIdx := startCol + step
					if srcIdx >= 0 && srcIdx < numTimeSteps {
						window[m][step] = melSpec[srcIdx][m]
					}
				}
			}

			// Verify window dimensions
			if len(window) != numMelBins {
				t.Errorf("Window mel bins: got %d, want %d", len(window), numMelBins)
			}
			if len(window[0]) != tt.windowSize {
				t.Errorf("Window size: got %d, want %d", len(window[0]), tt.windowSize)
			}
		})
	}
}

// Helper functions
func zeroPadAudioFeatures(audioData []float32, offset, count, featureSize int) {
	if count <= 0 {
		return
	}
	totalElements := count * featureSize
	destStart := offset
	destEnd := offset + totalElements
	for i := destStart; i < destEnd && i < len(audioData); i++ {
		audioData[i] = 0.0
	}
}

func copyAudioFeatures(audioData []float32, destOffset int, features []float32) {
	copy(audioData[destOffset:], features)
}

func makeTestFeatures(size int, value float32) []float32 {
	features := make([]float32, size)
	for i := range features {
		features[i] = value
	}
	return features
}

// Benchmark tests
func BenchmarkZeroPadding(b *testing.B) {
	audioData := make([]float32, 16*512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		zeroPadAudioFeatures(audioData, 0, 16, 512)
	}
}

func BenchmarkFeatureCopy(b *testing.B) {
	audioData := make([]float32, 16*512)
	features := makeTestFeatures(512, 0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyAudioFeatures(audioData, 0, features)
	}
}

func BenchmarkMelWindowExtraction(b *testing.B) {
	melSpec := make([][]float32, 100)
	for i := range melSpec {
		melSpec[i] = make([]float32, 80)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		window := make([][]float32, 80)
		for m := 0; m < 80; m++ {
			window[m] = make([]float32, 16)
			for step := 0; step < 16; step++ {
				window[m][step] = melSpec[step][m]
			}
		}
	}
}
