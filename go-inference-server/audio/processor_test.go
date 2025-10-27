package audio

import (
	"fmt"
	"testing"
)

func TestProcessorBasic(t *testing.T) {
	// Create processor with default config
	processor := NewProcessor(nil)

	// Generate test audio: 640ms at 16kHz = 10,240 samples
	sampleRate := 16000
	durationMs := 640
	numSamples := (sampleRate * durationMs) / 1000

	// Generate a simple sine wave (440 Hz - A4 note)
	testAudio := make([]float32, numSamples)
	frequency := 440.0
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		testAudio[i] = float32(0.5 * sinWave(2.0*3.14159*frequency*t))
	}

	// Process audio
	melSpec, err := processor.ProcessAudio(testAudio)
	if err != nil {
		t.Fatalf("ProcessAudio failed: %v", err)
	}

	// Verify output shape
	if len(melSpec) == 0 {
		t.Fatal("Empty mel-spectrogram")
	}

	numFrames := len(melSpec)
	numMelBins := len(melSpec[0])

	fmt.Printf("Mel-spectrogram shape: [%d, %d]\n", numFrames, numMelBins)

	// Expected: ~80 frames for 640ms audio with 200 hop length
	// (10240 - 800) / 200 + 1 = 48 frames
	expectedFrames := (numSamples-processor.config.WindowSize)/processor.config.HopLength + 1
	if numFrames < expectedFrames-2 || numFrames > expectedFrames+2 {
		t.Errorf("Unexpected number of frames: got %d, expected ~%d", numFrames, expectedFrames)
	}

	if numMelBins != processor.config.NumMelBins {
		t.Errorf("Unexpected mel bins: got %d, expected %d", numMelBins, processor.config.NumMelBins)
	}

	// Verify values are normalized [0, 1] range (approximately)
	for i := 0; i < numFrames; i++ {
		for j := 0; j < numMelBins; j++ {
			val := melSpec[i][j]
			if val < -0.5 || val > 1.5 {
				t.Errorf("Value out of expected range at [%d][%d]: %f", i, j, val)
			}
		}
	}

	t.Logf("✅ Mel-spectrogram generated: [%d, %d]", numFrames, numMelBins)
}

func TestExtractMelWindow(t *testing.T) {
	// Create a mock mel-spectrogram
	totalFrames := 50
	numMelBins := 80
	melSpec := make([][]float32, totalFrames)
	for i := 0; i < totalFrames; i++ {
		melSpec[i] = make([]float32, numMelBins)
		for j := 0; j < numMelBins; j++ {
			melSpec[i][j] = float32(i*numMelBins + j)
		}
	}

	// Extract 16-frame window starting at frame 10
	window, err := ExtractMelWindow(melSpec, 10, 16)
	if err != nil {
		t.Fatalf("ExtractMelWindow failed: %v", err)
	}

	if len(window) != 16 {
		t.Errorf("Wrong window size: got %d, expected 16", len(window))
	}

	if len(window[0]) != numMelBins {
		t.Errorf("Wrong mel bins: got %d, expected %d", len(window[0]), numMelBins)
	}

	// Verify first frame is frame 10
	if window[0][0] != float32(10*numMelBins+0) {
		t.Errorf("Wrong value at window start: got %f, expected %f", window[0][0], float32(10*numMelBins+0))
	}

	t.Logf("✅ Window extraction works correctly")
}

func TestTransposeForEncoder(t *testing.T) {
	// Create 16x80 mel window
	numFrames := 16
	numMelBins := 80
	melWindow := make([][]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		melWindow[i] = make([]float32, numMelBins)
		for j := 0; j < numMelBins; j++ {
			melWindow[i][j] = float32(i*100 + j)
		}
	}

	// Transpose to 80x16
	transposed := TransposeForEncoder(melWindow)

	if len(transposed) != numMelBins {
		t.Errorf("Wrong transposed rows: got %d, expected %d", len(transposed), numMelBins)
	}

	if len(transposed[0]) != numFrames {
		t.Errorf("Wrong transposed cols: got %d, expected %d", len(transposed[0]), numFrames)
	}

	// Verify transpose is correct
	if transposed[0][0] != melWindow[0][0] {
		t.Error("Transpose value mismatch at [0][0]")
	}

	if transposed[79][15] != melWindow[15][79] {
		t.Errorf("Transpose value mismatch at [79][15]: got %f, expected %f",
			transposed[79][15], melWindow[15][79])
	}

	t.Logf("✅ Transpose works correctly: [%d, %d] -> [%d, %d]",
		numFrames, numMelBins, numMelBins, numFrames)
}

func sinWave(x float64) float64 {
	// Simple sine approximation
	const pi = 3.14159265359
	// Normalize to [-pi, pi]
	for x > pi {
		x -= 2 * pi
	}
	for x < -pi {
		x += 2 * pi
	}

	// Taylor series approximation
	x2 := x * x
	x3 := x2 * x
	x5 := x3 * x2
	x7 := x5 * x2

	return x - x3/6.0 + x5/120.0 - x7/5040.0
}

func BenchmarkProcessAudio(b *testing.B) {
	processor := NewProcessor(nil)

	// 640ms at 16kHz
	numSamples := 10240
	testAudio := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		testAudio[i] = float32(i) / float32(numSamples)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = processor.ProcessAudio(testAudio)
	}
}
