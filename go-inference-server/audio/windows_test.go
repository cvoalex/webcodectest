package audio

import (
	"testing"
)

func TestExtractMelWindows(t *testing.T) {
	// Create a test mel-spectrogram [52, 80]
	melSpec := make([][]float32, 52)
	for i := 0; i < 52; i++ {
		melSpec[i] = make([]float32, 80)
		for j := 0; j < 80; j++ {
			melSpec[i][j] = float32(i*80 + j) // Unique value for each position
		}
	}

	// Extract windows
	windows, err := ExtractMelWindows(melSpec, 16, false)
	if err != nil {
		t.Fatalf("Failed to extract windows: %v", err)
	}

	// Should have 52 windows (one per frame)
	if len(windows) != 52 {
		t.Errorf("Expected 52 windows, got %d", len(windows))
	}

	// Check first window
	if len(windows[0]) != 80 {
		t.Errorf("Expected 80 mel bins in window, got %d", len(windows[0]))
	}
	if len(windows[0][0]) != 16 {
		t.Errorf("Expected 16 frames in window, got %d", len(windows[0][0]))
	}

	// Verify transposition: window[mel][frame] should equal melSpec[frame][mel]
	// First window uses frames 0-15
	for m := 0; m < 80; m++ {
		for f := 0; f < 16; f++ {
			expected := melSpec[f][m]
			actual := windows[0][m][f]
			if actual != expected {
				t.Errorf("Window[0][%d][%d] = %f, expected %f", m, f, actual, expected)
			}
		}
	}

	// Verify second window uses frames 1-16
	for m := 0; m < 80; m++ {
		for f := 0; f < 16; f++ {
			expected := melSpec[1+f][m]
			actual := windows[1][m][f]
			if actual != expected {
				t.Errorf("Window[1][%d][%d] = %f, expected %f", m, f, actual, expected)
			}
		}
	}

	t.Logf("✅ Extracted %d windows successfully", len(windows))
	t.Logf("   Window shape: [%d, %d]", len(windows[0]), len(windows[0][0]))
}

func TestExtractMelWindowsForBatch(t *testing.T) {
	// Create a test mel-spectrogram [52, 80]
	melSpec := make([][]float32, 52)
	for i := 0; i < 52; i++ {
		melSpec[i] = make([]float32, 80)
		for j := 0; j < 80; j++ {
			melSpec[i][j] = float32(i*80 + j)
		}
	}

	// Extract 24 windows starting from frame 0
	batchSize := 24
	windows, err := ExtractMelWindowsForBatch(melSpec, batchSize, 0)
	if err != nil {
		t.Fatalf("Failed to extract windows: %v", err)
	}

	if len(windows) != batchSize {
		t.Errorf("Expected %d windows, got %d", batchSize, len(windows))
	}

	// Verify shape
	if len(windows[0]) != 80 || len(windows[0][0]) != 16 {
		t.Errorf("Expected window shape [80, 16], got [%d, %d]", len(windows[0]), len(windows[0][0]))
	}

	t.Logf("✅ Extracted batch of %d windows", len(windows))
}

func TestCalculateExpectedFrames(t *testing.T) {
	// Test with typical 640ms audio @ 16kHz
	numSamples := 10240 // 640ms at 16kHz
	hopLength := 200
	windowSize := 800

	expectedFrames := CalculateExpectedFrames(numSamples, hopLength, windowSize)

	t.Logf("Audio samples: %d", numSamples)
	t.Logf("Expected frames: %d", expectedFrames)

	// Should produce ~52 frames
	if expectedFrames < 50 || expectedFrames > 55 {
		t.Errorf("Unexpected number of frames: %d (expected ~52)", expectedFrames)
	}
}

func TestValidateAudioLength(t *testing.T) {
	cfg := DefaultConfig()

	// Test valid audio length for batch size 24
	numSamples := 10240 // 640ms at 16kHz
	batchSize := 24

	err := ValidateAudioLength(numSamples, batchSize, cfg)
	if err != nil {
		t.Errorf("Valid audio length failed validation: %v", err)
	}

	// Test audio that's too short
	shortSamples := 1000
	err = ValidateAudioLength(shortSamples, batchSize, cfg)
	if err == nil {
		t.Error("Short audio should have failed validation")
	} else {
		t.Logf("✅ Correctly rejected short audio: %v", err)
	}
}

func TestEndToEndWindowExtraction(t *testing.T) {
	// Simulate real audio processing
	processor := NewProcessor(nil)

	// Generate 640ms of test audio
	sampleRate := 16000
	duration := 0.64
	numSamples := int(float64(sampleRate) * duration)
	audioSamples := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		audioSamples[i] = float32(0.5 * (1.0 + 0.5*2.0*3.14159*440.0*t))
	}

	// Process to mel-spectrogram
	melSpec, err := processor.ProcessAudio(audioSamples)
	if err != nil {
		t.Fatalf("Failed to process audio: %v", err)
	}

	t.Logf("Mel-spectrogram shape: [%d, %d]", len(melSpec), len(melSpec[0]))

	// Extract windows for a batch of 24
	batchSize := 24
	windows, err := ExtractMelWindowsForBatch(melSpec, batchSize, 0)
	if err != nil {
		t.Fatalf("Failed to extract windows: %v", err)
	}

	if len(windows) != batchSize {
		t.Errorf("Expected %d windows, got %d", batchSize, len(windows))
	}

	// Verify each window has correct shape [80, 16]
	for i, window := range windows {
		if len(window) != 80 {
			t.Errorf("Window %d has %d mel bins, expected 80", i, len(window))
		}
		if len(window[0]) != 16 {
			t.Errorf("Window %d has %d frames, expected 16", i, len(window[0]))
		}
	}

	t.Logf("✅ End-to-end test successful")
	t.Logf("   Audio: %d samples -> Mel: %d frames -> Windows: %d", numSamples, len(melSpec), len(windows))
}
