package audio

import (
	"testing"
)

const testLibraryPath = "C:/onnxruntime-1.22.0/lib/onnxruntime.dll"

func TestAudioEncoderLoad(t *testing.T) {
	encoder, err := NewAudioEncoder(testLibraryPath)
	if err != nil {
		t.Skipf("Skipping audio encoder test - ONNX model not found: %v", err)
		return
	}
	defer encoder.Close()

	t.Logf("✅ Audio encoder loaded successfully")
	t.Logf("   Input shape: %v", encoder.inputShape)
	t.Logf("   Output shape: %v", encoder.outputShape)
}

func TestAudioEncoderInference(t *testing.T) {
	encoder, err := NewAudioEncoder(testLibraryPath)
	if err != nil {
		t.Skipf("Skipping audio encoder test - ONNX model not found: %v", err)
		return
	}
	defer encoder.Close()

	// Create a test mel-spectrogram window [80, 16]
	testMelWindow := make([][]float32, 80)
	for i := 0; i < 80; i++ {
		testMelWindow[i] = make([]float32, 16)
		for j := 0; j < 16; j++ {
			// Fill with normalized values in range [-4, +4]
			testMelWindow[i][j] = -2.0 + float32(i+j)*0.05
		}
	}

	// Run inference
	features, err := encoder.Encode(testMelWindow)
	if err != nil {
		t.Fatalf("Failed to encode mel window: %v", err)
	}

	// Validate output
	if len(features) != 512 {
		t.Errorf("Expected 512 features, got %d", len(features))
	}

	t.Logf("✅ Audio encoder inference successful")
	t.Logf("   Output features: %d", len(features))
	t.Logf("   First 10 values: %v", features[:10])
}

func TestAudioEncoderWithRealData(t *testing.T) {
	// Load the audio encoder
	encoder, err := NewAudioEncoder(testLibraryPath)
	if err != nil {
		t.Skipf("Skipping audio encoder test - ONNX model not found: %v", err)
		return
	}
	defer encoder.Close()

	// Process real audio to mel-spectrogram
	processor := NewProcessor(nil)

	// Generate test audio (440 Hz sine wave for 640ms at 16kHz = 10,240 samples)
	sampleRate := 16000
	duration := 0.64 // 640ms
	numSamples := int(float64(sampleRate) * duration)
	audioSamples := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		audioSamples[i] = float32(0.5 * (1.0 + 0.5*2.0*3.14159*440.0*t))
	}

	// Convert to mel-spectrogram
	melSpec, err := processor.ProcessAudio(audioSamples)
	if err != nil {
		t.Fatalf("Failed to process audio: %v", err)
	}

	t.Logf("Mel-spectrogram shape: [%d, %d]", len(melSpec), len(melSpec[0]))

	// The mel-spectrogram is [numFrames, 80]
	// We need to extract 16-frame windows and transpose to [80, 16]
	if len(melSpec) < 16 {
		t.Fatalf("Not enough frames: got %d, need at least 16", len(melSpec))
	}

	// Extract first 16 frames and transpose
	melWindow := make([][]float32, 80)
	for i := 0; i < 80; i++ {
		melWindow[i] = make([]float32, 16)
		for j := 0; j < 16; j++ {
			melWindow[i][j] = melSpec[j][i]
		}
	}

	// Encode through ONNX model
	features, err := encoder.Encode(melWindow)
	if err != nil {
		t.Fatalf("Failed to encode mel window: %v", err)
	}

	// Validate
	if len(features) != 512 {
		t.Errorf("Expected 512 features, got %d", len(features))
	}

	t.Logf("✅ End-to-end audio processing successful")
	t.Logf("   Audio samples: %d", len(audioSamples))
	t.Logf("   Mel frames: %d", len(melSpec))
	t.Logf("   Audio features: %d", len(features))
	t.Logf("   First 10 features: %v", features[:10])
}
