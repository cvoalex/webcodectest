package audio

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

func TestDebugMelSteps(t *testing.T) {
	// Load test audio
	data, err := os.ReadFile("../../audio_test_data/reference_data.json")
	if err != nil {
		t.Skipf("Skipping - reference data not found")
		return
	}

	var refData struct {
		Audio []float32 `json:"audio"`
	}
	if err := json.Unmarshal(data, &refData); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	processor := NewProcessor(nil)

	// Step 1: Pre-emphasis
	emphasized := processor.preEmphasis(refData.Audio)
	fmt.Printf("\n1. Pre-emphasis:\n")
	fmt.Printf("   First 10: %v\n", emphasized[:10])

	// Step 2: Pad
	padSize := processor.config.NumFFT / 2
	paddedLen := len(emphasized) + 2*padSize
	padded := make([]float32, paddedLen)
	copy(padded[padSize:], emphasized)
	fmt.Printf("\n2. Padding:\n")
	fmt.Printf("   Original length: %d\n", len(emphasized))
	fmt.Printf("   Padded length: %d\n", len(padded))
	fmt.Printf("   Pad size: %d\n", padSize)

	// Step 3: STFT (first frame only)
	frame := padded[0:800]
	fmt.Printf("\n3. First frame extraction:\n")
	fmt.Printf("   Frame length: %d\n", len(frame))
	fmt.Printf("   First 10: %v\n", frame[:10])

	// Apply window
	windowed := make([]float32, 800)
	for i := 0; i < 800; i++ {
		windowed[i] = frame[i] * processor.window[i]
	}
	fmt.Printf("\n4. After windowing:\n")
	fmt.Printf("   First 10: %v\n", windowed[:10])

	// FFT
	input := make([]float64, 800)
	for i := 0; i < 800; i++ {
		input[i] = float64(windowed[i])
	}
	coeffs := processor.fftObj.Coefficients(nil, input)

	// Magnitudes
	fmt.Printf("\n5. FFT magnitudes (first 10 bins):\n")
	for i := 0; i < 10; i++ {
		mag := math.Sqrt(real(coeffs[i])*real(coeffs[i]) + imag(coeffs[i])*imag(coeffs[i]))
		fmt.Printf("   [%d]: %.8f\n", i, mag)
	}

	// Max magnitude
	maxMag := 0.0
	for i := 0; i <= 400; i++ {
		mag := math.Sqrt(real(coeffs[i])*real(coeffs[i]) + imag(coeffs[i])*imag(coeffs[i]))
		if mag > maxMag {
			maxMag = mag
		}
	}
	fmt.Printf("   Max magnitude: %.8f\n", maxMag)

	// Run full STFT
	spectrogram := processor.stft(padded)
	fmt.Printf("\n6. Full STFT:\n")
	fmt.Printf("   Shape: [%d, %d]\n", len(spectrogram), len(spectrogram[0]))
	fmt.Printf("   First frame (first 10): %v\n", spectrogram[0][:10])

	// Mel filterbank
	melSpec := processor.linearToMel(spectrogram)
	fmt.Printf("\n7. After mel filterbank:\n")
	fmt.Printf("   First frame (first 10): %v\n", melSpec[0][:10])

	// dB conversion
	melSpecDB := processor.ampToDB(melSpec)
	fmt.Printf("\n8. After dB conversion:\n")
	fmt.Printf("   First frame (first 10): %v\n", melSpecDB[0][:10])

	// Normalize
	normalized := processor.normalize(melSpecDB)
	fmt.Printf("\n9. After normalization:\n")
	fmt.Printf("   First frame (first 10): %v\n", normalized[0][:10])
}
