package audio

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

func TestWithLibrosaMelFilters(t *testing.T) {
	// Load librosa mel filters
	data, err := os.ReadFile("../../audio_test_data/librosa_mel_filters.json")
	if err != nil {
		t.Skipf("Skip - librosa filters not found")
		return
	}

	var filterData struct {
		Filters [][]float32 `json:"filters"`
		Shape   []int       `json:"shape"`
	}
	if err := json.Unmarshal(data, &filterData); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	t.Logf("Loaded librosa mel filters: shape=%v", filterData.Shape)

	// Load test audio
	refData, err := os.ReadFile("../../audio_test_data/reference_data.json")
	if err != nil {
		t.Fatalf("Failed to load audio: %v", err)
	}

	var ref struct {
		Audio []float32 `json:"audio"`
	}
	if err := json.Unmarshal(refData, &ref); err != nil {
		t.Fatalf("Failed to parse audio: %v", err)
	}

	// Create processor
	processor := NewProcessor(nil)

	// Replace mel filters with librosa's
	processor.melFilters = filterData.Filters

	// Process audio
	result, err := processor.ProcessAudio(ref.Audio)
	if err != nil {
		t.Fatalf("ProcessAudio failed: %v", err)
	}

	fmt.Printf("\nWith librosa mel filters:\n")
	fmt.Printf("  Shape: [%d, %d]\n", len(result), len(result[0]))
	fmt.Printf("  First frame (first 10): %v\n", result[0][:10])

	// Load Python reference
	correctData, err := os.ReadFile("../../audio_test_data/reference_data_correct.json")
	if err != nil {
		t.Fatalf("Failed to load reference: %v", err)
	}

	var correct struct {
		MelSpec [][]float32 `json:"mel_spectrogram"`
	}
	if err := json.Unmarshal(correctData, &correct); err != nil {
		t.Fatalf("Failed to parse reference: %v", err)
	}

	fmt.Printf("\nPython reference:\n")
	fmt.Printf("  First frame (first 10): %v\n", correct.MelSpec[0][:10])

	// Compare
	maxDiff := float32(0.0)
	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[i]); j++ {
			diff := abs32(result[i][j] - correct.MelSpec[i][j])
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}

	fmt.Printf("\nMax difference: %.6f\n", maxDiff)
	if maxDiff < 0.01 {
		fmt.Printf("✅ PASS: Matches Python reference!\n")
	} else {
		fmt.Printf("❌ FAIL: Still differs from Python\n")
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
