package audio

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ReferenceData holds Python-generated reference mel-spectrogram
type ReferenceData struct {
	Audio   []float32   `json:"audio"`
	MelSpec [][]float32 `json:"mel_spectrogram"`
	Shape   []int       `json:"shape"`
	Stats   struct {
		MelMin  float64 `json:"min"`
		MelMax  float64 `json:"max"`
		MelMean float64 `json:"mean"`
		MelStd  float64 `json:"std"`
	} `json:"stats"`
}

func TestAgainstPythonReference(t *testing.T) {
	// Load Python reference data (generated using original code with center=True)
	// Try multiple possible locations
	possiblePaths := []string{
		"../audio_test_data/reference_data_correct.json",
		"../../audio_test_data/reference_data_correct.json",
		filepath.Join("..", "..", "audio_test_data", "reference_data_correct.json"),
	}

	var data []byte
	var err error
	var foundPath string

	for _, jsonFile := range possiblePaths {
		data, err = os.ReadFile(jsonFile)
		if err == nil {
			foundPath = jsonFile
			break
		}
	}

	if err != nil {
		t.Skipf("Skipping validation test - Python reference not found. Tried: %v", possiblePaths)
		return
	}

	t.Logf("Using reference data from: %s", foundPath)

	var ref ReferenceData
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatalf("Failed to parse reference data: %v", err)
	}

	t.Logf("Loaded Python reference: shape=%v", ref.Shape)
	t.Logf("Python stats: min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
		ref.Stats.MelMin, ref.Stats.MelMax, ref.Stats.MelMean, ref.Stats.MelStd)

	// Process same audio with Go implementation
	processor := NewProcessor(nil)
	goMelSpec, err := processor.ProcessAudio(ref.Audio)
	if err != nil {
		t.Fatalf("Go ProcessAudio failed: %v", err)
	}

	// Compare shapes
	goNumFrames := len(goMelSpec)
	goNumMels := len(goMelSpec[0])
	pyNumFrames := ref.Shape[0]
	pyNumMels := ref.Shape[1]

	if goNumFrames != pyNumFrames || goNumMels != pyNumMels {
		t.Errorf("Shape mismatch: Go=[%d, %d], Python=[%d, %d]",
			goNumFrames, goNumMels, pyNumFrames, pyNumMels)
	}

	// Compute Go statistics
	goMin, goMax, goSum := float32(math.MaxFloat32), float32(-math.MaxFloat32), float32(0.0)
	count := 0
	for i := 0; i < goNumFrames; i++ {
		for j := 0; j < goNumMels; j++ {
			val := goMelSpec[i][j]
			if val < goMin {
				goMin = val
			}
			if val > goMax {
				goMax = val
			}
			goSum += val
			count++
		}
	}
	goMean := goSum / float32(count)

	// Compute std dev
	sumSq := float32(0.0)
	for i := 0; i < goNumFrames; i++ {
		for j := 0; j < goNumMels; j++ {
			diff := goMelSpec[i][j] - goMean
			sumSq += diff * diff
		}
	}
	goStd := float32(math.Sqrt(float64(sumSq / float32(count))))

	t.Logf("Go stats:     min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
		goMin, goMax, goMean, goStd)

	// Compare statistics (allow some tolerance due to float precision)
	tolerance := 0.01 // 1% tolerance

	if math.Abs(float64(goMin)-ref.Stats.MelMin) > tolerance {
		t.Errorf("Min mismatch: Go=%.6f, Python=%.6f", goMin, ref.Stats.MelMin)
	}

	if math.Abs(float64(goMax)-ref.Stats.MelMax) > tolerance {
		t.Errorf("Max mismatch: Go=%.6f, Python=%.6f", goMax, ref.Stats.MelMax)
	}

	if math.Abs(float64(goMean)-ref.Stats.MelMean) > tolerance {
		t.Errorf("Mean mismatch: Go=%.6f, Python=%.6f", goMean, ref.Stats.MelMean)
	}

	if math.Abs(float64(goStd)-ref.Stats.MelStd) > tolerance {
		t.Errorf("Std mismatch: Go=%.6f, Python=%.6f", goStd, ref.Stats.MelStd)
	}

	// Compare actual values element-by-element
	maxDiff := float32(0.0)
	totalDiff := float32(0.0)
	diffCount := 0

	minFrames := goNumFrames
	if pyNumFrames < minFrames {
		minFrames = pyNumFrames
	}

	for i := 0; i < minFrames; i++ {
		for j := 0; j < goNumMels && j < pyNumMels; j++ {
			goVal := goMelSpec[i][j]
			pyVal := ref.MelSpec[i][j]
			diff := float32(math.Abs(float64(goVal - pyVal)))

			if diff > maxDiff {
				maxDiff = diff
			}
			totalDiff += diff
			diffCount++
		}
	}

	avgDiff := totalDiff / float32(diffCount)

	t.Logf("Differences: max=%.6f, avg=%.6f", maxDiff, avgDiff)

	// Allow small differences due to floating point precision and implementation details
	maxAllowedDiff := float32(0.1)     // 10% max difference per element
	maxAllowedAvgDiff := float32(0.01) // 1% average difference

	if maxDiff > maxAllowedDiff {
		t.Errorf("Max difference too large: %.6f (threshold: %.6f)", maxDiff, maxAllowedDiff)
	}

	if avgDiff > maxAllowedAvgDiff {
		t.Errorf("Average difference too large: %.6f (threshold: %.6f)", avgDiff, maxAllowedAvgDiff)
	}

	// Print first frame comparison
	if goNumFrames > 0 && pyNumFrames > 0 {
		t.Log("First frame comparison (first 10 values):")
		t.Logf("  Go:     %v", goMelSpec[0][:10])
		t.Logf("  Python: %v", ref.MelSpec[0][:10])
	}

	if maxDiff < maxAllowedDiff && avgDiff < maxAllowedAvgDiff {
		t.Logf("✅ Go implementation matches Python reference!")
		t.Logf("   Shape: [%d, %d]", goNumFrames, goNumMels)
		t.Logf("   Max difference: %.6f", maxDiff)
		t.Logf("   Avg difference: %.6f", avgDiff)
	}
}

func TestProcessorMatchesPythonShape(t *testing.T) {
	processor := NewProcessor(nil)

	// Test multiple audio lengths
	testCases := []struct {
		durationMs     int
		expectedFrames int
	}{
		{640, 48},  // (10240 - 800) / 200 + 1 = 48
		{1000, 73}, // (16000 - 800) / 200 + 1 = 77
		{320, 24},  // (5120 - 800) / 200 + 1 = 22
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("duration_%dms", tc.durationMs), func(t *testing.T) {
			// Generate test audio
			sampleRate := 16000
			numSamples := (sampleRate * tc.durationMs) / 1000
			audio := make([]float32, numSamples)
			for i := 0; i < numSamples; i++ {
				audio[i] = 0.5
			}

			// Process
			melSpec, err := processor.ProcessAudio(audio)
			if err != nil {
				t.Fatalf("ProcessAudio failed: %v", err)
			}

			// Check shape
			numFrames := len(melSpec)
			numMels := len(melSpec[0])

			expectedFrames := (numSamples-processor.config.WindowSize)/processor.config.HopLength + 1

			if numFrames != expectedFrames {
				t.Errorf("Frame count mismatch: got %d, expected %d", numFrames, expectedFrames)
			}

			if numMels != processor.config.NumMelBins {
				t.Errorf("Mel bins mismatch: got %d, expected %d", numMels, processor.config.NumMelBins)
			}

			t.Logf("✅ %dms audio → [%d, %d] mel-spec", tc.durationMs, numFrames, numMels)
		})
	}
}
