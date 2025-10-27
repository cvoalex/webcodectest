package audio

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestWithVideoReference(t *testing.T) {
	// Load Python reference data from video
	jsonFile := filepath.Join("..", "..", "audio_test_data", "reference_data_video.json")
	data, err := os.ReadFile(jsonFile)
	if err != nil {
		t.Skipf("Skipping video validation test - reference not found at %s. Error: %v", jsonFile, err)
		return
	}

	var refData ReferenceData
	if err := json.Unmarshal(data, &refData); err != nil {
		t.Fatalf("Failed to parse reference data: %v", err)
	}

	t.Logf("Loaded Python reference from video: shape=%v", refData.Shape)
	t.Logf("Python stats: min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
		refData.Stats.MelMin, refData.Stats.MelMax, refData.Stats.MelMean, refData.Stats.MelStd)

	// Process same audio with Go implementation
	processor := NewProcessor(nil)
	goMelSpec, err := processor.ProcessAudio(refData.Audio)
	if err != nil {
		t.Fatalf("Go ProcessAudio failed: %v", err)
	}

	// Compare shapes
	goNumFrames := len(goMelSpec)
	if goNumFrames == 0 {
		t.Fatal("Go implementation produced empty spectrogram")
	}
	goNumMels := len(goMelSpec[0])
	pyNumFrames := refData.Shape[0]
	pyNumMels := refData.Shape[1]

	if goNumFrames != pyNumFrames || goNumMels != pyNumMels {
		t.Errorf("Shape mismatch: Go=[%d, %d], Python=[%d, %d]",
			goNumFrames, goNumMels, pyNumFrames, pyNumMels)
	}

	// Compute Go statistics
	goMin, goMax, goMean, goStd := computeStats(goMelSpec)
	t.Logf("Go stats:     min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
		goMin, goMax, goMean, goStd)

	// Compare stats
	if abs32(float32(goMin-refData.Stats.MelMin)) > 0.1 {
		t.Logf("Min mismatch: Go=%.6f, Python=%.6f", goMin, refData.Stats.MelMin)
	}
	if abs32(float32(goMax-refData.Stats.MelMax)) > 0.1 {
		t.Logf("Max mismatch: Go=%.6f, Python=%.6f", goMax, refData.Stats.MelMax)
	}
	if abs32(float32(goMean-refData.Stats.MelMean)) > 0.1 {
		t.Logf("Mean mismatch: Go=%.6f, Python=%.6f", goMean, refData.Stats.MelMean)
	}
	if abs32(float32(goStd-refData.Stats.MelStd)) > 0.1 {
		t.Logf("Std mismatch: Go=%.6f, Python=%.6f", goStd, refData.Stats.MelStd)
	}

	// Check element-wise differences
	maxDiff := float32(0.0)
	sumDiff := float64(0.0)
	totalElements := 0

	for i := 0; i < min(pyNumFrames, goNumFrames); i++ {
		for j := 0; j < min(pyNumMels, goNumMels); j++ {
			gVal := goMelSpec[i][j]
			pVal := refData.MelSpec[i][j]
			diff := abs32(gVal - pVal)

			if diff > maxDiff {
				maxDiff = diff
			}
			sumDiff += float64(diff)
			totalElements++
		}
	}

	avgDiff := float32(0.0)
	if totalElements > 0 {
		avgDiff = float32(sumDiff / float64(totalElements))
	}

	t.Logf("Differences: max=%.6f, avg=%.6f", maxDiff, avgDiff)

	// Relaxed thresholds
	maxThreshold := float32(0.5)
	avgThreshold := float32(0.5)

	if maxDiff > maxThreshold {
		t.Errorf("Max difference too large: %.6f (threshold: %.6f)", maxDiff, maxThreshold)
	}

	if avgDiff > avgThreshold {
		t.Errorf("Average difference too large: %.6f (threshold: %.6f)", avgDiff, avgThreshold)
	}

	// Log first frame for visual comparison
	if len(goMelSpec) > 0 && len(refData.MelSpec) > 0 {
		t.Logf("First frame comparison (first 10 values):")
		t.Logf("  Go:     %v", goMelSpec[0][:10])
		t.Logf("  Python: %v", refData.MelSpec[0][:10])
	}
}

func computeStats(spec [][]float32) (min, max, mean, std float64) {
	if len(spec) == 0 || len(spec[0]) == 0 {
		return 0, 0, 0, 0
	}

	minVal := float32(math.MaxFloat32)
	maxVal := float32(-math.MaxFloat32)
	sum := float64(0.0)
	count := 0

	for _, frame := range spec {
		for _, val := range frame {
			if val < minVal {
				minVal = val
			}
			if val > maxVal {
				maxVal = val
			}
			sum += float64(val)
			count++
		}
	}

	meanVal := sum / float64(count)

	sumSq := float64(0.0)
	for _, frame := range spec {
		for _, val := range frame {
			diff := float64(val) - meanVal
			sumSq += diff * diff
		}
	}
	stdVal := math.Sqrt(sumSq / float64(count))

	return float64(minVal), float64(maxVal), meanVal, stdVal
}
