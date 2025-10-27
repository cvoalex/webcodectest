package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"go-onnx-inference/lipsyncinfer"
)

// Summary holds inference statistics
type Summary struct {
	NumFrames    int     `json:"num_frames"`
	MeanTimeMs   float64 `json:"mean_time_ms"`
	MedianTimeMs float64 `json:"median_time_ms"`
	FPS          float64 `json:"fps"`
	TotalTimeS   float64 `json:"total_time_s"`
	OutputShape  []int   `json:"output_shape"`
	OutputMean   float64 `json:"output_mean"`
	OutputStd    float64 `json:"output_std"`
	OutputMin    float64 `json:"output_min"`
	OutputMax    float64 `json:"output_max"`
}

func main() {
	// Parse command line arguments
	_ = flag.String("audio", "d:/Projects/webcodecstest/aud.wav", "Path to audio file")
	modelPath := flag.String("model", "d:/Projects/webcodecstest/fast_service/models/default_model/models/99.onnx", "Path to ONNX model")
	outputDir := flag.String("output", "output_go_onnx", "Output directory")
	noSave := flag.Bool("no-save", false, "Do not save frames")
	audioFeaturesPath := flag.String("audio-features", "", "Path to pre-extracted audio features (.npy)")
	flag.Parse()

	fmt.Println("\n" + "================================================================================")
	fmt.Println("🔷 GO + ONNX RUNTIME - REAL AUDIO TEST")
	fmt.Println("================================================================================")

	// For now, we need pre-extracted audio features
	// In production, you'd extract these in Go or have them pre-processed
	if *audioFeaturesPath == "" {
		fmt.Println("\n⚠️  Note: Using simulated audio features for comparison.")
		fmt.Println("    Real audio extraction would be added for production use.")
		fmt.Println("    This demo processes the same number of frames (255) as Python test.")
	}

	// Load inferencer
	fmt.Printf("\n📦 Loading ONNX model: %s\n", *modelPath)
	inferencer, err := lipsyncinfer.NewInferencer(*modelPath)
	if err != nil {
		log.Fatalf("Failed to create inferencer: %v", err)
	}
	defer inferencer.Close()

	visualShape, audioShape := inferencer.GetInputShapes()
	outputShape := inferencer.GetOutputShape()

	fmt.Printf("   Visual input:  %v\n", visualShape)
	fmt.Printf("   Audio input:   %v\n", audioShape)
	fmt.Printf("   Output shape:  %v\n", outputShape)

	// TODO: Load audio features from .npy file
	// For now, we'll simulate with dummy data

	fmt.Println("\n⚠️  This is a demo with simulated frames")
	fmt.Println("    Full implementation requires audio feature extraction in Go")

	runDemoInference(inferencer, *outputDir, !*noSave)
}

func runDemoInference(inferencer *lipsyncinfer.Inferencer, outputDir string, saveFrames bool) {
	// Simulate processing same number of frames as audio would produce
	// Match Python test: 255 frames (10.22 seconds at 25 FPS)
	numFrames := 255

	visualShape, audioShape := inferencer.GetInputShapes()
	outputShape := inferencer.GetOutputShape()

	visualSize := int(visualShape[0] * visualShape[1] * visualShape[2] * visualShape[3])
	audioSize := int(audioShape[0] * audioShape[1] * audioShape[2] * audioShape[3])
	_ = int(outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3]) // outputSize unused

	fmt.Printf("\n🚀 Running inference on %d simulated frames...\n", numFrames)
	fmt.Println("================================================================================")

	// Create output directory if saving
	if saveFrames {
		os.MkdirAll(outputDir, 0755)
		fmt.Printf("\n💾 Saving frames to: %s\n", outputDir)
	}

	// Create dummy inputs (in real use, these would come from audio features)
	visualInput := make([]float32, visualSize)
	audioInput := make([]float32, audioSize)

	// Fill with test data (0.5 for demo)
	for i := range visualInput {
		visualInput[i] = 0.5
	}
	for i := range audioInput {
		audioInput[i] = 0.5
	}

	// Run inference on all frames
	var inferenceTimes []float64
	var allOutputs [][]float32

	for i := 0; i < numFrames; i++ {
		// In real implementation, vary visual and audio inputs per frame
		// For now, using same inputs to match Python test pattern

		start := time.Now()
		output, err := inferencer.Infer(visualInput, audioInput)
		elapsed := time.Since(start)

		if err != nil {
			log.Fatalf("Inference failed on frame %d: %v", i, err)
		}

		inferenceTimeMs := float64(elapsed.Microseconds()) / 1000.0
		inferenceTimes = append(inferenceTimes, inferenceTimeMs)
		allOutputs = append(allOutputs, output)

		// Save frame if requested
		if saveFrames && i%10 == 0 {
			// Convert output to image
			err := saveFrameAsImage(output, outputShape, filepath.Join(outputDir, fmt.Sprintf("frame_%04d.png", i)))
			if err != nil {
				fmt.Printf("⚠️  Warning: Failed to save frame %d: %v\n", i, err)
			}
		}

		// Progress indicator
		if (i+1)%10 == 0 || i == 0 || i == numFrames-1 {
			startIdx := len(inferenceTimes) - 10
			if startIdx < 0 {
				startIdx = 0
			}
			avgTime := mean(inferenceTimes[startIdx:])
			fmt.Printf("   Frame %d/%d: %.2fms (avg: %.2fms)\n", i+1, numFrames, inferenceTimeMs, avgTime)
		}
	}

	// Calculate statistics
	fmt.Println("\n================================================================================")
	fmt.Println("📊 INFERENCE STATISTICS")
	fmt.Println("================================================================================")

	meanTime := mean(inferenceTimes)
	medianTime := median(inferenceTimes)
	stdTime := stdDev(inferenceTimes, meanTime)
	minTime := min(inferenceTimes)
	maxTime := max(inferenceTimes)
	p95Time := percentile(inferenceTimes, 95)
	p99Time := percentile(inferenceTimes, 99)
	totalTime := sum(inferenceTimes) / 1000.0 // seconds

	fmt.Printf("Total frames:     %d\n", numFrames)
	fmt.Printf("Mean time:        %.3f ms\n", meanTime)
	fmt.Printf("Median time:      %.3f ms\n", medianTime)
	fmt.Printf("Std deviation:    %.3f ms\n", stdTime)
	fmt.Printf("Min time:         %.3f ms\n", minTime)
	fmt.Printf("Max time:         %.3f ms\n", maxTime)
	fmt.Printf("P95:              %.3f ms\n", p95Time)
	fmt.Printf("P99:              %.3f ms\n", p99Time)
	fmt.Printf("Average FPS:      %.1f\n", 1000.0/meanTime)
	fmt.Printf("Total time:       %.2f seconds\n", totalTime)

	if saveFrames {
		frameCount := 0
		files, _ := os.ReadDir(outputDir)
		for _, f := range files {
			if filepath.Ext(f.Name()) == ".png" {
				frameCount++
			}
		}
		fmt.Printf("\n✅ Frames saved to: %s/\n", outputDir)
		fmt.Printf("   Saved %d sample frames\n", frameCount)
	}

	// Calculate output statistics
	fmt.Println("\n📈 OUTPUT STATISTICS")
	fmt.Println("================================================================================")

	allValues := []float32{}
	for _, output := range allOutputs {
		allValues = append(allValues, output...)
	}

	outMean := meanF32(allValues)
	outStd := stdDevF32(allValues, outMean)
	outMin := minF32(allValues)
	outMax := maxF32(allValues)

	fmt.Printf("Output shape:     [%d, %d, %d, %d]\n", numFrames, outputShape[1], outputShape[2], outputShape[3])
	fmt.Printf("Output mean:      %.6f\n", outMean)
	fmt.Printf("Output std:       %.6f\n", outStd)
	fmt.Printf("Output min:       %.6f\n", outMin)
	fmt.Printf("Output max:       %.6f\n", outMax)

	// Create summary
	summary := Summary{
		NumFrames:    numFrames,
		MeanTimeMs:   meanTime,
		MedianTimeMs: medianTime,
		FPS:          1000.0 / meanTime,
		TotalTimeS:   totalTime,
		OutputShape:  []int{numFrames, int(outputShape[1]), int(outputShape[2]), int(outputShape[3])},
		OutputMean:   float64(outMean),
		OutputStd:    float64(outStd),
		OutputMin:    float64(outMin),
		OutputMax:    float64(outMax),
	}

	// Save summary
	summaryPath := filepath.Join(outputDir, "summary.json")
	summaryJSON, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile(summaryPath, summaryJSON, 0644)
	fmt.Printf("\n💾 Saved summary to: %s\n", summaryPath)

	fmt.Println("\n✅ Test completed successfully!")
}

func saveFrameAsImage(output []float32, shape []int64, filepath string) error {
	// Output is [1, 3, 320, 320] but we get it as flat array
	// Need to convert to image format

	width := int(shape[3])
	height := int(shape[2])

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Get RGB values from channels
			// Output format: [C, H, W] where C=3 (RGB)
			rIdx := y*width + x
			gIdx := width*height + y*width + x
			bIdx := 2*width*height + y*width + x

			// Scale from [-1, 1] to [0, 255]
			r := uint8(math.Max(0, math.Min(255, (float64(output[rIdx])+1)*127.5)))
			g := uint8(math.Max(0, math.Min(255, (float64(output[gIdx])+1)*127.5)))
			b := uint8(math.Max(0, math.Min(255, (float64(output[bIdx])+1)*127.5)))

			img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	// Save image
	f, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer f.Close()

	return png.Encode(f, img)
}

// Statistical helper functions
func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	return sum(values) / float64(len(values))
}

func sum(values []float64) float64 {
	total := 0.0
	for _, v := range values {
		total += v
	}
	return total
}

func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	// Simple bubble sort for median
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}

func stdDev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	return math.Sqrt(variance / float64(len(values)))
}

func min(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	minVal := values[0]
	for _, v := range values[1:] {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func max(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	idx := int(float64(len(sorted)-1) * p / 100.0)
	return sorted[idx]
}

// Float32 statistics
func meanF32(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	total := float32(0)
	for _, v := range values {
		total += v
	}
	return total / float32(len(values))
}

func stdDevF32(values []float32, mean float32) float32 {
	if len(values) == 0 {
		return 0
	}
	variance := float32(0)
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	return float32(math.Sqrt(float64(variance / float32(len(values)))))
}

func minF32(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	minVal := values[0]
	for _, v := range values[1:] {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func maxF32(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}
