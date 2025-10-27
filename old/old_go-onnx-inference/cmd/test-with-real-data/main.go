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

// Metadata matches the JSON structure from Python export
type Metadata struct {
	NumFrames   int    `json:"num_frames"`
	StartFrame  int    `json:"start_frame"`
	VisualShape []int  `json:"visual_shape"`
	AudioShape  []int  `json:"audio_shape"`
	Dtype       string `json:"dtype"`
}

func main() {
	// Parse command line arguments
	dataDir := flag.String("data", "d:/Projects/webcodecstest/fast_service/test_data_for_go", "Path to test data directory")
	modelPath := flag.String("model", "d:/Projects/webcodecstest/fast_service/models/default_model/models/99.onnx", "Path to ONNX model")
	outputDir := flag.String("output", "output_go_real_data", "Output directory")
	flag.Parse()

	fmt.Println("\n" + "================================================================================")
	fmt.Println("🔷 GO + ONNX - REAL DATA TEST")
	fmt.Println("================================================================================")

	// Load metadata
	fmt.Printf("\n📦 Loading test data from: %s\n", *dataDir)
	metadataPath := filepath.Join(*dataDir, "metadata.json")
	metadataBytes, err := os.ReadFile(metadataPath)
	if err != nil {
		log.Fatalf("Failed to read metadata: %v", err)
	}

	var metadata Metadata
	if err := json.Unmarshal(metadataBytes, &metadata); err != nil {
		log.Fatalf("Failed to parse metadata: %v", err)
	}

	fmt.Printf("   Frames: %d\n", metadata.NumFrames)
	fmt.Printf("   Visual shape: %v\n", metadata.VisualShape)
	fmt.Printf("   Audio shape: %v\n", metadata.AudioShape)

	// Load visual input data
	visualPath := filepath.Join(*dataDir, "visual_input.bin")
	visualData, err := loadFloat32Binary(visualPath)
	if err != nil {
		log.Fatalf("Failed to load visual data: %v", err)
	}
	fmt.Printf("✅ Loaded visual data: %d float32 values\n", len(visualData))

	// Load audio input data
	audioPath := filepath.Join(*dataDir, "audio_input.bin")
	audioData, err := loadFloat32Binary(audioPath)
	if err != nil {
		log.Fatalf("Failed to load audio data: %v", err)
	}
	fmt.Printf("✅ Loaded audio data: %d float32 values\n", len(audioData))

	// Load ONNX model
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

	// Calculate frame sizes
	visualFrameSize := int(visualShape[1] * visualShape[2] * visualShape[3]) // 6 * 320 * 320
	audioFrameSize := int(audioShape[1] * audioShape[2] * audioShape[3])     // 32 * 16 * 16
	_ = int(outputShape[1] * outputShape[2] * outputShape[3])                // 3 * 320 * 320 (outputFrameSize unused)

	fmt.Printf("\n🚀 Running inference on %d frames...\n", metadata.NumFrames)
	fmt.Println("================================================================================")

	// Create output directory
	os.MkdirAll(*outputDir, 0755)

	var inferenceTimes []float64
	var outputFrames [][]float32

	for i := 0; i < metadata.NumFrames; i++ {
		// Extract frame data
		visualOffset := i * visualFrameSize
		audioOffset := i * audioFrameSize

		visualFrame := visualData[visualOffset : visualOffset+visualFrameSize]
		audioFrame := audioData[audioOffset : audioOffset+audioFrameSize]

		// Run inference
		start := time.Now()
		output, err := inferencer.Infer(visualFrame, audioFrame)
		elapsed := time.Since(start)

		if err != nil {
			log.Fatalf("Inference failed on frame %d: %v", i, err)
		}

		inferenceTimeMs := float64(elapsed.Microseconds()) / 1000.0
		inferenceTimes = append(inferenceTimes, inferenceTimeMs)
		outputFrames = append(outputFrames, output)

		fmt.Printf("   Frame %d/%d: %.2fms\n", i+1, metadata.NumFrames, inferenceTimeMs)

		// Save output frame
		err = saveFrameAsImage(output, outputShape, filepath.Join(*outputDir, fmt.Sprintf("output_frame_%04d.png", i)))
		if err != nil {
			fmt.Printf("⚠️  Warning: Failed to save frame %d: %v\n", i, err)
		}
	}

	// Calculate statistics
	fmt.Println("\n================================================================================")
	fmt.Println("📊 INFERENCE STATISTICS")
	fmt.Println("================================================================================")

	meanTime := mean(inferenceTimes)
	medianTime := median(inferenceTimes)
	minTime := min(inferenceTimes)
	maxTime := max(inferenceTimes)

	fmt.Printf("Total frames:     %d\n", metadata.NumFrames)
	fmt.Printf("Mean time:        %.3f ms\n", meanTime)
	fmt.Printf("Median time:      %.3f ms\n", medianTime)
	fmt.Printf("Min time:         %.3f ms\n", minTime)
	fmt.Printf("Max time:         %.3f ms\n", maxTime)
	fmt.Printf("Average FPS:      %.1f\n", 1000.0/meanTime)

	// Analyze output quality
	fmt.Println("\n📈 OUTPUT QUALITY")
	fmt.Println("================================================================================")

	allOutputValues := []float32{}
	for _, frame := range outputFrames {
		allOutputValues = append(allOutputValues, frame...)
	}

	outMean := meanF32(allOutputValues)
	outStd := stdDevF32(allOutputValues, outMean)
	outMin := minF32(allOutputValues)
	outMax := maxF32(allOutputValues)

	fmt.Printf("Output shape:     [%d, %d, %d, %d]\n", metadata.NumFrames, outputShape[1], outputShape[2], outputShape[3])
	fmt.Printf("Output mean:      %.6f\n", outMean)
	fmt.Printf("Output std:       %.6f\n", outStd)
	fmt.Printf("Output min:       %.6f\n", outMin)
	fmt.Printf("Output max:       %.6f\n", outMax)

	// Check if values are in valid range
	inRange := 0
	for _, v := range allOutputValues {
		if v >= -1.1 && v <= 1.1 {
			inRange++
		}
	}
	pctInRange := float64(inRange) / float64(len(allOutputValues)) * 100.0
	fmt.Printf("Values in [-1.1, 1.1]: %.2f%%\n", pctInRange)

	fmt.Printf("\n✅ Output saved to: %s/\n", *outputDir)
	fmt.Println("   - output_frame_XXXX.png: Model outputs")
	fmt.Println("\n👀 Compare with input images in test_data_for_go/input_face_XXXX.png")
	fmt.Println("\n✅ Test completed successfully!")
}

func loadFloat32Binary(path string) ([]float32, error) {
	// Read binary file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Convert bytes to float32 array
	numFloats := len(data) / 4
	result := make([]float32, numFloats)

	for i := 0; i < numFloats; i++ {
		offset := i * 4
		// Little-endian byte order
		bits := uint32(data[offset]) | uint32(data[offset+1])<<8 |
			uint32(data[offset+2])<<16 | uint32(data[offset+3])<<24
		result[i] = math.Float32frombits(bits)
	}

	return result, nil
}

func saveFrameAsImage(output []float32, shape []int64, filepath string) error {
	// Output is [1, 3, 320, 320] but we get it as flat array
	width := int(shape[3])
	height := int(shape[2])

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Get RGB values from channels [C, H, W]
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
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func median(values []float64) float64 {
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
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
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

func meanF32(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	sum := float32(0)
	for _, v := range values {
		sum += v
	}
	return sum / float32(len(values))
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
