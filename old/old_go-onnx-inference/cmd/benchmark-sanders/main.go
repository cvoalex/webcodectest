package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"go-onnx-inference/lipsyncinfer"
)

type Metadata struct {
	NumFrames   int    `json:"num_frames"`
	StartFrame  int    `json:"start_frame"`
	VisualShape []int  `json:"visual_shape"`
	AudioShape  []int  `json:"audio_shape"`
	Dtype       string `json:"dtype"`
	Format      string `json:"format"`
	Note        string `json:"note"`
}

func main() {
	// Parse flags
	dataDir := flag.String("data", "d:/Projects/webcodecstest/test_data_sanders_for_go", "Test data directory")
	modelPath := flag.String("model", "d:/Projects/webcodecstest/minimal_server/models/sanders/checkpoint/model_best.onnx", "ONNX model path")
	outputDir := flag.String("output", "output_go_sanders_benchmark", "Output directory")
	flag.Parse()

	fmt.Println("\n" + "================================================================================")
	fmt.Println("🔷 GO + ONNX BENCHMARK (Sanders Dataset)")
	fmt.Println("================================================================================")

	// Create output directory
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Load metadata
	fmt.Printf("\n📦 Loading data from: %s\n", *dataDir)
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
	fmt.Printf("   Format: %s (CORRECT - matches cv2.imread)\n", metadata.Format)

	// Load binary data
	fmt.Printf("\n💾 Loading binary data...\n")
	visualData, err := loadFloat32Binary(filepath.Join(*dataDir, "visual_input.bin"))
	if err != nil {
		log.Fatalf("Failed to load visual data: %v", err)
	}

	audioData, err := loadFloat32Binary(filepath.Join(*dataDir, "audio_input.bin"))
	if err != nil {
		log.Fatalf("Failed to load audio data: %v", err)
	}

	fmt.Printf("   Visual data: %.2f MB\n", float64(len(visualData)*4)/(1024*1024))
	fmt.Printf("   Audio data: %.2f MB\n", float64(len(audioData)*4)/(1024*1024))

	// Initialize inferencer
	fmt.Printf("\n🚀 Initializing ONNX inferencer...\n")
	inferencer, err := lipsyncinfer.NewInferencer(*modelPath)
	if err != nil {
		log.Fatalf("Failed to create inferencer: %v", err)
	}
	defer inferencer.Close()

	fmt.Printf("   Model loaded: %s\n", *modelPath)

	// Process frames
	numFrames := metadata.NumFrames
	visualFrameSize := 6 * 320 * 320
	audioFrameSize := 32 * 16 * 16

	fmt.Printf("\n🎬 Processing %d frames...\n", numFrames)

	var totalInferenceTime float64
	var inferenceTimes []float64
	batchStart := time.Now()

	for i := 0; i < numFrames; i++ {
		// Extract frame data
		visualStart := i * visualFrameSize
		visualEnd := visualStart + visualFrameSize
		visualInput := visualData[visualStart:visualEnd]

		audioStart := i * audioFrameSize
		audioEnd := audioStart + audioFrameSize
		audioInput := audioData[audioStart:audioEnd]

		// Run inference
		inferenceStart := time.Now()
		output, err := inferencer.Infer(visualInput, audioInput)
		if err != nil {
			log.Printf("Inference failed for frame %d: %v", i, err)
			continue
		}
		inferenceTime := time.Since(inferenceStart).Seconds() * 1000

		totalInferenceTime += inferenceTime
		inferenceTimes = append(inferenceTimes, inferenceTime)

		// Save output image
		outputImg := convertToImage(output)
		outputPath := filepath.Join(*outputDir, fmt.Sprintf("frame_%04d.jpg", i))
		if err := saveImage(outputImg, outputPath); err != nil {
			log.Printf("Failed to save frame %d: %v", i, err)
		}

		if (i+1)%10 == 0 {
			fmt.Printf("   Processed %d/%d frames (%.2fms)\n", i+1, numFrames, inferenceTime)
		}
	}

	batchTime := time.Since(batchStart).Seconds()

	// Calculate statistics
	avgInference := totalInferenceTime / float64(numFrames)
	fps := float64(numFrames) / batchTime
	throughput := 1000.0 / avgInference

	fmt.Printf("\n📊 Performance Statistics:\n")
	fmt.Printf("   Total time: %.2fs\n", batchTime)
	fmt.Printf("   Frames processed: %d\n", numFrames)
	fmt.Printf("   FPS (overall): %.2f\n", fps)
	fmt.Printf("   Avg inference time: %.2fms\n", avgInference)
	fmt.Printf("   Throughput (inference only): %.1f FPS\n", throughput)

	// Find min/max inference times
	minTime := inferenceTimes[0]
	maxTime := inferenceTimes[0]
	for _, t := range inferenceTimes {
		if t < minTime {
			minTime = t
		}
		if t > maxTime {
			maxTime = t
		}
	}

	fmt.Printf("   Min inference time: %.2fms\n", minTime)
	fmt.Printf("   Max inference time: %.2fms\n", maxTime)

	fmt.Printf("\n✅ Frames saved to %s/\n", *outputDir)
}

func loadFloat32Binary(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	numFloats := len(data) / 4
	result := make([]float32, numFloats)

	for i := 0; i < numFloats; i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
		result[i] = float32frombits(bits)
	}

	return result, nil
}

func float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}

func convertToImage(output []float32) image.Image {
	// Output is [3, 320, 320] in BGR format
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))

	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// BGR order in output
			b := output[0*320*320+y*320+x]
			g := output[1*320*320+y*320+x]
			r := output[2*320*320+y*320+x]

			// Clamp and convert to uint8
			bVal := uint8(clamp(b*255.0, 0, 255))
			gVal := uint8(clamp(g*255.0, 0, 255))
			rVal := uint8(clamp(r*255.0, 0, 255))

			img.SetRGBA(x, y, color.RGBA{rVal, gVal, bVal, 255})
		}
	}

	return img
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func saveImage(img image.Image, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return jpeg.Encode(f, img, &jpeg.Options{Quality: 95})
}
