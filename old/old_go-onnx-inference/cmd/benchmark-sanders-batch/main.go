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
	outputDir := flag.String("output", "output_go_batch", "Output directory")
	batchSize := flag.Int("batch", 4, "Batch size for inference")
	flag.Parse()

	fmt.Println("\n" + "================================================================================")
	fmt.Println("🚀 GO + ONNX BATCH BENCHMARK (Sanders Dataset)")
	fmt.Println("================================================================================")
	fmt.Printf("   Batch size: %d\n", *batchSize)

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

	// Process frames in batches
	numFrames := metadata.NumFrames
	visualFrameSize := 6 * 320 * 320
	audioFrameSize := 32 * 16 * 16

	fmt.Printf("\n🎬 Processing %d frames in batches of %d...\n", numFrames, *batchSize)

	var totalInferenceTime float64
	frameCount := 0

	batchStart := time.Now()

	// Process in batches
	for batchIdx := 0; batchIdx < numFrames; batchIdx += *batchSize {
		batchEnd := batchIdx + *batchSize
		if batchEnd > numFrames {
			batchEnd = numFrames
		}
		currentBatchSize := batchEnd - batchIdx

		// Prepare batch inputs
		batchVisual := make([]float32, currentBatchSize*visualFrameSize)
		batchAudio := make([]float32, currentBatchSize*audioFrameSize)

		for i := 0; i < currentBatchSize; i++ {
			frameID := batchIdx + i
			visualStart := frameID * visualFrameSize
			copy(batchVisual[i*visualFrameSize:(i+1)*visualFrameSize],
				visualData[visualStart:visualStart+visualFrameSize])

			audioStart := frameID * audioFrameSize
			copy(batchAudio[i*audioFrameSize:(i+1)*audioFrameSize],
				audioData[audioStart:audioStart+audioFrameSize])
		}

		// Batch inference
		inferenceStart := time.Now()
		batchOutput, err := inferencer.InferBatch(batchVisual, batchAudio, currentBatchSize)
		if err != nil {
			log.Printf("Batch inference failed: %v", err)
			continue
		}
		inferenceTime := time.Since(inferenceStart).Seconds() * 1000
		totalInferenceTime += inferenceTime

		// Save output images
		outputFrameSize := 3 * 320 * 320
		for i := 0; i < currentBatchSize; i++ {
			frameID := batchIdx + i
			outputStart := i * outputFrameSize
			outputEnd := outputStart + outputFrameSize
			frameOutput := batchOutput[outputStart:outputEnd]

			outputImg := convertToImage(frameOutput)
			outputPath := filepath.Join(*outputDir, fmt.Sprintf("frame_%04d.jpg", frameID))
			if err := saveImage(outputImg, outputPath); err != nil {
				log.Printf("Failed to save frame %d: %v", frameID, err)
			}
		}

		frameCount += currentBatchSize

		if (batchEnd)%10 == 0 || batchEnd == numFrames {
			avgInf := inferenceTime / float64(currentBatchSize)
			fmt.Printf("   Processed %d/%d frames (%.2fms/frame)\n", batchEnd, numFrames, avgInf)
		}
	}

	totalTime := time.Since(batchStart).Seconds()

	// Calculate statistics
	avgInference := totalInferenceTime / float64(frameCount)
	fps := float64(frameCount) / totalTime
	throughput := 1000.0 / avgInference

	fmt.Printf("\n📊 Performance Statistics:\n")
	fmt.Printf("   Total time: %.2fs\n", totalTime)
	fmt.Printf("   Frames processed: %d\n", frameCount)
	fmt.Printf("   FPS (overall): %.2f\n", fps)
	fmt.Printf("   Avg inference time: %.2fms/frame\n", avgInference)
	fmt.Printf("   Throughput (inference only): %.1f FPS\n", throughput)

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
