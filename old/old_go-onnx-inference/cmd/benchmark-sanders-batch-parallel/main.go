package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"time"

	"go-onnx-inference/lipsyncinfer"
)

const (
	visualFrameSize = 6 * 320 * 320
	audioFrameSize  = 32 * 16 * 16
	outputFrameSize = 6 * 320 * 320
)

func loadBinaryFile(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	floatData := make([]float32, len(data)/4)
	for i := 0; i < len(floatData); i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		floatData[i] = math.Float32frombits(bits)
	}

	return floatData, nil
}

func saveFrame(outputData []float32, frameIdx int, outputDir string) error {
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))

	for c := 0; c < 3; c++ {
		channelOffset := c * 320 * 320
		for y := 0; y < 320; y++ {
			for x := 0; x < 320; x++ {
				idx := channelOffset + y*320 + x
				value := outputData[idx]

				normalized := (value + 1.0) / 2.0
				byteValue := uint8(normalized * 255.0)

				if byteValue > 255 {
					byteValue = 255
				}

				pixel := img.RGBAAt(x, y)
				switch c {
				case 0:
					pixel.B = byteValue
				case 1:
					pixel.G = byteValue
				case 2:
					pixel.R = byteValue
				}
				img.SetRGBA(x, y, pixel)
			}
		}
	}

	outputPath := filepath.Join(outputDir, fmt.Sprintf("frame_%04d.png", frameIdx))
	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer f.Close()

	return png.Encode(f, img)
}

type BatchJob struct {
	startIdx   int
	endIdx     int
	visualData []float32
	audioData  []float32
	batchSize  int
}

type BatchResult struct {
	startIdx int
	endIdx   int
	outputs  []float32
	duration time.Duration
	err      error
}

func processBatchWorker(inferencer *lipsyncinfer.Inferencer, job BatchJob, resultChan chan<- BatchResult) {
	startTime := time.Now()

	numFrames := job.endIdx - job.startIdx
	allOutputs := make([]float32, numFrames*outputFrameSize)

	// Process in batches
	for i := 0; i < numFrames; i += job.batchSize {
		batchEnd := i + job.batchSize
		if batchEnd > numFrames {
			batchEnd = numFrames
		}
		currentBatchSize := batchEnd - i

		// Extract batch data
		visualBatch := make([]float32, currentBatchSize*visualFrameSize)
		audioBatch := make([]float32, currentBatchSize*audioFrameSize)

		for j := 0; j < currentBatchSize; j++ {
			frameIdx := job.startIdx + i + j
			copy(visualBatch[j*visualFrameSize:(j+1)*visualFrameSize],
				job.visualData[frameIdx*visualFrameSize:(frameIdx+1)*visualFrameSize])
			copy(audioBatch[j*audioFrameSize:(j+1)*audioFrameSize],
				job.audioData[frameIdx*audioFrameSize:(frameIdx+1)*audioFrameSize])
		}

		// Run batch inference
		outputs, err := inferencer.InferBatch(visualBatch, audioBatch, currentBatchSize)
		if err != nil {
			resultChan <- BatchResult{
				startIdx: job.startIdx,
				endIdx:   job.endIdx,
				err:      err,
			}
			return
		}

		// Copy outputs
		copy(allOutputs[i*outputFrameSize:], outputs)
	}

	resultChan <- BatchResult{
		startIdx: job.startIdx,
		endIdx:   job.endIdx,
		outputs:  allOutputs,
		duration: time.Since(startTime),
		err:      nil,
	}
}

func main() {
	batchSize := flag.Int("batch", 4, "Batch size for inference")
	numWorkers := flag.Int("workers", 4, "Number of parallel workers")
	flag.Parse()

	fmt.Println("================================================================================")
	fmt.Println("🚀 GO + ONNX BATCH PARALLEL BENCHMARK (Sanders Dataset)")
	fmt.Println("================================================================================")
	fmt.Printf("   Batch size: %d\n", *batchSize)
	fmt.Printf("   Workers: %d\n\n", *numWorkers)

	dataDir := "d:/Projects/webcodecstest/test_data_sanders_for_go"

	fmt.Printf("📦 Loading data from: %s\n", dataDir)

	visualPath := filepath.Join(dataDir, "visual_input.bin")
	audioPath := filepath.Join(dataDir, "audio_input.bin")

	fmt.Println("\n💾 Loading binary data...")
	visualData, err := loadBinaryFile(visualPath)
	if err != nil {
		fmt.Printf("❌ Error loading visual data: %v\n", err)
		return
	}

	audioData, err := loadBinaryFile(audioPath)
	if err != nil {
		fmt.Printf("❌ Error loading audio data: %v\n", err)
		return
	}

	numFrames := len(visualData) / visualFrameSize
	fmt.Printf("   Frames: %d\n", numFrames)
	fmt.Printf("   Visual data: %.2f MB\n", float64(len(visualData)*4)/(1024*1024))
	fmt.Printf("   Audio data: %.2f MB\n", float64(len(audioData)*4)/(1024*1024))

	fmt.Println("\n🚀 Initializing ONNX inferencer...")
	modelPath := "d:/Projects/webcodecstest/minimal_server/models/sanders/checkpoint/model_best.onnx"

	inferencer, err := lipsyncinfer.NewInferencer(modelPath)
	if err != nil {
		fmt.Printf("❌ Error initializing inferencer: %v\n", err)
		return
	}
	defer inferencer.Close()

	fmt.Println("✅ ONNX Runtime session created")
	fmt.Printf("   Model loaded: %s\n", modelPath)

	// Create output directory
	outputDir := "output_go_batch_parallel"
	os.RemoveAll(outputDir)
	os.MkdirAll(outputDir, 0755)

	fmt.Printf("\n🎬 Processing %d frames with %d parallel workers (batch size: %d)...\n", numFrames, *numWorkers, *batchSize)

	// Calculate frames per worker
	framesPerWorker := numFrames / *numWorkers

	// Create jobs
	jobs := make([]BatchJob, *numWorkers)
	for i := 0; i < *numWorkers; i++ {
		startIdx := i * framesPerWorker
		endIdx := startIdx + framesPerWorker
		if i == *numWorkers-1 {
			endIdx = numFrames // Last worker gets remaining frames
		}

		jobs[i] = BatchJob{
			startIdx:   startIdx,
			endIdx:     endIdx,
			visualData: visualData,
			audioData:  audioData,
			batchSize:  *batchSize,
		}
	}

	// Launch workers
	startTime := time.Now()
	resultChan := make(chan BatchResult, *numWorkers)

	for _, job := range jobs {
		go processBatchWorker(inferencer, job, resultChan)
	}

	// Collect results
	results := make(map[int]BatchResult)
	var totalInferenceTime time.Duration

	for i := 0; i < *numWorkers; i++ {
		result := <-resultChan
		if result.err != nil {
			fmt.Printf("❌ Worker error (frames %d-%d): %v\n", result.startIdx, result.endIdx, result.err)
			return
		}

		results[result.startIdx] = result
		totalInferenceTime += result.duration

		framesProcessed := result.endIdx - result.startIdx
		fmt.Printf("   ✅ Worker completed frames %d-%d (%d frames in %s, %.2fms/frame)\n",
			result.startIdx, result.endIdx-1, framesProcessed,
			result.duration, float64(result.duration.Milliseconds())/float64(framesProcessed))
	}

	totalTime := time.Since(startTime)

	// Save frames
	fmt.Println("\n💾 Saving output frames...")
	for workerStart, result := range results {
		numWorkerFrames := result.endIdx - result.startIdx
		for i := 0; i < numWorkerFrames; i++ {
			frameIdx := workerStart + i
			frameOutput := result.outputs[i*outputFrameSize : (i+1)*outputFrameSize]
			if err := saveFrame(frameOutput, frameIdx, outputDir); err != nil {
				fmt.Printf("❌ Error saving frame %d: %v\n", frameIdx, err)
			}
		}
	}

	// Statistics
	avgInferenceTime := totalInferenceTime / time.Duration(*numWorkers)
	fps := float64(numFrames) / totalTime.Seconds()
	avgTimePerFrame := float64(totalTime.Milliseconds()) / float64(numFrames)
	throughputFPS := 1000.0 / (float64(avgInferenceTime.Milliseconds()) / float64(framesPerWorker))

	fmt.Println("\n📊 Performance Statistics:")
	fmt.Printf("   Total wall time: %.2fs\n", totalTime.Seconds())
	fmt.Printf("   Total inference time (all workers): %.2fs\n", totalInferenceTime.Seconds())
	fmt.Printf("   Avg inference time per worker: %.2fs\n", avgInferenceTime.Seconds())
	fmt.Printf("   Frames processed: %d\n", numFrames)
	fmt.Printf("   FPS (wall time): %.2f\n", fps)
	fmt.Printf("   Avg time per frame: %.2fms\n", avgTimePerFrame)
	fmt.Printf("   Speedup vs sequential: %.2fx\n", float64(totalInferenceTime)/float64(totalTime))
	fmt.Printf("   Throughput (inference only): %.1f FPS\n", throughputFPS)

	fmt.Printf("\n✅ Frames saved to %s/\n", outputDir)
}
