package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"time"

	"go-onnx-inference/lipsyncinfer"

	"gocv.io/x/gocv"
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

func outputToMat(outputData []float32) gocv.Mat {
	img := gocv.NewMatWithSize(320, 320, gocv.MatTypeCV8UC3)

	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// BGR order (OpenCV format)
			b := outputData[0*320*320+y*320+x]
			g := outputData[1*320*320+y*320+x]
			r := outputData[2*320*320+y*320+x]

			// Denormalize from [-1, 1] to [0, 255]
			bByte := uint8(((b + 1.0) / 2.0) * 255.0)
			gByte := uint8(((g + 1.0) / 2.0) * 255.0)
			rByte := uint8(((r + 1.0) / 2.0) * 255.0)

			img.SetUCharAt(y, x*3+0, bByte) // B
			img.SetUCharAt(y, x*3+1, gByte) // G
			img.SetUCharAt(y, x*3+2, rByte) // R
		}
	}

	return img
}

func compositeFrame(mouthRegion gocv.Mat, backgroundFrame gocv.Mat, x, y, w, h int) gocv.Mat {
	// Resize mouth region to target size
	resized := gocv.NewMat()
	gocv.Resize(mouthRegion, &resized, image.Point{X: w, Y: h}, 0, 0, gocv.InterpolationLinear)

	// Create output frame (copy of background)
	result := backgroundFrame.Clone()

	// Get ROI (Region of Interest) in the result frame
	roi := result.Region(image.Rect(x, y, x+w, y+h))

	// Copy resized mouth into ROI
	resized.CopyTo(&roi)

	resized.Close()
	roi.Close()

	return result
}

type BatchJob struct {
	startIdx    int
	endIdx      int
	visualData  []float32
	audioData   []float32
	batchSize   int
	backgrounds []gocv.Mat
}

type BatchResult struct {
	startIdx  int
	endIdx    int
	frames    []gocv.Mat
	inferTime time.Duration
	compTime  time.Duration
	totalTime time.Duration
	err       error
}

func processBatchWorkerWithComposite(inferencer *lipsyncinfer.Inferencer, job BatchJob, resultChan chan<- BatchResult) {
	totalStart := time.Now()

	numFrames := job.endIdx - job.startIdx
	outputFrames := make([]gocv.Mat, numFrames)

	var totalInferTime time.Duration
	var totalCompTime time.Duration

	// Process in batches
	frameIdx := 0
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
			globalFrameIdx := job.startIdx + i + j
			copy(visualBatch[j*visualFrameSize:(j+1)*visualFrameSize],
				job.visualData[globalFrameIdx*visualFrameSize:(globalFrameIdx+1)*visualFrameSize])
			copy(audioBatch[j*audioFrameSize:(j+1)*audioFrameSize],
				job.audioData[globalFrameIdx*audioFrameSize:(globalFrameIdx+1)*audioFrameSize])
		}

		// Run batch inference
		inferStart := time.Now()
		outputs, err := inferencer.InferBatch(visualBatch, audioBatch, currentBatchSize)
		inferTime := time.Since(inferStart)
		totalInferTime += inferTime

		if err != nil {
			resultChan <- BatchResult{
				startIdx: job.startIdx,
				endIdx:   job.endIdx,
				err:      err,
			}
			return
		}

		// Composite each frame in the batch
		compStart := time.Now()
		for j := 0; j < currentBatchSize; j++ {
			globalFrameIdx := job.startIdx + i + j
			frameOutput := outputs[j*outputFrameSize : (j+1)*outputFrameSize]

			// Convert output to OpenCV Mat
			mouthRegion := outputToMat(frameOutput)

			// Composite with background
			composited := compositeFrame(mouthRegion, job.backgrounds[globalFrameIdx],
				480, 240, 320, 320) // Center mouth region in 1280x720 frame

			outputFrames[frameIdx] = composited
			frameIdx++

			mouthRegion.Close()
		}
		compTime := time.Since(compStart)
		totalCompTime += compTime
	}

	resultChan <- BatchResult{
		startIdx:  job.startIdx,
		endIdx:    job.endIdx,
		frames:    outputFrames,
		inferTime: totalInferTime,
		compTime:  totalCompTime,
		totalTime: time.Since(totalStart),
		err:       nil,
	}
}

func main() {
	batchSize := flag.Int("batch", 4, "Batch size for inference")
	numWorkers := flag.Int("workers", 4, "Number of parallel workers")
	flag.Parse()

	fmt.Println("================================================================================")
	fmt.Println("🚀 GO + ONNX BATCH PARALLEL WITH COMPOSITING (Sanders Dataset)")
	fmt.Println("================================================================================")
	fmt.Printf("   Batch size: %d\n", *batchSize)
	fmt.Printf("   Workers: %d\n", *numWorkers)
	fmt.Printf("   Output: 1280x720 (full compositing pipeline)\n\n")

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

	// Load background frames into RAM cache
	fmt.Println("\n💾 Loading background frames into RAM cache...")
	bgDir := "d:/Projects/webcodecstest/realtime_lipsync/data/sanders/frames"
	backgrounds := make([]gocv.Mat, numFrames)

	cacheStart := time.Now()
	for i := 0; i < numFrames; i++ {
		framePath := filepath.Join(bgDir, fmt.Sprintf("frame_%04d.png", i))
		mat := gocv.IMRead(framePath, gocv.IMReadColor)
		if mat.Empty() {
			fmt.Printf("❌ Error loading background frame %d from %s\n", i, framePath)
			return
		}
		backgrounds[i] = mat

		if (i+1)%20 == 0 || i == numFrames-1 {
			fmt.Printf("   Loaded %d/%d frames (%.2f MB)\n", i+1, numFrames,
				float64((i+1)*1280*720*3)/(1024*1024))
		}
	}
	cacheTime := time.Since(cacheStart)
	fmt.Printf("✅ Background cache loaded in %.2fs\n", cacheTime.Seconds())

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
	outputDir := "output_go_batch_parallel_composite"
	os.RemoveAll(outputDir)
	os.MkdirAll(outputDir, 0755)

	fmt.Printf("\n🎬 Processing %d frames with %d parallel workers (batch size: %d)...\n",
		numFrames, *numWorkers, *batchSize)

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
			startIdx:    startIdx,
			endIdx:      endIdx,
			visualData:  visualData,
			audioData:   audioData,
			batchSize:   *batchSize,
			backgrounds: backgrounds,
		}
	}

	// Launch workers
	startTime := time.Now()
	resultChan := make(chan BatchResult, *numWorkers)

	for _, job := range jobs {
		go processBatchWorkerWithComposite(inferencer, job, resultChan)
	}

	// Collect results
	results := make(map[int]BatchResult)
	var totalInferenceTime time.Duration
	var totalCompositeTime time.Duration
	var totalWorkerTime time.Duration

	for i := 0; i < *numWorkers; i++ {
		result := <-resultChan
		if result.err != nil {
			fmt.Printf("❌ Worker error (frames %d-%d): %v\n", result.startIdx, result.endIdx, result.err)
			return
		}

		results[result.startIdx] = result
		totalInferenceTime += result.inferTime
		totalCompositeTime += result.compTime
		totalWorkerTime += result.totalTime

		framesProcessed := result.endIdx - result.startIdx
		avgInferMs := float64(result.inferTime.Milliseconds()) / float64(framesProcessed)
		avgCompMs := float64(result.compTime.Milliseconds()) / float64(framesProcessed)

		fmt.Printf("   ✅ Worker completed frames %d-%d (%d frames in %s)\n",
			result.startIdx, result.endIdx-1, framesProcessed, result.totalTime)
		fmt.Printf("      Inference: %.2fms/frame, Composite: %.2fms/frame\n",
			avgInferMs, avgCompMs)
	}

	totalTime := time.Since(startTime)

	// Save frames
	fmt.Println("\n💾 Saving output frames...")
	saveStart := time.Now()
	for workerStart, result := range results {
		for i, frame := range result.frames {
			frameIdx := workerStart + i
			outputPath := filepath.Join(outputDir, fmt.Sprintf("frame_%04d.png", frameIdx))
			gocv.IMWrite(outputPath, frame)
			frame.Close()
		}
	}
	saveTime := time.Since(saveStart)

	// Clean up background cache
	for _, bg := range backgrounds {
		bg.Close()
	}

	// Statistics
	avgInferTimePerWorker := totalInferenceTime / time.Duration(*numWorkers)
	avgCompTimePerWorker := totalCompositeTime / time.Duration(*numWorkers)
	fps := float64(numFrames) / totalTime.Seconds()
	avgTimePerFrame := float64(totalTime.Milliseconds()) / float64(numFrames)
	avgInferPerFrame := float64(totalInferenceTime.Milliseconds()) / float64(numFrames)
	avgCompPerFrame := float64(totalCompositeTime.Milliseconds()) / float64(numFrames)
	speedup := float64(totalWorkerTime) / float64(totalTime)

	fmt.Println("\n📊 Performance Statistics:")
	fmt.Println("\n⏱️  Timing Breakdown:")
	fmt.Printf("   Background cache load: %.2fs\n", cacheTime.Seconds())
	fmt.Printf("   Total wall time: %.2fs\n", totalTime.Seconds())
	fmt.Printf("   Frame save time: %.2fs\n", saveTime.Seconds())

	fmt.Println("\n🔄 Worker Times:")
	fmt.Printf("   Total inference time (all workers): %.2fs\n", totalInferenceTime.Seconds())
	fmt.Printf("   Total composite time (all workers): %.2fs\n", totalCompositeTime.Seconds())
	fmt.Printf("   Total worker time (all workers): %.2fs\n", totalWorkerTime.Seconds())
	fmt.Printf("   Avg inference time per worker: %.2fs\n", avgInferTimePerWorker.Seconds())
	fmt.Printf("   Avg composite time per worker: %.2fs\n", avgCompTimePerWorker.Seconds())

	fmt.Println("\n🎯 Per-Frame Averages:")
	fmt.Printf("   Frames processed: %d\n", numFrames)
	fmt.Printf("   Avg total time: %.2fms/frame\n", avgTimePerFrame)
	fmt.Printf("   Avg inference: %.2fms/frame\n", avgInferPerFrame)
	fmt.Printf("   Avg composite: %.2fms/frame\n", avgCompPerFrame)

	fmt.Println("\n🚀 Throughput:")
	fmt.Printf("   FPS (wall time): %.2f\n", fps)
	fmt.Printf("   Speedup vs sequential: %.2fx\n", speedup)

	fmt.Printf("\n✅ Frames saved to %s/\n", outputDir)
	fmt.Println("\n🏁 FULL COMPOSITING PIPELINE COMPLETE!")
}
