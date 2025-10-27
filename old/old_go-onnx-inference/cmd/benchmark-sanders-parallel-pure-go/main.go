package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"time"

	"go-onnx-inference/lipsyncinfer"
)

type CropRect struct {
	Rect []int `json:"rect"` // [x1, y1, x2, y2]
}

const (
	visualFrameSize = 6 * 320 * 320
	audioFrameSize  = 32 * 16 * 16
	outputFrameSize = 3 * 320 * 320 // Output is 3 channels (BGR)
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

// Convert ONNX output to Go image.Image
func outputToImage(outputData []float32) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))

	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// BGR order from ONNX
			b := outputData[0*320*320+y*320+x]
			g := outputData[1*320*320+y*320+x]
			r := outputData[2*320*320+y*320+x]

			// Output is already in [0, 1] range, just multiply by 255
			rByte := uint8(clamp(r * 255.0))
			gByte := uint8(clamp(g * 255.0))
			bByte := uint8(clamp(b * 255.0))

			img.SetRGBA(x, y, color.RGBA{R: rByte, G: gByte, B: bByte, A: 255})
		}
	}

	return img
}

func clamp(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 255 {
		return 255
	}
	return val
}

// Bilinear resize - pure Go implementation
func resizeImage(src *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	dst := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))

	xRatio := float32(srcWidth) / float32(targetWidth)
	yRatio := float32(srcHeight) / float32(targetHeight)

	for dstY := 0; dstY < targetHeight; dstY++ {
		for dstX := 0; dstX < targetWidth; dstX++ {
			srcX := float32(dstX) * xRatio
			srcY := float32(dstY) * yRatio

			x0 := int(srcX)
			y0 := int(srcY)
			x1 := x0 + 1
			y1 := y0 + 1

			if x1 >= srcWidth {
				x1 = srcWidth - 1
			}
			if y1 >= srcHeight {
				y1 = srcHeight - 1
			}

			xWeight := srcX - float32(x0)
			yWeight := srcY - float32(y0)

			// Get four neighboring pixels
			c00 := src.RGBAAt(x0, y0)
			c10 := src.RGBAAt(x1, y0)
			c01 := src.RGBAAt(x0, y1)
			c11 := src.RGBAAt(x1, y1)

			// Bilinear interpolation
			r := bilinearInterp(c00.R, c10.R, c01.R, c11.R, xWeight, yWeight)
			g := bilinearInterp(c00.G, c10.G, c01.G, c11.G, xWeight, yWeight)
			b := bilinearInterp(c00.B, c10.B, c01.B, c11.B, xWeight, yWeight)

			dst.SetRGBA(dstX, dstY, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return dst
}

func bilinearInterp(c00, c10, c01, c11 uint8, xWeight, yWeight float32) uint8 {
	top := float32(c00)*(1-xWeight) + float32(c10)*xWeight
	bottom := float32(c01)*(1-xWeight) + float32(c11)*xWeight
	result := top*(1-yWeight) + bottom*yWeight
	return uint8(result)
}

// Composite mouth region onto background - pure Go
func compositeFrame(mouthRegion *image.RGBA, background *image.RGBA, x, y, w, h int) *image.RGBA {
	// Resize mouth region
	resized := resizeImage(mouthRegion, w, h)

	// Clone background
	result := image.NewRGBA(background.Bounds())
	draw.Draw(result, result.Bounds(), background, image.Point{}, draw.Src)

	// Paste mouth region
	dstRect := image.Rect(x, y, x+w, y+h)
	draw.Draw(result, dstRect, resized, image.Point{}, draw.Src)

	return result
}

type BatchJob struct {
	startIdx    int
	endIdx      int
	visualData  []float32
	audioData   []float32
	batchSize   int
	backgrounds []*image.RGBA
	cropRects   [][]int // Array of [x1, y1, x2, y2] for each frame
}

type BatchResult struct {
	startIdx  int
	endIdx    int
	frames    []*image.RGBA
	inferTime time.Duration
	compTime  time.Duration
	totalTime time.Duration
	err       error
}

func processBatchWorkerWithComposite(inferencer *lipsyncinfer.Inferencer, job BatchJob, resultChan chan<- BatchResult) {
	totalStart := time.Now()

	numFrames := job.endIdx - job.startIdx
	outputFrames := make([]*image.RGBA, numFrames)

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

			// Convert output to image
			mouthRegion := outputToImage(frameOutput)

			// Get crop rectangle for this GLOBAL frame number (not relative to batch)
			cropRect := job.cropRects[globalFrameIdx]
			x1, y1, x2, y2 := cropRect[0], cropRect[1], cropRect[2], cropRect[3]
			w := x2 - x1
			h := y2 - y1

			// Composite with background using correct crop rectangle
			composited := compositeFrame(mouthRegion, job.backgrounds[globalFrameIdx],
				x1, y1, w, h)

			outputFrames[frameIdx] = composited
			frameIdx++
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
	fmt.Println("🚀 GO PURE - PARALLEL BATCH WITH COMPOSITING (Sanders Dataset)")
	fmt.Println("================================================================================")
	fmt.Printf("   Batch size: %d\n", *batchSize)
	fmt.Printf("   Workers: %d\n", *numWorkers)
	fmt.Printf("   Output: 1280x720 (PURE GO - NO PYTHON, NO OPENCV!)\n\n")

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
	bgDir := "d:/Projects/webcodecstest/minimal_server/models/sanders/frames"
	backgrounds := make([]*image.RGBA, numFrames)

	cacheStart := time.Now()
	for i := 0; i < numFrames; i++ {
		framePath := filepath.Join(bgDir, fmt.Sprintf("frame_%04d.png", i))
		file, err := os.Open(framePath)
		if err != nil {
			fmt.Printf("❌ Error loading background frame %d: %v\n", i, err)
			return
		}

		img, err := png.Decode(file)
		file.Close()
		if err != nil {
			fmt.Printf("❌ Error decoding background frame %d: %v\n", i, err)
			return
		}

		// Convert to RGBA
		rgba := image.NewRGBA(img.Bounds())
		draw.Draw(rgba, rgba.Bounds(), img, image.Point{}, draw.Src)
		backgrounds[i] = rgba

		if (i+1)%20 == 0 || i == numFrames-1 {
			fmt.Printf("   Loaded %d/%d frames (%.2f MB)\n", i+1, numFrames,
				float64((i+1)*1280*720*4)/(1024*1024))
		}
	}
	cacheTime := time.Since(cacheStart)
	fmt.Printf("✅ Background cache loaded in %.2fs\n", cacheTime.Seconds())

	// Load crop rectangles
	fmt.Println("\n💾 Loading crop rectangles...")
	cropRectsPath := "d:/Projects/webcodecstest/minimal_server/models/sanders/cache/crop_rectangles.json"
	cropRectsFile, err := os.ReadFile(cropRectsPath)
	if err != nil {
		fmt.Printf("❌ Error loading crop rectangles: %v\n", err)
		return
	}

	var cropRectsMap map[string]CropRect
	if err := json.Unmarshal(cropRectsFile, &cropRectsMap); err != nil {
		fmt.Printf("❌ Error parsing crop rectangles: %v\n", err)
		return
	}

	// Convert map to array indexed by frame number
	cropRects := make([][]int, numFrames)
	for i := 0; i < numFrames; i++ {
		frameKey := fmt.Sprintf("%d", i)
		if rect, ok := cropRectsMap[frameKey]; ok {
			cropRects[i] = rect.Rect
		} else {
			fmt.Printf("❌ No crop rectangle for frame %d\n", i)
			return
		}
	}
	fmt.Printf("✅ Loaded %d crop rectangles\n", len(cropRects))

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

	// WARMUP: Run one inference to warm up CUDA
	fmt.Println("\n🔥 WARMUP: Running one inference to warm up CUDA...")
	warmupStart := time.Now()
	_, err = inferencer.Infer(
		visualData[:visualFrameSize],
		audioData[:audioFrameSize],
	)
	warmupTime := time.Since(warmupStart)
	if err != nil {
		fmt.Printf("❌ Warmup error: %v\n", err)
		return
	}
	fmt.Printf("✅ Warmup complete: %.2fms (this time is NOT counted in benchmark)\n",
		float64(warmupTime.Milliseconds()))

	// Create output directory
	outputDir := "output_go_pure_parallel_composite"
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
			endIdx = numFrames
		}

		jobs[i] = BatchJob{
			startIdx:    startIdx,
			endIdx:      endIdx,
			visualData:  visualData,
			audioData:   audioData,
			batchSize:   *batchSize,
			backgrounds: backgrounds,
			cropRects:   cropRects,
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

			file, err := os.Create(outputPath)
			if err != nil {
				fmt.Printf("❌ Error creating file %d: %v\n", frameIdx, err)
				continue
			}

			png.Encode(file, frame)
			file.Close()
		}
	}
	saveTime := time.Since(saveStart)

	// Statistics
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

	fmt.Println("\n🎯 Per-Frame Averages:")
	fmt.Printf("   Frames processed: %d\n", numFrames)
	fmt.Printf("   Avg total time: %.2fms/frame\n", avgTimePerFrame)
	fmt.Printf("   Avg inference: %.2fms/frame\n", avgInferPerFrame)
	fmt.Printf("   Avg composite: %.2fms/frame\n", avgCompPerFrame)

	fmt.Println("\n🚀 Throughput:")
	fmt.Printf("   FPS (wall time): %.2f\n", fps)
	fmt.Printf("   Speedup vs sequential: %.2fx\n", speedup)

	fmt.Printf("\n✅ Frames saved to %s/\n", outputDir)
	fmt.Println("\n🏁 100% PURE GO - NO PYTHON - NO OPENCV - COMPLETE!")
}
