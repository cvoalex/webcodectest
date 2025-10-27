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
			bByte := uint8(b * 255.0)
			gByte := uint8(g * 255.0)
			rByte := uint8(r * 255.0)

			img.SetRGBA(x, y, color.RGBA{R: rByte, G: gByte, B: bByte, A: 255})
		}
	}

	return img
}

// Resize image using nearest neighbor (fast!)
func resizeNearestNeighbor(src *image.RGBA, width, height int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	srcBounds := src.Bounds()
	srcW := srcBounds.Dx()
	srcH := srcBounds.Dy()

	xRatio := float64(srcW) / float64(width)
	yRatio := float64(srcH) / float64(height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			srcX := int(float64(x) * xRatio)
			srcY := int(float64(y) * yRatio)
			dst.Set(x, y, src.At(srcX, srcY))
		}
	}

	return dst
}

// Composite mouth region onto background frame
func compositeFrame(mouthRegion *image.RGBA, background *image.RGBA, x, y, w, h int) *image.RGBA {
	// Resize mouth region to target size
	resized := resizeNearestNeighbor(mouthRegion, w, h)

	// Create output frame (copy of background)
	result := image.NewRGBA(background.Bounds())
	draw.Draw(result, result.Bounds(), background, image.Point{}, draw.Src)

	// Paste resized mouth into output
	rect := image.Rect(x, y, x+w, y+h)
	draw.Draw(result, rect, resized, image.Point{}, draw.Src)

	return result
}

type CompositeJob struct {
	frameIdx   int
	mouthImage *image.RGBA
	background *image.RGBA
	cropRect   []int // [x1, y1, x2, y2]
}

type CompositeResult struct {
	frameIdx int
	frame    *image.RGBA
	duration time.Duration
	err      error
}

func compositeWorker(jobs <-chan CompositeJob, results chan<- CompositeResult) {
	for job := range jobs {
		start := time.Now()
		// Calculate width and height from crop rectangle
		x1, y1, x2, y2 := job.cropRect[0], job.cropRect[1], job.cropRect[2], job.cropRect[3]
		w := x2 - x1
		h := y2 - y1
		composited := compositeFrame(job.mouthImage, job.background, x1, y1, w, h)
		results <- CompositeResult{
			frameIdx: job.frameIdx,
			frame:    composited,
			duration: time.Since(start),
			err:      nil,
		}
	}
}

func main() {
	numWorkers := flag.Int("workers", 5, "Number of parallel compositing workers")
	flag.Parse()

	fmt.Println("================================================================================")
	fmt.Println("🚀 GO BATCH INFERENCE → PARALLEL COMPOSITING")
	fmt.Println("================================================================================")
	fmt.Printf("   Compositing workers: %d\n", *numWorkers)
	fmt.Printf("   Strategy: Infer ALL frames in batch, then parallel composite\n")
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
	}
	cacheTime := time.Since(cacheStart)
	fmt.Printf("✅ Background cache loaded in %.2fs (%d frames)\n", cacheTime.Seconds(), numFrames)

	// Load crop rectangles
	fmt.Println("\n💾 Loading crop rectangles...")
	cropRectsPath := "d:/Projects/webcodecstest/minimal_server/models/sanders/cache/crop_rectangles.json"
	cropRectsFile, err := os.ReadFile(cropRectsPath)
	if err != nil {
		fmt.Printf("❌ Error loading crop rectangles: %v\n", err)
		return
	}

	var cropRects map[string]CropRect
	if err := json.Unmarshal(cropRectsFile, &cropRects); err != nil {
		fmt.Printf("❌ Error parsing crop rectangles: %v\n", err)
		return
	}
	fmt.Printf("✅ Loaded %d crop rectangles\n", len(cropRects))

	modelPath := "d:/Projects/webcodecstest/minimal_server/models/sanders/checkpoint/model_best.onnx"
	fmt.Printf("\n📁 Model path: %s\n", modelPath)

	// Create batch inferencer (supports dynamic batch sizes!)
	fmt.Println("\n🚀 Creating batch inferencer...")
	batchInferencer, err := lipsyncinfer.NewBatchInferencer(modelPath)
	if err != nil {
		fmt.Printf("❌ Error creating batch inferencer: %v\n", err)
		return
	}
	defer batchInferencer.Close()

	// WARMUP: Run one batch inference to warm up CUDA
	fmt.Println("\n🔥 WARMUP: Running one batch inference to warm up CUDA...")
	warmupStart := time.Now()
	_, err = batchInferencer.InferBatch(visualData, audioData, numFrames)
	warmupTime := time.Since(warmupStart)
	if err != nil {
		fmt.Printf("❌ Warmup error: %v\n", err)
		return
	}
	fmt.Printf("✅ Warmup complete: %.2fs (this time is NOT counted in benchmark)\n\n",
		warmupTime.Seconds())

	// STEP 1: TRUE Batch inference (ALL frames at once, ONE GPU call!)
	fmt.Printf("🎬 STEP 1: TRUE Batch inference (%d frames - ONE GPU call!)...\n", numFrames)
	inferStart := time.Now()
	outputs, err := batchInferencer.InferBatch(visualData, audioData, numFrames)
	inferTime := time.Since(inferStart)

	if err != nil {
		fmt.Printf("❌ Inference error: %v\n", err)
		return
	}

	fmt.Printf("✅ Batch inference complete: %.2fs (%.2fms/frame)\n",
		inferTime.Seconds(), float64(inferTime.Milliseconds())/float64(numFrames))

	// Convert outputs to images
	fmt.Println("\n🖼️  Converting outputs to images...")
	convertStart := time.Now()
	mouthImages := make([]*image.RGBA, numFrames)
	for i := 0; i < numFrames; i++ {
		frameOutput := outputs[i*outputFrameSize : (i+1)*outputFrameSize]
		mouthImages[i] = outputToImage(frameOutput)
	}
	convertTime := time.Since(convertStart)
	fmt.Printf("✅ Conversion complete: %.2fs\n", convertTime.Seconds())

	// STEP 2: Parallel compositing
	fmt.Printf("\n🎨 STEP 2: Parallel compositing (%d workers)...\n", *numWorkers)
	compStart := time.Now()

	// Create job and result channels
	jobs := make(chan CompositeJob, numFrames)
	results := make(chan CompositeResult, numFrames)

	// Start workers
	for w := 0; w < *numWorkers; w++ {
		go compositeWorker(jobs, results)
	}

	// Send jobs
	for i := 0; i < numFrames; i++ {
		// Get crop rectangle for this frame
		frameKey := fmt.Sprintf("%d", i)
		cropRect, ok := cropRects[frameKey]
		if !ok {
			fmt.Printf("❌ No crop rectangle for frame %d\n", i)
			return
		}

		jobs <- CompositeJob{
			frameIdx:   i,
			mouthImage: mouthImages[i],
			background: backgrounds[i],
			cropRect:   cropRect.Rect,
		}
	}
	close(jobs)

	// Collect results
	compositedFrames := make([]*image.RGBA, numFrames)
	var totalCompTime time.Duration
	for i := 0; i < numFrames; i++ {
		result := <-results
		if result.err != nil {
			fmt.Printf("❌ Composite error frame %d: %v\n", result.frameIdx, result.err)
			return
		}
		compositedFrames[result.frameIdx] = result.frame
		totalCompTime += result.duration
		fmt.Printf("   ✅ Frame %d composited (%.2fms)\n", result.frameIdx,
			float64(result.duration.Microseconds())/1000.0)
	}
	compTime := time.Since(compStart)

	fmt.Printf("✅ All compositing complete: %.2fs (wall time)\n", compTime.Seconds())
	fmt.Printf("   Total composite time (all workers): %.2fs\n", totalCompTime.Seconds())
	fmt.Printf("   Speedup: %.2fx\n", float64(totalCompTime)/float64(compTime))

	// Save frames
	fmt.Println("\n💾 Saving output frames...")
	outputDir := "output_batch_then_parallel_composite"
	os.RemoveAll(outputDir)
	os.MkdirAll(outputDir, 0755)

	saveStart := time.Now()
	for i, frame := range compositedFrames {
		outputPath := filepath.Join(outputDir, fmt.Sprintf("frame_%04d.png", i))
		file, err := os.Create(outputPath)
		if err != nil {
			fmt.Printf("❌ Error creating file %d: %v\n", i, err)
			continue
		}
		png.Encode(file, frame)
		file.Close()
	}
	saveTime := time.Since(saveStart)

	// Final statistics
	totalTime := inferTime + convertTime + compTime + saveTime
	avgInferPerFrame := float64(inferTime.Milliseconds()) / float64(numFrames)
	avgCompPerFrame := float64(totalCompTime.Milliseconds()) / float64(numFrames)
	fps := float64(numFrames) / (inferTime + compTime).Seconds()

	fmt.Println("\n📊 Performance Statistics:")
	fmt.Println("\n⏱️  Timing Breakdown:")
	fmt.Printf("   Background cache: %.2fs\n", cacheTime.Seconds())
	fmt.Printf("   Batch inference: %.2fs (%.2fms/frame)\n", inferTime.Seconds(), avgInferPerFrame)
	fmt.Printf("   Output conversion: %.2fs\n", convertTime.Seconds())
	fmt.Printf("   Parallel composite (wall): %.2fs\n", compTime.Seconds())
	fmt.Printf("   Parallel composite (total): %.2fs (%.2fms/frame)\n",
		totalCompTime.Seconds(), avgCompPerFrame)
	fmt.Printf("   Frame save: %.2fs\n", saveTime.Seconds())
	fmt.Printf("   Total time: %.2fs\n", totalTime.Seconds())

	fmt.Println("\n🚀 Throughput:")
	fmt.Printf("   FPS (inference + composite): %.2f\n", fps)
	fmt.Printf("   Composite speedup: %.2fx\n", float64(totalCompTime)/float64(compTime))

	fmt.Printf("\n✅ Frames saved to %s/\n", outputDir)
	fmt.Println("\n🏁 BATCH INFERENCE → PARALLEL COMPOSITE COMPLETE!")
}
