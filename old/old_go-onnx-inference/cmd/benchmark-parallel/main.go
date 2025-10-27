package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
	"unsafe"

	"go-onnx-inference/lipsyncinfer"

	"gocv.io/x/gocv"
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

type CropRectangle struct {
	Rect         []int  `json:"rect"`
	OriginalPath string `json:"original_path"`
	Crop328Path  string `json:"crop_328_path"`
}

type CompositeJob struct {
	FrameID    int
	Prediction []float32
}

type CompositeResult struct {
	FrameID int
	Frame   gocv.Mat
}

// CachedData holds all preloaded video frames and metadata
type CachedData struct {
	Crop328Frames  []gocv.Mat
	FullBodyFrames []gocv.Mat
	CropRectangles map[string]CropRectangle
	NumFrames      int
}

func main() {
	// Parse flags
	dataDir := flag.String("data", "d:/Projects/webcodecstest/test_data_sanders_for_go", "Test data directory")
	sandersDir := flag.String("sanders", "d:/Projects/webcodecstest/minimal_server/models/sanders", "Sanders directory with videos")
	modelPath := flag.String("model", "d:/Projects/webcodecstest/minimal_server/models/sanders/checkpoint/model_best.onnx", "ONNX model path")
	outputDir := flag.String("output", "output_go_parallel", "Output directory")
	batchSize := flag.Int("batch", 4, "Batch size for inference")
	numWorkers := flag.Int("workers", 4, "Number of parallel compositing workers")
	flag.Parse()

	fmt.Println("\n" + "================================================================================")
	fmt.Println("🚀 GO + ONNX PARALLEL BENCHMARK")
	fmt.Println("================================================================================")
	fmt.Printf("   Batch size: %d\n", *batchSize)
	fmt.Printf("   Composite workers: %d\n", *numWorkers)

	// Create output directory
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

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

	// Preload all video data
	fmt.Printf("\n💾 Preloading ALL data into RAM...\n")
	preloadStart := time.Now()

	cachedData, err := preloadAllData(*sandersDir, metadata.NumFrames)
	if err != nil {
		log.Fatalf("Failed to preload data: %v", err)
	}
	defer cleanupCachedData(cachedData)

	preloadTime := time.Since(preloadStart).Seconds()
	fmt.Printf("⚡ Preload completed in %.2fs\n", preloadTime)

	// Load binary data
	fmt.Printf("\n💾 Loading inference inputs...\n")
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

	// Process frames with parallel compositing
	fmt.Printf("\n🎬 Processing %d frames (Batch: %d, Workers: %d)...\n",
		metadata.NumFrames, *batchSize, *numWorkers)

	var totalInferenceTime float64
	var totalCompositeTime float64
	var totalSaveTime float64
	frameCount := 0

	visualFrameSize := 6 * 320 * 320
	audioFrameSize := 32 * 16 * 16

	batchStart := time.Now()

	// Process in batches
	for batchIdx := 0; batchIdx < metadata.NumFrames; batchIdx += *batchSize {
		batchEnd := batchIdx + *batchSize
		if batchEnd > metadata.NumFrames {
			batchEnd = metadata.NumFrames
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

		// Batch inference (run frames sequentially, but composite in parallel)
		inferenceStart := time.Now()
		batchOutput := make([]float32, currentBatchSize*3*320*320)
		for i := 0; i < currentBatchSize; i++ {
			visualStart := i * visualFrameSize
			visualFrame := batchVisual[visualStart : visualStart+visualFrameSize]

			audioStart := i * audioFrameSize
			audioFrame := batchAudio[audioStart : audioStart+audioFrameSize]

			output, err := inferencer.Infer(visualFrame, audioFrame)
			if err != nil {
				log.Printf("Inference failed for frame %d: %v", batchIdx+i, err)
				continue
			}

			outputStart := i * 3 * 320 * 320
			copy(batchOutput[outputStart:outputStart+3*320*320], output)
		}
		inferenceTime := time.Since(inferenceStart).Seconds() * 1000
		totalInferenceTime += inferenceTime

		// Parallel compositing
		compositeStart := time.Now()
		compositedFrames := parallelComposite(
			batchOutput,
			batchIdx,
			currentBatchSize,
			cachedData,
			*numWorkers,
		)
		compositeTime := time.Since(compositeStart).Seconds() * 1000
		totalCompositeTime += compositeTime

		// Save frames
		saveStart := time.Now()
		for i, mat := range compositedFrames {
			frameID := batchIdx + i
			outputPath := filepath.Join(*outputDir, fmt.Sprintf("frame_%04d.jpg", frameID))
			if ok := gocv.IMWrite(outputPath, mat); !ok {
				log.Printf("Failed to save frame %d", frameID)
			}
			mat.Close()
		}
		saveTime := time.Since(saveStart).Seconds() * 1000
		totalSaveTime += saveTime

		frameCount += currentBatchSize

		// Print progress
		if batchEnd%20 == 0 || batchEnd == metadata.NumFrames {
			avgInf := inferenceTime / float64(currentBatchSize)
			avgComp := compositeTime / float64(currentBatchSize)
			avgSave := saveTime / float64(currentBatchSize)
			fmt.Printf("   Processed %d/%d frames (%.2fms inf, %.2fms comp, %.2fms save)\n",
				batchEnd, metadata.NumFrames, avgInf, avgComp, avgSave)
		}
	}

	totalTime := time.Since(batchStart).Seconds()

	// Statistics
	avgInference := totalInferenceTime / float64(frameCount)
	avgComposite := totalCompositeTime / float64(frameCount)
	avgSave := totalSaveTime / float64(frameCount)
	fps := float64(frameCount) / totalTime
	throughput := 1000.0 / avgInference

	fmt.Printf("\n📊 Performance Statistics:\n")
	fmt.Printf("   Total time: %.2fs\n", totalTime)
	fmt.Printf("   Frames processed: %d\n", frameCount)
	fmt.Printf("   FPS (overall): %.2f\n", fps)
	fmt.Printf("   Avg inference time: %.2fms/frame\n", avgInference)
	fmt.Printf("   Avg composite time: %.2fms/frame\n", avgComposite)
	fmt.Printf("   Avg save time: %.2fms/frame\n", avgSave)
	fmt.Printf("   Throughput (inference only): %.1f FPS\n", throughput)

	fmt.Printf("\n✅ Frames saved to %s/\n", *outputDir)
}

func preloadAllData(sandersDir string, numFrames int) (*CachedData, error) {
	cached := &CachedData{
		Crop328Frames:  make([]gocv.Mat, 0, numFrames),
		FullBodyFrames: make([]gocv.Mat, 0, numFrames),
		CropRectangles: make(map[string]CropRectangle),
	}

	// Load crop rectangles
	fmt.Println("   Loading crop rectangles...")
	cropRectsPath := filepath.Join(sandersDir, "cache", "crop_rectangles.json")
	cropRectsBytes, err := os.ReadFile(cropRectsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read crop_rectangles.json: %w", err)
	}

	if err := json.Unmarshal(cropRectsBytes, &cached.CropRectangles); err != nil {
		return nil, fmt.Errorf("failed to parse crop_rectangles.json: %w", err)
	}
	fmt.Printf("      ✅ %d crop rectangles\n", len(cached.CropRectangles))

	// Load crops_328 video
	fmt.Println("   Loading crops_328_video.mp4...")
	crops328Path := filepath.Join(sandersDir, "crops_328_video.mp4")
	crops328Cap := gocv.VideoCaptureFile(crops328Path)
	if !crops328Cap.IsOpened() {
		return nil, fmt.Errorf("failed to open crops_328_video.mp4")
	}
	defer crops328Cap.Close()

	for {
		mat := gocv.NewMat()
		if ok := crops328Cap.Read(&mat); !ok {
			mat.Close()
			break
		}
		cached.Crop328Frames = append(cached.Crop328Frames, mat)
	}
	fmt.Printf("      ✅ %d frames\n", len(cached.Crop328Frames))

	// Load full_body video
	fmt.Println("   Loading full_body_video.mp4...")
	fullBodyPath := filepath.Join(sandersDir, "full_body_video.mp4")
	fullBodyCap := gocv.VideoCaptureFile(fullBodyPath)
	if !fullBodyCap.IsOpened() {
		return nil, fmt.Errorf("failed to open full_body_video.mp4")
	}
	defer fullBodyCap.Close()

	for {
		mat := gocv.NewMat()
		if ok := fullBodyCap.Read(&mat); !ok {
			mat.Close()
			break
		}
		cached.FullBodyFrames = append(cached.FullBodyFrames, mat)
	}
	fmt.Printf("      ✅ %d frames\n", len(cached.FullBodyFrames))

	cached.NumFrames = len(cached.Crop328Frames)
	return cached, nil
}

func cleanupCachedData(cached *CachedData) {
	for _, mat := range cached.Crop328Frames {
		mat.Close()
	}
	for _, mat := range cached.FullBodyFrames {
		mat.Close()
	}
}

func parallelComposite(
	predictions []float32,
	startFrameID int,
	batchSize int,
	cached *CachedData,
	numWorkers int,
) []gocv.Mat {

	// Create job channel and result channel
	jobs := make(chan CompositeJob, batchSize)
	results := make(chan CompositeResult, batchSize)

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go compositeWorker(jobs, results, cached, &wg)
	}

	// Send jobs
	predictionSize := 3 * 320 * 320
	for i := 0; i < batchSize; i++ {
		frameID := startFrameID + i
		predStart := i * predictionSize
		predEnd := predStart + predictionSize
		prediction := predictions[predStart:predEnd]

		jobs <- CompositeJob{
			FrameID:    frameID,
			Prediction: prediction,
		}
	}
	close(jobs)

	// Wait for workers to finish
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results and sort by frame ID
	resultMap := make(map[int]gocv.Mat)
	for result := range results {
		resultMap[result.FrameID] = result.Frame
	}

	// Return frames in order
	orderedFrames := make([]gocv.Mat, batchSize)
	for i := 0; i < batchSize; i++ {
		frameID := startFrameID + i
		orderedFrames[i] = resultMap[frameID]
	}

	return orderedFrames
}

func compositeWorker(
	jobs <-chan CompositeJob,
	results chan<- CompositeResult,
	cached *CachedData,
	wg *sync.WaitGroup,
) {
	defer wg.Done()

	for job := range jobs {
		composited := compositeSingleFrame(job, cached)
		results <- CompositeResult{
			FrameID: job.FrameID,
			Frame:   composited,
		}
	}
}

func compositeSingleFrame(job CompositeJob, cached *CachedData) gocv.Mat {
	frameID := job.FrameID
	prediction := job.Prediction

	// Get crop_328 frame (clone to avoid modifying original)
	crop328 := cached.Crop328Frames[frameID].Clone()

	// Convert prediction to Mat [320, 320, 3] BGR [0,255]
	predMat := gocv.NewMatWithSize(320, 320, gocv.MatTypeCV8UC3)
	defer predMat.Close()

	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// Prediction is in CHW format: [B, G, R][H][W]
			b := uint8(clamp(prediction[0*320*320+y*320+x]*255.0, 0, 255))
			g := uint8(clamp(prediction[1*320*320+y*320+x]*255.0, 0, 255))
			r := uint8(clamp(prediction[2*320*320+y*320+x]*255.0, 0, 255))
			predMat.SetUCharAt(y, x*3+0, b)
			predMat.SetUCharAt(y, x*3+1, g)
			predMat.SetUCharAt(y, x*3+2, r)
		}
	}

	// Place prediction in center of crop_328 [4:324, 4:324]
	roi := crop328.Region(image.Rect(4, 4, 324, 324))
	predMat.CopyTo(&roi)
	roi.Close()

	// Get original crop rectangle
	cropRect := cached.CropRectangles[fmt.Sprintf("%d", frameID)]
	x1, y1, x2, y2 := cropRect.Rect[0], cropRect.Rect[1], cropRect.Rect[2], cropRect.Rect[3]
	origWidth := x2 - x1
	origHeight := y2 - y1

	// Resize crop_328 back to original size
	cropResized := gocv.NewMat()
	gocv.Resize(crop328, &cropResized, image.Pt(origWidth, origHeight), 0, 0, gocv.InterpolationLinear)
	crop328.Close()

	// Get full body frame (clone to avoid modifying original)
	fullFrame := cached.FullBodyFrames[frameID].Clone()

	// Composite at original position
	fullRoi := fullFrame.Region(image.Rect(x1, y1, x2, y2))
	cropResized.CopyTo(&fullRoi)
	fullRoi.Close()
	cropResized.Close()

	return fullFrame
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

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}
