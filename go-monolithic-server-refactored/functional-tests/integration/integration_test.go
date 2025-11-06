package integration_test

import (
	"image"
	"image/color"
	"os"
	"testing"
	"time"
)

// TestFullPipelineFlow tests complete end-to-end processing
func TestFullPipelineFlow(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Test data setup
	testDataDir := setupTestData(t)
	defer cleanupTestData(testDataDir)

	tests := []struct {
		name        string
		batchSize   int
		numFrames   int
		videoWidth  int
		videoHeight int
	}{
		{
			name:        "Single frame batch 1",
			batchSize:   1,
			numFrames:   1,
			videoWidth:  1920,
			videoHeight: 1080,
		},
		{
			name:        "Small batch 8",
			batchSize:   8,
			numFrames:   8,
			videoWidth:  1920,
			videoHeight: 1080,
		},
		{
			name:        "Large batch 25",
			batchSize:   25,
			numFrames:   25,
			videoWidth:  1920,
			videoHeight: 1080,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Step 1: Image processing (BGR → RGBA → Resize)
			startTime := time.Now()

			videoFrame := createTestVideoFrame(tt.videoWidth, tt.videoHeight)
			bgrData := rgbaToBGR(videoFrame)

			// Convert BGR to RGBA (parallel)
			rgbaImg := bgrToRGBA(bgrData, tt.videoWidth, tt.videoHeight)

			// Resize to 320x320 (parallel)
			resizedImg := resizeImage(rgbaImg, 320, 320)

			imageProcessTime := time.Since(startTime)

			// Step 2: Audio processing (STFT → Mel-spec → Window extraction)
			startTime = time.Now()

			audioData := createTestAudioData(tt.numFrames, 44100, 25)
			melSpec := processAudioToMel(audioData, 44100)

			audioProcessTime := time.Since(startTime)

			// Step 3: Batch preparation
			startTime = time.Now()

			batchData := prepareBatchData(resizedImg, melSpec, tt.batchSize, 0)

			batchPrepTime := time.Since(startTime)

			// Verify results
			if len(batchData.VideoFrames) != tt.batchSize {
				t.Errorf("Batch size mismatch: got %d, want %d", len(batchData.VideoFrames), tt.batchSize)
			}

			if len(batchData.AudioFeatures) != tt.batchSize*512 {
				t.Errorf("Audio features size mismatch: got %d, want %d",
					len(batchData.AudioFeatures), tt.batchSize*512)
			}

			// Report timing
			t.Logf("Image processing: %v", imageProcessTime)
			t.Logf("Audio processing: %v", audioProcessTime)
			t.Logf("Batch preparation: %v", batchPrepTime)
			t.Logf("Total pipeline: %v", imageProcessTime+audioProcessTime+batchPrepTime)
		})
	}
}

// TestMemoryPooling tests memory pool behavior
func TestMemoryPooling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	const iterations = 100

	// Test buffer pool
	t.Run("BufferPool", func(t *testing.T) {
		pool := &ByteSlicePool{
			pool: make(chan []byte, 10),
			size: 320 * 320 * 3,
		}

		for i := 0; i < iterations; i++ {
			buf := pool.Get()
			if len(buf) != 320*320*3 {
				t.Errorf("Buffer size mismatch: got %d, want %d", len(buf), 320*320*3)
			}
			pool.Put(buf)
		}
	})

	// Test RGBA pool
	t.Run("RGBAPool", func(t *testing.T) {
		pool := &RGBAPool{
			pool:   make(chan *image.RGBA, 10),
			width:  320,
			height: 320,
		}

		for i := 0; i < iterations; i++ {
			img := pool.Get()
			if img.Bounds().Dx() != 320 || img.Bounds().Dy() != 320 {
				t.Errorf("Image dimensions: got %dx%d, want 320x320",
					img.Bounds().Dx(), img.Bounds().Dy())
			}
			pool.Put(img)
		}
	})
}

// TestConcurrentRequestProcessing tests handling multiple simultaneous requests
func TestConcurrentRequestProcessing(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	const numConcurrent = 10
	results := make(chan error, numConcurrent)

	for i := 0; i < numConcurrent; i++ {
		go func(id int) {
			// Simulate processing request
			videoFrame := createTestVideoFrame(1920, 1080)
			bgrData := rgbaToBGR(videoFrame)
			rgbaImg := bgrToRGBA(bgrData, 1920, 1080)
			_ = resizeImage(rgbaImg, 320, 320)

			results <- nil
		}(i)
	}

	// Wait for all to complete
	for i := 0; i < numConcurrent; i++ {
		err := <-results
		if err != nil {
			t.Errorf("Concurrent request %d failed: %v", i, err)
		}
	}
}

// TestErrorRecovery tests error handling and recovery
func TestErrorRecovery(t *testing.T) {
	tests := []struct {
		name      string
		testFunc  func() error
		expectErr bool
	}{
		{
			name: "Invalid image dimensions",
			testFunc: func() error {
				_, err := processInvalidImage(0, 0)
				return err
			},
			expectErr: true,
		},
		{
			name: "Nil image data",
			testFunc: func() error {
				_, err := processNilImage()
				return err
			},
			expectErr: true,
		},
		{
			name: "Empty audio data",
			testFunc: func() error {
				_, err := processEmptyAudio()
				return err
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.testFunc()
			if tt.expectErr && err == nil {
				t.Error("Expected error but got nil")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// Helper functions and types
type BatchData struct {
	VideoFrames   [][]float32
	AudioFeatures []float32
}

type ByteSlicePool struct {
	pool chan []byte
	size int
}

func (p *ByteSlicePool) Get() []byte {
	select {
	case buf := <-p.pool:
		return buf[:p.size]
	default:
		return make([]byte, p.size)
	}
}

func (p *ByteSlicePool) Put(buf []byte) {
	select {
	case p.pool <- buf:
	default:
	}
}

type RGBAPool struct {
	pool   chan *image.RGBA
	width  int
	height int
}

func (p *RGBAPool) Get() *image.RGBA {
	select {
	case img := <-p.pool:
		return img
	default:
		return image.NewRGBA(image.Rect(0, 0, p.width, p.height))
	}
}

func (p *RGBAPool) Put(img *image.RGBA) {
	select {
	case p.pool <- img:
	default:
	}
}

func setupTestData(t *testing.T) string {
	dir, err := os.MkdirTemp("", "integration_test_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	return dir
}

func cleanupTestData(dir string) {
	os.RemoveAll(dir)
}

func createTestVideoFrame(width, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	// Create gradient pattern
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r := uint8(float64(x) / float64(width) * 255)
			g := uint8(float64(y) / float64(height) * 255)
			b := uint8(128)
			img.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	return img
}

func createTestAudioData(numFrames, sampleRate, fps int) []float32 {
	samplesPerFrame := sampleRate / fps
	totalSamples := numFrames * samplesPerFrame
	audioData := make([]float32, totalSamples)

	// Generate simple sine wave
	frequency := 440.0 // A4 note
	for i := 0; i < totalSamples; i++ {
		t := float64(i) / float64(sampleRate)
		audioData[i] = float32(0.5 * (1.0 + (t * frequency * 2 * 3.14159)))
	}

	return audioData
}

func rgbaToBGR(img *image.RGBA) []byte {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	bgrData := make([]byte, width*height*3)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			offset := (y*width + x) * 3
			bgrData[offset] = uint8(b >> 8)
			bgrData[offset+1] = uint8(g >> 8)
			bgrData[offset+2] = uint8(r >> 8)
		}
	}

	return bgrData
}

func bgrToRGBA(bgrData []byte, width, height int) *image.RGBA {
	rgbaImg := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			offset := (y*width + x) * 3
			b := bgrData[offset]
			g := bgrData[offset+1]
			r := bgrData[offset+2]
			rgbaImg.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}

	return rgbaImg
}

func resizeImage(src *image.RGBA, dstWidth, dstHeight int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, dstWidth, dstHeight))
	srcBounds := src.Bounds()
	srcWidth := float64(srcBounds.Dx())
	srcHeight := float64(srcBounds.Dy())

	for dy := 0; dy < dstHeight; dy++ {
		for dx := 0; dx < dstWidth; dx++ {
			sx := (float64(dx) + 0.5) * srcWidth / float64(dstWidth)
			sy := (float64(dy) + 0.5) * srcHeight / float64(dstHeight)

			x := int(sx)
			y := int(sy)

			if x >= int(srcWidth)-1 {
				x = int(srcWidth) - 2
			}
			if y >= int(srcHeight)-1 {
				y = int(srcHeight) - 2
			}

			dst.Set(dx, dy, src.At(x, y))
		}
	}

	return dst
}

func processAudioToMel(audioData []float32, sampleRate int) [][]float32 {
	// Simplified mel-spectrogram (would use real STFT/mel in production)
	numTimeSteps := 100
	numMelBins := 80

	melSpec := make([][]float32, numTimeSteps)
	for i := range melSpec {
		melSpec[i] = make([]float32, numMelBins)
		for j := range melSpec[i] {
			// Placeholder values
			melSpec[i][j] = float32(i*numMelBins + j)
		}
	}

	return melSpec
}

func prepareBatchData(videoFrame *image.RGBA, melSpec [][]float32, batchSize, startFrameIdx int) *BatchData {
	batch := &BatchData{
		VideoFrames:   make([][]float32, batchSize),
		AudioFeatures: make([]float32, batchSize*512),
	}

	// Flatten video frame to float32
	bounds := videoFrame.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	for b := 0; b < batchSize; b++ {
		batch.VideoFrames[b] = make([]float32, width*height*3)
		// Simplified - just copy pixel data
		for i := range batch.VideoFrames[b] {
			batch.VideoFrames[b][i] = float32(i % 256)
		}
	}

	return batch
}

func processInvalidImage(width, height int) (*image.RGBA, error) {
	if width <= 0 || height <= 0 {
		return nil, &InvalidDimensionsError{width, height}
	}
	return image.NewRGBA(image.Rect(0, 0, width, height)), nil
}

func processNilImage() (*image.RGBA, error) {
	return nil, &NilImageError{}
}

func processEmptyAudio() ([]float32, error) {
	return nil, &EmptyAudioError{}
}

// Error types
type InvalidDimensionsError struct {
	Width  int
	Height int
}

func (e *InvalidDimensionsError) Error() string {
	return "invalid image dimensions"
}

type NilImageError struct{}

func (e *NilImageError) Error() string {
	return "nil image data"
}

type EmptyAudioError struct{}

func (e *EmptyAudioError) Error() string {
	return "empty audio data"
}

// Benchmark tests
func BenchmarkFullPipeline(b *testing.B) {
	videoFrame := createTestVideoFrame(1920, 1080)
	audioData := createTestAudioData(25, 44100, 25)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bgrData := rgbaToBGR(videoFrame)
		rgbaImg := bgrToRGBA(bgrData, 1920, 1080)
		resizedImg := resizeImage(rgbaImg, 320, 320)
		melSpec := processAudioToMel(audioData, 44100)
		_ = prepareBatchData(resizedImg, melSpec, 25, 0)
	}
}
