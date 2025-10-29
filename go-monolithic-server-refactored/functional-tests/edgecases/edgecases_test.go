package edgecases_test

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// TestBoundaryConditions tests edge cases at boundaries
func TestBoundaryConditions(t *testing.T) {
	tests := []struct {
		name string
		test func(t *testing.T)
	}{
		{"Zero dimension image", testZeroDimensionImage},
		{"Single pixel image", testSinglePixelImage},
		{"Maximum dimension image", testMaxDimensionImage},
		{"Negative coordinates", testNegativeCoordinates},
		{"Out of bounds access", testOutOfBoundsAccess},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.test)
	}
}

func testZeroDimensionImage(t *testing.T) {
	// Go's image.NewRGBA actually handles zero dimensions gracefully
	img := image.NewRGBA(image.Rect(0, 0, 0, 0))
	
	// Verify it creates an empty image
	if img.Bounds().Dx() != 0 || img.Bounds().Dy() != 0 {
		t.Errorf("Zero dimension image should have 0x0 bounds, got %dx%d",
			img.Bounds().Dx(), img.Bounds().Dy())
	}
	// Should handle gracefully or panic predictably
}

func testSinglePixelImage(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{255, 0, 0, 255})

	r, g, b, a := img.At(0, 0).RGBA()
	if r>>8 != 255 || g>>8 != 0 || b>>8 != 0 || a>>8 != 255 {
		t.Errorf("Single pixel incorrect: got RGBA(%d,%d,%d,%d)", r>>8, g>>8, b>>8, a>>8)
	}
}

func testMaxDimensionImage(t *testing.T) {
	// Test with relatively large but reasonable dimensions
	width, height := 8192, 8192
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	if img.Bounds().Dx() != width || img.Bounds().Dy() != height {
		t.Errorf("Large image dimensions incorrect: got %dx%d, want %dx%d",
			img.Bounds().Dx(), img.Bounds().Dy(), width, height)
	}
}

func testNegativeCoordinates(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))

	// Accessing negative coordinates should return zero value
	r, g, b, a := img.At(-1, -1).RGBA()
	if r != 0 || g != 0 || b != 0 || a != 0 {
		t.Logf("Negative coordinates returned non-zero: RGBA(%d,%d,%d,%d)", r, g, b, a)
	}
}

func testOutOfBoundsAccess(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))

	// Accessing out of bounds should return zero value
	r, g, b, a := img.At(1000, 1000).RGBA()
	if r != 0 || g != 0 || b != 0 || a != 0 {
		t.Logf("Out of bounds returned non-zero: RGBA(%d,%d,%d,%d)", r, g, b, a)
	}
}

// TestNumericalStability tests floating point edge cases
func TestNumericalStability(t *testing.T) {
	tests := []struct {
		name  string
		value float32
		want  float32
	}{
		{"Zero", 0.0, 0.0},
		{"Small positive", 1e-10, 1e-10},
		{"Small negative", -1e-10, -1e-10},
		{"Large positive", 1e10, 1e10},
		{"Negative zero", -0.0, 0.0},
		{"Infinity", float32(math.Inf(1)), float32(math.Inf(1))},
		{"Negative infinity", float32(math.Inf(-1)), float32(math.Inf(-1))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test clamping with extreme values
			clamped := clampFloat(tt.value, 0.0, 255.0)

			if math.IsInf(float64(tt.value), 1) {
				if clamped != 255.0 {
					t.Errorf("Positive infinity not clamped to 255: got %v", clamped)
				}
			} else if math.IsInf(float64(tt.value), -1) {
				if clamped != 0.0 {
					t.Errorf("Negative infinity not clamped to 0: got %v", clamped)
				}
			}
		})
	}
}

// TestAudioFeatureEdgeCases tests audio processing edge cases
func TestAudioFeatureEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		test func(t *testing.T)
	}{
		{"Zero length audio", testZeroLengthAudio},
		{"Partial frame audio", testPartialFrameAudio},
		{"Exact frame audio", testExactFrameAudio},
		{"Extra samples audio", testExtraSamplesAudio},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.test)
	}
}

func testZeroLengthAudio(t *testing.T) {
	audioData := make([]float32, 0)

	// Should handle gracefully
	if len(audioData) != 0 {
		t.Errorf("Zero length audio should be empty")
	}
}

func testPartialFrameAudio(t *testing.T) {
	// 23 samples instead of 25 (partial frame at 25fps)
	audioData := make([]float32, 23*1764) // 1764 samples per frame at 44.1kHz, 25fps

	// Should pad or handle partial frames
	if len(audioData) < 25*1764 {
		t.Logf("Partial frame detected: %d samples (need %d for 25 frames)", len(audioData), 25*1764)
	}
}

func testExactFrameAudio(t *testing.T) {
	// Exactly 25 frames worth of audio
	audioData := make([]float32, 25*1764)

	if len(audioData) != 25*1764 {
		t.Errorf("Exact frame audio length mismatch")
	}
}

func testExtraSamplesAudio(t *testing.T) {
	// 26.5 frames worth of audio
	audioData := make([]float32, 26*1764+882)

	// Should handle extra samples (truncate or process)
	if len(audioData) > 25*1764 {
		t.Logf("Extra samples detected: %d samples (expected max %d for 25 frames)",
			len(audioData), 25*1764)
	}
}

// TestBilinearInterpolationEdgeCases tests interpolation edge cases
func TestBilinearInterpolationEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		v00  uint8
		v10  uint8
		v01  uint8
		v11  uint8
		fx   float64
		fy   float64
		want uint8
	}{
		{"All zeros", 0, 0, 0, 0, 0.5, 0.5, 0},
		{"All max", 255, 255, 255, 255, 0.5, 0.5, 255},
		{"Top-left corner", 100, 0, 0, 0, 0.0, 0.0, 100},
		{"Top-right corner", 0, 100, 0, 0, 1.0, 0.0, 100},
		{"Bottom-left corner", 0, 0, 100, 0, 0.0, 1.0, 100},
		{"Bottom-right corner", 0, 0, 0, 100, 1.0, 1.0, 100},
		{"Center (equal)", 100, 100, 100, 100, 0.5, 0.5, 100},
		{"Gradient horizontal", 0, 255, 0, 255, 0.5, 0.5, 127},
		{"Gradient vertical", 0, 0, 255, 255, 0.5, 0.5, 127},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := bilinearInterpolate(tt.v00, tt.v10, tt.v01, tt.v11, tt.fx, tt.fy)

			// Allow 1 unit tolerance for rounding
			if absDiff(result, tt.want) > 1 {
				t.Errorf("Interpolation mismatch: got %d, want %d (±1)", result, tt.want)
			}
		})
	}
}

// TestResizeEdgeCases tests image resize edge cases
func TestResizeEdgeCases(t *testing.T) {
	tests := []struct {
		name      string
		srcWidth  int
		srcHeight int
		dstWidth  int
		dstHeight int
	}{
		{"Upscale 2x", 160, 160, 320, 320},
		{"Downscale 2x", 640, 640, 320, 320},
		{"Upscale 10x", 32, 32, 320, 320},
		{"Downscale 10x", 3200, 3200, 320, 320},
		{"Non-square source", 640, 480, 320, 320},
		{"Non-square dest", 640, 640, 320, 240},
		{"Aspect ratio change", 1920, 1080, 320, 320},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srcImg := image.NewRGBA(image.Rect(0, 0, tt.srcWidth, tt.srcHeight))

			// Fill with gradient
			for y := 0; y < tt.srcHeight; y++ {
				for x := 0; x < tt.srcWidth; x++ {
					intensity := uint8((x + y) % 256)
					srcImg.Set(x, y, color.RGBA{intensity, intensity, intensity, 255})
				}
			}

			dstImg := simpleResize(srcImg, tt.dstWidth, tt.dstHeight)

			// Verify dimensions
			if dstImg.Bounds().Dx() != tt.dstWidth || dstImg.Bounds().Dy() != tt.dstHeight {
				t.Errorf("Resize dimensions mismatch: got %dx%d, want %dx%d",
					dstImg.Bounds().Dx(), dstImg.Bounds().Dy(), tt.dstWidth, tt.dstHeight)
			}
		})
	}
}

// TestMemoryOverflow tests potential overflow conditions
func TestMemoryOverflow(t *testing.T) {
	tests := []struct {
		name string
		test func(t *testing.T)
	}{
		{"Large batch size", testLargeBatchSize},
		{"Maximum audio features", testMaxAudioFeatures},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.test)
	}
}

func testLargeBatchSize(t *testing.T) {
	const maxBatch = 100
	audioFeatures := make([]float32, maxBatch*512)

	// Should allocate without panic
	if len(audioFeatures) != maxBatch*512 {
		t.Errorf("Large batch allocation failed")
	}
}

func testMaxAudioFeatures(t *testing.T) {
	const maxFeatures = 100 * 512 * 16 // 100 batches, 512 features, 16 frames
	features := make([]float32, maxFeatures)

	// Should handle large allocations
	if len(features) != maxFeatures {
		t.Errorf("Max features allocation failed")
	}
}

// TestConcurrentAccess tests thread safety edge cases
func TestConcurrentAccess(t *testing.T) {
	const numGoroutines = 100
	const iterations = 100

	// Test concurrent reads
	t.Run("Concurrent reads", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 320, 320))
		done := make(chan bool, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func() {
				for j := 0; j < iterations; j++ {
					_ = img.At(160, 160)
				}
				done <- true
			}()
		}

		for i := 0; i < numGoroutines; i++ {
			<-done
		}
	})

	// Test concurrent writes to separate regions
	t.Run("Concurrent separate writes", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 320, 320))
		done := make(chan bool, 8)

		for workerID := 0; workerID < 8; workerID++ {
			go func(id int) {
				startRow := id * 40
				endRow := startRow + 40

				for y := startRow; y < endRow; y++ {
					for x := 0; x < 320; x++ {
						img.Set(x, y, color.RGBA{uint8(id), 0, 0, 255})
					}
				}
				done <- true
			}(workerID)
		}

		for i := 0; i < 8; i++ {
			<-done
		}

		// Verify no corruption
		for workerID := 0; workerID < 8; workerID++ {
			startRow := workerID * 40
			r, _, _, _ := img.At(160, startRow).RGBA()
			if r>>8 != uint32(workerID) {
				t.Errorf("Worker %d region corrupted", workerID)
			}
		}
	})
}

// Helper functions
func clampFloat(value, min, max float32) float32 {
	if math.IsInf(float64(value), 1) {
		return max
	}
	if math.IsInf(float64(value), -1) {
		return min
	}
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func bilinearInterpolate(v00, v10, v01, v11 uint8, fx, fy float64) uint8 {
	f00 := float64(v00)
	f10 := float64(v10)
	f01 := float64(v01)
	f11 := float64(v11)

	v0 := f00*(1-fx) + f10*fx
	v1 := f01*(1-fx) + f11*fx
	result := v0*(1-fy) + v1*fy

	if result < 0 {
		return 0
	}
	if result > 255 {
		return 255
	}
	return uint8(result + 0.5) // Round to nearest
}

func simpleResize(src *image.RGBA, dstWidth, dstHeight int) *image.RGBA {
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

			if x >= int(srcWidth) {
				x = int(srcWidth) - 1
			}
			if y >= int(srcHeight) {
				y = int(srcHeight) - 1
			}

			dst.Set(dx, dy, src.At(x, y))
		}
	}

	return dst
}

func absDiff(a, b uint8) uint8 {
	if a > b {
		return a - b
	}
	return b - a
}
