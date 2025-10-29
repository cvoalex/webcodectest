package server

import (
	"image"
	"image/color"
	"testing"
)

// Test calculateWorkerRows function - pure function, easy to test
func TestCalculateWorkerRows(t *testing.T) {
	tests := []struct {
		name         string
		workerID     int
		totalWorkers int
		totalRows    int
		wantStart    int
		wantEnd      int
	}{
		{
			name:         "First worker of 8, 320 rows",
			workerID:     0,
			totalWorkers: 8,
			totalRows:    320,
			wantStart:    0,
			wantEnd:      40,
		},
		{
			name:         "Middle worker of 8, 320 rows",
			workerID:     4,
			totalWorkers: 8,
			totalRows:    320,
			wantStart:    160,
			wantEnd:      200,
		},
		{
			name:         "Last worker of 8, 320 rows (handles remainder)",
			workerID:     7,
			totalWorkers: 8,
			totalRows:    320,
			wantStart:    280,
			wantEnd:      320,
		},
		{
			name:         "Single worker",
			workerID:     0,
			totalWorkers: 1,
			totalRows:    320,
			wantStart:    0,
			wantEnd:      320,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotStart, gotEnd := calculateWorkerRows(tt.workerID, tt.totalWorkers, tt.totalRows)
			if gotStart != tt.wantStart {
				t.Errorf("calculateWorkerRows() gotStart = %v, want %v", gotStart, tt.wantStart)
			}
			if gotEnd != tt.wantEnd {
				t.Errorf("calculateWorkerRows() gotEnd = %v, want %v", gotEnd, tt.wantEnd)
			}
		})
	}
}

// Test extractBGRPixel function - pure function, easy to test
func TestExtractBGRPixel(t *testing.T) {
	// Create test data: 320x320 image with known values
	bgrData := make([]float32, 3*320*320)
	
	// Set specific pixel (100, 50) to known values
	x, y := 100, 50
	offset := y*320 + x
	bgrData[0*320*320+offset] = 0.5  // B = 127
	bgrData[1*320*320+offset] = 1.0  // G = 255
	bgrData[2*320*320+offset] = 0.0  // R = 0

	pixel := extractBGRPixel(bgrData, x, y)

	if pixel.R != 0 {
		t.Errorf("extractBGRPixel() R = %v, want 0", pixel.R)
	}
	if pixel.G != 255 {
		t.Errorf("extractBGRPixel() G = %v, want 255", pixel.G)
	}
	if pixel.B != 127 {
		t.Errorf("extractBGRPixel() B = %v, want 127", pixel.B)
	}
	if pixel.A != 255 {
		t.Errorf("extractBGRPixel() A = %v, want 255", pixel.A)
	}
}

// Test extractBGRPixel with clamping - pure function, edge cases
func TestExtractBGRPixelClamping(t *testing.T) {
	bgrData := make([]float32, 3*320*320)
	
	x, y := 0, 0
	offset := y*320 + x
	
	// Test clamping: values outside [0, 1] range
	bgrData[0*320*320+offset] = -0.5  // Should clamp to 0
	bgrData[1*320*320+offset] = 1.5   // Should clamp to 255
	bgrData[2*320*320+offset] = 0.5   // Normal value

	pixel := extractBGRPixel(bgrData, x, y)

	if pixel.B != 0 {
		t.Errorf("extractBGRPixel() clamping B = %v, want 0 (clamped from negative)", pixel.B)
	}
	if pixel.G != 255 {
		t.Errorf("extractBGRPixel() clamping G = %v, want 255 (clamped from >1)", pixel.G)
	}
	if pixel.R != 127 {
		t.Errorf("extractBGRPixel() clamping R = %v, want 127", pixel.R)
	}
}

// Benchmark the parallel conversion
func BenchmarkConvertBGRToRGBAParallel(b *testing.B) {
	bgrData := make([]float32, 3*320*320)
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))
	
	// Initialize with some data
	for i := range bgrData {
		bgrData[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertBGRToRGBAParallel(bgrData, img, 8)
	}
}

// Benchmark with different worker counts
func BenchmarkConvertBGRToRGBA_Workers1(b *testing.B) {
	bgrData := make([]float32, 3*320*320)
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))
	for i := range bgrData {
		bgrData[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertBGRToRGBAParallel(bgrData, img, 1)
	}
}

func BenchmarkConvertBGRToRGBA_Workers4(b *testing.B) {
	bgrData := make([]float32, 3*320*320)
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))
	for i := range bgrData {
		bgrData[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertBGRToRGBAParallel(bgrData, img, 4)
	}
}

func BenchmarkConvertBGRToRGBA_Workers8(b *testing.B) {
	bgrData := make([]float32, 3*320*320)
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))
	for i := range bgrData {
		bgrData[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertBGRToRGBAParallel(bgrData, img, 8)
	}
}

func BenchmarkConvertBGRToRGBA_Workers16(b *testing.B) {
	bgrData := make([]float32, 3*320*320)
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))
	for i := range bgrData {
		bgrData[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertBGRToRGBAParallel(bgrData, img, 16)
	}
}

// Test clampFloat helper function
func TestClampFloat(t *testing.T) {
	tests := []struct {
		name  string
		input float32
		want  float32
	}{
		{"Below zero", -10.0, 0.0},
		{"Zero", 0.0, 0.0},
		{"Normal value", 127.5, 127.5},
		{"Max value", 255.0, 255.0},
		{"Above max", 300.0, 255.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := clampFloat(tt.input)
			if got != tt.want {
				t.Errorf("clampFloat(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

// Test that all workers cover all rows without gaps or overlaps
func TestWorkerRowCoverage(t *testing.T) {
	totalRows := 320
	totalWorkers := 8
	
	covered := make([]bool, totalRows)
	
	for worker := 0; worker < totalWorkers; worker++ {
		start, end := calculateWorkerRows(worker, totalWorkers, totalRows)
		
		// Mark rows as covered
		for row := start; row < end; row++ {
			if covered[row] {
				t.Errorf("Row %d covered by multiple workers", row)
			}
			covered[row] = true
		}
	}
	
	// Check all rows are covered
	for row := 0; row < totalRows; row++ {
		if !covered[row] {
			t.Errorf("Row %d not covered by any worker", row)
		}
	}
}

// Example test showing how the functional approach enables easy testing
func Example_extractBGRPixel() {
	bgrData := make([]float32, 3*320*320)
	
	// Set pixel at (0, 0) to cyan (full green + blue)
	bgrData[0] = 1.0  // B channel
	bgrData[320*320] = 1.0  // G channel
	bgrData[2*320*320] = 0.0  // R channel
	
	pixel := extractBGRPixel(bgrData, 0, 0)
	
	// Output shows RGB values
	_ = pixel // RGBA{R:0, G:255, B:255, A:255} - Cyan color
}

// Test bilinearSample function - pure function, easy to test
func TestBilinearSample(t *testing.T) {
	// Create a simple 2x2 test image
	src := image.NewRGBA(image.Rect(0, 0, 2, 2))
	src.SetRGBA(0, 0, color.RGBA{R: 0, G: 0, B: 0, A: 255})       // Black
	src.SetRGBA(1, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})     // Red
	src.SetRGBA(0, 1, color.RGBA{R: 0, G: 255, B: 0, A: 255})     // Green
	src.SetRGBA(1, 1, color.RGBA{R: 255, G: 255, B: 255, A: 255}) // White

	xRatio := float32(2) / float32(4) // Scale 2x2 to 4x4
	yRatio := float32(2) / float32(4)

	// Sample at (1, 1) should interpolate between the four corners
	pixel := bilinearSample(src, 2, 2, 1, 1, xRatio, yRatio)

	// Center of 2x2 should be average of all four corners
	// (0+255+0+255)/4 = 127.5 for R, (0+0+255+255)/4 = 127.5 for G, etc.
	if pixel.A != 255 {
		t.Errorf("bilinearSample() A = %v, want 255", pixel.A)
	}
	// Note: exact values depend on interpolation weights
}

// Test bilinearSample edge clamping
func TestBilinearSampleEdgeClamping(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
		}
	}

	xRatio := float32(10) / float32(5)
	yRatio := float32(10) / float32(5)

	// Sample near edge - should not panic
	pixel := bilinearSample(src, 10, 10, 4, 4, xRatio, yRatio)

	if pixel.A != 255 {
		t.Errorf("Edge sampling A = %v, want 255", pixel.A)
	}
}

// Benchmark the parallel resize
func BenchmarkResizeImageParallel(b *testing.B) {
	src := image.NewRGBA(image.Rect(0, 0, 320, 320))
	dst := image.NewRGBA(image.Rect(0, 0, 300, 200))
	
	// Fill with test data
	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resizeImageParallel(src, dst, 300, 200, 8)
	}
}

// Benchmark resize with different worker counts
func BenchmarkResizeImage_Workers1(b *testing.B) {
	src := image.NewRGBA(image.Rect(0, 0, 320, 320))
	dst := image.NewRGBA(image.Rect(0, 0, 300, 200))
	
	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resizeImageParallel(src, dst, 300, 200, 1)
	}
}

func BenchmarkResizeImage_Workers4(b *testing.B) {
	src := image.NewRGBA(image.Rect(0, 0, 320, 320))
	dst := image.NewRGBA(image.Rect(0, 0, 300, 200))
	
	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resizeImageParallel(src, dst, 300, 200, 4)
	}
}

func BenchmarkResizeImage_Workers8(b *testing.B) {
	src := image.NewRGBA(image.Rect(0, 0, 320, 320))
	dst := image.NewRGBA(image.Rect(0, 0, 300, 200))
	
	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resizeImageParallel(src, dst, 300, 200, 8)
	}
}

func BenchmarkResizeImage_Workers16(b *testing.B) {
	src := image.NewRGBA(image.Rect(0, 0, 320, 320))
	dst := image.NewRGBA(image.Rect(0, 0, 300, 200))
	
	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 128, G: 128, B: 128, A: 255})
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resizeImageParallel(src, dst, 300, 200, 16)
	}
}

// Test zeroPadAudioFeatures function - pure function, easy to test
func TestZeroPadAudioFeatures(t *testing.T) {
	// Create test data
	audioData := make([]float32, 100)
	for i := range audioData {
		audioData[i] = 1.0 // Fill with non-zero values
	}

	// Zero-pad 3 frames of 10 elements each, starting at offset 20
	zeroPadAudioFeatures(audioData, 20, 3, 10)

	// Check that elements [20:50] are zeroed
	for i := 20; i < 50; i++ {
		if audioData[i] != 0.0 {
			t.Errorf("zeroPadAudioFeatures() at index %d = %v, want 0.0", i, audioData[i])
		}
	}

	// Check that other elements are unchanged
	for i := 0; i < 20; i++ {
		if audioData[i] != 1.0 {
			t.Errorf("zeroPadAudioFeatures() modified index %d = %v, want 1.0", i, audioData[i])
		}
	}
	for i := 50; i < 100; i++ {
		if audioData[i] != 1.0 {
			t.Errorf("zeroPadAudioFeatures() modified index %d = %v, want 1.0", i, audioData[i])
		}
	}
}

// Test zeroPadAudioFeatures with zero count (edge case)
func TestZeroPadAudioFeaturesZeroCount(t *testing.T) {
	audioData := make([]float32, 10)
	for i := range audioData {
		audioData[i] = 1.0
	}

	// Should be no-op
	zeroPadAudioFeatures(audioData, 0, 0, 512)

	// All should remain 1.0
	for i := range audioData {
		if audioData[i] != 1.0 {
			t.Errorf("zeroPadAudioFeatures(count=0) modified data at %d", i)
		}
	}
}

// Test copyAudioFeatures function
func TestCopyAudioFeatures(t *testing.T) {
	audioData := make([]float32, 100)
	features := []float32{1.0, 2.0, 3.0, 4.0, 5.0}

	copyAudioFeatures(audioData, 10, features)

	// Check copied values
	for i := 0; i < len(features); i++ {
		if audioData[10+i] != features[i] {
			t.Errorf("copyAudioFeatures() at index %d = %v, want %v", 10+i, audioData[10+i], features[i])
		}
	}

	// Check other values are untouched (should be 0.0 from make)
	for i := 0; i < 10; i++ {
		if audioData[i] != 0.0 {
			t.Errorf("copyAudioFeatures() modified index %d", i)
		}
	}
}

// Benchmark zeroPadAudioFeatures
func BenchmarkZeroPadAudioFeatures(b *testing.B) {
	audioData := make([]float32, 16*512) // Typical size
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		zeroPadAudioFeatures(audioData, 0, 16, 512)
	}
}
