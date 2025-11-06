package parallel_processing_test

import (
	"image"
	"image/color"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestWorkerRowCalculation tests row distribution logic
func TestWorkerRowCalculation(t *testing.T) {
	tests := []struct {
		name          string
		totalRows     int
		numWorkers    int
		expectedSplit []int // Expected rows per worker
	}{
		{
			name:          "320 rows, 8 workers (even split)",
			totalRows:     320,
			numWorkers:    8,
			expectedSplit: []int{40, 40, 40, 40, 40, 40, 40, 40},
		},
		{
			name:          "321 rows, 8 workers (uneven split)",
			totalRows:     321,
			numWorkers:    8,
			expectedSplit: []int{41, 40, 40, 40, 40, 40, 40, 40},
		},
		{
			name:          "100 rows, 8 workers (some workers idle)",
			totalRows:     100,
			numWorkers:    8,
			expectedSplit: []int{13, 13, 13, 13, 12, 12, 12, 12},
		},
		{
			name:          "1920 rows, 8 workers (large image)",
			totalRows:     1920,
			numWorkers:    8,
			expectedSplit: []int{240, 240, 240, 240, 240, 240, 240, 240},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for workerID := 0; workerID < tt.numWorkers; workerID++ {
				startRow, endRow := calculateWorkerRows(tt.totalRows, tt.numWorkers, workerID)
				actualRows := endRow - startRow

				expectedRows := tt.expectedSplit[workerID]
				if actualRows != expectedRows {
					t.Errorf("Worker %d: got %d rows, want %d rows (start=%d, end=%d)",
						workerID, actualRows, expectedRows, startRow, endRow)
				}
			}

			// Verify full coverage (no gaps or overlaps)
			allRows := make(map[int]bool)
			for workerID := 0; workerID < tt.numWorkers; workerID++ {
				startRow, endRow := calculateWorkerRows(tt.totalRows, tt.numWorkers, workerID)
				for row := startRow; row < endRow; row++ {
					if allRows[row] {
						t.Errorf("Row %d processed by multiple workers", row)
					}
					allRows[row] = true
				}
			}

			if len(allRows) != tt.totalRows {
				t.Errorf("Coverage mismatch: processed %d rows, expected %d", len(allRows), tt.totalRows)
			}
		})
	}
}

// TestParallelExecution tests parallel worker coordination
func TestParallelExecution(t *testing.T) {
	const numWorkers = 8
	const workItems = 320

	// Test 1: Verify all workers execute
	workerExecuted := make([]atomic.Bool, numWorkers)
	var wg sync.WaitGroup

	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			workerExecuted[id].Store(true)
			time.Sleep(1 * time.Millisecond) // Simulate work
		}(workerID)
	}

	wg.Wait()

	for i := 0; i < numWorkers; i++ {
		if !workerExecuted[i].Load() {
			t.Errorf("Worker %d did not execute", i)
		}
	}

	// Test 2: Verify no data races with shared counter
	var counter atomic.Int32
	wg = sync.WaitGroup{}

	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				counter.Add(1)
			}
		}()
	}

	wg.Wait()

	expectedCount := int32(numWorkers * 100)
	if counter.Load() != expectedCount {
		t.Errorf("Counter mismatch: got %d, want %d", counter.Load(), expectedCount)
	}
}

// TestParallelImageProcessing tests parallel image conversion
func TestParallelImageProcessing(t *testing.T) {
	// Create test image
	width, height := 320, 320
	bgrData := make([]byte, width*height*3)

	// Fill with pattern: red in top half, blue in bottom half
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			offset := (y*width + x) * 3
			if y < height/2 {
				bgrData[offset] = 0     // B
				bgrData[offset+1] = 0   // G
				bgrData[offset+2] = 255 // R
			} else {
				bgrData[offset] = 255 // B
				bgrData[offset+1] = 0 // G
				bgrData[offset+2] = 0 // R
			}
		}
	}

	// Process in parallel
	rgbaImg := processImageParallel(bgrData, width, height, 8)

	// Verify results
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, a := rgbaImg.At(x, y).RGBA()

			if y < height/2 {
				// Top half should be red
				if r>>8 != 255 || g>>8 != 0 || b>>8 != 0 || a>>8 != 255 {
					t.Errorf("Top half pixel at (%d,%d): got RGBA(%d,%d,%d,%d), want (255,0,0,255)",
						x, y, r>>8, g>>8, b>>8, a>>8)
					return
				}
			} else {
				// Bottom half should be blue
				if r>>8 != 0 || g>>8 != 0 || b>>8 != 255 || a>>8 != 255 {
					t.Errorf("Bottom half pixel at (%d,%d): got RGBA(%d,%d,%d,%d), want (0,0,255,255)",
						x, y, r>>8, g>>8, b>>8, a>>8)
					return
				}
			}
		}
	}
}

// TestParallelResize tests parallel resize operation
func TestParallelResize(t *testing.T) {
	// Create checkerboard pattern
	srcWidth, srcHeight := 640, 640
	dstWidth, dstHeight := 320, 320

	srcImg := image.NewRGBA(image.Rect(0, 0, srcWidth, srcHeight))
	for y := 0; y < srcHeight; y++ {
		for x := 0; x < srcWidth; x++ {
			if (x/64+y/64)%2 == 0 {
				srcImg.Set(x, y, color.RGBA{255, 255, 255, 255})
			} else {
				srcImg.Set(x, y, color.RGBA{0, 0, 0, 255})
			}
		}
	}

	// Resize in parallel
	dstImg := resizeImageParallel(srcImg, dstWidth, dstHeight, 8)

	// Verify dimensions
	if dstImg.Bounds().Dx() != dstWidth || dstImg.Bounds().Dy() != dstHeight {
		t.Errorf("Resize dimensions: got %dx%d, want %dx%d",
			dstImg.Bounds().Dx(), dstImg.Bounds().Dy(), dstWidth, dstHeight)
	}

	// Verify pattern continuity (sample a few points)
	testPoints := []struct{ x, y int }{
		{32, 32},   // Top-left quadrant
		{288, 32},  // Top-right quadrant
		{32, 288},  // Bottom-left quadrant
		{288, 288}, // Bottom-right quadrant
	}

	for _, pt := range testPoints {
		r, g, b, _ := dstImg.At(pt.x, pt.y).RGBA()
		// Should be either white or black (or close due to interpolation)
		if !((r>>8 > 200 && g>>8 > 200 && b>>8 > 200) || (r>>8 < 55 && g>>8 < 55 && b>>8 < 55)) {
			t.Errorf("Unexpected color at (%d,%d): RGB(%d,%d,%d)", pt.x, pt.y, r>>8, g>>8, b>>8)
		}
	}
}

// TestRaceConditions tests for data races in parallel processing
func TestRaceConditions(t *testing.T) {
	// This test is designed to be run with -race flag
	const iterations = 100

	for iter := 0; iter < iterations; iter++ {
		// Test shared slice access pattern
		data := make([]float32, 8*512)
		var wg sync.WaitGroup

		for workerID := 0; workerID < 8; workerID++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				start := id * 512
				end := start + 512
				for i := start; i < end; i++ {
					data[i] = float32(id)
				}
			}(workerID)
		}

		wg.Wait()

		// Verify each worker wrote to its own section
		for workerID := 0; workerID < 8; workerID++ {
			start := workerID * 512
			end := start + 512
			for i := start; i < end; i++ {
				if data[i] != float32(workerID) {
					t.Errorf("Data race detected: worker %d region has value %v at index %d",
						workerID, data[i], i)
					return
				}
			}
		}
	}
}

// Helper functions
func calculateWorkerRows(totalRows, numWorkers, workerID int) (startRow, endRow int) {
	rowsPerWorker := totalRows / numWorkers
	extraRows := totalRows % numWorkers

	startRow = workerID * rowsPerWorker
	if workerID < extraRows {
		startRow += workerID
	} else {
		startRow += extraRows
	}

	endRow = startRow + rowsPerWorker
	if workerID < extraRows {
		endRow++
	}

	return startRow, endRow
}

func processImageParallel(bgrData []byte, width, height, numWorkers int) *image.RGBA {
	rgbaImg := image.NewRGBA(image.Rect(0, 0, width, height))
	var wg sync.WaitGroup

	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			startRow, endRow := calculateWorkerRows(height, numWorkers, id)

			for y := startRow; y < endRow; y++ {
				for x := 0; x < width; x++ {
					offset := (y*width + x) * 3
					b := bgrData[offset]
					g := bgrData[offset+1]
					r := bgrData[offset+2]
					rgbaImg.Set(x, y, color.RGBA{r, g, b, 255})
				}
			}
		}(workerID)
	}

	wg.Wait()
	return rgbaImg
}

func resizeImageParallel(src *image.RGBA, dstWidth, dstHeight, numWorkers int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, dstWidth, dstHeight))
	var wg sync.WaitGroup

	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			startRow, endRow := calculateWorkerRows(dstHeight, numWorkers, id)

			srcBounds := src.Bounds()
			srcWidth := float64(srcBounds.Dx())
			srcHeight := float64(srcBounds.Dy())

			for dy := startRow; dy < endRow; dy++ {
				for dx := 0; dx < dstWidth; dx++ {
					// Bilinear interpolation
					sx := (float64(dx) + 0.5) * srcWidth / float64(dstWidth)
					sy := (float64(dy) + 0.5) * srcHeight / float64(dstHeight)

					x0 := int(sx)
					y0 := int(sy)
					x1 := x0 + 1
					y1 := y0 + 1

					if x1 >= int(srcWidth) {
						x1 = int(srcWidth) - 1
					}
					if y1 >= int(srcHeight) {
						y1 = int(srcHeight) - 1
					}

					fx := sx - float64(x0)
					fy := sy - float64(y0)

					c00 := src.RGBAAt(x0, y0)
					c10 := src.RGBAAt(x1, y0)
					c01 := src.RGBAAt(x0, y1)
					c11 := src.RGBAAt(x1, y1)

					r := bilinear(c00.R, c10.R, c01.R, c11.R, fx, fy)
					g := bilinear(c00.G, c10.G, c01.G, c11.G, fx, fy)
					b := bilinear(c00.B, c10.B, c01.B, c11.B, fx, fy)

					dst.Set(dx, dy, color.RGBA{r, g, b, 255})
				}
			}
		}(workerID)
	}

	wg.Wait()
	return dst
}

func bilinear(v00, v10, v01, v11 uint8, fx, fy float64) uint8 {
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
	return uint8(result)
}

// Benchmark tests
func BenchmarkWorkerCoordination(b *testing.B) {
	const numWorkers = 8
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		for workerID := 0; workerID < numWorkers; workerID++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
			}()
		}
		wg.Wait()
	}
}

func BenchmarkParallelImageProcessing(b *testing.B) {
	bgrData := make([]byte, 320*320*3)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		processImageParallel(bgrData, 320, 320, 8)
	}
}

func BenchmarkParallelResize(b *testing.B) {
	srcImg := image.NewRGBA(image.Rect(0, 0, 640, 640))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		resizeImageParallel(srcImg, 320, 320, 8)
	}
}
