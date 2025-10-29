package performance_test

import (
	"fmt"
	"image"
	"image/color"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestFPSThroughput measures frames per second for different batch sizes
func TestFPSThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	tests := []struct {
		name          string
		batchSize     int
		numIterations int
		targetFPS     float64
	}{
		{
			name:          "Batch 1",
			batchSize:     1,
			numIterations: 100,
			targetFPS:     60.0,
		},
		{
			name:          "Batch 8",
			batchSize:     8,
			numIterations: 100,
			targetFPS:     25.0,
		},
		{
			name:          "Batch 25",
			batchSize:     25,
			numIterations: 100,
			targetFPS:     47.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Prepare test data
			videoFrame := createVideoFrame(1920, 1080)
			bgrData := convertToBGR(videoFrame)

			// Warm up
			for i := 0; i < 10; i++ {
				processFrame(bgrData, 1920, 1080, tt.batchSize)
			}

			// Measure FPS
			runtime.GC() // Clean start
			startTime := time.Now()

			for i := 0; i < tt.numIterations; i++ {
				processFrame(bgrData, 1920, 1080, tt.batchSize)
			}

			elapsed := time.Since(startTime)
			totalFrames := tt.numIterations * tt.batchSize
			fps := float64(totalFrames) / elapsed.Seconds()

			t.Logf("Batch %d: %.2f FPS (target: %.2f FPS)", tt.batchSize, fps, tt.targetFPS)
			t.Logf("  Total frames: %d", totalFrames)
			t.Logf("  Elapsed time: %v", elapsed)
			t.Logf("  Time per batch: %v", elapsed/time.Duration(tt.numIterations))

			if fps < tt.targetFPS {
				t.Logf("WARNING: FPS below target (%.2f < %.2f)", fps, tt.targetFPS)
			}
		})
	}
}

// TestMemoryAllocation measures memory allocation patterns
func TestMemoryAllocation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	tests := []struct {
		name           string
		operation      func()
		maxAllocsMB    float64
		maxAllocsBytes uint64
	}{
		{
			name: "BGR to RGBA conversion",
			operation: func() {
				bgrData := make([]byte, 320*320*3)
				convertBGRToRGBA(bgrData, 320, 320)
			},
			maxAllocsMB:    1.0,
			maxAllocsBytes: 1024 * 1024,
		},
		{
			name: "Image resize",
			operation: func() {
				srcImg := image.NewRGBA(image.Rect(0, 0, 640, 640))
				resizeImageBilinear(srcImg, 320, 320)
			},
			maxAllocsMB:    1.0,
			maxAllocsBytes: 1024 * 1024,
		},
		{
			name: "Zero padding",
			operation: func() {
				audioData := make([]float32, 16*512)
				zeroPadAudio(audioData, 0, 16, 512)
			},
			maxAllocsMB:    0.05,
			maxAllocsBytes: 50 * 1024,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var memBefore, memAfter runtime.MemStats

			runtime.GC()
			runtime.ReadMemStats(&memBefore)

			// Run operation multiple times
			for i := 0; i < 1000; i++ {
				tt.operation()
			}

			runtime.GC()
			runtime.ReadMemStats(&memAfter)

			allocatedBytes := memAfter.TotalAlloc - memBefore.TotalAlloc
			allocatedMB := float64(allocatedBytes) / (1024 * 1024)

			t.Logf("Total allocated: %.2f MB (%d bytes)", allocatedMB, allocatedBytes)
			t.Logf("Allocs per operation: %.0f bytes", float64(allocatedBytes)/1000)

			avgAllocPerOp := allocatedBytes / 1000
			if avgAllocPerOp > tt.maxAllocsBytes {
				t.Logf("WARNING: High allocation per operation (avg: %d bytes, max: %d bytes)",
					avgAllocPerOp, tt.maxAllocsBytes)
			}
		})
	}
}

// TestParallelScaling measures speedup from parallel processing
func TestParallelScaling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	workerCounts := []int{1, 2, 4, 8, 16}
	bgrData := make([]byte, 320*320*3)

	results := make(map[int]time.Duration)

	for _, numWorkers := range workerCounts {
		// Warm up
		for i := 0; i < 5; i++ {
			convertBGRToRGBAParallel(bgrData, 320, 320, numWorkers)
		}

		// Measure
		runtime.GC()
		startTime := time.Now()

		for i := 0; i < 100; i++ {
			convertBGRToRGBAParallel(bgrData, 320, 320, numWorkers)
		}

		elapsed := time.Since(startTime)
		results[numWorkers] = elapsed

		t.Logf("%d workers: %v (%.0f μs per operation)",
			numWorkers, elapsed, float64(elapsed.Microseconds())/100)
	}

	// Calculate speedup
	baseline := results[1]
	for _, numWorkers := range workerCounts[1:] {
		speedup := float64(baseline) / float64(results[numWorkers])
		efficiency := speedup / float64(numWorkers) * 100

		t.Logf("Speedup with %d workers: %.2fx (efficiency: %.1f%%)",
			numWorkers, speedup, efficiency)
	}
}

// TestCachingEffectiveness tests memory pool effectiveness
func TestCachingEffectiveness(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	const iterations = 1000

	// Test without pooling
	t.Run("Without pooling", func(t *testing.T) {
		runtime.GC()
		var memBefore runtime.MemStats
		runtime.ReadMemStats(&memBefore)

		startTime := time.Now()
		for i := 0; i < iterations; i++ {
			buf := make([]byte, 320*320*3)
			_ = buf
		}
		elapsed := time.Since(startTime)

		var memAfter runtime.MemStats
		runtime.ReadMemStats(&memAfter)

		t.Logf("Time: %v", elapsed)
		t.Logf("Allocations: %d", memAfter.Mallocs-memBefore.Mallocs)
		t.Logf("Allocated: %.2f MB", float64(memAfter.TotalAlloc-memBefore.TotalAlloc)/(1024*1024))
	})

	// Test with pooling
	t.Run("With pooling", func(t *testing.T) {
		pool := &SimplePool{
			pool: make(chan []byte, 10),
			size: 320 * 320 * 3,
		}

		runtime.GC()
		var memBefore runtime.MemStats
		runtime.ReadMemStats(&memBefore)

		startTime := time.Now()
		for i := 0; i < iterations; i++ {
			buf := pool.Get()
			pool.Put(buf)
		}
		elapsed := time.Since(startTime)

		var memAfter runtime.MemStats
		runtime.ReadMemStats(&memAfter)

		t.Logf("Time: %v", elapsed)
		t.Logf("Allocations: %d", memAfter.Mallocs-memBefore.Mallocs)
		t.Logf("Allocated: %.2f MB", float64(memAfter.TotalAlloc-memBefore.TotalAlloc)/(1024*1024))
	})
}

// TestConcurrentThroughput measures throughput under concurrent load
func TestConcurrentThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	concurrencyLevels := []int{1, 2, 4, 8, 16}
	const operationsPerWorker = 100

	for _, concurrency := range concurrencyLevels {
		t.Run(fmt.Sprintf("Concurrency %d", concurrency), func(t *testing.T) {
			var wg sync.WaitGroup
			startTime := time.Now()

			for i := 0; i < concurrency; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					bgrData := make([]byte, 320*320*3)
					for j := 0; j < operationsPerWorker; j++ {
						convertBGRToRGBA(bgrData, 320, 320)
					}
				}()
			}

			wg.Wait()
			elapsed := time.Since(startTime)

			totalOps := concurrency * operationsPerWorker
			opsPerSec := float64(totalOps) / elapsed.Seconds()

			t.Logf("Concurrency %d: %v total", concurrency, elapsed)
			t.Logf("  Operations: %d", totalOps)
			t.Logf("  Throughput: %.0f ops/sec", opsPerSec)
		})
	}
}

// Helper functions
func createVideoFrame(width, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.RGBA{uint8(x % 256), uint8(y % 256), 128, 255})
		}
	}
	return img
}

func convertToBGR(img *image.RGBA) []byte {
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

func processFrame(bgrData []byte, width, height, batchSize int) {
	// BGR to RGBA
	rgbaImg := convertBGRToRGBA(bgrData, width, height)

	// Resize
	resized := resizeImageBilinear(rgbaImg, 320, 320)

	// Audio features (simplified)
	audioFeatures := make([]float32, batchSize*512)
	zeroPadAudio(audioFeatures, 0, batchSize, 512)

	_ = resized
}

func convertBGRToRGBA(bgrData []byte, width, height int) *image.RGBA {
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

func convertBGRToRGBAParallel(bgrData []byte, width, height, numWorkers int) *image.RGBA {
	rgbaImg := image.NewRGBA(image.Rect(0, 0, width, height))
	var wg sync.WaitGroup

	rowsPerWorker := height / numWorkers
	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			startRow := id * rowsPerWorker
			endRow := startRow + rowsPerWorker
			if id == numWorkers-1 {
				endRow = height
			}

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

func resizeImageBilinear(src *image.RGBA, dstWidth, dstHeight int) *image.RGBA {
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

func zeroPadAudio(audioData []float32, offset, count, featureSize int) {
	totalElements := count * featureSize
	for i := 0; i < totalElements; i++ {
		audioData[offset+i] = 0.0
	}
}

type SimplePool struct {
	pool chan []byte
	size int
}

func (p *SimplePool) Get() []byte {
	select {
	case buf := <-p.pool:
		return buf[:p.size]
	default:
		return make([]byte, p.size)
	}
}

func (p *SimplePool) Put(buf []byte) {
	select {
	case p.pool <- buf:
	default:
	}
}

// Benchmark tests
func BenchmarkBatch8FPS(b *testing.B) {
	videoFrame := createVideoFrame(1920, 1080)
	bgrData := convertToBGR(videoFrame)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		processFrame(bgrData, 1920, 1080, 8)
	}

	elapsed := b.Elapsed()
	fps := float64(b.N*8) / elapsed.Seconds()
	b.ReportMetric(fps, "fps")
}

func BenchmarkBatch25FPS(b *testing.B) {
	videoFrame := createVideoFrame(1920, 1080)
	bgrData := convertToBGR(videoFrame)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		processFrame(bgrData, 1920, 1080, 25)
	}

	elapsed := b.Elapsed()
	fps := float64(b.N*25) / elapsed.Seconds()
	b.ReportMetric(fps, "fps")
}
