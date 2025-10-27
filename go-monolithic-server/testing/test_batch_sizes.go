package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	pb "go-monolithic-server/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	serverAddr = "localhost:50053"
	modelID    = "sanders"
	maxMsgSize = 100 * 1024 * 1024
)

func main() {
	fmt.Println("🧪 Monolithic Server - Batch Size Performance Test")
	fmt.Println("=" + string(make([]byte, 70)))

	// Connect to monolithic server
	fmt.Printf("🔌 Connecting to monolithic server at %s...\n", serverAddr)
	conn, err := grpc.NewClient(
		serverAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewMonolithicServiceClient(conn)
	fmt.Println("✅ Connected successfully")

	// Check server health
	fmt.Println("\n📊 Checking server health...")
	healthResp, err := client.Health(context.Background(), &pb.HealthRequest{})
	if err != nil {
		log.Fatalf("❌ Health check failed: %v", err)
	}
	fmt.Printf("✅ Server Status: Healthy (Models: %d/%d, GPUs: %v)\n",
		healthResp.LoadedModels, healthResp.MaxModels, healthResp.GpuIds)

	// Prepare output directory
	os.MkdirAll("test_output", 0755)

	// Test different batch sizes
	batchSizes := []int{1, 4, 8, 25}

	fmt.Println("\n⚠️  Using mock audio features (model expects old 32x16x16 format)")
	fmt.Println("\n🚀 Testing batch sizes: 1, 4, 8, 25")
	fmt.Println("   Running 3 iterations per batch size (first may include loading)")
	fmt.Println("=" + string(make([]byte, 70)))

	for _, batchSize := range batchSizes {
		fmt.Printf("\n📊 TESTING BATCH SIZE: %d\n", batchSize)
		fmt.Println("─" + string(make([]byte, 70)))

		var totalInferenceMs float32
		var totalCompositeMs float32
		var totalTimeMs float32
		iterations := 3

		for i := 0; i < iterations; i++ {
			frameIdx := i * batchSize

			// Generate mock visual frames
			visualFrames := generateMockVisualFrames(batchSize)

			// Generate mock audio features (old format: 32x16x16 = 8192 floats)
			audioFeatures := generateMockAudioFeatures()

			// Call monolithic server
			req := &pb.CompositeBatchRequest{
				ModelId:       modelID,
				VisualFrames:  visualFrames,
				AudioFeatures: audioFeatures,
				BatchSize:     int32(batchSize),
				StartFrameIdx: int32(frameIdx),
			}

			batchStart := time.Now()
			resp, err := client.InferBatchComposite(context.Background(), req)
			batchDuration := time.Since(batchStart)

			if err != nil {
				log.Fatalf("❌ Batch failed: %v", err)
			}

			if !resp.Success {
				log.Fatalf("❌ Batch returned success=false: %s", resp.Error)
			}

			// Print iteration results
			iterLabel := "WARM"
			if i == 0 {
				iterLabel = "COLD (w/ loading)"
			}

			fmt.Printf("  Iter %d/%d [%s]:\n", i+1, iterations, iterLabel)
			fmt.Printf("    ⚡ Inference:   %7.2f ms\n", resp.InferenceTimeMs)
			fmt.Printf("    🎨 Compositing: %7.2f ms\n", resp.CompositeTimeMs)
			fmt.Printf("    📊 Total:       %7.2f ms (actual: %d ms)\n",
				resp.TotalTimeMs, batchDuration.Milliseconds())

			if len(resp.CompositedFrames) > 0 {
				fmt.Printf("    💾 Frame size:  %d bytes avg\n",
					len(resp.CompositedFrames[0]))
			}

			// Accumulate stats (skip first iteration for cold start)
			if i > 0 {
				totalInferenceMs += resp.InferenceTimeMs
				totalCompositeMs += resp.CompositeTimeMs
				totalTimeMs += resp.TotalTimeMs
			}

			// Save first frame of first iteration for quality check
			if i == 0 && len(resp.CompositedFrames) > 0 {
				filename := fmt.Sprintf("test_output/batch_%d_sample.jpg", batchSize)
				if err := os.WriteFile(filename, resp.CompositedFrames[0], 0644); err != nil {
					log.Printf("⚠️  Failed to save sample: %v", err)
				}
			}

			// Small delay between iterations
			time.Sleep(100 * time.Millisecond)
		}

		// Calculate averages (excluding first cold iteration)
		warmIterations := iterations - 1
		avgInference := totalInferenceMs / float32(warmIterations)
		avgComposite := totalCompositeMs / float32(warmIterations)
		avgTotal := totalTimeMs / float32(warmIterations)

		fmt.Println("\n  📈 WARM Performance (avg of last 2 iterations):")
		fmt.Printf("    ⚡ Inference:   %7.2f ms  (%.2f ms/frame)\n",
			avgInference, avgInference/float32(batchSize))
		fmt.Printf("    🎨 Compositing: %7.2f ms  (%.2f ms/frame)\n",
			avgComposite, avgComposite/float32(batchSize))
		fmt.Printf("    📊 Total:       %7.2f ms  (%.2f ms/frame)\n",
			avgTotal, avgTotal/float32(batchSize))
		fmt.Printf("    🚀 Throughput:  %.1f FPS\n",
			float32(batchSize)*1000.0/avgTotal)
	}

	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("✅ Batch size performance test complete!")
	fmt.Println("\n💡 Check test_output/ for sample frames from each batch size")
}

// generateMockVisualFrames creates mock visual input data
func generateMockVisualFrames(batchSize int) []byte {
	// Visual frames: batch_size × 6 × 320 × 320 float32 values
	numFloats := batchSize * 6 * 320 * 320
	data := make([]byte, numFloats*4)

	// Fill with pseudo-random but deterministic pattern
	for i := 0; i < numFloats; i++ {
		val := float32(math.Sin(float64(i)*0.01)) * 0.5
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(val))
	}

	return data
}

// generateMockAudioFeatures creates mock audio features in old format
func generateMockAudioFeatures() []byte {
	// Old audio format: 32 × 16 × 16 = 8192 float32 values
	numFloats := 32 * 16 * 16
	data := make([]byte, numFloats*4)

	// Fill with pseudo-random audio features
	for i := 0; i < numFloats; i++ {
		val := float32(math.Sin(float64(i)*0.1)) * 0.3
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(val))
	}

	return data
}
