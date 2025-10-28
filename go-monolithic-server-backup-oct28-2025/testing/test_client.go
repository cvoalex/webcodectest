package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	pb "go-monolithic-server/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	serverAddr = "localhost:50053" // Monolithic server port
	modelID    = "sanders"         // Change to your model ID
	batchSize  = 24                // Batch size 24 (max is 25 on current server)
	numBatches = 5                 // Run 5 batches for testing
	maxMsgSize = 100 * 1024 * 1024 // 100MB message size limit
)

func main() {
	fmt.Println("🧪 Monolithic Server Test Client")
	fmt.Println("=" + string(make([]byte, 60)))

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
	if healthResp.Healthy {
		fmt.Println("✅ Server Status: Healthy")
	} else {
		fmt.Println("⚠️  Server Status: Unhealthy")
	}
	fmt.Printf("   Loaded Models: %d/%d\n", healthResp.LoadedModels, healthResp.MaxModels)
	fmt.Printf("   GPUs: %v\n", healthResp.GpuIds)

	// Prepare output directory
	os.MkdirAll("test_output", 0755)
	fmt.Println("\n📁 Output directory: test_output/")

	// Note: Using mock audio features in old format (32x16x16) for compatibility
	// The current model expects this format, not the new 512-dim audio encoder output
	fmt.Println("\n⚠️  Using mock audio features (model expects old 32x16x16 format)")

	// Run test batches
	fmt.Printf("\n🚀 Running %d batches (batch_size=%d)...\n", numBatches, batchSize)
	fmt.Println(string(make([]byte, 60)))

	var totalInferenceMs float32
	var totalCompositeMs float32
	var totalTimeMs float32
	var totalFrames int
	startTime := time.Now()

	for batchNum := 0; batchNum < numBatches; batchNum++ {
		frameIdx := batchNum * batchSize

		// Generate mock visual frames
		visualFrames := generateMockVisualFrames(batchSize)

		// Extract audio chunk for this batch
		// Generate mock audio features (old format: 32x16x16 = 8192 floats)
		audioFeatures := generateMockAudioFeatures()

		// Call monolithic server with pre-computed audio features
		req := &pb.CompositeBatchRequest{
			ModelId:       modelID,
			VisualFrames:  visualFrames,
			AudioFeatures: audioFeatures, // Use pre-computed features for now
			BatchSize:     int32(batchSize),
			StartFrameIdx: int32(frameIdx),
		}

		batchStart := time.Now()
		resp, err := client.InferBatchComposite(context.Background(), req)
		batchDuration := time.Since(batchStart).Milliseconds()

		if err != nil {
			log.Fatalf("❌ Batch %d failed: %v", batchNum, err)
		}

		if !resp.Success {
			log.Fatalf("❌ Batch %d returned success=false: %s", batchNum, resp.Error)
		}

		// Accumulate stats
		totalInferenceMs += resp.InferenceTimeMs
		totalCompositeMs += resp.CompositeTimeMs
		totalTimeMs += resp.TotalTimeMs
		totalFrames += len(resp.CompositedFrames)

		// Calculate overhead
		overhead := resp.TotalTimeMs - resp.InferenceTimeMs

		fmt.Printf("Batch %d/%d: GPU=%d, frames=%d\n",
			batchNum+1, numBatches, resp.GpuId, len(resp.CompositedFrames))
		if resp.AudioProcessingMs > 0 {
			fmt.Printf("  🎵 Audio:       %6.2f ms\n", resp.AudioProcessingMs)
		}
		fmt.Printf("  ⚡ Inference:   %6.2f ms\n", resp.InferenceTimeMs)
		fmt.Printf("  🎨 Compositing: %6.2f ms\n", resp.CompositeTimeMs)
		fmt.Printf("  📊 Total:       %6.2f ms (actual: %d ms)\n", resp.TotalTimeMs, batchDuration)
		fmt.Printf("  📈 Overhead:    %6.2f ms (%.1f%%)\n",
			overhead, (overhead/resp.InferenceTimeMs)*100)

		// Save all frames from this batch
		for i, frameData := range resp.CompositedFrames {
			filename := fmt.Sprintf("test_output/batch_%d_frame_%d.jpg", batchNum+1, frameIdx+i)
			err = os.WriteFile(filename, frameData, 0644)
			if err != nil {
				log.Printf("⚠️  Failed to save frame %d: %v", frameIdx+i, err)
			}
		}

		// Report first frame save
		if len(resp.CompositedFrames) > 0 {
			fmt.Printf("  💾 Saved %d frames to test_output/ (%d bytes avg)\n",
				len(resp.CompositedFrames), len(resp.CompositedFrames[0]))
		}

		fmt.Println()
	}

	totalDuration := time.Since(startTime)

	// Print summary
	fmt.Println(string(make([]byte, 60)))
	fmt.Println("📈 PERFORMANCE SUMMARY")
	fmt.Println(string(make([]byte, 60)))
	fmt.Printf("Total frames processed:  %d\n", totalFrames)
	fmt.Printf("Total duration:          %.2f seconds\n", totalDuration.Seconds())
	fmt.Printf("\n⚡ Average Inference:      %.2f ms\n", totalInferenceMs/float32(numBatches))
	fmt.Printf("🎨 Average Compositing:   %.2f ms\n", totalCompositeMs/float32(numBatches))
	fmt.Printf("📊 Average Total:         %.2f ms\n", totalTimeMs/float32(numBatches))

	avgInference := totalInferenceMs / float32(numBatches)
	avgTotal := totalTimeMs / float32(numBatches)
	avgOverhead := avgTotal - avgInference
	overheadPct := (avgOverhead / avgInference) * 100

	fmt.Printf("\n📈 Separation Overhead:   %.2f ms (%.1f%% of inference time)\n",
		avgOverhead, overheadPct)

	// Throughput calculations
	fps := float64(totalFrames) / totalDuration.Seconds()
	fmt.Printf("\n🚀 Throughput:            %.1f FPS\n", fps)
	fmt.Printf("   Frames per batch:      %d\n", batchSize)
	fmt.Printf("   Batches per second:    %.1f\n", float64(numBatches)/totalDuration.Seconds())

	// Verdict
	fmt.Println("\n" + string(make([]byte, 60)))
	if avgOverhead < 5.0 {
		fmt.Println("✅ SUCCESS: Overhead < 5ms target!")
	} else if avgOverhead < 10.0 {
		fmt.Println("⚠️  WARNING: Overhead is higher than 5ms target but acceptable")
	} else {
		fmt.Println("❌ CONCERN: Overhead exceeds 10ms, investigation recommended")
	}

	fmt.Println("\n💡 Next Steps:")
	fmt.Println("   1. Check test_output/ for sample frames")
	fmt.Println("   2. Verify compositing quality")
	fmt.Println("   3. Run with different batch sizes")
	fmt.Println("   4. Test with multiple concurrent clients")
}

// generateMockAudioFeatures creates mock audio features
// Format: 32x16x16 float32 values (old format expected by model)
func generateMockAudioFeatures() []byte {
	totalFloats := 32 * 16 * 16         // 8192 floats
	data := make([]byte, totalFloats*4) // 4 bytes per float32

	// Generate random float32 values between -1 and 1
	for i := 0; i < totalFloats; i++ {
		value := float32(rand.Float64()*2.0 - 1.0)
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}

	return data
}

// generateMockVisualFrames creates random visual frame data
// Format: 6 frames of 320x320 float32 values
func generateMockVisualFrames(batchSize int) []byte {
	numFrames := 6
	height := 320
	width := 320

	totalFloats := batchSize * numFrames * height * width
	data := make([]byte, totalFloats*4) // 4 bytes per float32

	// Generate random float32 values between -1 and 1
	for i := 0; i < totalFloats; i++ {
		value := float32(rand.Float64()*2.0 - 1.0) // Range: -1 to 1
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}

	return data
}
