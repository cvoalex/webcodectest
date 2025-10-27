package main

import (
	"context"
	"encoding/binary"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	pb "go-onnx-inference/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	visualFrameSize = 6 * 320 * 320
	audioFrameSize  = 32 * 16 * 16
	outputFrameSize = 3 * 320 * 320
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

func main() {
	serverAddr := "localhost:50052"
	dataDir := "d:/Projects/webcodecstest/test_data_sanders_for_go"

	log.Println("================================================================================")
	log.Println("🧪 gRPC Composite Server Test Client")
	log.Println("================================================================================")
	log.Printf("   Server: %s", serverAddr)
	log.Printf("   Data: %s", dataDir)
	log.Println()

	// Load test data
	log.Println("📦 Loading test data...")
	visualPath := filepath.Join(dataDir, "visual_input.bin")
	audioPath := filepath.Join(dataDir, "audio_input.bin")

	visualData, err := loadBinaryFile(visualPath)
	if err != nil {
		log.Fatalf("❌ Error loading visual data: %v", err)
	}

	audioData, err := loadBinaryFile(audioPath)
	if err != nil {
		log.Fatalf("❌ Error loading audio data: %v", err)
	}

	numFrames := len(visualData) / visualFrameSize
	log.Printf("✓ Loaded %d frames (%.2f MB visual, %.2f MB audio)\n",
		numFrames,
		float64(len(visualData)*4)/(1024*1024),
		float64(len(audioData)*4)/(1024*1024))

	// Connect to server
	log.Printf("\n🔌 Connecting to %s...", serverAddr)
	conn, err := grpc.NewClient(
		serverAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(100*1024*1024),
			grpc.MaxCallSendMsgSize(100*1024*1024),
		),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewLipSyncClient(conn)
	log.Println("✓ Connected")

	// Health check
	log.Println("\n❤️  Health check...")
	healthResp, err := client.Health(context.Background(), &pb.HealthRequest{})
	if err != nil {
		log.Fatalf("❌ Health check failed: %v", err)
	}
	log.Printf("✓ Server healthy: %v\n", healthResp.Healthy)

	// Test different batch sizes
	batchSizes := []int{1, 5, 10}

	for _, batchSize := range batchSizes {
		log.Printf("\n" + strings.Repeat("=", 80))
		log.Printf("Testing batch size: %d", batchSize)
		log.Printf(strings.Repeat("=", 80))

		// Prepare batch
		startIdx := 0
		visualBatch := visualData[startIdx*visualFrameSize : (startIdx+batchSize)*visualFrameSize]
		audioWindow := audioData[0:audioFrameSize] // One audio window

		// Make request
		req := &pb.BatchRequest{
			VisualFrames:  visualBatch,
			AudioFeatures: audioWindow,
			BatchSize:     int32(batchSize),
			StartFrameIdx: int32(startIdx),
		}

		log.Printf("📤 Sending request (batch=%d, start_frame=%d)...", batchSize, startIdx)
		start := time.Now()

		resp, err := client.InferBatch(context.Background(), req)

		elapsed := time.Since(start)

		if err != nil {
			log.Printf("❌ Request failed: %v", err)
			continue
		}

		if !resp.Success {
			log.Printf("❌ Server error: %s", resp.Error)
			continue
		}

		log.Printf("✓ Success!")
		log.Printf("   Inference time: %.2f ms", resp.InferenceTimeMs)
		log.Printf("   Total time: %.2f ms", elapsed.Seconds()*1000)
		log.Printf("   Info: %s", resp.Error) // Contains composite timing info
		log.Printf("   Throughput: %.2f FPS", float64(batchSize)/(elapsed.Seconds()))
	}

	log.Println("\n" + strings.Repeat("=", 80))
	log.Println("✅ All tests complete!")
	log.Println(strings.Repeat("=", 80))
}
