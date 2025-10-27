package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	pb "go-onnx-inference/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
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
	fmt.Println("================================================================================")
	fmt.Println("🧪 LipSync gRPC Client (Protobuf)")
	fmt.Println("================================================================================")

	// Connect to server
	serverAddr := "localhost:50051"
	fmt.Printf("\n🔌 Connecting to server: %s\n", serverAddr)

	conn, err := grpc.NewClient(serverAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(200*1024*1024),
			grpc.MaxCallSendMsgSize(200*1024*1024),
		),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewLipSyncClient(conn)

	// Health check
	ctx := context.Background()
	healthResp, err := client.Health(ctx, &pb.HealthRequest{})
	if err != nil {
		log.Fatalf("❌ Health check failed: %v", err)
	}

	fmt.Printf("✅ Connected to server\n")
	fmt.Printf("   Model: %s\n", healthResp.ModelPath)
	fmt.Printf("   CUDA: %v\n", healthResp.CudaAvailable)

	// Load test data
	dataDir := "d:/Projects/webcodecstest/test_data_sanders_for_go"
	fmt.Printf("\n📦 Loading test data from: %s\n", dataDir)

	visualData, err := loadBinaryFile(filepath.Join(dataDir, "visual_input.bin"))
	if err != nil {
		log.Fatalf("❌ Error loading visual data: %v", err)
	}

	audioData, err := loadBinaryFile(filepath.Join(dataDir, "audio_input.bin"))
	if err != nil {
		log.Fatalf("❌ Error loading audio data: %v", err)
	}

	visualFrameSize := 6 * 320 * 320
	audioFrameSize := 32 * 16 * 16
	numFrames := len(visualData) / visualFrameSize

	fmt.Printf("✅ Loaded %d frames\n", numFrames)
	fmt.Printf("   Visual: %.2f MB\n", float64(len(visualData)*4)/(1024*1024))
	fmt.Printf("   Audio: %.2f MB\n", float64(len(audioData)*4)/(1024*1024))

	// Test different batch sizes
	testBatchSizes := []int{1, 5, 10, 25}

	for _, batchSize := range testBatchSizes {
		if batchSize > numFrames {
			continue
		}

		fmt.Printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("🔬 Testing batch size: %d\n", batchSize)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

		// Prepare batch
		visualBatch := visualData[:batchSize*visualFrameSize]

		// Audio: send ONE audio window for the entire batch
		// The server will replicate it for each frame as needed
		// (one audio window covers ~16 visual frames)
		audioBatch := audioData[:audioFrameSize]

		// Send request
		requestStart := time.Now()

		request := &pb.BatchRequest{
			VisualFrames:  visualBatch,
			AudioFeatures: audioBatch,
			BatchSize:     int32(batchSize),
			StartFrameIdx: 0,
		}

		response, err := client.InferBatch(ctx, request)
		if err != nil {
			fmt.Printf("❌ Request failed: %v\n", err)
			continue
		}

		requestTime := time.Since(requestStart)

		if !response.Success {
			fmt.Printf("❌ Server error: %s\n", response.Error)
			continue
		}

		// Print results
		outputFrameSize := 3 * 320 * 320
		numOutputFrames := len(response.OutputFrames) / outputFrameSize

		fmt.Printf("   ✅ Inference successful\n")
		fmt.Printf("      Batch size: %d\n", batchSize)
		fmt.Printf("      Server inference time: %.2fms (%.2fms/frame)\n",
			response.InferenceTimeMs, response.InferenceTimeMs/float64(batchSize))
		fmt.Printf("      Round-trip time: %.2fms\n", float64(requestTime.Milliseconds()))
		fmt.Printf("      Network overhead: %.2fms\n", float64(requestTime.Milliseconds())-response.InferenceTimeMs)
		fmt.Printf("      Output frames: %d\n", numOutputFrames)
		fmt.Printf("      Output data size: %.2f MB\n", float64(len(response.OutputFrames)*4)/(1024*1024))

		fps := float64(batchSize) / (response.InferenceTimeMs / 1000.0)
		fpsWithNetwork := float64(batchSize) / (float64(requestTime.Milliseconds()) / 1000.0)
		fmt.Printf("      📊 Server FPS: %.2f\n", fps)
		fmt.Printf("      📊 End-to-end FPS: %.2f (including network)\n", fpsWithNetwork)
	}

	fmt.Println("\n✅ All tests completed!")
}
