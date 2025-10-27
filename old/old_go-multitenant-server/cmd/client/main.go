package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	pb "go-multitenant-server/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	visualFrameSize = 6 * 320 * 320
	audioFrameSize  = 32 * 16 * 16
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
	serverAddr := "localhost:50053"
	dataDir := "d:/Projects/webcodecstest/test_data_sanders_for_go"

	log.Println(strings.Repeat("=", 80))
	log.Println("🧪 Multi-Tenant LipSync gRPC Client")
	log.Println(strings.Repeat("=", 80))
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
			grpc.MaxCallRecvMsgSize(200*1024*1024),
			grpc.MaxCallSendMsgSize(200*1024*1024),
		),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewMultiTenantLipSyncClient(conn)
	log.Println("✓ Connected")

	// Health check
	log.Println("\n❤️  Health check...")
	healthResp, err := client.Health(context.Background(), &pb.HealthRequest{})
	if err != nil {
		log.Fatalf("❌ Health check failed: %v", err)
	}
	log.Printf("✓ Server healthy")
	log.Printf("   Loaded models: %d/%d", healthResp.LoadedModels, healthResp.MaxModels)
	log.Printf("   CUDA: %v", healthResp.CudaAvailable)
	log.Printf("   Version: %s", healthResp.Version)

	// List available models
	log.Println("\n📋 Listing available models...")
	listResp, err := client.ListModels(context.Background(), &pb.ListModelsRequest{})
	if err != nil {
		log.Fatalf("❌ Failed to list models: %v", err)
	}

	log.Printf("✓ Found %d configured models:", len(listResp.Models))
	for _, model := range listResp.Models {
		status := "⚪ not loaded"
		if model.Loaded {
			status = fmt.Sprintf("✅ loaded (used %d times)", model.Stats.UsageCount)
		}
		log.Printf("   • %s: %s", model.ModelId, status)
	}

	// Test inference with sanders model
	log.Println("\n" + strings.Repeat("=", 80))
	log.Println("🎬 Testing inference with 'sanders' model")
	log.Println(strings.Repeat("=", 80))

	testInference(client, "sanders", visualData, audioData, 5)

	// Test with different batch sizes
	batchSizes := []int{1, 5, 10}
	for _, batchSize := range batchSizes {
		log.Println("\n" + strings.Repeat("-", 80))
		log.Printf("Testing batch size: %d", batchSize)
		log.Println(strings.Repeat("-", 80))
		testInference(client, "sanders", visualData, audioData, batchSize)
	}

	// Get model statistics
	log.Println("\n" + strings.Repeat("=", 80))
	log.Println("📊 Model Statistics")
	log.Println(strings.Repeat("=", 80))

	statsResp, err := client.GetModelStats(context.Background(), &pb.GetModelStatsRequest{})
	if err != nil {
		log.Printf("❌ Failed to get stats: %v", err)
	} else {
		log.Printf("Capacity: %d/%d models loaded", statsResp.LoadedModels, statsResp.MaxModels)
		log.Printf("Memory: %.2f MB / %.2f GB",
			float64(statsResp.TotalMemoryBytes)/(1024*1024),
			float64(statsResp.MaxMemoryBytes)/(1024*1024*1024))

		for _, model := range statsResp.Models {
			if model.Stats != nil {
				log.Printf("\n📌 Model '%s':", model.ModelId)
				log.Printf("   Usage count: %d", model.Stats.UsageCount)
				log.Printf("   Last used: %s ago",
					time.Since(time.UnixMilli(model.Stats.LastUsedUnixMs)).Round(time.Second))
				log.Printf("   Total inference: %.2f seconds", model.Stats.TotalInferenceTimeMs/1000)
				log.Printf("   Avg per inference: %.2f ms",
					model.Stats.TotalInferenceTimeMs/float64(model.Stats.UsageCount))
				log.Printf("   Memory: %.2f MB", float64(model.Stats.MemoryBytes)/(1024*1024))
				log.Printf("   Loaded at: %s",
					time.UnixMilli(model.Stats.LoadedUnixMs).Format("15:04:05"))
			}
		}
	}

	// Test model loading/unloading
	log.Println("\n" + strings.Repeat("=", 80))
	log.Println("🔄 Testing Model Management")
	log.Println(strings.Repeat("=", 80))

	// Unload sanders
	log.Println("\n🗑️  Unloading 'sanders' model...")
	unloadResp, err := client.UnloadModel(context.Background(), &pb.UnloadModelRequest{
		ModelId: "sanders",
	})
	if err != nil {
		log.Printf("❌ Unload failed: %v", err)
	} else if unloadResp.Success {
		log.Println("✓ Model unloaded")
	} else {
		log.Printf("❌ Unload failed: %s", unloadResp.Error)
	}

	// Reload sanders
	log.Println("\n📦 Reloading 'sanders' model...")
	loadResp, err := client.LoadModel(context.Background(), &pb.LoadModelRequest{
		ModelId:     "sanders",
		ForceReload: false,
	})
	if err != nil {
		log.Printf("❌ Load failed: %v", err)
	} else if loadResp.Success {
		log.Printf("✓ Model loaded in %.2f seconds", loadResp.LoadTimeMs/1000)
	} else {
		log.Printf("❌ Load failed: %s", loadResp.Error)
	}

	log.Println("\n" + strings.Repeat("=", 80))
	log.Println("✅ All tests complete!")
	log.Println(strings.Repeat("=", 80))
}

func testInference(client pb.MultiTenantLipSyncClient, modelID string, visualData, audioData []float32, batchSize int) {
	startIdx := 0
	visualBatch := visualData[startIdx*visualFrameSize : (startIdx+batchSize)*visualFrameSize]
	audioWindow := audioData[0:audioFrameSize]

	req := &pb.CompositeBatchRequest{
		ModelId:       modelID,
		VisualFrames:  visualBatch,
		AudioFeatures: audioWindow,
		BatchSize:     int32(batchSize),
		StartFrameIdx: int32(startIdx),
	}

	log.Printf("📤 Sending request (model=%s, batch=%d, start_frame=%d)...",
		modelID, batchSize, startIdx)
	start := time.Now()

	resp, err := client.InferBatchComposite(context.Background(), req)

	elapsed := time.Since(start)

	if err != nil {
		log.Printf("❌ Request failed: %v", err)
		return
	}

	if !resp.Success {
		log.Printf("❌ Server error: %s", resp.Error)
		return
	}

	log.Printf("✓ Success!")
	if resp.ModelLoaded {
		log.Printf("   Model loaded: %.2f ms", resp.ModelLoadTimeMs)
	}
	log.Printf("   Inference: %.2f ms", resp.InferenceTimeMs)
	log.Printf("   Compositing: %.2f ms", resp.CompositeTimeMs)
	log.Printf("   Total (server): %.2f ms", resp.TotalTimeMs)
	log.Printf("   Total (client): %.2f ms", elapsed.Seconds()*1000)
	log.Printf("   Throughput: %.2f FPS", float64(batchSize)/(elapsed.Seconds()))
	log.Printf("   Output: %d PNG frames (avg %.2f KB each)",
		len(resp.CompositedFrames),
		float64(len(resp.CompositedFrames[0]))/1024)
}
