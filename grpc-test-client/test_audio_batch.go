package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	pb "github.com/cvoalex/lipsync-proxy/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func testAudioBatchInference() {
	// Command line flags
	server := flag.String("server", "localhost:50051", "gRPC server address")
	model := flag.String("model", "sanders", "Model name")
	startFrame := flag.Int("start", 100, "Starting frame ID")
	count := flag.Int("count", 4, "Number of consecutive frames")
	flag.Parse()

	fmt.Println("\n" + "=================================================================")
	fmt.Println("🎵 AUDIO BATCH INFERENCE TEST")
	fmt.Println("=================================================================")
	fmt.Printf("🔌 Server: %s\n", *server)
	fmt.Printf("📦 Model: %s\n", *model)
	fmt.Printf("📊 Frames: %d-%d (%d frames)\n", *startFrame, *startFrame+*count-1, *count)

	// Calculate audio chunks needed
	audioChunksNeeded := *count + 15  // 8 before + N frames + 7 after
	oldMethod := *count * 16
	savings := float64(oldMethod-audioChunksNeeded) / float64(oldMethod) * 100

	fmt.Printf("🎵 Audio chunks: %d (vs %d old method = %.1f%% savings)\n", audioChunksNeeded, oldMethod, savings)
	fmt.Println()

	// Connect to gRPC server
	fmt.Printf("🔌 Connecting to gRPC server at %s...\n", *server)
	conn, err := grpc.Dial(*server,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(50*1024*1024)),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewOptimizedLipSyncServiceClient(conn)
	fmt.Println("✅ Connected!")

	// Generate dummy audio chunks (in real scenario, these would be actual audio data)
	audioChunks := make([][]byte, audioChunksNeeded)
	for i := 0; i < audioChunksNeeded; i++ {
		// Each chunk is ~16KB of dummy audio data (40ms)
		chunk := make([]byte, 16384)
		for j := range chunk {
			chunk[j] = byte(i % 256) // Dummy pattern
		}
		audioChunks[i] = chunk
	}

	totalAudioSize := len(audioChunks) * len(audioChunks[0])
	fmt.Printf("\n🎵 Generated %d audio chunks (%d bytes total = %.2f MB)\n",
		len(audioChunks), totalAudioSize, float64(totalAudioSize)/(1024*1024))

	fmt.Printf("\n🎯 Sending audio batch request...\n\n")

	// Send audio batch request
	batchStart := time.Now()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := &pb.BatchInferenceWithAudioRequest{
		ModelName:    *model,
		StartFrameId: int32(*startFrame),
		FrameCount:   int32(*count),
		AudioChunks:  audioChunks,
	}

	resp, err := client.GenerateBatchWithAudio(ctx, req)
	if err != nil {
		log.Fatalf("❌ Audio batch inference failed: %v", err)
	}

	batchDuration := time.Since(batchStart)

	// Print results
	fmt.Println("======================================================================")
	fmt.Println("📊 AUDIO BATCH RESULTS")
	fmt.Println("======================================================================")
	fmt.Printf("\n✅ Received %d responses\n\n", len(resp.Responses))

	totalSize := 0
	successCount := 0

	for i, r := range resp.Responses {
		frameId := *startFrame + i
		if r.Success {
			successCount++
			size := len(r.PredictionData)
			totalSize += size

			fmt.Printf("  ✅ Frame %d: %dms (%.2fms inference) - %d bytes (%.2f KB)\n",
				frameId,
				r.ProcessingTimeMs,
				r.InferenceTimeMs,
				size,
				float64(size)/1024.0)

			// Save frame to file
			filename := fmt.Sprintf("audio_batch_frame_%d.jpg", frameId)
			err := os.WriteFile(filename, r.PredictionData, 0644)
			if err != nil {
				fmt.Printf("     ⚠️  Failed to save: %v\n", err)
			} else {
				fmt.Printf("     💾 Saved: %s\n", filename)
			}
		} else {
			errorMsg := "unknown error"
			if r.Error != nil {
				errorMsg = *r.Error
			}
			fmt.Printf("  ❌ Frame %d: ERROR - %s\n", frameId, errorMsg)
		}
	}

	fmt.Println("\n======================================================================")
	fmt.Println("📈 PERFORMANCE SUMMARY")
	fmt.Println("======================================================================")
	fmt.Printf("\n🎯 Batch Stats:\n")
	fmt.Printf("   Total Time: %.2fms\n", float64(batchDuration.Milliseconds()))
	fmt.Printf("   Server Total: %dms\n", resp.TotalProcessingTimeMs)
	fmt.Printf("   Server Avg: %.2fms per frame\n", resp.AvgFrameTimeMs)
	fmt.Printf("   Success Rate: %d/%d frames\n", successCount, *count)

	if batchDuration.Seconds() > 0 {
		fps := float64(*count) / batchDuration.Seconds()
		fmt.Printf("   Throughput: %.2f FPS\n", fps)
	}

	if totalSize > 0 {
		fmt.Printf("   Frame Data: %d bytes (%.2f MB)\n", totalSize, float64(totalSize)/(1024*1024))
		dataRate := float64(totalSize) / (1024 * 1024) / batchDuration.Seconds()
		fmt.Printf("   Data Rate: %.2f MB/s\n", dataRate)
	}

	// Bandwidth comparison
	fmt.Println("\n======================================================================")
	fmt.Println("📊 BANDWIDTH ANALYSIS")
	fmt.Println("======================================================================")
	fmt.Printf("\n🎵 Audio Transfer:\n")
	fmt.Printf("   Chunks Sent: %d\n", audioChunksNeeded)
	fmt.Printf("   Old Method: %d chunks\n", oldMethod)
	fmt.Printf("   Savings: %d chunks (%.1f%%)\n", oldMethod-audioChunksNeeded, savings)
	fmt.Printf("   Audio Size: %.2f MB\n", float64(totalAudioSize)/(1024*1024))

	oldAudioSize := oldMethod * 16384
	fmt.Printf("   Old Method Size: %.2f MB\n", float64(oldAudioSize)/(1024*1024))
	fmt.Printf("   Bandwidth Saved: %.2f MB (%.1f%%)\n",
		float64(oldAudioSize-totalAudioSize)/(1024*1024), savings)

	fmt.Println("\n======================================================================")
	fmt.Println()
}

func main() {
	testAudioBatchInference()
}
