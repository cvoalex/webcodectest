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

func testBatchInference() {
	// Command line flags
	server := flag.String("server", "localhost:50051", "gRPC server address")
	model := flag.String("model", "sanders", "Model name")
	startFrame := flag.Int("start", 95, "Starting frame ID")
	count := flag.Int("count", 4, "Number of frames to request in batch")
	flag.Parse()

	fmt.Println("\n" + "=================================================================")
	fmt.Println("🚀 BATCH INFERENCE TEST")
	fmt.Println("=================================================================")
	fmt.Printf("🔌 Server: %s\n", *server)
	fmt.Printf("📦 Model: %s\n", *model)
	fmt.Printf("📊 Batch: %d frames (starting at frame %d)\n", *count, *startFrame)
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

	// Prepare batch request
	frameIds := make([]int32, *count)
	for i := 0; i < *count; i++ {
		frameIds[i] = int32(*startFrame + i)
	}

	fmt.Printf("\n🎯 Sending batch request for frames: %v\n\n", frameIds)

	// Send batch request
	batchStart := time.Now()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := &pb.BatchInferenceRequest{
		ModelName: *model,
		FrameIds:  frameIds,
	}

	resp, err := client.GenerateBatchInference(ctx, req)
	if err != nil {
		log.Fatalf("❌ Batch inference failed: %v", err)
	}

	batchDuration := time.Since(batchStart)

	// Print results
	fmt.Println("======================================================================")
	fmt.Println("📊 BATCH RESULTS")
	fmt.Println("======================================================================")
	fmt.Printf("\n✅ Received %d responses\n\n", len(resp.Responses))

	totalSize := 0
	successCount := 0

	for i, r := range resp.Responses {
		if r.Success {
			successCount++
			size := len(r.PredictionData)
			totalSize += size

			fmt.Printf("  ✅ Frame %d: %dms (%.2fms inference) - %d bytes (%.2f KB)\n",
				r.FrameId,
				r.ProcessingTimeMs,
				r.InferenceTimeMs,
				size,
				float64(size)/1024.0)

			// Save frame to file
			filename := fmt.Sprintf("batch_frame_%d.jpg", r.FrameId)
			err := os.WriteFile(filename, r.PredictionData, 0644)
			if err != nil {
				fmt.Printf("     ⚠️  Failed to save: %v\n", err)
			} else {
				fmt.Printf("     💾 Saved: %s\n", filename)
			}
		} else {
			fmt.Printf("  ❌ Frame %d: ERROR - %s\n", frameIds[i], r.Error)
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
		fmt.Printf("   Total Data: %d bytes (%.2f MB)\n", totalSize, float64(totalSize)/(1024*1024))
		dataRate := float64(totalSize) / (1024 * 1024) / batchDuration.Seconds()
		fmt.Printf("   Data Rate: %.2f MB/s\n", dataRate)
	}

	fmt.Println("\n======================================================================")
	fmt.Println()
}

func main() {
	testBatchInference()
}
