package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/cvoalex/lipsync-proxy/pb"
)

type FrameResult struct {
	FrameID        int32
	ProcessingTime float64
	InferenceTime  float64
	TotalTime      time.Duration
	ImageSize      int
	Success        bool
	Error          string
}

func main() {
	// Command-line flags
	serverAddr := flag.String("server", "localhost:50051", "gRPC server address")
	modelName := flag.String("model", "sanders", "Model name to use")
	startFrame := flag.Int("start", 95, "Starting frame ID")
	count := flag.Int("count", 5, "Number of frames to generate")
	flag.Parse()

	fmt.Printf("🔌 Connecting to gRPC server at %s...\n", *serverAddr)

	// Connect to the gRPC server
	conn, err := grpc.Dial(*serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewOptimizedLipSyncServiceClient(conn)

	fmt.Printf("✅ Connected!\n\n")
	fmt.Printf("📊 Generating %d frames (frames %d-%d)...\n", *count, *startFrame, *startFrame+*count-1)
	fmt.Printf("   Model: %s\n", *modelName)
	fmt.Printf("   Note: Uses pre-extracted audio features\n\n")

	// Generate multiple frames and collect stats
	results := make([]FrameResult, 0, *count)
	totalStartTime := time.Now()

	for i := 0; i < *count; i++ {
		frameID := int32(*startFrame + i)
		
		// Create request
		req := &pb.OptimizedInferenceRequest{
			ModelName: *modelName,
			FrameId:   frameID,
		}

		// Call the gRPC method
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		startTime := time.Now()
		resp, err := client.GenerateInference(ctx, req)
		elapsed := time.Since(startTime)
		cancel()

		result := FrameResult{
			FrameID:   frameID,
			TotalTime: elapsed,
		}

		if err != nil {
			result.Success = false
			result.Error = err.Error()
			fmt.Printf("❌ Frame %d: Failed - %v\n", frameID, err)
		} else {
			result.Success = resp.Success
			result.ProcessingTime = float64(resp.ProcessingTimeMs)
			result.InferenceTime = resp.InferenceTimeMs
			result.ImageSize = len(resp.PredictionData)

			if resp.Success {
				// Save the image
				filename := fmt.Sprintf("frame_%d.jpg", frameID)
				err = os.WriteFile(filename, resp.PredictionData, 0644)
				if err != nil {
					fmt.Printf("⚠️  Frame %d: Generated but failed to save - %v\n", frameID, err)
				} else {
					fmt.Printf("✅ Frame %d: %.2fms (%.2fms inference) - %d bytes - saved to %s\n",
						frameID, elapsed.Seconds()*1000, resp.InferenceTimeMs, len(resp.PredictionData), filename)
				}
			} else {
				result.Error = "Server returned success=false"
				fmt.Printf("❌ Frame %d: Server error\n", frameID)
			}
		}

		results = append(results, result)
	}

	totalElapsed := time.Since(totalStartTime)

	// Calculate statistics
	fmt.Printf("\n" + strings.Repeat("=", 70) + "\n")
	fmt.Printf("📈 PERFORMANCE STATISTICS\n")
	fmt.Printf(strings.Repeat("=", 70) + "\n\n")

	successCount := 0
	var totalProcessing, totalInference, totalTotal float64
	var minProcessing, maxProcessing, minInference, maxInference float64
	var totalSize int

	minProcessing = 999999
	maxProcessing = 0
	minInference = 999999
	maxInference = 0

	for _, result := range results {
		if result.Success {
			successCount++
			totalProcessing += result.ProcessingTime
			totalInference += result.InferenceTime
			totalTotal += result.TotalTime.Seconds() * 1000
			totalSize += result.ImageSize

			if result.ProcessingTime < minProcessing {
				minProcessing = result.ProcessingTime
			}
			if result.ProcessingTime > maxProcessing {
				maxProcessing = result.ProcessingTime
			}
			if result.InferenceTime < minInference {
				minInference = result.InferenceTime
			}
			if result.InferenceTime > maxInference {
				maxInference = result.InferenceTime
			}
		}
	}

	fmt.Printf("📊 Summary:\n")
	fmt.Printf("   Total Frames: %d\n", len(results))
	fmt.Printf("   Successful: %d\n", successCount)
	fmt.Printf("   Failed: %d\n", len(results)-successCount)
	fmt.Printf("   Total Time: %.2fms (%.2fs)\n", totalElapsed.Seconds()*1000, totalElapsed.Seconds())
	fmt.Printf("   Total Data: %d bytes (%.2f MB)\n", totalSize, float64(totalSize)/1024/1024)

	if successCount > 0 {
		avgProcessing := totalProcessing / float64(successCount)
		avgInference := totalInference / float64(successCount)
		avgTotal := totalTotal / float64(successCount)
		avgSize := totalSize / successCount
		fps := float64(successCount) / totalElapsed.Seconds()

		fmt.Printf("\n⚡ Processing Time:\n")
		fmt.Printf("   Average: %.2fms\n", avgProcessing)
		fmt.Printf("   Min: %.2fms\n", minProcessing)
		fmt.Printf("   Max: %.2fms\n", maxProcessing)

		fmt.Printf("\n🧠 Inference Time:\n")
		fmt.Printf("   Average: %.2fms\n", avgInference)
		fmt.Printf("   Min: %.2fms\n", minInference)
		fmt.Printf("   Max: %.2fms\n", maxInference)

		fmt.Printf("\n🌐 Total Time (including network):\n")
		fmt.Printf("   Average: %.2fms\n", avgTotal)

		fmt.Printf("\n💾 Image Size:\n")
		fmt.Printf("   Average: %d bytes (%.2f KB)\n", avgSize, float64(avgSize)/1024)

		fmt.Printf("\n🚀 Throughput:\n")
		fmt.Printf("   Frames per second: %.2f FPS\n", fps)
		fmt.Printf("   Data rate: %.2f MB/s\n", float64(totalSize)/1024/1024/totalElapsed.Seconds())
	}

	fmt.Printf("\n" + strings.Repeat("=", 70) + "\n")

	if successCount < len(results) {
		fmt.Printf("\n⚠️  Some frames failed:\n")
		for _, result := range results {
			if !result.Success {
				fmt.Printf("   Frame %d: %s\n", result.FrameID, result.Error)
			}
		}
		os.Exit(1)
	}

	fmt.Printf("\n✅ All frames generated successfully!\n")
}
