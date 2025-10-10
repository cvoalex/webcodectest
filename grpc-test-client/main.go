package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/cvoalex/lipsync-proxy/pb"
)

func main() {
	// Command-line flags
	serverAddr := flag.String("server", "localhost:50051", "gRPC server address")
	modelName := flag.String("model", "sanders", "Model name to use")
	frameID := flag.Int("frame", 100, "Frame ID to request")
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

	// Create request (no audio needed - uses pre-extracted features!)
	req := &pb.OptimizedInferenceRequest{
		ModelName: *modelName,
		FrameId:   int32(*frameID),
	}

	fmt.Printf("📤 Sending inference request:\n")
	fmt.Printf("   Model: %s\n", *modelName)
	fmt.Printf("   Frame: %d\n", *frameID)
	fmt.Printf("   Note: Uses pre-extracted audio features (no audio data needed)\n\n")

	// Call the gRPC method
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	startTime := time.Now()
	resp, err := client.GenerateInference(ctx, req)
	elapsed := time.Since(startTime)

	if err != nil {
		log.Fatalf("❌ gRPC call failed: %v", err)
	}

	// Display results
	fmt.Printf("✅ Success!\n\n")
	fmt.Printf("📊 Response Details:\n")
	fmt.Printf("   Success: %v\n", resp.Success)
	fmt.Printf("   Frame ID: %d\n", resp.FrameId)
	fmt.Printf("   Processing Time: %.2fms\n", float64(resp.ProcessingTimeMs))
	fmt.Printf("   Prepare Time: %.2fms\n", resp.PrepareTimeMs)
	fmt.Printf("   Inference Time: %.2fms\n", resp.InferenceTimeMs)
	fmt.Printf("   Composite Time: %.2fms\n", resp.CompositeTimeMs)
	fmt.Printf("   Total Time: %.2fms\n", elapsed.Seconds()*1000)
	fmt.Printf("   Image Size: %d bytes (%.2f KB)\n", len(resp.PredictionData), float64(len(resp.PredictionData))/1024)
	fmt.Printf("   Bounds: %v\n", resp.Bounds)

	if !resp.Success {
		fmt.Printf("\n❌ Error: %s\n", resp.Error)
		os.Exit(1)
	}

	// Optionally save the image
	if len(resp.PredictionData) > 0 {
		filename := fmt.Sprintf("frame_%d.jpg", *frameID)
		err = os.WriteFile(filename, resp.PredictionData, 0644)
		if err != nil {
			log.Printf("⚠️  Failed to save image: %v", err)
		} else {
			fmt.Printf("\n💾 Image saved to: %s\n", filename)
		}
	}
}
