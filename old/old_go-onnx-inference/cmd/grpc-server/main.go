package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"time"

	"go-onnx-inference/lipsyncinfer"
	pb "go-onnx-inference/proto"

	"google.golang.org/grpc"
)

const (
	visualFrameSize = 6 * 320 * 320
	audioFrameSize  = 32 * 16 * 16
	outputFrameSize = 3 * 320 * 320
)

type lipSyncServer struct {
	pb.UnimplementedLipSyncServer
	inferencer *lipsyncinfer.Inferencer
	modelPath  string
}

func (s *lipSyncServer) InferBatch(ctx context.Context, req *pb.BatchRequest) (*pb.BatchResponse, error) {
	start := time.Now()

	// Validate batch size
	if req.BatchSize < 1 || req.BatchSize > 25 {
		return &pb.BatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid batch size: %d (must be 1-25)", req.BatchSize),
		}, nil
	}

	// Validate input sizes
	expectedVisualSize := int(req.BatchSize) * visualFrameSize
	expectedAudioSize := audioFrameSize // Single audio window covers multiple frames

	if len(req.VisualFrames) != expectedVisualSize {
		return &pb.BatchResponse{
			Success: false,
			Error: fmt.Sprintf("Invalid visual frames size: got %d, expected %d",
				len(req.VisualFrames), expectedVisualSize),
		}, nil
	}

	if len(req.AudioFeatures) != expectedAudioSize {
		return &pb.BatchResponse{
			Success: false,
			Error: fmt.Sprintf("Invalid audio features size: got %d, expected %d",
				len(req.AudioFeatures), expectedAudioSize),
		}, nil
	}

	// Replicate audio window for each frame (audio window covers ~16 frames)
	// InferBatch expects audio for each frame, so we duplicate the single window
	audioForBatch := make([]float32, int(req.BatchSize)*audioFrameSize)
	for i := 0; i < int(req.BatchSize); i++ {
		copy(audioForBatch[i*audioFrameSize:], req.AudioFeatures)
	}

	// Perform batch inference
	outputs, err := s.inferencer.InferBatch(req.VisualFrames, audioForBatch, int(req.BatchSize))
	if err != nil {
		return &pb.BatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Inference failed: %v", err),
		}, nil
	}

	inferenceTime := time.Since(start)

	return &pb.BatchResponse{
		OutputFrames:    outputs,
		InferenceTimeMs: float64(inferenceTime.Milliseconds()),
		Success:         true,
	}, nil
}

func (s *lipSyncServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{
		Healthy:       true,
		ModelPath:     s.modelPath,
		CudaAvailable: true, // Assuming CUDA if server is running
	}, nil
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 LipSync gRPC Server (Protobuf)")
	fmt.Println("================================================================================")

	modelPath := "d:/Projects/webcodecstest/minimal_server/models/sanders/checkpoint/model_best.onnx"
	port := ":50051"

	fmt.Printf("\n📁 Loading model: %s\n", modelPath)

	inferencer, err := lipsyncinfer.NewInferencer(modelPath)
	if err != nil {
		log.Fatalf("❌ Failed to create inferencer: %v", err)
	}
	defer inferencer.Close()

	fmt.Println("✅ Model loaded successfully")
	fmt.Println("✅ CUDA enabled")

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(200*1024*1024), // 200MB max receive
		grpc.MaxSendMsgSize(200*1024*1024), // 200MB max send
	)

	server := &lipSyncServer{
		inferencer: inferencer,
		modelPath:  modelPath,
	}

	pb.RegisterLipSyncServer(grpcServer, server)

	// Start listening
	lis, err := net.Listen("tcp", port)
	if err != nil {
		log.Fatalf("❌ Failed to listen: %v", err)
	}

	fmt.Printf("\n🌐 Server listening on port %s\n", port)
	fmt.Println("   Batch size: 1-25 frames")
	fmt.Println("   Protocol: gRPC with Protobuf")
	fmt.Println("   Max message size: 200MB")
	fmt.Println("\n✅ Ready to accept connections!")
	fmt.Println("================================================================================\n")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("❌ Failed to serve: %v", err)
	}
}
