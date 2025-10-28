package server

import (
	"context"
	"fmt"
	"time"

	pb "go-monolithic-server/proto"
)

// ListModels returns all configured models - EXACT copy from original main.go
func (s *Server) ListModels(ctx context.Context, req *pb.ListModelsRequest) (*pb.ListModelsResponse, error) {
	models := s.modelRegistry.ListModels()
	modelInfos := make([]*pb.ModelInfo, len(models))

	for i, m := range models {
		avgInferenceMs := float32(0)
		if m.UsageCount > 0 {
			avgInferenceMs = float32(m.TotalInferenceMs / float64(m.UsageCount))
		}

		modelInfos[i] = &pb.ModelInfo{
			ModelId:        m.ID,
			Loaded:         true, // All returned models are loaded
			GpuId:          int32(m.GPUID),
			MemoryMb:       m.MemoryBytes / (1024 * 1024),
			RequestCount:   int32(m.UsageCount),
			AvgInferenceMs: avgInferenceMs,
		}
	}

	return &pb.ListModelsResponse{
		Models: modelInfos,
	}, nil
}

// LoadModel explicitly loads a model - EXACT copy from original main.go
func (s *Server) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	startTime := time.Now()

	modelInstance, err := s.modelRegistry.GetOrLoadModel(req.ModelId)
	if err != nil {
		return &pb.LoadModelResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to load model: %v", err),
		}, nil
	}

	// Also load backgrounds for this model
	_, err = s.imageRegistry.GetModelData(req.ModelId)
	if err != nil {
		return &pb.LoadModelResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to load backgrounds: %v", err),
		}, nil
	}

	loadTime := time.Since(startTime)

	return &pb.LoadModelResponse{
		Success:    true,
		GpuId:      int32(modelInstance.GPUID),
		LoadTimeMs: float32(loadTime.Seconds() * 1000),
	}, nil
}

// UnloadModel explicitly unloads a model - EXACT copy from original main.go
func (s *Server) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*pb.UnloadModelResponse, error) {
	err := s.modelRegistry.UnloadModel(req.ModelId)
	if err != nil {
		return &pb.UnloadModelResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to unload model: %v", err),
		}, nil
	}

	// Also unload backgrounds
	s.imageRegistry.UnloadModel(req.ModelId)

	return &pb.UnloadModelResponse{
		Success: true,
	}, nil
}
