package server

import (
	"context"

	pb "go-monolithic-server/proto"
)

// Health check - EXACT copy from original main.go
func (s *Server) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	gpuInfo := s.modelRegistry.GetGPUInfo()
	gpuIDs := make([]int32, len(gpuInfo))
	gpuMemory := make([]float32, len(gpuInfo))

	for i, gpu := range gpuInfo {
		gpuIDs[i] = int32(gpu.GPUID)
		gpuMemory[i] = float32(gpu.UsedMemory) / (1024 * 1024 * 1024) // Convert to GB
	}

	return &pb.HealthResponse{
		Healthy:         true,
		LoadedModels:    int32(s.modelRegistry.GetLoadedCount()),
		MaxModels:       int32(s.cfg.Capacity.MaxModels),
		GpuIds:          gpuIDs,
		GpuMemoryUsedGb: gpuMemory,
	}, nil
}

// GetModelStats returns statistics for all models - EXACT copy from original main.go
func (s *Server) GetModelStats(ctx context.Context, req *pb.GetModelStatsRequest) (*pb.GetModelStatsResponse, error) {
	stats, err := s.modelRegistry.GetStats("") // Empty string = all models
	if err != nil {
		return nil, err
	}

	modelStats := make([]*pb.ModelStats, len(stats))

	for i, m := range stats {
		avgInferenceMs := float32(0)
		if m.UsageCount > 0 {
			avgInferenceMs = float32(m.TotalInferenceMs / float64(m.UsageCount))
		}

		modelStats[i] = &pb.ModelStats{
			ModelId:           m.ID,
			Loaded:            true,
			GpuId:             int32(m.GPUID),
			MemoryMb:          m.MemoryBytes / (1024 * 1024),
			RequestCount:      int32(m.UsageCount),
			TotalInferenceMs:  int64(m.TotalInferenceMs),
			AvgInferenceMs:    avgInferenceMs,
			LastUsedTimestamp: m.LastUsed.Unix(),
		}
	}

	return &pb.GetModelStatsResponse{
		Models: modelStats,
	}, nil
}
