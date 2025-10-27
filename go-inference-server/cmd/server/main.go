package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"net"
	"time"
	"unsafe"

	"go-inference-server/audio"
	"go-inference-server/config"
	pb "go-inference-server/proto"
	"go-inference-server/registry"

	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
)

const (
	visualFrameSize = 6 * 320 * 320
	audioFrameSize  = 32 * 16 * 16
	outputFrameSize = 3 * 320 * 320
)

type inferenceServer struct {
	pb.UnimplementedInferenceServiceServer
	registry      *registry.ModelRegistry
	cfg           *config.Config
	audioProcessor *audio.Processor
	audioEncoder   *audio.AudioEncoder
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 Multi-GPU Inference Server (Inference ONLY)")
	fmt.Println("================================================================================")

	// Load configuration
	cfgPath := "config.yaml"
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("❌ Failed to load config: %v", err)
	}

	log.Printf("✅ Configuration loaded from %s", cfgPath)
	log.Printf("   GPUs: %d × %dGB", cfg.GPUs.Count, cfg.GPUs.MemoryGBPerGPU)
	log.Printf("   Workers per GPU: %d (total: %d workers)",
		cfg.Server.WorkerCountPerGPU, cfg.GPUs.Count*cfg.Server.WorkerCountPerGPU)
	log.Printf("   Max models: %d", cfg.Capacity.MaxModels)
	log.Printf("   Max memory: %d GB", cfg.Capacity.MaxMemoryGB)
	log.Printf("   Eviction policy: %s", cfg.Capacity.EvictionPolicy)
	log.Printf("   Configured models: %d", len(cfg.Models))

	// Create model registry
	log.Println("\n📦 Initializing model registry...")
	reg, err := registry.NewModelRegistry(cfg)
	if err != nil {
		log.Fatalf("❌ Failed to create model registry: %v", err)
	}
	defer reg.Close()

	log.Printf("✅ Model registry initialized (%d models preloaded)\n", reg.GetLoadedCount())

	// Display GPU info
	gpuInfo := reg.GetGPUInfo()
	fmt.Println("🎮 GPU Status:")
	for _, gpu := range gpuInfo {
		fmt.Printf("   GPU %d: %d models, %d MB used / %d MB total\n",
			gpu.GPUID,
			gpu.LoadedModels,
			gpu.UsedMemory/(1024*1024),
			gpu.TotalMemory/(1024*1024))
	}

	// Create gRPC server with keep-alive enforcement
	maxSize := cfg.Server.MaxMessageSizeMB * 1024 * 1024

	// Configure server-side keep-alive enforcement
	var kaep = keepalive.EnforcementPolicy{
		MinTime:             5 * time.Second, // Min time between pings
		PermitWithoutStream: true,            // Allow pings even without active streams
	}

	var kasp = keepalive.ServerParameters{
		Time:    10 * time.Second, // Ping client if idle for 10 seconds
		Timeout: 3 * time.Second,  // Wait 3 seconds for ping ack
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxSize),
		grpc.MaxSendMsgSize(maxSize),
		grpc.KeepaliveEnforcementPolicy(kaep),
		grpc.KeepaliveParams(kasp),
	)

	server := &inferenceServer{
		registry: reg,
		cfg:      cfg,
	}

	pb.RegisterInferenceServiceServer(grpcServer, server)

	// Initialize audio processing pipeline
	log.Println("\n🎵 Initializing audio processing pipeline...")
	audioProcessor := audio.NewProcessor(nil)
	log.Println("✅ Mel-spectrogram processor initialized")

	audioEncoder, err := audio.NewAudioEncoder(cfg.ONNX.LibraryPath)
	if err != nil {
		log.Printf("⚠️  Warning: Audio encoder not available: %v", err)
		log.Println("   Server will require pre-computed audio features")
		audioEncoder = nil
	} else {
		log.Println("✅ Audio encoder initialized (ONNX)")
	}

	server.audioProcessor = audioProcessor
	server.audioEncoder = audioEncoder

	// Start listening
	lis, err := net.Listen("tcp", cfg.Server.Port)
	if err != nil {
		log.Fatalf("❌ Failed to listen: %v", err)
	}

	fmt.Printf("\n🌐 Inference server listening on port %s\n", cfg.Server.Port)
	fmt.Println("   Protocol: gRPC with Protobuf")
	fmt.Println("   Features:")
	fmt.Println("      • Multi-GPU inference (8× GPUs)")
	fmt.Println("      • Multi-model support (1200+ models)")
	fmt.Println("      • Dynamic model loading")
	fmt.Println("      • Automatic eviction (LRU/LFU)")
	fmt.Println("      • Round-robin GPU assignment")
	fmt.Println("      • Raw float32 output (no compositing)")
	fmt.Println("\n✅ Ready to accept connections!")
	fmt.Println("================================================================================")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("❌ Failed to serve: %v", err)
	}
}

// InferBatch performs inference and returns raw float32 mouth regions
func (s *inferenceServer) InferBatch(ctx context.Context, req *pb.InferBatchRequest) (*pb.InferBatchResponse, error) {
	startTime := time.Now()

	// Validate request
	if req.ModelId == "" {
		return &pb.InferBatchResponse{
			Success: false,
			Error:   "model_id is required",
		}, nil
	}

	if req.BatchSize < 1 || req.BatchSize > 50 {
		return &pb.InferBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid batch size: %d (must be 1-50)", req.BatchSize),
		}, nil
	}

	// Validate input sizes
	expectedVisualSize := int(req.BatchSize) * visualFrameSize

	if len(req.VisualFrames) != expectedVisualSize*4 { // *4 for float32 bytes
		return &pb.InferBatchResponse{
			Success: false,
			Error: fmt.Sprintf("Invalid visual frames size: got %d bytes, expected %d bytes",
				len(req.VisualFrames), expectedVisualSize*4),
		}, nil
	}

	// Process audio: either raw audio or pre-computed features
	var audioData []float32
	var audioProcessingTime time.Duration

	if len(req.RawAudio) > 0 {
		// NEW: Process raw audio through mel-spectrogram -> audio encoder
		if s.audioProcessor == nil || s.audioEncoder == nil {
			return &pb.InferBatchResponse{
				Success: false,
				Error:   "Raw audio processing not available (audio encoder not initialized)",
			}, nil
		}

		audioStart := time.Now()

		// Convert bytes to float32 audio samples
		rawAudioSamples := bytesToFloat32(req.RawAudio)

		// 1. Convert to mel-spectrogram
		melSpec, err := s.audioProcessor.ProcessAudio(rawAudioSamples)
		if err != nil {
			return &pb.InferBatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Failed to process audio to mel-spectrogram: %v", err),
			}, nil
		}

		// 2. Extract mel windows for the batch
		melWindows, err := audio.ExtractMelWindowsForBatch(melSpec, int(req.BatchSize), 0)
		if err != nil {
			return &pb.InferBatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Failed to extract mel windows: %v", err),
			}, nil
		}

		// 3. Encode each window to audio features
		audioFeatures, err := s.audioEncoder.EncodeBatch(melWindows)
		if err != nil {
			return &pb.InferBatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Failed to encode audio features: %v", err),
			}, nil
		}

		// 4. Flatten audio features for inference [batchSize][512] -> [batchSize*512]
		audioData = make([]float32, int(req.BatchSize)*512)
		for i := 0; i < int(req.BatchSize); i++ {
			copy(audioData[i*512:], audioFeatures[i])
		}

		audioProcessingTime = time.Since(audioStart)

		if s.cfg.Logging.LogInferenceTimes {
			log.Printf("🎵 Audio processing: %d samples -> %d frames -> %d features (%.2fms)",
				len(rawAudioSamples), len(melSpec), len(audioFeatures), audioProcessingTime.Seconds()*1000)
		}

	} else if len(req.AudioFeatures) > 0 {
		// BACKWARD COMPAT: Use pre-computed audio features
		expectedAudioSize := audioFrameSize
		if len(req.AudioFeatures) != expectedAudioSize*4 {
			return &pb.InferBatchResponse{
				Success: false,
				Error: fmt.Sprintf("Invalid audio features size: got %d bytes, expected %d bytes",
					len(req.AudioFeatures), expectedAudioSize*4),
			}, nil
		}

		audioData = bytesToFloat32(req.AudioFeatures)

		// Replicate audio for batch
		audioForBatch := make([]float32, int(req.BatchSize)*audioFrameSize)
		for i := 0; i < int(req.BatchSize); i++ {
			copy(audioForBatch[i*audioFrameSize:], audioData)
		}
		audioData = audioForBatch

	} else {
		return &pb.InferBatchResponse{
			Success: false,
			Error:   "Either raw_audio or audio_features must be provided",
		}, nil
	}

	// Get or load model
	modelInstance, err := s.registry.GetOrLoadModel(req.ModelId)
	if err != nil {
		return &pb.InferBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to load model '%s': %v", req.ModelId, err),
		}, nil
	}

	// Convert visual frames bytes to float32
	visualData := bytesToFloat32(req.VisualFrames)

	// Run inference
	inferStart := time.Now()
	outputs, err := modelInstance.Inferencer.InferBatch(visualData, audioData, int(req.BatchSize))
	inferTime := time.Since(inferStart)

	if err != nil {
		return &pb.InferBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Inference failed: %v", err),
		}, nil
	}

	// Convert output float32 to bytes
	rawOutputs := make([]*pb.RawMouthRegion, req.BatchSize)
	for i := 0; i < int(req.BatchSize); i++ {
		frameOutput := outputs[i*outputFrameSize : (i+1)*outputFrameSize]
		rawOutputs[i] = &pb.RawMouthRegion{
			Data: float32ToBytes(frameOutput),
		}
	}

	// Record statistics
	s.registry.RecordInference(req.ModelId, inferTime.Seconds()*1000)

	totalTime := time.Since(startTime)

	if s.cfg.Logging.LogInferenceTimes {
		log.Printf("⚡ Inference: model=%s, batch=%d, gpu=%d, time=%.2fms, total=%.2fms",
			req.ModelId, req.BatchSize, modelInstance.GPUID,
			inferTime.Seconds()*1000, totalTime.Seconds()*1000)
	}

	return &pb.InferBatchResponse{
		Outputs:         rawOutputs,
		InferenceTimeMs: float32(inferTime.Seconds() * 1000),
		Success:         true,
		WorkerId:        0, // TODO: add worker ID when implementing worker pool
		GpuId:           int32(modelInstance.GPUID),
	}, nil
}

// ListModels returns all configured models
func (s *inferenceServer) ListModels(ctx context.Context, req *pb.ListModelsRequest) (*pb.ListModelsResponse, error) {
	instances := s.registry.ListModels()

	models := make([]*pb.ModelInfo, 0, len(s.cfg.Models))
	loadedMap := make(map[string]*registry.ModelInstance)

	for _, inst := range instances {
		loadedMap[inst.ID] = inst
	}

	// Include all configured models
	for modelID, modelCfg := range s.cfg.Models {
		info := &pb.ModelInfo{
			ModelId:   modelID,
			ModelPath: modelCfg.ModelPath,
		}

		if inst, loaded := loadedMap[modelID]; loaded {
			info.Loaded = true
			info.GpuId = int32(inst.GPUID)
			inst.Mu.RLock()
			info.Stats = &pb.ModelStats{
				UsageCount:           inst.UsageCount,
				LastUsedUnixMs:       inst.LastUsed.UnixMilli(),
				TotalInferenceTimeMs: inst.TotalInferenceMs,
				MemoryBytes:          inst.MemoryBytes,
				LoadedUnixMs:         inst.LoadedAt.UnixMilli(),
			}
			inst.Mu.RUnlock()
		}

		models = append(models, info)
	}

	return &pb.ListModelsResponse{
		Models: models,
	}, nil
}

// LoadModel explicitly loads a model
func (s *inferenceServer) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	if req.ModelId == "" {
		return &pb.LoadModelResponse{
			Success: false,
			Error:   "model_id is required",
		}, nil
	}

	startLoad := time.Now()

	if req.ForceReload {
		s.registry.UnloadModel(req.ModelId)
	}

	err := s.registry.LoadModel(req.ModelId)
	loadTime := time.Since(startLoad)

	if err != nil {
		return &pb.LoadModelResponse{
			Success:    false,
			Error:      fmt.Sprintf("Failed to load model: %v", err),
			LoadTimeMs: float32(loadTime.Seconds() * 1000),
		}, nil
	}

	// Get stats
	instances, _ := s.registry.GetStats(req.ModelId)
	var stats *pb.ModelStats
	if len(instances) > 0 {
		inst := instances[0]
		inst.Mu.RLock()
		stats = &pb.ModelStats{
			UsageCount:           inst.UsageCount,
			LastUsedUnixMs:       inst.LastUsed.UnixMilli(),
			TotalInferenceTimeMs: inst.TotalInferenceMs,
			MemoryBytes:          inst.MemoryBytes,
			LoadedUnixMs:         inst.LoadedAt.UnixMilli(),
		}
		inst.Mu.RUnlock()
	}

	return &pb.LoadModelResponse{
		Success:    true,
		LoadTimeMs: float32(loadTime.Seconds() * 1000),
		Stats:      stats,
	}, nil
}

// UnloadModel explicitly unloads a model
func (s *inferenceServer) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*pb.UnloadModelResponse, error) {
	if req.ModelId == "" {
		return &pb.UnloadModelResponse{
			Success: false,
			Error:   "model_id is required",
		}, nil
	}

	err := s.registry.UnloadModel(req.ModelId)
	if err != nil {
		return &pb.UnloadModelResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to unload model: %v", err),
		}, nil
	}

	return &pb.UnloadModelResponse{
		Success: true,
	}, nil
}

// GetModelStats returns statistics for models
func (s *inferenceServer) GetModelStats(ctx context.Context, req *pb.GetModelStatsRequest) (*pb.GetModelStatsResponse, error) {
	instances, err := s.registry.GetStats(req.ModelId)
	if err != nil {
		return &pb.GetModelStatsResponse{}, err
	}

	models := make([]*pb.ModelInfo, len(instances))
	for i, inst := range instances {
		inst.Mu.RLock()
		models[i] = &pb.ModelInfo{
			ModelId:   inst.ID,
			Loaded:    true,
			ModelPath: inst.Config.ModelPath,
			GpuId:     int32(inst.GPUID),
			Stats: &pb.ModelStats{
				UsageCount:           inst.UsageCount,
				LastUsedUnixMs:       inst.LastUsed.UnixMilli(),
				TotalInferenceTimeMs: inst.TotalInferenceMs,
				MemoryBytes:          inst.MemoryBytes,
				LoadedUnixMs:         inst.LoadedAt.UnixMilli(),
			},
		}
		inst.Mu.RUnlock()
	}

	// Get GPU info
	gpuInfo := s.registry.GetGPUInfo()
	gpus := make([]*pb.GPUInfo, len(gpuInfo))
	for i, gpu := range gpuInfo {
		gpus[i] = &pb.GPUInfo{
			GpuId:            int32(gpu.GPUID),
			Name:             fmt.Sprintf("NVIDIA RTX 6000 Blackwell %d", gpu.GPUID),
			TotalMemoryBytes: gpu.TotalMemory,
			UsedMemoryBytes:  gpu.UsedMemory,
			LoadedModels:     int32(gpu.LoadedModels),
		}
	}

	return &pb.GetModelStatsResponse{
		Models:           models,
		MaxModels:        int32(s.cfg.Capacity.MaxModels),
		LoadedModels:     int32(s.registry.GetLoadedCount()),
		TotalMemoryBytes: s.registry.GetTotalMemory(),
		MaxMemoryBytes:   int64(s.cfg.Capacity.MaxMemoryGB * 1024 * 1024 * 1024),
	}, nil
}

// Health returns server health status
func (s *inferenceServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	gpuInfo := s.registry.GetGPUInfo()
	gpus := make([]*pb.GPUInfo, len(gpuInfo))
	for i, gpu := range gpuInfo {
		gpus[i] = &pb.GPUInfo{
			GpuId:            int32(gpu.GPUID),
			Name:             fmt.Sprintf("NVIDIA RTX 6000 Blackwell %d", gpu.GPUID),
			TotalMemoryBytes: gpu.TotalMemory,
			UsedMemoryBytes:  gpu.UsedMemory,
			LoadedModels:     int32(gpu.LoadedModels),
		}
	}

	return &pb.HealthResponse{
		Healthy:       true,
		CudaAvailable: true,
		LoadedModels:  int32(s.registry.GetLoadedCount()),
		MaxModels:     int32(s.cfg.Capacity.MaxModels),
		GpuCount:      int32(s.cfg.GPUs.Count),
		Version:       "1.0.0",
		Gpus:          gpus,
	}, nil
}

// Helper functions

func bytesToFloat32(b []byte) []float32 {
	floats := make([]float32, len(b)/4)
	for i := 0; i < len(floats); i++ {
		floats[i] = float32frombytes(b[i*4 : (i+1)*4])
	}
	return floats
}

func float32ToBytes(floats []float32) []byte {
	bytes := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(bytes[i*4:], float32bits(f))
	}
	return bytes
}

func float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	return float32frombits(bits)
}

func float32bits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}

func float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}
