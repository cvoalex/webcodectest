package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"net"
	"time"

	"go-multitenant-server/config"
	pb "go-multitenant-server/proto"
	"go-multitenant-server/registry"
	"go-multitenant-server/workers"

	"google.golang.org/grpc"
)

const (
	visualFrameSize  = 6 * 320 * 320
	audioFrameSize   = 32 * 16 * 16
	outputFrameSize  = 3 * 320 * 320
	backgroundWidth  = 1280
	backgroundHeight = 720
)

type multiTenantServer struct {
	pb.UnimplementedMultiTenantLipSyncServer
	registry   *registry.ModelRegistry
	workerPool *workers.WorkerPool
	cfg        *config.Config
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🏢 Multi-Tenant LipSync gRPC Server")
	fmt.Println("================================================================================")

	// Load configuration
	cfgPath := "config.yaml"
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("❌ Failed to load config: %v", err)
	}

	log.Printf("✅ Configuration loaded from %s", cfgPath)
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

	// Create worker pool
	log.Println("🔧 Initializing worker pool...")
	workerPool, err := workers.NewWorkerPool(reg, cfg.Server.WorkerCount, cfg.Server.QueueSize)
	if err != nil {
		log.Fatalf("❌ Failed to create worker pool: %v", err)
	}
	defer workerPool.Shutdown()

	log.Printf("✅ Worker pool initialized (%d workers, queue size: %d)\n", cfg.Server.WorkerCount, cfg.Server.QueueSize)

	// Create gRPC server
	maxSize := cfg.Server.MaxMessageSizeMB * 1024 * 1024
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxSize),
		grpc.MaxSendMsgSize(maxSize),
	)

	server := &multiTenantServer{
		registry:   reg,
		workerPool: workerPool,
		cfg:        cfg,
	}

	pb.RegisterMultiTenantLipSyncServer(grpcServer, server)

	// Start listening
	lis, err := net.Listen("tcp", cfg.Server.Port)
	if err != nil {
		log.Fatalf("❌ Failed to listen: %v", err)
	}

	fmt.Printf("\n🌐 Server listening on port %s\n", cfg.Server.Port)
	fmt.Println("   Protocol: gRPC with Protobuf")
	fmt.Println("   Features:")
	fmt.Println("      • Multi-model support")
	fmt.Println("      • Dynamic model loading")
	fmt.Println("      • Automatic eviction (LRU/LFU)")
	fmt.Println("      • Usage statistics tracking")
	fmt.Println("      • Compositing with backgrounds")
	fmt.Println("      • Parallel worker pool inference")
	fmt.Println("      • Lazy-loading background cache")
	fmt.Println("\n✅ Ready to accept connections!")
	fmt.Println("================================================================================")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("❌ Failed to serve: %v", err)
	}
}

// InferBatchComposite handles inference + compositing for a specific model
func (s *multiTenantServer) InferBatchComposite(ctx context.Context, req *pb.CompositeBatchRequest) (*pb.CompositeBatchResponse, error) {
	totalStart := time.Now()

	// Validate request
	if req.ModelId == "" {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   "model_id is required",
		}, nil
	}

	if req.BatchSize < 1 || req.BatchSize > 25 {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid batch size: %d (must be 1-25)", req.BatchSize),
		}, nil
	}

	// Validate input sizes
	expectedVisualSize := int(req.BatchSize) * visualFrameSize
	expectedAudioSize := audioFrameSize

	if len(req.VisualFrames) != expectedVisualSize {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error: fmt.Sprintf("Invalid visual frames size: got %d, expected %d",
				len(req.VisualFrames), expectedVisualSize),
		}, nil
	}

	if len(req.AudioFeatures) != expectedAudioSize {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error: fmt.Sprintf("Invalid audio features size: got %d, expected %d",
				len(req.AudioFeatures), expectedAudioSize),
		}, nil
	}

	// Get or load model
	loadStart := time.Now()
	modelInstance, err := s.registry.GetOrLoadModel(req.ModelId)
	loadTime := time.Since(loadStart)
	modelLoaded := loadTime > 100*time.Millisecond // Consider "loaded" if took >100ms

	if err != nil {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to load model '%s': %v", req.ModelId, err),
		}, nil
	}

	if modelLoaded {
		log.Printf("🔄 Model '%s' loaded on-demand in %.2fs", req.ModelId, loadTime.Seconds())
	}

	// Replicate audio window for each frame in batch
	audioForBatch := make([]float32, int(req.BatchSize)*audioFrameSize)
	for i := 0; i < int(req.BatchSize); i++ {
		copy(audioForBatch[i*audioFrameSize:], req.AudioFeatures)
	}

	// Submit to worker pool for parallel inference
	inferStart := time.Now()
	resultChan := make(chan *workers.InferenceResult, 1)

	inferReq := &workers.InferenceRequest{
		ModelID:       req.ModelId,
		VisualFrames:  req.VisualFrames,
		AudioFeatures: audioForBatch,
		BatchSize:     int(req.BatchSize),
		StartFrameIdx: int(req.StartFrameIdx),
		ResultChan:    resultChan,
	}

	err = s.workerPool.Submit(inferReq)
	if err != nil {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to submit to worker pool: %v", err),
		}, nil
	}

	// Wait for result
	result := <-resultChan
	inferTime := time.Since(inferStart)

	if !result.Success {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Inference failed: %v", result.Error),
		}, nil
	}

	outputs := result.Outputs

	// Composite each frame
	compStart := time.Now()
	compositedPNGs := make([][]byte, req.BatchSize)
	startFrameIdx := int(req.StartFrameIdx)

	for i := 0; i < int(req.BatchSize); i++ {
		frameIdx := startFrameIdx + i
		frameOutput := outputs[i*outputFrameSize : (i+1)*outputFrameSize]

		// Convert output to image
		mouthRegion := outputToImage(frameOutput)

		// Get crop rectangle for this frame
		cropRect, exists := modelInstance.CropRects[frameIdx]
		if !exists {
			return &pb.CompositeBatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Crop rectangle not found for frame %d", frameIdx),
			}, nil
		}

		// Get background from cache (lazy-loaded)
		background, err := modelInstance.BackgroundCache.Get(frameIdx)
		if err != nil {
			return &pb.CompositeBatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Failed to load background for frame %d: %v", frameIdx, err),
			}, nil
		}

		x1, y1, x2, y2 := cropRect[0], cropRect[1], cropRect[2], cropRect[3]
		w := x2 - x1
		h := y2 - y1

		// Composite with background
		composited := compositeFrame(mouthRegion, background, x1, y1, w, h)

		// Encode to PNG
		var buf bytes.Buffer
		if err := png.Encode(&buf, composited); err != nil {
			return &pb.CompositeBatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Failed to encode frame %d: %v", i, err),
			}, nil
		}

		compositedPNGs[i] = buf.Bytes()
	}

	compTime := time.Since(compStart)
	totalTime := time.Since(totalStart)

	// Record statistics
	s.registry.RecordInference(req.ModelId, inferTime.Seconds()*1000)

	return &pb.CompositeBatchResponse{
		CompositedFrames: compositedPNGs,
		InferenceTimeMs:  inferTime.Seconds() * 1000,
		CompositeTimeMs:  compTime.Seconds() * 1000,
		TotalTimeMs:      totalTime.Seconds() * 1000,
		ModelLoaded:      modelLoaded,
		ModelLoadTimeMs:  loadTime.Seconds() * 1000,
		Success:          true,
	}, nil
}

// ListModels returns all configured models
func (s *multiTenantServer) ListModels(ctx context.Context, req *pb.ListModelsRequest) (*pb.ListModelsResponse, error) {
	instances := s.registry.ListModels()

	models := make([]*pb.ModelInfo, 0, len(s.cfg.Models))
	loadedMap := make(map[string]*registry.ModelInstance)

	for _, inst := range instances {
		loadedMap[inst.ID] = inst
	}

	// Include all configured models, not just loaded ones
	for modelID, modelCfg := range s.cfg.Models {
		info := &pb.ModelInfo{
			ModelId:       modelID,
			ModelPath:     modelCfg.ModelPath,
			BackgroundDir: modelCfg.BackgroundDir,
			CropRectsPath: modelCfg.CropRectsPath,
		}

		if inst, loaded := loadedMap[modelID]; loaded {
			info.Loaded = true
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
func (s *multiTenantServer) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	if req.ModelId == "" {
		return &pb.LoadModelResponse{
			Success: false,
			Error:   "model_id is required",
		}, nil
	}

	startLoad := time.Now()

	if req.ForceReload {
		// Unload first if already loaded
		s.registry.UnloadModel(req.ModelId) // Ignore error
	}

	err := s.registry.LoadModel(req.ModelId)
	loadTime := time.Since(startLoad)

	if err != nil {
		return &pb.LoadModelResponse{
			Success:    false,
			Error:      fmt.Sprintf("Failed to load model: %v", err),
			LoadTimeMs: loadTime.Seconds() * 1000,
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
		LoadTimeMs: loadTime.Seconds() * 1000,
		Stats:      stats,
	}, nil
}

// UnloadModel explicitly unloads a model
func (s *multiTenantServer) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*pb.UnloadModelResponse, error) {
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
func (s *multiTenantServer) GetModelStats(ctx context.Context, req *pb.GetModelStatsRequest) (*pb.GetModelStatsResponse, error) {
	instances, err := s.registry.GetStats(req.ModelId)
	if err != nil {
		return &pb.GetModelStatsResponse{}, err
	}

	models := make([]*pb.ModelInfo, len(instances))
	for i, inst := range instances {
		inst.Mu.RLock()
		models[i] = &pb.ModelInfo{
			ModelId:       inst.ID,
			Loaded:        true,
			ModelPath:     inst.Config.ModelPath,
			BackgroundDir: inst.Config.BackgroundDir,
			CropRectsPath: inst.Config.CropRectsPath,
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

	return &pb.GetModelStatsResponse{
		Models:           models,
		MaxModels:        int32(s.cfg.Capacity.MaxModels),
		LoadedModels:     int32(s.registry.GetLoadedCount()),
		TotalMemoryBytes: s.registry.GetTotalMemory(),
		MaxMemoryBytes:   int64(s.cfg.Capacity.MaxMemoryGB * 1024 * 1024 * 1024),
	}, nil
}

// Health returns server health status
func (s *multiTenantServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{
		Healthy:       true,
		CudaAvailable: true,
		LoadedModels:  int32(s.registry.GetLoadedCount()),
		MaxModels:     int32(s.cfg.Capacity.MaxModels),
		Version:       "1.0.0",
	}, nil
}

// Helper functions for image processing

func outputToImage(outputData []float32) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))

	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// BGR order from ONNX
			b := outputData[0*320*320+y*320+x]
			g := outputData[1*320*320+y*320+x]
			r := outputData[2*320*320+y*320+x]

			// Output is already in [0, 1] range
			rByte := uint8(clamp(r * 255.0))
			gByte := uint8(clamp(g * 255.0))
			bByte := uint8(clamp(b * 255.0))

			img.SetRGBA(x, y, color.RGBA{R: rByte, G: gByte, B: bByte, A: 255})
		}
	}

	return img
}

func clamp(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 255 {
		return 255
	}
	return val
}

func resizeImage(src *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	dst := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))

	xRatio := float32(srcWidth) / float32(targetWidth)
	yRatio := float32(srcHeight) / float32(targetHeight)

	for dstY := 0; dstY < targetHeight; dstY++ {
		for dstX := 0; dstX < targetWidth; dstX++ {
			srcX := float32(dstX) * xRatio
			srcY := float32(dstY) * yRatio

			x0 := int(srcX)
			y0 := int(srcY)
			x1 := x0 + 1
			y1 := y0 + 1

			if x1 >= srcWidth {
				x1 = srcWidth - 1
			}
			if y1 >= srcHeight {
				y1 = srcHeight - 1
			}

			xWeight := srcX - float32(x0)
			yWeight := srcY - float32(y0)

			c00 := src.RGBAAt(x0, y0)
			c10 := src.RGBAAt(x1, y0)
			c01 := src.RGBAAt(x0, y1)
			c11 := src.RGBAAt(x1, y1)

			r := bilinearInterp(c00.R, c10.R, c01.R, c11.R, xWeight, yWeight)
			g := bilinearInterp(c00.G, c10.G, c01.G, c11.G, xWeight, yWeight)
			b := bilinearInterp(c00.B, c10.B, c01.B, c11.B, xWeight, yWeight)

			dst.SetRGBA(dstX, dstY, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return dst
}

func bilinearInterp(c00, c10, c01, c11 uint8, xWeight, yWeight float32) uint8 {
	top := float32(c00)*(1-xWeight) + float32(c10)*xWeight
	bottom := float32(c01)*(1-xWeight) + float32(c11)*xWeight
	result := top*(1-yWeight) + bottom*yWeight
	return uint8(result)
}

func compositeFrame(mouthRegion *image.RGBA, background *image.RGBA, x, y, w, h int) *image.RGBA {
	// Resize mouth region
	resized := resizeImage(mouthRegion, w, h)

	// Clone background
	result := image.NewRGBA(background.Bounds())
	draw.Draw(result, result.Bounds(), background, image.Point{}, draw.Src)

	// Paste mouth region
	dstRect := image.Rect(x, y, x+w, y+h)
	draw.Draw(result, dstRect, resized, image.Point{}, draw.Src)

	return result
}
