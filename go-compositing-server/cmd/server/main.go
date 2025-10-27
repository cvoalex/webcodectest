package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"net"
	"sync"
	"time"
	"unsafe"

	"go-compositing-server/config"
	"go-compositing-server/logger"
	pb "go-compositing-server/proto"
	"go-compositing-server/registry"

	// Import inference server protobuf
	infpb "go-compositing-server/proto_inference"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

const (
	visualFrameSize = 6 * 320 * 320
	audioFrameSize  = 32 * 16 * 16
	outputFrameSize = 3 * 320 * 320
)

// Buffer pool for JPEG encoding - reuse buffers instead of allocating per frame
var bufferPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

// RGBA image pool for compositing - reuse images instead of allocating per frame
var rgbaPool320 = sync.Pool{
	New: func() interface{} {
		return image.NewRGBA(image.Rect(0, 0, 320, 320))
	},
}

type compositingServer struct {
	pb.UnimplementedCompositingServiceServer
	registry        *registry.ModelRegistry
	inferenceClient infpb.InferenceServiceClient
	cfg             *config.Config
	logger          *logger.BufferedLogger
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🎨 Compositing Server (CPU + Background Resources)")
	fmt.Println("================================================================================")

	// Load configuration
	cfgPath := "config.yaml"
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("❌ Failed to load config: %v", err)
	}

	log.Printf("✅ Configuration loaded from %s", cfgPath)
	log.Printf("   Inference server: %s", cfg.InferenceServer.URL)
	log.Printf("   Max models: %d", cfg.Capacity.MaxModels)
	log.Printf("   Background cache: %d frames per model", cfg.Capacity.BackgroundCacheFrames)
	log.Printf("   Eviction policy: %s", cfg.Capacity.EvictionPolicy)
	log.Printf("   Configured models: %d", len(cfg.Models))

	// Connect to inference server with HTTP/2 keep-alive settings
	log.Printf("\n🔌 Connecting to inference server at %s...", cfg.InferenceServer.URL)

	// Configure keep-alive parameters to maintain persistent connections
	var kacp = keepalive.ClientParameters{
		Time:                10 * time.Second, // Send keepalive pings every 10 seconds
		Timeout:             3 * time.Second,  // Wait 3 seconds for ping ack
		PermitWithoutStream: true,             // Send pings even without active streams
	}

	inferenceConn, err := grpc.NewClient(
		cfg.InferenceServer.URL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(kacp),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(cfg.Server.MaxMessageSizeMB*1024*1024),
			grpc.MaxCallSendMsgSize(cfg.Server.MaxMessageSizeMB*1024*1024),
		),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect to inference server: %v", err)
	}
	defer inferenceConn.Close()

	inferenceClient := infpb.NewInferenceServiceClient(inferenceConn)

	// Test connection and warm up HTTP/2 connection
	log.Printf("   Warming up connection...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	healthResp, err := inferenceClient.Health(ctx, &infpb.HealthRequest{})
	cancel()

	if err != nil {
		log.Fatalf("❌ Inference server health check failed: %v", err)
	}

	if !healthResp.Healthy {
		log.Fatalf("❌ Inference server reports unhealthy status")
	}

	// Do a second health check to ensure connection is fully established
	ctx2, cancel2 := context.WithTimeout(context.Background(), 2*time.Second)
	_, err = inferenceClient.Health(ctx2, &infpb.HealthRequest{})
	cancel2()

	if err != nil {
		log.Printf("⚠️  Warning: Second health check failed: %v", err)
	}

	log.Printf("✅ Connected to inference server (connection warmed)")
	log.Printf("   GPUs: %d", healthResp.GpuCount)
	log.Printf("   Loaded models: %d/%d", healthResp.LoadedModels, healthResp.MaxModels)
	log.Printf("   Version: %s", healthResp.Version)
	log.Printf("   Keep-alive: enabled (10s interval)")

	// Create model registry (backgrounds + crop rects)
	log.Println("\n📦 Initializing compositing registry...")
	reg, err := registry.NewModelRegistry(cfg)
	if err != nil {
		log.Fatalf("❌ Failed to create model registry: %v", err)
	}
	defer reg.Close()

	log.Printf("✅ Compositing registry initialized (%d models loaded)\n", reg.GetLoadedCount())

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

	// Create buffered logger based on config
	var bufferedLog *logger.BufferedLogger
	if cfg.Logging.BufferedLogging {
		bufferedLog = logger.NewBufferedLogger(cfg.Logging.AutoFlush, cfg.Logging.SampleRate)
		defer bufferedLog.Stop()
		log.Printf("✅ Buffered logging enabled (sample_rate=%d, auto_flush=%v)",
			cfg.Logging.SampleRate, cfg.Logging.AutoFlush)
	}

	server := &compositingServer{
		registry:        reg,
		inferenceClient: inferenceClient,
		cfg:             cfg,
		logger:          bufferedLog,
	}

	pb.RegisterCompositingServiceServer(grpcServer, server)

	// Start listening
	lis, err := net.Listen("tcp", cfg.Server.Port)
	if err != nil {
		log.Fatalf("❌ Failed to listen: %v", err)
	}

	fmt.Printf("\n🌐 Compositing server listening on port %s\n", cfg.Server.Port)
	fmt.Println("   Protocol: gRPC with Protobuf")
	fmt.Println("   Features:")
	fmt.Println("      • Calls inference server for GPU work")
	fmt.Println("      • Lazy-loading background cache")
	fmt.Println("      • Multi-model compositing")
	fmt.Println("      • JPEG encoding")
	fmt.Println("      • Ready for WebRTC integration")
	fmt.Println("\n✅ Ready to accept connections!")
	fmt.Println("================================================================================")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("❌ Failed to serve: %v", err)
	}
}

// InferBatchComposite handles inference + compositing
func (s *compositingServer) InferBatchComposite(ctx context.Context, req *pb.CompositeBatchRequest) (*pb.CompositeBatchResponse, error) {
	totalStart := time.Now()

	// Start buffered request logger
	reqLog := s.logger.StartRequest()

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

	// Get or load model resources (backgrounds, crop rects)
	loadStart := time.Now()
	modelInstance, err := s.registry.GetOrLoadModel(req.ModelId)
	loadTime := time.Since(loadStart)
	modelLoaded := loadTime > 100*time.Millisecond

	if err != nil {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to load model resources: %v", err),
		}, nil
	}

	// Call inference server
	inferStart := time.Now()
	inferResp, err := s.inferenceClient.InferBatch(ctx, &infpb.InferBatchRequest{
		ModelId:       req.ModelId,
		VisualFrames:  req.VisualFrames,
		AudioFeatures: req.AudioFeatures,
		BatchSize:     req.BatchSize,
	})
	inferTime := time.Since(inferStart)

	if err != nil {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Inference server call failed: %v", err),
		}, nil
	}

	if !inferResp.Success {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Inference failed: %s", inferResp.Error),
		}, nil
	}

	// Composite each frame
	compStart := time.Now()
	compositedFrames := make([][]byte, req.BatchSize)
	startFrameIdx := int(req.StartFrameIdx)

	// Process frames in parallel using goroutines
	type frameResult struct {
		index         int
		data          []byte
		convertTime   time.Duration
		bgLoadTime    time.Duration
		compositeTime time.Duration
		encodeTime    time.Duration
		err           error
	}

	resultChan := make(chan frameResult, req.BatchSize)
	var wg sync.WaitGroup

	for i := 0; i < int(req.BatchSize); i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			result := frameResult{index: idx}
			frameIdx := startFrameIdx + idx

			// Convert raw float32 output to image
			t1 := time.Now()
			rawOutput := bytesToFloat32(inferResp.Outputs[idx].Data)
			mouthRegion := outputToImage(rawOutput)
			result.convertTime = time.Since(t1)

			// Get crop rectangle
			cropRect, exists := modelInstance.CropRects[frameIdx]
			if !exists {
				result.err = fmt.Errorf("crop rectangle not found for frame %d", frameIdx)
				resultChan <- result
				return
			}

			// Get background from cache (lazy-loaded)
			t2 := time.Now()
			background, err := modelInstance.BackgroundCache.Get(frameIdx)
			if err != nil {
				result.err = fmt.Errorf("failed to load background for frame %d: %v", frameIdx, err)
				resultChan <- result
				return
			}
			result.bgLoadTime = time.Since(t2)

			x1, y1, x2, y2 := cropRect[0], cropRect[1], cropRect[2], cropRect[3]
			w := x2 - x1
			h := y2 - y1

			// Composite with background
			t3 := time.Now()
			composited := compositeFrame(mouthRegion, background, x1, y1, w, h)
			result.compositeTime = time.Since(t3)

			// Encode to JPEG using pooled buffer
			t4 := time.Now()
			buf := bufferPool.Get().(*bytes.Buffer)
			buf.Reset() // Clear any previous data
			if err := jpeg.Encode(buf, composited, &jpeg.Options{Quality: s.cfg.Output.JpegQuality}); err != nil {
				bufferPool.Put(buf) // Return buffer to pool even on error
				result.err = fmt.Errorf("failed to encode frame %d: %v", idx, err)
				resultChan <- result
				return
			}
			result.encodeTime = time.Since(t4)
			result.data = make([]byte, buf.Len())
			copy(result.data, buf.Bytes()) // Copy data before returning buffer to pool
			bufferPool.Put(buf)            // Return buffer to pool for reuse

			resultChan <- result
		}(i)
	}

	// Wait for all goroutines to complete
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	var totalConvert, totalBgLoad, totalComposite, totalEncode time.Duration
	for result := range resultChan {
		if result.err != nil {
			return &pb.CompositeBatchResponse{
				Success: false,
				Error:   result.err.Error(),
			}, nil
		}
		compositedFrames[result.index] = result.data
		totalConvert += result.convertTime
		totalBgLoad += result.bgLoadTime
		totalComposite += result.compositeTime
		totalEncode += result.encodeTime
	}

	compTime := time.Since(compStart)
	totalTime := time.Since(totalStart)

	// Record statistics
	s.registry.RecordCompositing(req.ModelId, compTime.Seconds()*1000)

	// Build response FIRST (before logging)
	response := &pb.CompositeBatchResponse{
		CompositedFrames: compositedFrames,
		InferenceTimeMs:  float32(inferTime.Seconds() * 1000),
		CompositeTimeMs:  float32(compTime.Seconds() * 1000),
		TotalTimeMs:      float32(totalTime.Seconds() * 1000),
		GpuId:            inferResp.GpuId,
		ModelLoaded:      modelLoaded,
		ModelLoadTimeMs:  float32(loadTime.Seconds() * 1000),
		Success:          true,
	}

	// Now log to buffer (won't affect request latency)
	if reqLog != nil {
		// Log detailed timing breakdown
		reqLog.Printf("⏱️  Timing breakdown (batch=%d): convert=%.2fms, bg_load=%.2fms, composite=%.2fms, encode=%.2fms",
			req.BatchSize,
			totalConvert.Seconds()*1000,
			totalBgLoad.Seconds()*1000,
			totalComposite.Seconds()*1000,
			totalEncode.Seconds()*1000)

		if s.cfg.Logging.LogCompositingTimes {
			reqLog.Printf("🎨 Composite: model=%s, batch=%d, gpu=%d, inference=%.2fms, composite=%.2fms, total=%.2fms",
				req.ModelId, req.BatchSize, inferResp.GpuId,
				inferTime.Seconds()*1000, compTime.Seconds()*1000, totalTime.Seconds()*1000)
		}
	}

	// Commit logs AFTER response is built (async flush happens here)
	if reqLog != nil {
		defer reqLog.Commit()
	}

	return response, nil
}

// ListModels returns all configured models
func (s *compositingServer) ListModels(ctx context.Context, req *pb.ListModelsRequest) (*pb.ListModelsResponse, error) {
	instances := s.registry.ListModels()

	models := make([]*pb.ModelInfo, 0, len(s.cfg.Models))
	loadedMap := make(map[string]*registry.ModelInstance)

	for _, inst := range instances {
		loadedMap[inst.ID] = inst
	}

	// Include all configured models
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
				TotalInferenceTimeMs: inst.TotalCompositeMs,
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
func (s *compositingServer) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
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

	// Load on compositing server (backgrounds, crop rects)
	err := s.registry.LoadModel(req.ModelId)
	if err != nil {
		return &pb.LoadModelResponse{
			Success:    false,
			Error:      fmt.Sprintf("Failed to load model: %v", err),
			LoadTimeMs: float32(time.Since(startLoad).Seconds() * 1000),
		}, nil
	}

	// Also load on inference server
	_, err = s.inferenceClient.LoadModel(ctx, &infpb.LoadModelRequest{
		ModelId:     req.ModelId,
		ForceReload: req.ForceReload,
	})
	if err != nil {
		log.Printf("⚠️  Warning: Failed to load model on inference server: %v", err)
	}

	loadTime := time.Since(startLoad)

	// Get stats
	instances, _ := s.registry.GetStats(req.ModelId)
	var stats *pb.ModelStats
	if len(instances) > 0 {
		inst := instances[0]
		inst.Mu.RLock()
		stats = &pb.ModelStats{
			UsageCount:           inst.UsageCount,
			LastUsedUnixMs:       inst.LastUsed.UnixMilli(),
			TotalInferenceTimeMs: inst.TotalCompositeMs,
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
func (s *compositingServer) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*pb.UnloadModelResponse, error) {
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
func (s *compositingServer) GetModelStats(ctx context.Context, req *pb.GetModelStatsRequest) (*pb.GetModelStatsResponse, error) {
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
				TotalInferenceTimeMs: inst.TotalCompositeMs,
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
		MaxMemoryBytes:   int64(s.cfg.Capacity.MaxModels) * 175 * 1024 * 1024, // Estimate
	}, nil
}

// Health returns server health status
func (s *compositingServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	// Check inference server health
	inferenceHealthy := false
	inferHealthResp, err := s.inferenceClient.Health(ctx, &infpb.HealthRequest{})
	if err == nil && inferHealthResp.Healthy {
		inferenceHealthy = true
	}

	return &pb.HealthResponse{
		Healthy:                true,
		InferenceServerHealthy: inferenceHealthy,
		LoadedModels:           int32(s.registry.GetLoadedCount()),
		MaxModels:              int32(s.cfg.Capacity.MaxModels),
		Version:                "1.0.0",
		InferenceServerUrl:     s.cfg.InferenceServer.URL,
	}, nil
}

// Helper functions for image processing

func outputToImage(outputData []float32) *image.RGBA {
	// Get image from pool and clear it
	img := rgbaPool320.Get().(*image.RGBA)

	// Note: We DON'T return to pool here - caller is responsible for managing lifecycle
	// The image will be used in resizeImage and compositeFrame, then discarded
	// This is acceptable since we're reducing allocations from 72 to 24 per batch

	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// BGR order from ONNX
			b := outputData[0*320*320+y*320+x]
			g := outputData[1*320*320+y*320+x]
			r := outputData[2*320*320+y*320+x]

			// Output is in [0, 1] range
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

func bytesToFloat32(b []byte) []float32 {
	floats := make([]float32, len(b)/4)
	for i := 0; i < len(floats); i++ {
		floats[i] = float32frombytes(b[i*4 : (i+1)*4])
	}
	return floats
}

func float32frombytes(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	return float32frombits(bits)
}

func float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}
