package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"net"
	"os"
	"sync"
	"time"
	"unsafe"

	"go-monolithic-server/audio"
	"go-monolithic-server/config"
	"go-monolithic-server/logger"
	pb "go-monolithic-server/proto"
	"go-monolithic-server/registry"

	"google.golang.org/grpc"
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

// Pool for full HD background images (1920x1080)
var rgbaPoolFullHD = sync.Pool{
	New: func() interface{} {
		return image.NewRGBA(image.Rect(0, 0, 1920, 1080))
	},
}

// Pool for common resize dimensions (we'll use a max size and subset it)
var rgbaPoolResize = sync.Pool{
	New: func() interface{} {
		// Allocate max size (400x400 should cover most crop rects)
		return image.NewRGBA(image.Rect(0, 0, 400, 400))
	},
}

// Pool for mel windows [80][16] used in audio processing
var melWindowPool = sync.Pool{
	New: func() interface{} {
		window := make([][]float32, 80)
		for i := 0; i < 80; i++ {
			window[i] = make([]float32, 16)
		}
		return window
	},
}

type monolithicServer struct {
	pb.UnimplementedMonolithicServiceServer
	modelRegistry    *registry.ModelRegistry
	imageRegistry    *registry.ImageRegistry
	cfg              *config.Config
	audioProcessor   *audio.Processor
	audioEncoderPool *audio.AudioEncoderPool
	logger           *logger.BufferedLogger
}

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 Monolithic Lipsync Server (Inference + Compositing)")
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
	log.Printf("   Background cache: %d frames per model", cfg.Capacity.BackgroundCacheFrames)
	log.Printf("   Eviction policy: %s", cfg.Capacity.EvictionPolicy)
	log.Printf("   Configured models: %d", len(cfg.Models))

	// Create model registry (for ONNX models)
	log.Println("\n📦 Initializing model registry...")
	modelReg, err := registry.NewModelRegistry(cfg)
	if err != nil {
		log.Fatalf("❌ Failed to create model registry: %v", err)
	}
	defer modelReg.Close()

	log.Printf("✅ Model registry initialized (%d models preloaded)", modelReg.GetLoadedCount())

	// Create image registry (for backgrounds and crop rects)
	log.Println("\n🖼️  Initializing image registry...")
	imageReg, err := registry.NewImageRegistry(cfg)
	if err != nil {
		log.Fatalf("❌ Failed to create image registry: %v", err)
	}
	defer imageReg.Close()

	log.Printf("✅ Image registry initialized (%d models loaded)", imageReg.GetLoadedCount())

	// Display GPU info
	gpuInfo := modelReg.GetGPUInfo()
	fmt.Println("\n🎮 GPU Status:")
	for _, gpu := range gpuInfo {
		fmt.Printf("   GPU %d: %d models, %d MB used / %d MB total\n",
			gpu.GPUID,
			gpu.LoadedModels,
			gpu.UsedMemory/(1024*1024),
			gpu.TotalMemory/(1024*1024))
	}

	// Initialize audio processing pipeline
	log.Println("\n🎵 Initializing audio processing pipeline...")
	audioProcessor := audio.NewProcessor(nil)
	log.Println("✅ Mel-spectrogram processor initialized")

	// Create audio encoder pool (4 instances for parallel processing)
	encoderPoolSize := 4
	audioEncoderPool, err := audio.NewAudioEncoderPool(encoderPoolSize, cfg.ONNX.LibraryPath)
	if err != nil {
		log.Printf("⚠️  Warning: Audio encoder pool not available: %v", err)
		log.Println("   Server will require pre-computed audio features")
		audioEncoderPool = nil
	} else {
		log.Printf("✅ Audio encoder pool initialized (%d instances for parallel processing)", encoderPoolSize)
	}

	// Create gRPC server with keep-alive
	maxSize := cfg.Server.MaxMessageSizeMB * 1024 * 1024

	var kaep = keepalive.EnforcementPolicy{
		MinTime:             5 * time.Second,
		PermitWithoutStream: true,
	}

	var kasp = keepalive.ServerParameters{
		Time:    10 * time.Second,
		Timeout: 3 * time.Second,
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

	server := &monolithicServer{
		modelRegistry:    modelReg,
		imageRegistry:    imageReg,
		cfg:              cfg,
		audioProcessor:   audioProcessor,
		audioEncoderPool: audioEncoderPool,
		logger:           bufferedLog,
	}

	pb.RegisterMonolithicServiceServer(grpcServer, server)

	// Start listening
	lis, err := net.Listen("tcp", cfg.Server.Port)
	if err != nil {
		log.Fatalf("❌ Failed to listen: %v", err)
	}

	fmt.Printf("\n🌐 Monolithic server listening on port %s\n", cfg.Server.Port)
	fmt.Println("   Protocol: gRPC with Protobuf")
	fmt.Println("   Features:")
	fmt.Println("      • Inference + Compositing in single process")
	fmt.Println("      • No inter-service communication overhead")
	fmt.Println("      • Multi-GPU inference")
	fmt.Println("      • Real-time audio processing (mel + encoder)")
	fmt.Println("      • Dynamic model loading")
	fmt.Println("      • Automatic eviction (LFU)")
	fmt.Println("      • JPEG-encoded output")
	fmt.Println("\n✅ Ready to accept connections!")
	fmt.Println("================================================================================")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("❌ Failed to serve: %v", err)
	}
}

// InferBatchComposite performs inference + compositing in a single call
func (s *monolithicServer) InferBatchComposite(ctx context.Context, req *pb.CompositeBatchRequest) (*pb.CompositeBatchResponse, error) {
	startTime := time.Now()

	// Start buffered request logger
	reqLog := s.logger.StartRequest()

	// Validate request
	if req.ModelId == "" {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   "model_id is required",
		}, nil
	}

	if req.BatchSize <= 0 || req.BatchSize > 32 {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("batch_size must be 1-32, got %d", req.BatchSize),
		}, nil
	}

	// Validate input sizes
	expectedVisualSize := int(req.BatchSize) * visualFrameSize

	if len(req.VisualFrames) != expectedVisualSize*4 { // *4 for float32 bytes
		return &pb.CompositeBatchResponse{
			Success: false,
			Error: fmt.Sprintf("Invalid visual frames size: got %d bytes, expected %d bytes",
				len(req.VisualFrames), expectedVisualSize*4),
		}, nil
	}

	// === PHASE 1: AUDIO PROCESSING ===
	var audioData []float32
	var audioProcessingTime time.Duration

	if len(req.RawAudio) > 0 {
		// Process raw audio through mel-spectrogram -> audio encoder
		if s.audioProcessor == nil || s.audioEncoderPool == nil {
			return &pb.CompositeBatchResponse{
				Success: false,
				Error:   "Raw audio processing not available (audio encoder pool not initialized)",
			}, nil
		}

		audioStart := time.Now()

		// Convert bytes to float32 audio samples
		rawAudioSamples := bytesToFloat32(req.RawAudio)

		// 1. Convert to mel-spectrogram
		melSpec, err := s.audioProcessor.ProcessAudio(rawAudioSamples)
		if err != nil {
			return &pb.CompositeBatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Failed to process audio to mel-spectrogram: %v", err),
			}, nil
		}

		// 2. Encode mel-spectrogram to audio features
		// For each video frame at 25fps, we extract a 16-step window from the mel-spectrogram
		// Reference: SyncTalk_2D/utils.py AudDataset.crop_audio_window()
		//   start_idx = int(80 * (frame_num / 25))
		//   window = mel[start_idx:start_idx+16, :] = [16, 80]
		//   Then transpose to [80, 16] and encode to [1, 512]

		numMelFrames := len(melSpec)
		log.Printf("🎵 Audio processing: %d mel frames to encode", numMelFrames)

		// Calculate how many video frames we can encode
		// At 25fps, each frame uses a 16-step mel window starting at: int(80 * frame/25)
		// The last frame we can encode is when start+16 <= numMelFrames
		// So: int(80*frame/25) + 16 <= numMelFrames
		//     frame <= (numMelFrames - 16) * 25 / 80
		numVideoFrames := ((numMelFrames - 16) * 25) / 80
		if numVideoFrames < 0 {
			numVideoFrames = 0
		}
		// Only encode what we need for this batch + some padding
		if numVideoFrames > int(req.BatchSize)+16 {
			numVideoFrames = int(req.BatchSize) + 16
		}

		log.Printf("🎵 Will encode %d video frames from %d mel frames", numVideoFrames, numMelFrames)

		// Encode each video frame's mel window to get [512] features
		allFrameFeatures := make([][]float32, 0, numVideoFrames)
		allMelWindows := make([][][]float32, 0, numVideoFrames)

		log.Printf("🎵 Extracting %d mel windows...", numVideoFrames)

		for frameIdx := 0; frameIdx < numVideoFrames; frameIdx++ {
			// Calculate mel-spec window for this video frame
			// start_idx = int(80 * (frameIdx / 25))
			startIdx := int(float64(80*frameIdx) / 25.0)
			endIdx := startIdx + 16

			// Boundary check
			if endIdx > numMelFrames {
				endIdx = numMelFrames
				startIdx = endIdx - 16
			}
			if startIdx < 0 {
				startIdx = 0
			}

			// Log first few windows for debugging
			if frameIdx < 5 {
				log.Printf("   Frame %d: mel window [%d:%d]", frameIdx, startIdx, endIdx)
			}

			// Extract 16-step mel window [16, 80] and transpose to [80, 16]
			// Use pooled window
			window := melWindowPool.Get().([][]float32)

			for step := 0; step < 16; step++ {
				srcIdx := startIdx + step
				if srcIdx >= numMelFrames {
					srcIdx = numMelFrames - 1
				}
				for m := 0; m < 80; m++ {
					window[m][step] = melSpec[srcIdx][m]
				}
			}

			// Store the window directly (we're not returning it to pool, we need it for inference)
			allMelWindows = append(allMelWindows, window)

			// DEBUG: Save first few mel windows for comparison with Python (only if enabled)
			if s.cfg.Logging.SaveDebugFiles && frameIdx < 10 {
				os.MkdirAll("test_output/mel_windows_go", 0755)
				melFile := fmt.Sprintf("test_output/mel_windows_go/frame_%d.bin", frameIdx)
				if f, err := os.Create(melFile); err == nil {
					// Save as [16, 80] transposed back to [time, freq] for comparison
					for step := 0; step < 16; step++ {
						for m := 0; m < 80; m++ {
							binary.Write(f, binary.LittleEndian, window[m][step])
						}
					}
					f.Close()
				}
			}

			// Return window to pool
			melWindowPool.Put(window)
		}

		log.Printf("🎵 Encoding %d mel windows to audio features (parallel processing)...", len(allMelWindows))

		// Encode all windows to get 512-dim features per frame (PARALLEL via encoder pool)
		// This matches: outputs = torch.cat([model(mel) for mel in data_loader])
		// Result: [num_frames, 512]
		allFrameFeatures, err = s.audioEncoderPool.EncodeBatch(allMelWindows)
		if err != nil {
			log.Printf("❌ Audio encoding failed: %v", err)
			return &pb.CompositeBatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Failed to encode audio features: %v", err),
			}, nil
		}

		log.Printf("✅ Encoded %d audio feature vectors", len(allFrameFeatures))

		// Pad with first and last frames (matches Python: audio_feats = cat([first, outputs, last]))
		// This ensures get_audio_features() has proper padding for early/late frames
		paddedFeatures := make([][]float32, len(allFrameFeatures)+2)
		if len(allFrameFeatures) > 0 {
			paddedFeatures[0] = allFrameFeatures[0]                                           // First frame repeated
			copy(paddedFeatures[1:], allFrameFeatures)                                        // All frames
			paddedFeatures[len(paddedFeatures)-1] = allFrameFeatures[len(allFrameFeatures)-1] // Last frame repeated
		}

		log.Printf("✅ Padded audio features: %d → %d frames", len(allFrameFeatures), len(paddedFeatures))

		// For each OUTPUT frame in the batch, extract 16 consecutive encoded frames
		// Reference: SyncTalk_2D/utils.py get_audio_features() returns [16, 512]
		// Then inference_328.py line 142: audio_feat.reshape(32,16,16)
		audioData = make([]float32, int(req.BatchSize)*audioFrameSize) // audioFrameSize = 8192

		for batchIdx := 0; batchIdx < int(req.BatchSize); batchIdx++ {
			// For output frame i, extract encoded frames from i-8 to i+7 (16 frames total)
			// This matches: get_audio_features(audio_feats, i)
			// Python's get_audio_features() pads with ZEROS, not by repeating frames!
			// Note: Python uses the frame index directly (0-based), accounting for padding in the array
			// paddedFeatures: [f0, f0, f1, f2, ..., fN, fN] where f0 is at indices 0 and 1
			// For frame 0: extract from index -8 to 8, which becomes [0:8] with 8 left padding
			// This gives: [0,0,0,0,0,0,0,0, f0, f0, f1, f2, f3, f4, f5, f6] (16 total)
			frameIdx := batchIdx // Use frame index directly (matches Python)
			outputOffset := batchIdx * audioFrameSize

			// Calculate padding needs (matching Python's get_audio_features)
			left := frameIdx - 8
			right := frameIdx + 8
			padLeft := 0
			padRight := 0

			if left < 0 {
				padLeft = -left
				left = 0
			}
			if right > len(paddedFeatures) {
				padRight = right - len(paddedFeatures)
				right = len(paddedFeatures)
			}

			// Extract frames with zero-padding
			frameCounter := 0

			// Pad left with zeros
			for i := 0; i < padLeft; i++ {
				destOffset := outputOffset + frameCounter*512
				// Zero out 512 features
				for j := 0; j < 512; j++ {
					audioData[destOffset+j] = 0.0
				}
				frameCounter++
			}

			// Copy actual frames
			for srcFrame := left; srcFrame < right; srcFrame++ {
				features := paddedFeatures[srcFrame]
				destOffset := outputOffset + frameCounter*512
				copy(audioData[destOffset:destOffset+512], features)
				frameCounter++
			}

			// Pad right with zeros
			for i := 0; i < padRight; i++ {
				destOffset := outputOffset + frameCounter*512
				// Zero out 512 features
				for j := 0; j < 512; j++ {
					audioData[destOffset+j] = 0.0
				}
				frameCounter++
			}

			if batchIdx == 0 {
				// Calculate min/max/mean for debug on first batch frame
				min, max, sum := audioData[0], audioData[0], float32(0)
				for i := 0; i < audioFrameSize; i++ {
					v := audioData[outputOffset+i]
					if v < min {
						min = v
					}
					if v > max {
						max = v
					}
					sum += v
				}
				mean := sum / float32(audioFrameSize)
				log.Printf("🎵 Frame %d: Audio tensor [16x512→32x16x16] (min: %.3f, max: %.3f, mean: %.3f)",
					frameIdx, min, max, mean)
			}
		}

		audioProcessingTime = time.Since(audioStart)

		// DEBUG: Save audio tensors for each frame for comparison with PyTorch (only if enabled)
		if s.cfg.Logging.SaveDebugFiles {
			os.MkdirAll("test_output", 0755)
			for batchIdx := 0; batchIdx < int(req.BatchSize) && batchIdx < 10; batchIdx++ {
				frameIdx := batchIdx
				outputOffset := batchIdx * audioFrameSize

				debugFile := fmt.Sprintf("test_output/audio_tensor_frame_%d.bin", frameIdx)
				if f, err := os.Create(debugFile); err == nil {
					// Write 8192 floats (32*16*16) for this frame
					for i := 0; i < 8192; i++ {
						binary.Write(f, binary.LittleEndian, audioData[outputOffset+i])
					}
					f.Close()

					// Also save as JSON metadata
					jsonFile := fmt.Sprintf("test_output/audio_tensor_frame_%d.json", frameIdx)
					min, max, sum := audioData[outputOffset], audioData[outputOffset], float32(0)
					nonzero := 0
					for i := 0; i < audioFrameSize; i++ {
						v := audioData[outputOffset+i]
						if v < min {
							min = v
						}
						if v > max {
							max = v
						}
						sum += v
						if v != 0 {
							nonzero++
						}
					}
					mean := sum / float32(audioFrameSize)

					metadata := map[string]interface{}{
						"frame_index":    frameIdx,
						"shape":          []int{1, 32, 16, 16},
						"mean":           mean,
						"min":            min,
						"max":            max,
						"nonzero_count":  nonzero,
						"total_elements": audioFrameSize,
					}
					if jsonData, err := json.MarshalIndent(metadata, "", "  "); err == nil {
						os.WriteFile(jsonFile, jsonData, 0644)
					}
				}
			}
			if req.BatchSize > 0 {
				log.Printf("🔍 DEBUG: Saved %d audio tensors to test_output/", min(int(req.BatchSize), 10))
			}
		}

		if s.cfg.Logging.LogInferenceTimes {
			log.Printf("🎵 Audio processing: %d samples -> %d mel frames -> %d output frames with 8192 features each (%.2fms)",
				len(rawAudioSamples), len(melSpec), req.BatchSize, audioProcessingTime.Seconds()*1000)
		}

	} else if len(req.AudioFeatures) > 0 {
		// BACKWARD COMPAT: Use pre-computed audio features
		expectedAudioSize := audioFrameSize
		if len(req.AudioFeatures) != expectedAudioSize*4 {
			return &pb.CompositeBatchResponse{
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
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   "Either raw_audio or audio_features must be provided",
		}, nil
	}

	// === PHASE 2: MODEL INFERENCE ===
	// Get or load model
	modelInstance, err := s.modelRegistry.GetOrLoadModel(req.ModelId)
	if err != nil {
		return &pb.CompositeBatchResponse{
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
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Inference failed: %v", err),
		}, nil
	}

	if s.cfg.Logging.LogInferenceTimes {
		log.Printf("⚡ Inference: model=%s, batch=%d, gpu=%d, time=%.2fms",
			req.ModelId, req.BatchSize, modelInstance.GPUID,
			inferTime.Seconds()*1000)
	}

	// === PHASE 3: COMPOSITING (PARALLEL) ===
	compositeStart := time.Now()

	// Get background images and crop rects
	modelData, err := s.imageRegistry.GetModelData(req.ModelId)
	if err != nil {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to get model data: %v", err),
		}, nil
	}

	// Process frames in parallel using goroutines
	type frameResult struct {
		index int
		data  []byte
		err   error
	}

	resultChan := make(chan frameResult, req.BatchSize)
	var wg sync.WaitGroup
	frameIdx := int(req.StartFrameIdx)

	for i := 0; i < int(req.BatchSize); i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			result := frameResult{index: idx}

			// Get background for this frame (backgrounds can cycle)
			bgIdx := (frameIdx + idx) % len(modelData.Backgrounds)
			background := modelData.Backgrounds[bgIdx]

			// Get crop rect for this frame (crop rects are per-frame, don't cycle)
			actualFrameIdx := frameIdx + idx
			if actualFrameIdx >= len(modelData.CropRects) {
				result.err = fmt.Errorf("frame index %d exceeds crop rects length %d", actualFrameIdx, len(modelData.CropRects))
				resultChan <- result
				return
			}
			cropRect := modelData.CropRects[actualFrameIdx]

			// Extract mouth region from inference output
			mouthRegionStart := idx * outputFrameSize
			mouthRegionEnd := mouthRegionStart + outputFrameSize
			mouthRegion := outputs[mouthRegionStart:mouthRegionEnd]

			// Composite frame
			compositedFrame, err := compositeFrame(background, mouthRegion, cropRect, s.cfg.Output.JPEGQuality)
			if err != nil {
				result.err = fmt.Errorf("failed to composite frame %d: %v", idx, err)
				resultChan <- result
				return
			}

			result.data = compositedFrame
			resultChan <- result
		}(i)
	}

	// Wait for all goroutines to complete
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results in order
	compositedFrames := make([][]byte, req.BatchSize)
	var compositeErr error
	for result := range resultChan {
		if result.err != nil {
			compositeErr = result.err
			break
		}
		compositedFrames[result.index] = result.data
	}

	if compositeErr != nil {
		return &pb.CompositeBatchResponse{
			Success: false,
			Error:   compositeErr.Error(),
		}, nil
	}

	compositeTime := time.Since(compositeStart)
	totalTime := time.Since(startTime)

	// Log timing breakdown using buffered logger
	if s.cfg.Logging.LogCompositingTimes {
		reqLog.Printf("⏱️  Timing breakdown (batch=%d): audio=%.2fms, inference=%.2fms, composite=%.2fms, total=%.2fms",
			req.BatchSize,
			audioProcessingTime.Seconds()*1000,
			inferTime.Seconds()*1000,
			compositeTime.Seconds()*1000,
			totalTime.Seconds()*1000)
	}

	if s.cfg.Logging.LogInferenceTimes {
		reqLog.Printf("🎨 Monolithic: model=%s, batch=%d, gpu=%d, inference=%.2fms, composite=%.2fms, total=%.2fms",
			req.ModelId, req.BatchSize, modelInstance.GPUID,
			inferTime.Seconds()*1000,
			compositeTime.Seconds()*1000,
			totalTime.Seconds()*1000)

		log.Printf("🎨 Compositing: %d frames, %.2fms (%.2fms/frame)",
			req.BatchSize, compositeTime.Seconds()*1000,
			compositeTime.Seconds()*1000/float64(req.BatchSize))
	}

	// Prepare response
	response := &pb.CompositeBatchResponse{
		CompositedFrames:  compositedFrames,
		InferenceTimeMs:   float32(inferTime.Seconds() * 1000),
		CompositeTimeMs:   float32(compositeTime.Seconds() * 1000),
		TotalTimeMs:       float32(totalTime.Seconds() * 1000),
		AudioProcessingMs: float32(audioProcessingTime.Seconds() * 1000),
		Success:           true,
		GpuId:             int32(modelInstance.GPUID),
	}

	// Commit buffered logs AFTER response is ready (zero latency impact)
	defer reqLog.Commit()

	return response, nil
}

// compositeFrame composites a mouth region onto a background image at the crop rect
func compositeFrame(background *image.RGBA, mouthRegion []float32, cropRect image.Rectangle, jpegQuality int) ([]byte, error) {
	// Convert float32 output to RGBA image (uses pooled 320x320 image)
	mouthImg := outputToImage(mouthRegion)
	defer rgbaPool320.Put(mouthImg) // Return to pool when done

	// Get crop rect dimensions
	x := cropRect.Min.X
	y := cropRect.Min.Y
	w := cropRect.Dx()
	h := cropRect.Dy()

	// Resize mouth region to match crop rect (uses pooled image)
	resized := resizeImagePooled(mouthImg, w, h)
	defer rgbaPoolResize.Put(resized) // Return to pool when done

	// Clone background using pooled image
	bgBounds := background.Bounds()
	result := getPooledImageForSize(bgBounds.Dx(), bgBounds.Dy())
	defer returnPooledImageForSize(result, bgBounds.Dx(), bgBounds.Dy())

	draw.Draw(result, result.Bounds(), background, image.Point{}, draw.Src)

	// Paste resized mouth region onto background
	dstRect := image.Rect(x, y, x+w, y+h)
	// Create a sub-image view to only draw the needed portion
	resizedView := resized.SubImage(image.Rect(0, 0, w, h)).(*image.RGBA)
	draw.Draw(result, dstRect, resizedView, image.Point{}, draw.Src)

	// Encode to JPEG using pooled buffer
	buf := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufferPool.Put(buf)

	err := jpeg.Encode(buf, result, &jpeg.Options{Quality: jpegQuality})
	if err != nil {
		return nil, err
	}

	// Copy to new slice (buf will be returned to pool)
	jpegData := make([]byte, buf.Len())
	copy(jpegData, buf.Bytes())

	return jpegData, nil
}

// outputToImage converts model output float32 data to RGBA image
func outputToImage(outputData []float32) *image.RGBA {
	// Get image from pool
	img := rgbaPool320.Get().(*image.RGBA)

	// Convert BGR float32 [0,1] to RGB bytes [0,255]
	// Model outputs BGR in [C, H, W] format: [3, 320, 320] (same as OpenCV input)
	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// BGR order from ONNX model (OpenCV convention)
			b := outputData[0*320*320+y*320+x]
			g := outputData[1*320*320+y*320+x]
			r := outputData[2*320*320+y*320+x]

			// Clamp and convert to bytes
			rByte := uint8(clampFloat(r * 255.0))
			gByte := uint8(clampFloat(g * 255.0))
			bByte := uint8(clampFloat(b * 255.0))

			img.SetRGBA(x, y, color.RGBA{R: rByte, G: gByte, B: bByte, A: 255})
		}
	}

	return img
}

// resizeImage resizes an image using bilinear interpolation
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

// resizeImagePooled resizes an image using bilinear interpolation with pooled destination
func resizeImagePooled(src *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	// Get pooled image (400x400 max)
	dst := rgbaPoolResize.Get().(*image.RGBA)

	// Verify the pooled image is large enough
	if targetWidth > 400 || targetHeight > 400 {
		// Fallback to regular allocation for oversized requests
		log.Printf("⚠️  Resize target (%dx%d) exceeds pool size (400x400), allocating", targetWidth, targetHeight)
		return resizeImage(src, targetWidth, targetHeight)
	}

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

// getPooledImageForSize returns a pooled image appropriate for the given size
func getPooledImageForSize(width, height int) *image.RGBA {
	// Full HD (1920x1080) - most common for backgrounds
	if width == 1920 && height == 1080 {
		return rgbaPoolFullHD.Get().(*image.RGBA)
	}

	// For other sizes, allocate (could add more pools for common sizes)
	// In production you might want to add pools for 1280x720, 2560x1440, etc.
	return image.NewRGBA(image.Rect(0, 0, width, height))
}

// returnPooledImageForSize returns an image to the appropriate pool
func returnPooledImageForSize(img *image.RGBA, width, height int) {
	// Full HD (1920x1080)
	if width == 1920 && height == 1080 {
		rgbaPoolFullHD.Put(img)
	}
	// For other sizes, let GC handle it (no pool available)
}

// bilinearInterp performs bilinear interpolation for a single channel
func bilinearInterp(c00, c10, c01, c11 uint8, xWeight, yWeight float32) uint8 {
	top := float32(c00)*(1-xWeight) + float32(c10)*xWeight
	bottom := float32(c01)*(1-xWeight) + float32(c11)*xWeight
	result := top*(1-yWeight) + bottom*yWeight
	return uint8(result)
}

// clampFloat clamps a float32 value between 0 and 255
func clampFloat(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 255 {
		return 255
	}
	return val
}

// bytesToFloat32 converts a byte slice to float32 slice (zero-copy using unsafe)
func bytesToFloat32(b []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// Health check
func (s *monolithicServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
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

// ListModels returns all configured models
func (s *monolithicServer) ListModels(ctx context.Context, req *pb.ListModelsRequest) (*pb.ListModelsResponse, error) {
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

// LoadModel explicitly loads a model
func (s *monolithicServer) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
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

// UnloadModel explicitly unloads a model
func (s *monolithicServer) UnloadModel(ctx context.Context, req *pb.UnloadModelRequest) (*pb.UnloadModelResponse, error) {
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

// GetModelStats returns statistics for all models
func (s *monolithicServer) GetModelStats(ctx context.Context, req *pb.GetModelStatsRequest) (*pb.GetModelStatsResponse, error) {
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
