package server

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	pb "go-monolithic-server/proto"
)

// InferBatchComposite performs inference + compositing in a single call
// This is the EXACT original code from main.go - NO algorithmic changes
func (s *Server) InferBatchComposite(ctx context.Context, req *pb.CompositeBatchRequest) (*pb.CompositeBatchResponse, error) {
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
		// At 25fps, each frame uses a 16-step window starting at: int(80 * frame/25)
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
		allMelWindows := make([][][]float32, numVideoFrames)

		log.Printf("🎵 Extracting %d mel windows (parallel)...", numVideoFrames)

		// OPTIMIZATION #4: Parallelize mel window extraction
		extractMelWindowsParallel(melSpec, numMelFrames, numVideoFrames, allMelWindows, s.cfg.Logging.SaveDebugFiles)

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

			// Pad left with zeros (optimized bulk operation)
			if padLeft > 0 {
				destOffset := outputOffset + frameCounter*512
				zeroPadAudioFeatures(audioData, destOffset, padLeft, 512)
				frameCounter += padLeft
			}

			// Copy actual frames
			for srcFrame := left; srcFrame < right; srcFrame++ {
				features := paddedFeatures[srcFrame]
				destOffset := outputOffset + frameCounter*512
				copyAudioFeatures(audioData, destOffset, features)
				frameCounter++
			}

			// Pad right with zeros (optimized bulk operation)
			if padRight > 0 {
				destOffset := outputOffset + frameCounter*512
				zeroPadAudioFeatures(audioData, destOffset, padRight, 512)
				frameCounter += padRight
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

// min is a helper function to return the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
