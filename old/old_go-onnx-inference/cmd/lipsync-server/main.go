package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"go-onnx-inference/lipsyncinfer"
)

const (
	visualFrameSize = 6 * 320 * 320 // Per-frame visual input size
	audioFrameSize  = 32 * 16 * 16  // Audio features size
	outputFrameSize = 3 * 320 * 320 // Per-frame output size
	maxBatchSize    = 25            // Maximum batch size
)

type BatchRequest struct {
	VisualFrames  []float32 `json:"visual_frames"`   // [batch_size * visualFrameSize]
	AudioFeatures []float32 `json:"audio_features"`  // [audioFrameSize]
	BatchSize     int       `json:"batch_size"`      // 1-25
	StartFrameIdx int       `json:"start_frame_idx"` // For crop rectangle lookup (optional)
}

type BatchResponse struct {
	OutputFrames    []float32 `json:"output_frames"` // [batch_size * outputFrameSize]
	InferenceTimeMs float64   `json:"inference_time_ms"`
	Success         bool      `json:"success"`
	Error           string    `json:"error,omitempty"`
}

type LipSyncServer struct {
	inferencer *lipsyncinfer.Inferencer
	mu         sync.RWMutex
	modelPath  string
}

func NewLipSyncServer(modelPath string) (*LipSyncServer, error) {
	inferencer, err := lipsyncinfer.NewInferencer(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create inferencer: %w", err)
	}

	// Warmup
	fmt.Println("🔥 Warming up CUDA...")
	warmupVisual := make([]float32, visualFrameSize)
	warmupAudio := make([]float32, audioFrameSize)
	_, err = inferencer.Infer(warmupVisual, warmupAudio)
	if err != nil {
		return nil, fmt.Errorf("warmup failed: %w", err)
	}
	fmt.Println("✅ Warmup complete")

	return &LipSyncServer{
		inferencer: inferencer,
		modelPath:  modelPath,
	}, nil
}

func (s *LipSyncServer) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.inferencer != nil {
		s.inferencer.Close()
	}
}

func (s *LipSyncServer) handleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to read body: %v", err), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var req BatchRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, fmt.Sprintf("Failed to parse JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Validate batch size
	if req.BatchSize < 1 || req.BatchSize > maxBatchSize {
		http.Error(w, fmt.Sprintf("Invalid batch size: %d (must be 1-%d)", req.BatchSize, maxBatchSize), http.StatusBadRequest)
		return
	}

	// Validate input sizes
	expectedVisualSize := req.BatchSize * visualFrameSize
	if len(req.VisualFrames) != expectedVisualSize {
		http.Error(w, fmt.Sprintf("Invalid visual frames size: got %d, expected %d", len(req.VisualFrames), expectedVisualSize), http.StatusBadRequest)
		return
	}

	if len(req.AudioFeatures) != audioFrameSize {
		http.Error(w, fmt.Sprintf("Invalid audio features size: got %d, expected %d", len(req.AudioFeatures), audioFrameSize), http.StatusBadRequest)
		return
	}

	// Process batch using parallel workers
	start := time.Now()
	outputFrames, err := s.processBatchParallel(req.VisualFrames, req.AudioFeatures, req.BatchSize)
	inferenceTime := time.Since(start)

	if err != nil {
		resp := BatchResponse{
			Success: false,
			Error:   err.Error(),
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(resp)
		return
	}

	resp := BatchResponse{
		OutputFrames:    outputFrames,
		InferenceTimeMs: float64(inferenceTime.Milliseconds()),
		Success:         true,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *LipSyncServer) processBatchParallel(visualData, audioData []float32, batchSize int) ([]float32, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Use parallel workers for batches > 1
	numWorkers := batchSize
	if numWorkers > 8 {
		numWorkers = 8
	}

	if batchSize == 1 {
		// Single frame - process directly
		return s.inferencer.Infer(visualData, audioData)
	}

	// Parallel processing
	type result struct {
		frameIdx int
		output   []float32
		err      error
	}

	results := make(chan result, batchSize)
	var wg sync.WaitGroup

	// Launch workers
	for i := 0; i < batchSize; i++ {
		wg.Add(1)
		go func(frameIdx int) {
			defer wg.Done()

			visualStart := frameIdx * visualFrameSize
			visualEnd := visualStart + visualFrameSize
			frameVisual := visualData[visualStart:visualEnd]

			output, err := s.inferencer.Infer(frameVisual, audioData)
			results <- result{
				frameIdx: frameIdx,
				output:   output,
				err:      err,
			}
		}(i)
	}

	// Wait for all workers
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	outputData := make([]float32, batchSize*outputFrameSize)
	for res := range results {
		if res.err != nil {
			return nil, fmt.Errorf("frame %d inference failed: %w", res.frameIdx, res.err)
		}

		outputStart := res.frameIdx * outputFrameSize
		copy(outputData[outputStart:outputStart+outputFrameSize], res.output)
	}

	return outputData, nil
}

func (s *LipSyncServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"model":  s.modelPath,
	})
}

func main() {
	modelPath := "d:/Projects/webcodecstest/minimal_server/models/sanders/checkpoint/model_best.onnx"
	port := 8080

	fmt.Println("================================================================================")
	fmt.Println("🚀 LIP SYNC INFERENCE SERVER (Go + ONNX)")
	fmt.Println("================================================================================")
	fmt.Printf("   Model: %s\n", modelPath)
	fmt.Printf("   Max batch size: %d\n", maxBatchSize)
	fmt.Printf("   Port: %d\n\n", port)

	server, err := NewLipSyncServer(modelPath)
	if err != nil {
		fmt.Printf("❌ Failed to create server: %v\n", err)
		os.Exit(1)
	}
	defer server.Close()

	http.HandleFunc("/batch", server.handleBatch)
	http.HandleFunc("/health", server.handleHealth)

	fmt.Printf("✅ Server ready at http://localhost:%d\n", port)
	fmt.Println("   POST /batch  - Process batch of frames")
	fmt.Println("   GET  /health - Health check")
	fmt.Println("")

	if err := http.ListenAndServe(fmt.Sprintf(":%d", port), nil); err != nil {
		fmt.Printf("❌ Server error: %v\n", err)
		os.Exit(1)
	}
}
