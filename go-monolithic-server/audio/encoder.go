package audio

import (
	"fmt"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

// AudioEncoder wraps the ONNX audio encoder model
type AudioEncoder struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputShape   []int64 // [1, 1, 80, 16]
	outputShape  []int64 // [1, 512]
	libraryPath  string
}

// NewAudioEncoder creates a new audio encoder from ONNX model
func NewAudioEncoder(libraryPath string) (*AudioEncoder, error) {
	// Set library path if provided
	if libraryPath != "" {
		ort.SetSharedLibraryPath(libraryPath)
	}

	// Initialize ONNX Runtime (might already be initialized, that's OK)
	_ = ort.InitializeEnvironment()

	// Find the ONNX model file
	possiblePaths := []string{
		"../audio_encoder.onnx",
		"../../audio_encoder.onnx",
		"audio_encoder.onnx",
		"model/audio_encoder.onnx",
	}

	var modelPath string
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			modelPath = path
			break
		}
	}

	if modelPath == "" {
		return nil, fmt.Errorf("audio_encoder.onnx not found in any of the expected paths: %v", possiblePaths)
	}

	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path for model: %w", err)
	}

	fmt.Printf("Loading audio encoder from: %s\n", absPath)

	// Create persistent tensors
	inputData := make([]float32, 1*1*80*16)
	outputData := make([]float32, 1*512)

	inputTensor, err := ort.NewTensor([]int64{1, 1, 80, 16}, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	outputTensor, err := ort.NewTensor([]int64{1, 512}, outputData)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Create session options
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	// Try to enable CUDA (optional, will fall back to CPU if not available)
	cudaOptions, err := ort.NewCUDAProviderOptions()
	if err == nil {
		defer cudaOptions.Destroy()
		err = cudaOptions.Update(map[string]string{
			"device_id": "0",
		})
		if err == nil {
			_ = sessionOptions.AppendExecutionProviderCUDA(cudaOptions)
		}
	}

	// Create advanced session (pre-allocated tensors)
	session, err := ort.NewAdvancedSession(
		absPath,
		[]string{"mel_spectrogram"}, // Input name from ONNX export
		[]string{"audio_features"},  // Output name from ONNX export
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		sessionOptions,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	encoder := &AudioEncoder{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		inputShape:   []int64{1, 1, 80, 16},
		outputShape:  []int64{1, 512},
		libraryPath:  libraryPath,
	}

	fmt.Printf("✅ Audio encoder loaded successfully\n")
	fmt.Printf("   Input shape: %v\n", encoder.inputShape)
	fmt.Printf("   Output shape: %v\n", encoder.outputShape)

	return encoder, nil
}

// Encode processes a mel-spectrogram window through the audio encoder
// Input: melWindow [80, 16] (mel_bins × frames)
// Output: features [512] (audio feature vector)
func (e *AudioEncoder) Encode(melWindow [][]float32) ([]float32, error) {
	// Validate input shape
	if len(melWindow) != 80 {
		return nil, fmt.Errorf("invalid mel window: expected 80 mel bins, got %d", len(melWindow))
	}
	if len(melWindow[0]) != 16 {
		return nil, fmt.Errorf("invalid mel window: expected 16 frames, got %d", len(melWindow[0]))
	}

	// Copy mel-spectrogram to input tensor (reshape from [80, 16] to [1, 1, 80, 16])
	inputData := e.inputTensor.GetData()
	idx := 0
	for i := 0; i < 80; i++ {
		for j := 0; j < 16; j++ {
			inputData[idx] = melWindow[i][j]
			idx++
		}
	}

	// Run inference
	err := e.session.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run ONNX inference: %w", err)
	}

	// Get output data and make a copy
	outputData := e.outputTensor.GetData()
	features := make([]float32, 512)
	copy(features, outputData)

	// Debug: Check if output looks reasonable (first frame only, occasionally)
	if len(features) > 0 {
		min, max, sum := features[0], features[0], float32(0)
		for _, v := range features {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
			sum += v
		}
		// Only log occasionally to avoid spam
		if min == 0 && max == 0 {
			fmt.Printf("⚠️  Audio encoder output is all zeros!\n")
		}
	}

	return features, nil
}

// EncodeBatch processes multiple mel-spectrogram windows in parallel
// Input: melWindows [][80][16]float32 (array of mel windows)
// Output: [][]float32 (array of 512-dimensional feature vectors)
func (e *AudioEncoder) EncodeBatch(melWindows [][][]float32) ([][]float32, error) {
	// CRITICAL: ONNX Runtime sessions are NOT thread-safe for concurrent inference
	// We must run sequentially. Parallelization would require multiple session instances.
	// TODO: Consider creating a pool of encoder instances for true parallelization

	features := make([][]float32, len(melWindows))

	for i, window := range melWindows {
		feat, err := e.Encode(window)
		if err != nil {
			return nil, fmt.Errorf("failed to encode window %d: %w", i, err)
		}
		features[i] = feat
	}

	return features, nil
}

// Close releases ONNX resources
func (e *AudioEncoder) Close() error {
	if e.session != nil {
		e.session.Destroy()
	}
	if e.inputTensor != nil {
		e.inputTensor.Destroy()
	}
	if e.outputTensor != nil {
		e.outputTensor.Destroy()
	}
	return nil
}

// Destroy is an alias for Close to match ONNX runtime naming
func (e *AudioEncoder) Destroy() error {
	return e.Close()
}
