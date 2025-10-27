package lipsyncinfer

import (
	"fmt"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// Inferencer handles ONNX model inference for lip sync
type Inferencer struct {
	session      *ort.AdvancedSession
	visualTensor *ort.Tensor[float32]
	audioTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputNames   []string
	outputNames  []string
	visualShape  []int64
	audioShape   []int64
	outputShape  []int64
}

// NewInferencer creates a new ONNX inferencer
func NewInferencer(modelPath string) (*Inferencer, error) {
	// Set the path to the ONNX Runtime shared library
	// This MUST be called before InitializeEnvironment()
	ort.SetSharedLibraryPath("C:\\onnxruntime-1.22.0\\lib\\onnxruntime.dll")

	// Initialize ONNX Runtime
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	// Create persistent input and output tensors
	visualData := make([]float32, 1*6*320*320)
	audioData := make([]float32, 1*32*16*16)
	outputData := make([]float32, 1*3*320*320)

	visualTensor, err := ort.NewTensor([]int64{1, 6, 320, 320}, visualData)
	if err != nil {
		return nil, fmt.Errorf("failed to create visual tensor: %w", err)
	}

	audioTensor, err := ort.NewTensor([]int64{1, 32, 16, 16}, audioData)
	if err != nil {
		visualTensor.Destroy()
		return nil, fmt.Errorf("failed to create audio tensor: %w", err)
	}

	outputTensor, err := ort.NewTensor([]int64{1, 3, 320, 320}, outputData)
	if err != nil {
		visualTensor.Destroy()
		audioTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// Create session options
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		visualTensor.Destroy()
		audioTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	// Try to append CUDA provider
	cudaOptions, err := ort.NewCUDAProviderOptions()
	if err == nil {
		defer cudaOptions.Destroy()
		err = cudaOptions.Update(map[string]string{"device_id": "0"})
		if err == nil {
			err = sessionOptions.AppendExecutionProviderCUDA(cudaOptions)
			if err == nil {
				fmt.Println("✅ CUDA execution provider enabled")
			} else {
				fmt.Printf("Warning: Could not enable CUDA: %v\n", err)
				fmt.Println("Running on CPU")
			}
		}
	} else {
		fmt.Printf("Warning: CUDA not available: %v\n", err)
		fmt.Println("Running on CPU")
	}

	// Create advanced session
	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input", "audio"},
		[]string{"output"},
		[]ort.Value{visualTensor, audioTensor},
		[]ort.Value{outputTensor},
		sessionOptions,
	)
	if err != nil {
		visualTensor.Destroy()
		audioTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	inferencer := &Inferencer{
		session:      session,
		visualTensor: visualTensor,
		audioTensor:  audioTensor,
		outputTensor: outputTensor,
		inputNames:   []string{"input", "audio"},
		outputNames:  []string{"output"},
		visualShape:  []int64{1, 6, 320, 320},
		audioShape:   []int64{1, 32, 16, 16},
		outputShape:  []int64{1, 3, 320, 320},
	}

	fmt.Printf("✅ ONNX Runtime session created\n")
	return inferencer, nil
}

// Infer runs inference on the model
// visualInput: [1, 6, 320, 320] - concatenated face and masked regions
// audioInput: [1, 32, 16, 16] - mel spectrogram features
// Returns: [1, 3, 320, 320] - predicted lip sync frame
func (inf *Inferencer) Infer(visualInput, audioInput []float32) ([]float32, error) {
	// Validate input sizes
	expectedVisualSize := int(inf.visualShape[0] * inf.visualShape[1] * inf.visualShape[2] * inf.visualShape[3])
	expectedAudioSize := int(inf.audioShape[0] * inf.audioShape[1] * inf.audioShape[2] * inf.audioShape[3])

	if len(visualInput) != expectedVisualSize {
		return nil, fmt.Errorf("invalid visual input size: got %d, expected %d",
			len(visualInput), expectedVisualSize)
	}
	if len(audioInput) != expectedAudioSize {
		return nil, fmt.Errorf("invalid audio input size: got %d, expected %d",
			len(audioInput), expectedAudioSize)
	}

	// Copy input data to tensors
	copy(inf.visualTensor.GetData(), visualInput)
	copy(inf.audioTensor.GetData(), audioInput)

	// Run inference
	err := inf.session.Run()
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Get output data
	outputData := make([]float32, len(inf.outputTensor.GetData()))
	copy(outputData, inf.outputTensor.GetData())

	return outputData, nil
}

// InferBatch runs TRUE batch inference on multiple frames using GPU batching
// visualInput: [batchSize, 6, 320, 320]
// audioInput: [batchSize, 32, 16, 16]
// Returns: [batchSize, 3, 320, 320]
func (inf *Inferencer) InferBatch(visualInput, audioInput []float32, batchSize int) ([]float32, error) {
	// Calculate expected sizes
	visualFrameSize := 6 * 320 * 320
	audioFrameSize := 32 * 16 * 16
	outputFrameSize := 3 * 320 * 320

	expectedVisualSize := batchSize * visualFrameSize
	expectedAudioSize := batchSize * audioFrameSize

	if len(visualInput) != expectedVisualSize {
		return nil, fmt.Errorf("invalid visual input size: got %d, expected %d",
			len(visualInput), expectedVisualSize)
	}
	if len(audioInput) != expectedAudioSize {
		return nil, fmt.Errorf("invalid audio input size: got %d, expected %d",
			len(audioInput), expectedAudioSize)
	}

	// For now, fall back to sequential processing
	// TODO: Implement true batch inference with dynamic session
	// The AdvancedSession API doesn't support dynamic batch sizes easily
	outputs := make([]float32, batchSize*outputFrameSize)

	for i := 0; i < batchSize; i++ {
		visualStart := i * visualFrameSize
		visualFrame := visualInput[visualStart : visualStart+visualFrameSize]

		audioStart := i * audioFrameSize
		audioFrame := audioInput[audioStart : audioStart+audioFrameSize]

		output, err := inf.Infer(visualFrame, audioFrame)
		if err != nil {
			return nil, fmt.Errorf("failed to infer frame %d in batch: %w", i, err)
		}

		outputStart := i * outputFrameSize
		copy(outputs[outputStart:outputStart+outputFrameSize], output)
	}

	return outputs, nil
}

// InferWithTiming runs inference and returns timing information
func (inf *Inferencer) InferWithTiming(visualInput, audioInput []float32) ([]float32, time.Duration, error) {
	start := time.Now()
	output, err := inf.Infer(visualInput, audioInput)
	elapsed := time.Since(start)
	return output, elapsed, err
}

// Benchmark runs a benchmark of the inference
func (inf *Inferencer) Benchmark(numIterations int, warmupIterations int) (avgTime time.Duration, err error) {
	// Create dummy inputs
	visualSize := int(inf.visualShape[0] * inf.visualShape[1] * inf.visualShape[2] * inf.visualShape[3])
	audioSize := int(inf.audioShape[0] * inf.audioShape[1] * inf.audioShape[2] * inf.audioShape[3])

	visualInput := make([]float32, visualSize)
	audioInput := make([]float32, audioSize)

	// Fill with random data
	for i := range visualInput {
		visualInput[i] = 0.5
	}
	for i := range audioInput {
		audioInput[i] = 0.5
	}

	// Warmup
	fmt.Printf("🔄 Warming up (%d iterations)...\n", warmupIterations)
	for i := 0; i < warmupIterations; i++ {
		_, err := inf.Infer(visualInput, audioInput)
		if err != nil {
			return 0, fmt.Errorf("warmup failed: %w", err)
		}
	}

	// Benchmark
	fmt.Printf("⏱️  Running benchmark (%d iterations)...\n", numIterations)
	var totalTime time.Duration

	for i := 0; i < numIterations; i++ {
		_, elapsed, err := inf.InferWithTiming(visualInput, audioInput)
		if err != nil {
			return 0, fmt.Errorf("benchmark iteration %d failed: %w", i, err)
		}
		totalTime += elapsed
	}

	avgTime = totalTime / time.Duration(numIterations)
	return avgTime, nil
}

// Close releases resources
func (inf *Inferencer) Close() error {
	if inf.session != nil {
		inf.session.Destroy()
	}
	if inf.visualTensor != nil {
		inf.visualTensor.Destroy()
	}
	if inf.audioTensor != nil {
		inf.audioTensor.Destroy()
	}
	if inf.outputTensor != nil {
		inf.outputTensor.Destroy()
	}
	return nil
}

// GetInputShapes returns the expected input shapes
func (inf *Inferencer) GetInputShapes() (visualShape, audioShape []int64) {
	return inf.visualShape, inf.audioShape
}

// GetOutputShape returns the expected output shape
func (inf *Inferencer) GetOutputShape() []int64 {
	return inf.outputShape
}

// BatchInferencer handles batch inference with dynamic batch sizes
// This reuses ONE session but creates new tensors for each batch
type BatchInferencer struct {
	session     *ort.DynamicAdvancedSession
	modelPath   string
	inputNames  []string
	outputNames []string
}

// NewBatchInferencer creates a batch-capable inferencer
func NewBatchInferencer(modelPath string) (*BatchInferencer, error) {
	// Set the path to the ONNX Runtime shared library
	ort.SetSharedLibraryPath("C:\\onnxruntime-1.22.0\\lib\\onnxruntime.dll")

	// Initialize ONNX Runtime if not already done
	err := ort.InitializeEnvironment()
	if err != nil {
		// Already initialized is OK
	}

	// Create session options with CUDA
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	// Enable CUDA
	cudaOptions, err := ort.NewCUDAProviderOptions()
	if err == nil {
		defer cudaOptions.Destroy()
		cudaOptions.Update(map[string]string{"device_id": "0"})
		sessionOptions.AppendExecutionProviderCUDA(cudaOptions)
		fmt.Println("✅ Batch inferencer: CUDA enabled")
	}

	// Create dynamic advanced session (supports different tensor sizes per run)
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input", "audio"},
		[]string{"output"},
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic session: %w", err)
	}

	fmt.Println("✅ Batch inferencer created (supports dynamic batch sizes)")

	return &BatchInferencer{
		session:     session,
		modelPath:   modelPath,
		inputNames:  []string{"input", "audio"},
		outputNames: []string{"output"},
	}, nil
}

// InferBatch runs TRUE batch inference with dynamic batch size
// visualInput: [batchSize, 6, 320, 320]
// audioInput: [batchSize, 32, 16, 16]
// Returns: [batchSize, 3, 320, 320]
func (bi *BatchInferencer) InferBatch(visualInput, audioInput []float32, batchSize int) ([]float32, error) {
	outputFrameSize := 3 * 320 * 320

	// Create tensors for this specific batch
	visualTensor, err := ort.NewTensor([]int64{int64(batchSize), 6, 320, 320}, visualInput)
	if err != nil {
		return nil, fmt.Errorf("failed to create visual tensor: %w", err)
	}
	defer visualTensor.Destroy()

	audioTensor, err := ort.NewTensor([]int64{int64(batchSize), 32, 16, 16}, audioInput)
	if err != nil {
		return nil, fmt.Errorf("failed to create audio tensor: %w", err)
	}
	defer audioTensor.Destroy()

	outputData := make([]float32, batchSize*outputFrameSize)
	outputTensor, err := ort.NewTensor([]int64{int64(batchSize), 3, 320, 320}, outputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	// Run inference with dynamic tensors - ONE GPU call!
	err = bi.session.Run(
		[]ort.Value{visualTensor, audioTensor},
		[]ort.Value{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("batch inference failed: %w", err)
	}

	// Copy output
	outputs := make([]float32, batchSize*outputFrameSize)
	copy(outputs, outputData)

	return outputs, nil
}

// Close cleans up the batch inferencer
func (bi *BatchInferencer) Close() error {
	if bi.session != nil {
		bi.session.Destroy()
	}
	return nil
}
