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
	libraryPath  string
}

// NewInferencer creates a new ONNX inferencer with custom library path
func NewInferencer(modelPath string, libraryPath string) (*Inferencer, error) {
	// Set the path to the ONNX Runtime shared library
	// This MUST be called before InitializeEnvironment()
	ort.SetSharedLibraryPath(libraryPath)

	// Initialize ONNX Runtime
	err := ort.InitializeEnvironment()
	if err != nil {
		// Already initialized is OK
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
			if err != nil {
				fmt.Printf("Warning: Could not enable CUDA: %v\n", err)
			}
		}
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
		libraryPath:  libraryPath,
	}

	return inferencer, nil
}

// Infer runs inference on the model
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

// InferBatch runs batch inference sequentially (fake batch)
func (inf *Inferencer) InferBatch(visualInput, audioInput []float32, batchSize int) ([]float32, error) {
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

	// Sequential processing (parallel single-frame is faster than true batch)
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
