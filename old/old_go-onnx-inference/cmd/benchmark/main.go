package main

import (
	"fmt"
	"log"

	"go-onnx-inference/lipsyncinfer"
)

func main() {
	fmt.Println("🎯 Go + ONNX Runtime Benchmark: RTX 4090")
	fmt.Println("=" + string(make([]byte, 60)))

	// Model path
	modelPath := "d:/Projects/webcodecstest/fast_service/models/default_model/models/99.onnx"

	// Create inferencer
	fmt.Println("\n🔄 Loading ONNX model...")
	inferencer, err := lipsyncinfer.NewInferencer(modelPath)
	if err != nil {
		log.Fatalf("Failed to create inferencer: %v", err)
	}
	defer inferencer.Close()

	visualShape, audioShape := inferencer.GetInputShapes()
	outputShape := inferencer.GetOutputShape()

	fmt.Printf("📊 Model Info:\n")
	fmt.Printf("   Visual input:  %v\n", visualShape)
	fmt.Printf("   Audio input:   %v\n", audioShape)
	fmt.Printf("   Output shape:  %v\n", outputShape)

	// Run benchmark
	fmt.Println("\n🚀 Starting benchmark...")
	avgTime, err := inferencer.Benchmark(500, 50)
	if err != nil {
		log.Fatalf("Benchmark failed: %v", err)
	}

	// Calculate FPS
	fps := 1.0 / avgTime.Seconds()

	// Print results
	fmt.Println("\n" + "=" + string(make([]byte, 60)))
	fmt.Println("📊 BENCHMARK RESULTS")
	fmt.Println("=" + string(make([]byte, 60)))
	fmt.Printf("Average time:  %.3f ms\n", avgTime.Seconds()*1000)
	fmt.Printf("FPS:           %.1f\n", fps)
	fmt.Println()

	// Compare with Python results
	fmt.Println("📈 COMPARISON WITH PYTHON:")
	fmt.Println("   Python + PyTorch:      8.784 ms  (113.8 FPS)")
	fmt.Println("   Python + ONNX + CUDA:  3.164 ms  (316.1 FPS)")
	fmt.Printf("   Go + ONNX + CUDA:      %.3f ms  (%.1f FPS)\n",
		avgTime.Seconds()*1000, fps)

	// Calculate speedup
	pytorchTime := 8.784
	pythonOnnxTime := 3.164
	goTime := avgTime.Seconds() * 1000

	speedupVsPyTorch := pytorchTime / goTime
	speedupVsPythonOnnx := pythonOnnxTime / goTime

	fmt.Println("\n🚀 SPEEDUP:")
	fmt.Printf("   vs PyTorch:        %.2fx faster\n", speedupVsPyTorch)
	fmt.Printf("   vs Python+ONNX:    %.2fx faster\n", speedupVsPythonOnnx)

	fmt.Println("\n✅ Benchmark complete!")
}
