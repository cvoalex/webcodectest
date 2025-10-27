package main

import (
	"fmt"
	"log"

	"go-onnx-inference/lipsyncinfer"
)

func main() {
	fmt.Println("🚀 Simple ONNX Inference Test")
	fmt.Println()

	// Model path - adjust if needed
	modelPath := "d:/Projects/webcodecstest/fast_service/models/default_model/models/99.onnx"

	// Create inferencer
	fmt.Println("Loading model...")
	inferencer, err := lipsyncinfer.NewInferencer(modelPath)
	if err != nil {
		log.Fatalf("Failed to create inferencer: %v", err)
	}
	defer inferencer.Close()

	fmt.Println("Model loaded successfully!")
	fmt.Println()

	// Get input shapes
	visualShape, audioShape := inferencer.GetInputShapes()
	fmt.Printf("Visual input shape: %v\n", visualShape)
	fmt.Printf("Audio input shape: %v\n", audioShape)
	fmt.Println()

	// Calculate sizes
	visualSize := int(visualShape[0] * visualShape[1] * visualShape[2] * visualShape[3])
	audioSize := int(audioShape[0] * audioShape[1] * audioShape[2] * audioShape[3])

	// Create dummy inputs
	fmt.Println("Creating test inputs...")
	visualInput := make([]float32, visualSize)
	audioInput := make([]float32, audioSize)

	// Fill with dummy data (0.5 for demo)
	for i := range visualInput {
		visualInput[i] = 0.5
	}
	for i := range audioInput {
		audioInput[i] = 0.5
	}

	// Run inference
	fmt.Println("Running inference...")
	output, elapsed, err := inferencer.InferWithTiming(visualInput, audioInput)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	fmt.Printf("✅ Inference successful!\n")
	fmt.Printf("   Time taken: %.3f ms\n", elapsed.Seconds()*1000)
	fmt.Printf("   Output size: %d values\n", len(output))
	fmt.Printf("   FPS: %.1f\n", 1.0/elapsed.Seconds())
	fmt.Println()

	// Show sample output values
	fmt.Println("Sample output values (first 10):")
	for i := 0; i < 10 && i < len(output); i++ {
		fmt.Printf("   [%d]: %.6f\n", i, output[i])
	}
}
