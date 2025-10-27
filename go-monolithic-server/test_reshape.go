package main

import (
	"fmt"
	"math"
)

func main() {
	// Create test features [512] with sequential values 0..511
	features := make([]float32, 512)
	for i := 0; i < 512; i++ {
		features[i] = float32(i)
	}

	// Reshape to [32, 16, 16] using our Go logic
	audioData := make([]float32, 32*16*16)

	for c := 0; c < 32; c++ {
		for h := 0; h < 16; h++ {
			featureValue := features[c*16+h]
			// Repeat this value 16 times in width dimension
			for w := 0; w < 16; w++ {
				audioData[c*16*16+h*16+w] = featureValue
			}
		}
	}

	// Verify the pattern
	fmt.Println("Go Reshape Test:")
	fmt.Printf("audioData[0:16] (should be 16x value 0): %v\n", audioData[0:16])
	fmt.Printf("audioData[16:32] (should be 16x value 1): %v\n", audioData[16:32])
	fmt.Printf("audioData[256:272] (should be 16x value 16): %v\n", audioData[256:272])

	// Check if diagonal pattern appears
	fmt.Println("\nChecking for diagonal pattern (should NOT appear):")
	hasDiagonal := false
	for c := 0; c < 32 && c < 16; c++ {
		for h := 0; h < 16; h++ {
			idx := c*16*16 + h*16
			val := audioData[idx]
			expectedVal := float32(c*16 + h)
			if math.Abs(float64(val-expectedVal)) > 0.001 {
				fmt.Printf("Mismatch at [%d,%d,0]: got %.1f, expected %.1f\n", c, h, val, expectedVal)
				hasDiagonal = true
			}
		}
	}

	if !hasDiagonal {
		fmt.Println("✅ No diagonal pattern detected - reshape is correct!")
	} else {
		fmt.Println("❌ Diagonal pattern detected - reshape has issues")
	}
}
