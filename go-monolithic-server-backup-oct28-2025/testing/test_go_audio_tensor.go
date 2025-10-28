package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"strings"

	"go-monolithic-server/audio"
)

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("GO FULL AUDIO ENCODER TEST")
	fmt.Println(strings.Repeat("=", 80))

	// Create output directory
	os.MkdirAll("debug_output", 0755)

	// Load same audio file
	wavPath := "../aud.wav"
	fmt.Printf("\n1. Loading audio: %s\n", wavPath)

	samples, sampleRate, err := audio.LoadWAV(wavPath)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   Loaded: %d samples at %d Hz (%.2fs)\n",
		len(samples), sampleRate, float64(len(samples))/float64(sampleRate))

	// Compute mel spectrogram (full audio)
	fmt.Println("\n2. Computing mel-spectrogram...")

	melProc, err := audio.NewMelProcessor()
	if err != nil {
		log.Fatal(err)
	}

	melSpec, err := melProc.ComputeMelSpectrogram(samples)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   Mel-spectrogram: %d frames × 80 bins\n", len(melSpec), 80)

	// Extract first 16 frames for encoder (or pad if less)
	targetFrames := 16
	var mel16 [][]float32

	if len(melSpec) >= targetFrames {
		mel16 = melSpec[:targetFrames]
	} else {
		mel16 = make([][]float32, targetFrames)
		copy(mel16, melSpec)
		// Pad with -4.0
		for i := len(melSpec); i < targetFrames; i++ {
			mel16[i] = make([]float32, 80)
			for j := 0; j < 80; j++ {
				mel16[i][j] = -4.0
			}
		}
	}

	fmt.Printf("   Using first 16 frames for encoder\n")

	// Load audio encoder
	fmt.Println("\n3. Loading ONNX audio encoder...")
	encoder, err := audio.NewAudioEncoder("")
	if err != nil {
		log.Fatal(err)
	}
	defer encoder.Close()

	fmt.Println("   ✅ Audio encoder loaded")

	// Create 16-frame window in [80][16] format (transpose)
	fmt.Println("\n4. Preparing input for encoder...")

	melWindow := make([][]float32, 80)
	for i := 0; i < 80; i++ {
		melWindow[i] = make([]float32, 16)
		for j := 0; j < 16; j++ {
			melWindow[i][j] = mel16[j][i]
		}
	}

	// Encode to 512 features
	fmt.Println("\n5. Running ONNX inference...")

	features, err := encoder.Encode(melWindow)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   Output shape: [%d]\n", len(features))

	// Calculate stats
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
	mean := sum / float32(len(features))

	fmt.Printf("   Min: %.8f, Max: %.8f, Mean: %.8f\n", min, max, mean)
	fmt.Printf("   First 10 features: %v\n", features[:10])

	// Save embedding for comparison
	saveFloat32Array("debug_output/go_audio_embedding.bin", features)
	fmt.Println("   Saved: debug_output/go_audio_embedding.bin")

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("✅ SUCCESS - Go audio encoder inference complete")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("\nNext step: Run Python comparison")
	fmt.Println("  python test_full_audio_encoder.py")
}

func saveFloat32Array(filepath string, data []float32) error {
	f, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer f.Close()

	length := int32(len(data))
	binary.Write(f, binary.LittleEndian, length)

	for _, val := range data {
		binary.Write(f, binary.LittleEndian, val)
	}

	return nil
}
