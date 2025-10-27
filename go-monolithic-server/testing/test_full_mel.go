package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"

	"go-monolithic-server/audio"
)

func main() {
	// Load FULL sanders audio
	wavPath := "../aud.wav"
	audioSamples, sampleRate, err := readWAVFile(wavPath)
	if err != nil {
		log.Fatalf("Failed to load audio: %v", err)
	}

	fmt.Printf("Loaded %d samples at %d Hz (%.2f seconds)\n",
		len(audioSamples), sampleRate, float64(len(audioSamples))/float64(sampleRate))

	// Convert to float32
	float32Samples := make([]float32, len(audioSamples))
	for i, s := range audioSamples {
		float32Samples[i] = float32(s) / 32768.0
	}

	fmt.Printf("First 5 samples: %v\n", float32Samples[:5])

	// Process through mel-spec pipeline
	processor := audio.NewProcessor(nil)
	melSpec, err := processor.ProcessAudio(float32Samples)
	if err != nil {
		log.Fatalf("Failed to process audio: %v", err)
	}

	fmt.Printf("\nMel-spec shape: [%d, %d]\n", len(melSpec), len(melSpec[0]))

	// Extract frame 8 window (mel frames 25:41)
	startIdx := 25 // int(80 * 8 / 25) = 25
	endIdx := startIdx + 16

	fmt.Printf("Frame 8 window: mel[%d:%d]\n", startIdx, endIdx)

	if endIdx > len(melSpec) {
		log.Fatalf("Not enough mel frames! Need %d, have %d", endIdx, len(melSpec))
	}

	window := melSpec[startIdx:endIdx]

	//Calculate stats
	mean := float32(0)
	min_val := window[0][0]
	max_val := window[0][0]
	count_neg4 := 0

	for i := 0; i < len(window); i++ {
		for j := 0; j < len(window[i]); j++ {
			v := window[i][j]
			mean += v
			if v < min_val {
				min_val = v
			}
			if v > max_val {
				max_val = v
			}
			if v == -4.0 {
				count_neg4++
			}
		}
	}
	mean /= float32(len(window) * len(window[0]))

	fmt.Printf("\nFrame 8 window stats:\n")
	fmt.Printf("  Mean: %.6f\n", mean)
	fmt.Printf("  Min:  %.6f\n", min_val)
	fmt.Printf("  Max:  %.6f\n", max_val)
	fmt.Printf("  Count at -4.0: %d\n", count_neg4)
	fmt.Printf("  First row, first 10: %v\n", window[0][:10])
	fmt.Printf("  Value at [0, 10]: %.6f\n", window[0][10])
	fmt.Printf("  Value at [1, 10]: %.6f\n", window[1][10])

	// Save for comparison
	os.MkdirAll("test_output", 0755)
	f, _ := os.Create("test_output/go_frame8_window_FULL.bin")
	for i := 0; i < len(window); i++ {
		for j := 0; j < len(window[i]); j++ {
			binary.Write(f, binary.LittleEndian, window[i][j])
		}
	}
	f.Close()

	fmt.Printf("\n✓ Saved to test_output/go_frame8_window_FULL.bin\n")

	// Compare with Python expected
	fmt.Printf("\nExpected from Python:\n")
	fmt.Printf("  Mean: -1.633603\n")
	fmt.Printf("  Min:  -3.689235\n")
	fmt.Printf("  Max:  0.254146\n")
	fmt.Printf("  Count at -4.0: 0\n")
	fmt.Printf("  Value at [0, 10]: -0.065241\n")
	fmt.Printf("  Value at [1, 10]: -0.012176\n")
}

func readWAVFile(filename string) ([]int16, int, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	// Read RIFF header (12 bytes)
	riffHeader := make([]byte, 12)
	_, err = f.Read(riffHeader)
	if err != nil {
		return nil, 0, err
	}

	// Verify RIFF/WAVE
	if string(riffHeader[0:4]) != "RIFF" || string(riffHeader[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a valid WAV file")
	}

	var sampleRate int
	var numChannels int
	var bitsPerSample int
	var dataStart int64
	var dataSize int

	// Read chunks until we find "data"
	for {
		chunkHeader := make([]byte, 8)
		n, err := f.Read(chunkHeader)
		if err != nil || n < 8 {
			break
		}

		chunkID := string(chunkHeader[0:4])
		chunkSize := int(binary.LittleEndian.Uint32(chunkHeader[4:8]))

		if chunkID == "fmt " {
			// Read format chunk
			fmtData := make([]byte, chunkSize)
			f.Read(fmtData)

			numChannels = int(binary.LittleEndian.Uint16(fmtData[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(fmtData[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(fmtData[14:16]))

			fmt.Printf("   📊 WAV: %d Hz, %d-bit, %d channel(s)\n", sampleRate, bitsPerSample, numChannels)
		} else if chunkID == "data" {
			// Found data chunk!
			dataSize = chunkSize
			dataStart, _ = f.Seek(0, 1) // Current position
			break
		} else {
			// Skip unknown chunk
			f.Seek(int64(chunkSize), 1)
		}
	}

	if dataStart == 0 {
		return nil, 0, fmt.Errorf("no data chunk found")
	}

	// Seek to data start
	f.Seek(dataStart, 0)

	// Read all audio samples
	numSamples := dataSize / 2 // 16-bit samples
	samples := make([]int16, numSamples)
	err = binary.Read(f, binary.LittleEndian, samples)
	if err != nil {
		return nil, 0, err
	}

	// Convert stereo to mono if needed
	if numChannels == 2 {
		fmt.Printf("   ↔️  Converted stereo to mono\n")
		mono := make([]int16, len(samples)/2)
		for i := 0; i < len(mono); i++ {
			left := int32(samples[i*2])
			right := int32(samples[i*2+1])
			mono[i] = int16((left + right) / 2)
		}
		return mono, sampleRate, nil
	}

	return samples, sampleRate, nil
}
