package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"time"

	pb "go-monolithic-server/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	serverAddr := "localhost:50053"
	modelID := "sanders"

	fmt.Println("🧪 BATCH SIZE PERFORMANCE TEST - Monolithic Server (REAL DATA)")
	fmt.Println("=" + string(make([]byte, 70)))

	// Connect to server
	fmt.Printf("🔌 Connecting to monolithic server at %s...\n", serverAddr)
	conn, err := grpc.Dial(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewMonolithicServiceClient(conn)
	fmt.Println("✅ Connected successfully")

	// Health check
	fmt.Println("\n📊 Checking server health...")
	healthReq := &pb.HealthRequest{}
	healthResp, err := client.Health(context.Background(), healthReq)
	if err != nil {
		log.Fatalf("Health check failed: %v", err)
	}
	fmt.Printf("✅ Server Status: Healthy (Models: %d/%d, GPUs: %v)\n",
		healthResp.LoadedModels, healthResp.MaxModels, healthResp.GpuIds)

	// Load audio
	audioFile := "../aud.wav"
	fmt.Printf("\n🎵 Loading REAL audio file: %s\n", audioFile)
	audioSamples, sampleRate, err := readWAVFile(audioFile)
	if err != nil {
		log.Fatalf("Failed to load audio: %v", err)
	}

	audioDuration := float64(len(audioSamples)) / float64(sampleRate)
	fmt.Printf("✅ Loaded %d samples (%.2f seconds at %d Hz)\n",
		len(audioSamples), audioDuration, sampleRate)

	// Create output directory
	if err := os.MkdirAll("test_output", 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Test different batch sizes
	batchSizes := []int{1, 4, 8, 25}

	fmt.Println("\n� Testing batch sizes with REAL audio + visual data: 1, 4, 8, 25")
	fmt.Println("   Running 3 iterations per batch size (first may include loading)")
	fmt.Println("=" + string(make([]byte, 70)))

	for _, batchSize := range batchSizes {
		fmt.Printf("\n� TESTING BATCH SIZE: %d\n", batchSize)
		fmt.Println("─" + string(make([]byte, 70)))

		var totalAudioMs float32
		var totalInferenceMs float32
		var totalCompositeMs float32
		var totalTimeMs float32
		iterations := 3

		for i := 0; i < iterations; i++ {
			frameIdx := i * batchSize

			// Load REAL visual frames
			crops, rois, err := loadRealVisualFrames(frameIdx, batchSize)
			if err != nil {
				log.Fatalf("❌ Failed to load real visual frames: %v", err)
			}

			// Flatten crops and ROIs into single byte array
			visualFrames := make([]byte, 0, len(crops[0])*batchSize+len(rois[0])*batchSize)
			for j := 0; j < batchSize; j++ {
				visualFrames = append(visualFrames, crops[j]...)
			}
			for j := 0; j < batchSize; j++ {
				visualFrames = append(visualFrames, rois[j]...)
			}

			// Extract REAL audio chunk
			rawAudio := extractAudioChunk(audioSamples, frameIdx, batchSize, sampleRate)

			// Call server
			req := &pb.CompositeBatchRequest{
				ModelId:       modelID,
				VisualFrames:  visualFrames,
				RawAudio:      rawAudio,
				BatchSize:     int32(batchSize),
				StartFrameIdx: int32(frameIdx),
			}

			batchStart := time.Now()
			resp, err := client.InferBatchComposite(context.Background(), req)
			batchDuration := time.Since(batchStart)

			if err != nil {
				log.Fatalf("❌ Batch failed: %v", err)
			}

			if !resp.Success {
				log.Fatalf("❌ Batch returned success=false: %s", resp.Error)
			}

			// Print iteration results
			iterLabel := "WARM"
			if i == 0 {
				iterLabel = "COLD (w/ loading)"
			}

			fmt.Printf("  Iter %d/%d [%s]:\n", i+1, iterations, iterLabel)
			fmt.Printf("    🎵 Audio Proc:  %7.2f ms\n", resp.AudioProcessingMs)
			fmt.Printf("    ⚡ Inference:   %7.2f ms\n", resp.InferenceTimeMs)
			fmt.Printf("    🎨 Compositing: %7.2f ms\n", resp.CompositeTimeMs)
			fmt.Printf("    📊 Total:       %7.2f ms (actual: %d ms)\n",
				resp.TotalTimeMs, batchDuration.Milliseconds())

			if len(resp.CompositedFrames) > 0 {
				fmt.Printf("    � Frame size:  %d bytes avg\n", len(resp.CompositedFrames[0]))
			}

			// Accumulate stats (skip first iteration for cold start)
			if i > 0 {
				totalAudioMs += resp.AudioProcessingMs
				totalInferenceMs += resp.InferenceTimeMs
				totalCompositeMs += resp.CompositeTimeMs
				totalTimeMs += resp.TotalTimeMs
			}

			// Save first frame of first iteration for quality check
			if i == 0 && len(resp.CompositedFrames) > 0 {
				filename := fmt.Sprintf("test_output/batch_%d_sample.jpg", batchSize)
				if err := os.WriteFile(filename, resp.CompositedFrames[0], 0644); err != nil {
					log.Printf("⚠️  Failed to save sample: %v", err)
				}
			}

			// Small delay between iterations
			time.Sleep(100 * time.Millisecond)
		}

		// Calculate averages (excluding first cold iteration)
		warmIterations := iterations - 1
		avgAudio := totalAudioMs / float32(warmIterations)
		avgInference := totalInferenceMs / float32(warmIterations)
		avgComposite := totalCompositeMs / float32(warmIterations)
		avgTotal := totalTimeMs / float32(warmIterations)

		fmt.Println("\n  📈 WARM Performance (avg of last 2 iterations):")
		fmt.Printf("    🎵 Audio Proc:  %7.2f ms\n", avgAudio)
		fmt.Printf("    ⚡ Inference:   %7.2f ms  (%.2f ms/frame)\n",
			avgInference, avgInference/float32(batchSize))
		fmt.Printf("    🎨 Compositing: %7.2f ms  (%.2f ms/frame)\n",
			avgComposite, avgComposite/float32(batchSize))
		fmt.Printf("    📊 Total:       %7.2f ms  (%.2f ms/frame)\n",
			avgTotal, avgTotal/float32(batchSize))
		fmt.Printf("    � Throughput:  %.1f FPS\n",
			float32(batchSize)*1000.0/avgTotal)
	}

	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("✅ Batch size performance test complete!")
	fmt.Println("\n💡 Check test_output/ for sample frames from each batch size")
}

// loadRealVisualFrames loads actual frames from the sanders crop and ROI videos using Python helper
func loadRealVisualFrames(startFrame, batchSize int) ([][]byte, [][]byte, error) {
	// Call Python script to load frames
	cmd := exec.Command("python", "load_frames.py", fmt.Sprintf("%d", startFrame), fmt.Sprintf("%d", batchSize))
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, nil, fmt.Errorf("Python script failed: %v\nStderr: %s", err, string(exitErr.Stderr))
		}
		return nil, nil, fmt.Errorf("failed to run Python script: %v", err)
	}

	// Calculate expected sizes (both crops and ROIs are 320x320 now)
	cropSize := batchSize * 3 * 320 * 320 * 4 // float32
	roiSize := batchSize * 3 * 320 * 320 * 4  // float32
	expectedSize := cropSize + roiSize

	if len(output) != expectedSize {
		return nil, nil, fmt.Errorf("unexpected output size: got %d, expected %d", len(output), expectedSize)
	}

	// Split output into crops and ROIs
	crops := make([][]byte, batchSize)
	rois := make([][]byte, batchSize)

	singleCropSize := 3 * 320 * 320 * 4
	singleRoiSize := 3 * 320 * 320 * 4

	for i := 0; i < batchSize; i++ {
		cropStart := i * singleCropSize
		cropEnd := cropStart + singleCropSize
		crops[i] = output[cropStart:cropEnd]
	}

	roiDataStart := cropSize
	for i := 0; i < batchSize; i++ {
		roiStart := roiDataStart + i*singleRoiSize
		roiEnd := roiStart + singleRoiSize
		rois[i] = output[roiStart:roiEnd]
	}

	return crops, rois, nil
}

// extractAudioChunk extracts 640ms of audio for a batch
func extractAudioChunk(audioSamples []int16, startFrame, batchSize, sampleRate int) []byte {
	// At 25 FPS: 40ms per frame
	numFramesNeeded := batchSize + 15
	batchDurationMs := numFramesNeeded * 40
	samplesNeeded := (sampleRate * batchDurationMs) / 1000

	// Calculate starting sample (need to start 8 frames earlier for context)
	startFrameWithContext := startFrame - 8
	if startFrameWithContext < 0 {
		startFrameWithContext = 0
	}
	startSample := (startFrameWithContext * sampleRate * 40) / 1000

	// Ensure we don't go past the end of the audio
	if startSample >= len(audioSamples) {
		startSample = startSample % len(audioSamples)
	}
	if startSample+samplesNeeded > len(audioSamples) {
		samplesNeeded = len(audioSamples) - startSample
	}

	// Convert int16 samples to float32 bytes
	audioChunk := audioSamples[startSample : startSample+samplesNeeded]
	audioBytes := make([]byte, len(audioChunk)*4)

	for i, sample := range audioChunk {
		// Normalize int16 to float32 [-1.0, 1.0]
		normalized := float32(sample) / 32768.0
		binary.LittleEndian.PutUint32(audioBytes[i*4:], math.Float32bits(normalized))
	}

	return audioBytes
}

// readWAVFile reads a WAV file and returns int16 PCM samples
func readWAVFile(filename string) ([]int16, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Read RIFF header
	var riffHeader [12]byte
	if _, err := io.ReadFull(file, riffHeader[:]); err != nil {
		return nil, 0, fmt.Errorf("failed to read RIFF header: %w", err)
	}

	if string(riffHeader[0:4]) != "RIFF" || string(riffHeader[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a valid WAV file")
	}

	var sampleRate int
	var numChannels int
	var bitsPerSample int

	// Read chunks
	for {
		var chunkHeader [8]byte
		if _, err := io.ReadFull(file, chunkHeader[:]); err != nil {
			if err == io.EOF {
				break
			}
			return nil, 0, fmt.Errorf("failed to read chunk header: %w", err)
		}

		chunkID := string(chunkHeader[0:4])
		chunkSize := int(binary.LittleEndian.Uint32(chunkHeader[4:8]))

		if chunkID == "fmt " {
			var fmtData [16]byte
			if _, err := io.ReadFull(file, fmtData[:]); err != nil {
				return nil, 0, fmt.Errorf("failed to read fmt chunk: %w", err)
			}

			numChannels = int(binary.LittleEndian.Uint16(fmtData[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(fmtData[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(fmtData[14:16]))

			if chunkSize > 16 {
				file.Seek(int64(chunkSize-16), io.SeekCurrent)
			}

		} else if chunkID == "data" {
			rawData := make([]byte, chunkSize)
			if _, err := io.ReadFull(file, rawData); err != nil {
				return nil, 0, fmt.Errorf("failed to read PCM data: %w", err)
			}

			numSamples := chunkSize / (bitsPerSample / 8)
			samples := make([]int16, numSamples)

			if bitsPerSample == 16 {
				for i := 0; i < numSamples; i++ {
					samples[i] = int16(binary.LittleEndian.Uint16(rawData[i*2:]))
				}
			} else {
				return nil, 0, fmt.Errorf("unsupported bit depth: %d", bitsPerSample)
			}

			// Convert stereo to mono if needed
			if numChannels == 2 {
				monoSamples := make([]int16, numSamples/2)
				for i := 0; i < len(monoSamples); i++ {
					left := int32(samples[i*2])
					right := int32(samples[i*2+1])
					monoSamples[i] = int16((left + right) / 2)
				}
				return monoSamples, sampleRate, nil
			}

			return samples, sampleRate, nil

		} else {
			file.Seek(int64(chunkSize), io.SeekCurrent)
		}
	}

	return nil, 0, fmt.Errorf("no audio data found in WAV file")
}
