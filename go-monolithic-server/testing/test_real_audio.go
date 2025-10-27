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
	"path/filepath"
	"time"

	pb "go-monolithic-server/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	fmt.Println("🎵 REAL AUDIO TEST - Monolithic Server")
	fmt.Println("=" + string(make([]byte, 70)))

	// Connect to monolithic server
	fmt.Printf("🔌 Connecting to monolithic server at localhost:50053...\n")
	conn, err := grpc.NewClient(
		"localhost:50053",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(100*1024*1024),
			grpc.MaxCallSendMsgSize(100*1024*1024),
		),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewMonolithicServiceClient(conn)
	fmt.Println("✅ Connected successfully")

	// Check server health
	fmt.Println("\n📊 Checking server health...")
	healthResp, err := client.Health(context.Background(), &pb.HealthRequest{})
	if err != nil {
		log.Fatalf("❌ Health check failed: %v", err)
	}
	fmt.Printf("✅ Server Status: Healthy (Models: %d/%d, GPUs: %v)\n",
		healthResp.LoadedModels, healthResp.MaxModels, healthResp.GpuIds)

	// Load REAL audio file
	audioPath := filepath.Join("..", "aud.wav")
	fmt.Printf("\n🎵 Loading REAL audio file: %s\n", audioPath)
	audioSamples, sampleRate, err := readWAVFile(audioPath)
	if err != nil {
		log.Fatalf("❌ Failed to load audio: %v", err)
	}

	fmt.Printf("✅ Loaded %d samples (%.2f seconds at %d Hz)\n",
		len(audioSamples), float64(len(audioSamples))/float64(sampleRate), sampleRate)

	// DEBUG: Print first 5 samples to verify correct loading
	fmt.Print("   🔍 First 5 samples (int16): [")
	for i := 0; i < 5 && i < len(audioSamples); i++ {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%d", audioSamples[i])
	}
	fmt.Println("]")
	fmt.Print("   🔍 First 5 samples (float32 normalized): [")
	for i := 0; i < 5 && i < len(audioSamples); i++ {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.8f", float32(audioSamples[i])/32768.0)
	}
	fmt.Println("]")

	// Prepare output directory
	os.MkdirAll("test_output", 0755)

	// Test different batch sizes with REAL audio
	batchSizes := []int{1, 4, 8, 25}
	modelID := "sanders"

	fmt.Println("\n🚀 Testing batch sizes with REAL AUDIO: 1, 4, 8, 25")
	fmt.Println("   Running 3 iterations per batch size")
	fmt.Println("=" + string(make([]byte, 70)))

	for _, batchSize := range batchSizes {
		fmt.Printf("\n📊 TESTING BATCH SIZE: %d (REAL AUDIO)\n", batchSize)
		fmt.Println("─" + string(make([]byte, 70)))

		var totalInferenceMs float32
		var totalAudioMs float32
		var totalCompositeMs float32
		var totalTimeMs float32
		iterations := 3

		for i := 0; i < iterations; i++ {
			frameIdx := i * batchSize

			// Load REAL visual frames from video
			crops, rois, err := loadRealVisualFrames(frameIdx, batchSize)
			if err != nil {
				log.Fatalf("❌ Failed to load real visual frames: %v", err)
			}

			// Flatten crops and ROIs into single byte array: [crops..., rois...]
			visualFrames := make([]byte, 0, len(crops[0])*batchSize+len(rois[0])*batchSize)
			for j := 0; j < batchSize; j++ {
				visualFrames = append(visualFrames, crops[j]...)
			}
			for j := 0; j < batchSize; j++ {
				visualFrames = append(visualFrames, rois[j]...)
			}

			// Extract REAL audio chunk for this batch (640ms per batch)
			rawAudio := extractAudioChunk(audioSamples, frameIdx, batchSize, sampleRate)

			// Call monolithic server with REAL AUDIO + REAL VISUAL
			req := &pb.CompositeBatchRequest{
				ModelId:       modelID,
				VisualFrames:  visualFrames,
				RawAudio:      rawAudio, // ← REAL AUDIO!
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
				fmt.Printf("    💾 Frame size:  %d bytes avg\n",
					len(resp.CompositedFrames[0]))
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
				filename := fmt.Sprintf("test_output/real_audio_batch_%d_sample.jpg", batchSize)
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
		fmt.Printf("    🎵 Audio Proc:  %7.2f ms  (%.2f ms/frame)\n",
			avgAudio, avgAudio/float32(batchSize))
		fmt.Printf("    ⚡ Inference:   %7.2f ms  (%.2f ms/frame)\n",
			avgInference, avgInference/float32(batchSize))
		fmt.Printf("    🎨 Compositing: %7.2f ms  (%.2f ms/frame)\n",
			avgComposite, avgComposite/float32(batchSize))
		fmt.Printf("    📊 Total:       %7.2f ms  (%.2f ms/frame)\n",
			avgTotal, avgTotal/float32(batchSize))
		fmt.Printf("    🚀 Throughput:  %.1f FPS\n",
			float32(batchSize)*1000.0/avgTotal)
	}

	fmt.Println("\n" + "=" + string(make([]byte, 70)))
	fmt.Println("✅ REAL AUDIO test complete!")
	fmt.Println("\n💡 Output:")
	fmt.Println("   - Check test_output/real_audio_batch_*.jpg for sample frames")
	fmt.Println("   - Audio processing times shown above confirm real audio pipeline")
}

// generateMockVisualFrames creates mock visual input data
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
	// Calculate audio samples needed for this batch
	// Each output frame needs a 16-frame context window (centered: frame-8 to frame+7)
	// For batch starting at frame F with size B:
	//   - First frame F needs audio from F-8 to F+7 (16 frames)
	//   - Last frame F+B-1 needs audio from F+B-9 to F+B+6 (16 frames)
	//   - Total range: F-8 to F+B+6 = B+14 frames
	// But for simplicity and to match SyncTalk, we'll send more: B+15 frames to be safe

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

			fmt.Printf("   📊 WAV: %d Hz, %d-bit, %d channel(s)\n",
				sampleRate, bitsPerSample, numChannels)

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
				fmt.Printf("   ↔️  Converted stereo to mono\n")
				return monoSamples, sampleRate, nil
			}

			return samples, sampleRate, nil

		} else {
			file.Seek(int64(chunkSize), io.SeekCurrent)
		}
	}

	return nil, 0, fmt.Errorf("no audio data found in WAV file")
}
