package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	pb "go-compositing-server/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	serverAddr = "localhost:50052"
	modelID    = "sanders"         // Change to your model ID
	batchSize  = 24                // Batch size 24 (max is 25 on current server)
	numBatches = 5                 // Run 5 batches for testing
	maxMsgSize = 100 * 1024 * 1024 // 100MB message size limit
)

func main() {
	fmt.Println("🧪 Compositing Server Test Client")
	fmt.Println("=" + string(make([]byte, 60)))

	// Connect to compositing server
	fmt.Printf("🔌 Connecting to compositing server at %s...\n", serverAddr)
	conn, err := grpc.NewClient(
		serverAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewCompositingServiceClient(conn)
	fmt.Println("✅ Connected successfully")

	// Check server health
	fmt.Println("\n📊 Checking server health...")
	healthResp, err := client.Health(context.Background(), &pb.HealthRequest{})
	if err != nil {
		log.Fatalf("❌ Health check failed: %v", err)
	}
	if healthResp.Healthy {
		fmt.Println("✅ Server Status: Healthy")
	} else {
		fmt.Println("⚠️  Server Status: Unhealthy")
	}
	fmt.Printf("   Loaded Models: %d/%d\n", healthResp.LoadedModels, healthResp.MaxModels)
	fmt.Printf("   Inference Server: %s (%v)\n", healthResp.InferenceServerUrl, healthResp.InferenceServerHealthy)

	// Prepare output directory
	os.MkdirAll("test_output", 0755)
	fmt.Println("\n📁 Output directory: test_output/")

	// Load audio file
	audioPath := filepath.Join("..", "aud.wav")
	fmt.Printf("\n🎵 Loading audio file: %s\n", audioPath)
	audioSamples, sampleRate, err := readWAVFile(audioPath)
	if err != nil {
		log.Fatalf("❌ Failed to load audio: %v", err)
	}

	// Verify sample rate (should be 16kHz for the pipeline)
	if sampleRate != 16000 {
		log.Printf("⚠️  Warning: Audio is %d Hz, but pipeline expects 16kHz. Resampling required!", sampleRate)
		// For now, we'll continue but results may be incorrect
	}

	// Run test batches
	fmt.Printf("\n🚀 Running %d batches (batch_size=%d)...\n", numBatches, batchSize)
	fmt.Println(string(make([]byte, 60)))

	var totalInferenceMs float32
	var totalCompositeMs float32
	var totalTimeMs float32
	var totalFrames int
	startTime := time.Now()

	for batchNum := 0; batchNum < numBatches; batchNum++ {
		frameIdx := batchNum * batchSize

		// Generate mock visual frames
		visualFrames := generateMockVisualFrames(batchSize)

		// Extract audio chunk for this batch
		rawAudio := extractAudioChunk(audioSamples, batchNum, batchSize, sampleRate)

		// Call compositing server with raw audio
		req := &pb.CompositeBatchRequest{
			ModelId:       modelID,
			VisualFrames:  visualFrames,
			RawAudio:      rawAudio, // Use raw audio instead of audio_features
			BatchSize:     int32(batchSize),
			StartFrameIdx: int32(frameIdx),
		}

		batchStart := time.Now()
		resp, err := client.InferBatchComposite(context.Background(), req)
		batchDuration := time.Since(batchStart).Milliseconds()

		if err != nil {
			log.Fatalf("❌ Batch %d failed: %v", batchNum, err)
		}

		if !resp.Success {
			log.Fatalf("❌ Batch %d returned success=false: %s", batchNum, resp.Error)
		}

		// Accumulate stats
		totalInferenceMs += resp.InferenceTimeMs
		totalCompositeMs += resp.CompositeTimeMs
		totalTimeMs += resp.TotalTimeMs
		totalFrames += len(resp.CompositedFrames)

		// Calculate overhead
		overhead := resp.TotalTimeMs - resp.InferenceTimeMs

		fmt.Printf("Batch %d/%d: GPU=%d, frames=%d\n",
			batchNum+1, numBatches, resp.GpuId, len(resp.CompositedFrames))
		if resp.AudioProcessingMs > 0 {
			fmt.Printf("  🎵 Audio:       %6.2f ms\n", resp.AudioProcessingMs)
		}
		fmt.Printf("  ⚡ Inference:   %6.2f ms\n", resp.InferenceTimeMs)
		fmt.Printf("  🎨 Compositing: %6.2f ms\n", resp.CompositeTimeMs)
		fmt.Printf("  📊 Total:       %6.2f ms (actual: %d ms)\n", resp.TotalTimeMs, batchDuration)
		fmt.Printf("  📈 Overhead:    %6.2f ms (%.1f%%)\n",
			overhead, (overhead/resp.InferenceTimeMs)*100)

		// Save all frames from this batch
		for i, frameData := range resp.CompositedFrames {
			filename := fmt.Sprintf("test_output/batch_%d_frame_%d.jpg", batchNum+1, frameIdx+i)
			err = os.WriteFile(filename, frameData, 0644)
			if err != nil {
				log.Printf("⚠️  Failed to save frame %d: %v", frameIdx+i, err)
			}
		}

		// Report first frame save
		if len(resp.CompositedFrames) > 0 {
			fmt.Printf("  💾 Saved %d frames to test_output/ (%d bytes avg)\n",
				len(resp.CompositedFrames), len(resp.CompositedFrames[0]))
		}

		fmt.Println()
	}

	totalDuration := time.Since(startTime)

	// Print summary
	fmt.Println(string(make([]byte, 60)))
	fmt.Println("📈 PERFORMANCE SUMMARY")
	fmt.Println(string(make([]byte, 60)))
	fmt.Printf("Total frames processed:  %d\n", totalFrames)
	fmt.Printf("Total duration:          %.2f seconds\n", totalDuration.Seconds())
	fmt.Printf("\n⚡ Average Inference:      %.2f ms\n", totalInferenceMs/float32(numBatches))
	fmt.Printf("🎨 Average Compositing:   %.2f ms\n", totalCompositeMs/float32(numBatches))
	fmt.Printf("📊 Average Total:         %.2f ms\n", totalTimeMs/float32(numBatches))

	avgInference := totalInferenceMs / float32(numBatches)
	avgTotal := totalTimeMs / float32(numBatches)
	avgOverhead := avgTotal - avgInference
	overheadPct := (avgOverhead / avgInference) * 100

	fmt.Printf("\n📈 Separation Overhead:   %.2f ms (%.1f%% of inference time)\n",
		avgOverhead, overheadPct)

	// Throughput calculations
	fps := float64(totalFrames) / totalDuration.Seconds()
	fmt.Printf("\n🚀 Throughput:            %.1f FPS\n", fps)
	fmt.Printf("   Frames per batch:      %d\n", batchSize)
	fmt.Printf("   Batches per second:    %.1f\n", float64(numBatches)/totalDuration.Seconds())

	// Verdict
	fmt.Println("\n" + string(make([]byte, 60)))
	if avgOverhead < 5.0 {
		fmt.Println("✅ SUCCESS: Overhead < 5ms target!")
	} else if avgOverhead < 10.0 {
		fmt.Println("⚠️  WARNING: Overhead is higher than 5ms target but acceptable")
	} else {
		fmt.Println("❌ CONCERN: Overhead exceeds 10ms, investigation recommended")
	}

	fmt.Println("\n💡 Next Steps:")
	fmt.Println("   1. Check test_output/ for sample frames")
	fmt.Println("   2. Verify compositing quality")
	fmt.Println("   3. Run with different batch sizes")
	fmt.Println("   4. Test with multiple concurrent clients")
}

// generateMockVisualFrames creates random visual frame data
// Format: 6 frames of 320x320 float32 values
func generateMockVisualFrames(batchSize int) []byte {
	numFrames := 6
	height := 320
	width := 320

	totalFloats := batchSize * numFrames * height * width
	data := make([]byte, totalFloats*4) // 4 bytes per float32

	// Generate random float32 values between -1 and 1
	for i := 0; i < totalFloats; i++ {
		value := float32(rand.Float64()*2.0 - 1.0) // Range: -1 to 1
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}

	return data
}

// readWAVFile reads a WAV file and returns the raw PCM audio samples as float32
func readWAVFile(filename string) ([]float32, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open WAV file: %v", err)
	}
	defer file.Close()

	// Read WAV header (44 bytes for standard PCM WAV)
	header := make([]byte, 44)
	_, err = io.ReadFull(file, header)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to read WAV header: %v", err)
	}

	// Verify RIFF header
	if string(header[0:4]) != "RIFF" {
		return nil, 0, fmt.Errorf("not a valid WAV file (missing RIFF)")
	}

	// Verify WAVE format
	if string(header[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a valid WAV file (missing WAVE)")
	}

	// Read audio format (offset 20-21, should be 1 for PCM)
	audioFormat := binary.LittleEndian.Uint16(header[20:22])
	if audioFormat != 1 && audioFormat != 3 { // 1 = PCM, 3 = IEEE float
		return nil, 0, fmt.Errorf("unsupported audio format: %d (only PCM and IEEE float supported)", audioFormat)
	}

	// Read sample rate (offset 24-27)
	sampleRate := int(binary.LittleEndian.Uint32(header[24:28]))

	// Read bits per sample (offset 34-35)
	bitsPerSample := binary.LittleEndian.Uint16(header[34:36])

	// Read number of channels (offset 22-23)
	numChannels := binary.LittleEndian.Uint16(header[22:24])

	fmt.Printf("📊 WAV Info: %d Hz, %d-bit, %d channel(s), format=%d\n",
		sampleRate, bitsPerSample, numChannels, audioFormat)

	// Read the rest of the file (audio data)
	audioData, err := io.ReadAll(file)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to read audio data: %v", err)
	}

	// Convert to float32 based on format
	var samples []float32

	if audioFormat == 3 { // IEEE float
		// Convert byte array to float32 array
		numSamples := len(audioData) / 4
		samples = make([]float32, numSamples)
		for i := 0; i < numSamples; i++ {
			bits := binary.LittleEndian.Uint32(audioData[i*4 : (i+1)*4])
			samples[i] = math.Float32frombits(bits)
		}
	} else if bitsPerSample == 16 {
		// Convert 16-bit PCM to float32 (range -1 to 1)
		numSamples := len(audioData) / 2
		samples = make([]float32, numSamples)
		for i := 0; i < numSamples; i++ {
			sample := int16(binary.LittleEndian.Uint16(audioData[i*2 : (i+1)*2]))
			samples[i] = float32(sample) / 32768.0 // Normalize to -1 to 1
		}
	} else {
		return nil, 0, fmt.Errorf("unsupported bits per sample: %d", bitsPerSample)
	}

	// If stereo, convert to mono by averaging channels
	if numChannels == 2 {
		monoSamples := make([]float32, len(samples)/2)
		for i := 0; i < len(monoSamples); i++ {
			monoSamples[i] = (samples[i*2] + samples[i*2+1]) / 2.0
		}
		samples = monoSamples
		fmt.Printf("   Converted stereo to mono (%d samples)\n", len(samples))
	}

	fmt.Printf("   Loaded %d samples (%.2f seconds)\n", len(samples), float64(len(samples))/float64(sampleRate))

	return samples, sampleRate, nil
}

// extractAudioChunk extracts a chunk of audio for a specific batch
// For batch size 24, we need 10,240 samples (640ms at 16kHz)
func extractAudioChunk(audioSamples []float32, batchNum int, batchSize int, sampleRate int) []byte {
	// Calculate samples needed for this batch
	// Each frame is 40ms (640 samples at 16kHz)
	samplesPerFrame := sampleRate / 25 // 25 fps -> 40ms per frame
	samplesNeeded := batchSize * samplesPerFrame

	startSample := batchNum * samplesNeeded
	endSample := startSample + samplesNeeded

	// Handle end of audio by looping or padding
	var chunk []float32
	if endSample <= len(audioSamples) {
		chunk = audioSamples[startSample:endSample]
	} else if startSample < len(audioSamples) {
		// Partial chunk, pad with zeros
		chunk = make([]float32, samplesNeeded)
		available := len(audioSamples) - startSample
		copy(chunk, audioSamples[startSample:])
		// Rest is already zeros
		fmt.Printf("   ⚠️  Padded with %d zero samples\n", samplesNeeded-available)
	} else {
		// Loop back to beginning
		startSample = startSample % len(audioSamples)
		endSample = startSample + samplesNeeded
		if endSample <= len(audioSamples) {
			chunk = audioSamples[startSample:endSample]
		} else {
			// Wrap around
			chunk = make([]float32, samplesNeeded)
			copy(chunk, audioSamples[startSample:])
			copy(chunk[len(audioSamples)-startSample:], audioSamples[0:endSample-len(audioSamples)])
		}
	}

	// Convert float32 slice to bytes
	data := make([]byte, len(chunk)*4)
	for i, sample := range chunk {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(sample))
	}

	return data
}

// generateMockAudioFeatures creates random audio feature data (DEPRECATED - use raw_audio)
// Format: 32 features of 16x16 float32 values (for entire batch, not per frame)
// Kept for backward compatibility testing
/*
func generateMockAudioFeatures(batchSize int) []byte {
	numFeatures := 32
	height := 16
	width := 16

	// Audio features are for the whole batch, not per frame
	totalFloats := numFeatures * height * width
	data := make([]byte, totalFloats*4) // 4 bytes per float32

	// Generate random float32 values between -1 and 1
	for i := 0; i < totalFloats; i++ {
		value := float32(rand.Float64()*2.0 - 1.0) // Range: -1 to 1
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}

	return data
}
*/
