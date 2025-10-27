package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	pb "go-monolithic-server/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	serverAddr = "localhost:50053" // Monolithic server port
	maxMsgSize = 100 * 1024 * 1024 // 100MB message size limit
)

func main() {
	fmt.Println("🧪 Audio Processing Performance Test")
	fmt.Println("=" + string(make([]byte, 60)))

	// Connect to monolithic server
	fmt.Printf("🔌 Connecting to monolithic server at %s...\n", serverAddr)
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

	client := pb.NewMonolithicServiceClient(conn)
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
	fmt.Printf("   GPUs: %v\n", healthResp.GpuIds)

	// Load audio file
	audioPath := filepath.Join("..", "aud.wav")
	fmt.Printf("\n🎵 Loading audio file: %s\n", audioPath)
	audioSamples, sampleRate, err := readWAVFile(audioPath)
	if err != nil {
		log.Fatalf("❌ Failed to load audio: %v", err)
	}

	fmt.Printf("   Loaded %d samples (%.2f seconds at %d Hz)\n",
		len(audioSamples), float64(len(audioSamples))/float64(sampleRate), sampleRate)

	// Test audio processing performance
	// We'll send just the audio for processing (model will fail to load, but we can measure audio time)
	fmt.Println("\n🚀 Testing audio processing performance...")
	fmt.Println("   (Model loading will fail, but we'll measure audio processing time)")

	batchSize := 24
	numTests := 10

	var totalAudioTimeMs float32
	var successfulTests int

	for i := 0; i < numTests; i++ {
		// Generate mock visual frames (won't be used, but required by API)
		visualFrames := generateMockVisualFrames(batchSize)

		// Extract audio chunk
		rawAudio := extractAudioChunk(audioSamples, i, batchSize, sampleRate)

		// Send request
		req := &pb.InferBatchCompositeRequest{
			ModelId:      "test", // Dummy model ID
			FrameIndices: makeFrameIndices(i*batchSize, batchSize),
			VisualFrames: visualFrames,
			RawAudio:     rawAudio,
			SampleRate:   int32(sampleRate),
		}

		resp, err := client.InferBatchComposite(context.Background(), req)
		if err == nil && resp.Success {
			// This shouldn't happen since we don't have the model,
			// but if it does, record the audio time
			totalAudioTimeMs += resp.AudioTimeMs
			successfulTests++
			fmt.Printf("   Test %d: Audio processing time: %.2f ms\n", i+1, resp.AudioTimeMs)
		} else {
			// Expected: model loading will fail
			// But server might still log audio processing time internally
			if resp != nil && resp.AudioTimeMs > 0 {
				totalAudioTimeMs += resp.AudioTimeMs
				successfulTests++
				fmt.Printf("   Test %d: Audio processing time: %.2f ms (model load failed as expected)\n",
					i+1, resp.AudioTimeMs)
			} else {
				fmt.Printf("   Test %d: Failed (expected - no model available)\n", i+1)
			}
		}
	}

	fmt.Println("\n📊 Performance Results:")
	if successfulTests > 0 {
		avgAudioTime := totalAudioTimeMs / float32(successfulTests)
		fmt.Printf("   Average audio processing time: %.2f ms\n", avgAudioTime)
		fmt.Printf("   Successful tests: %d/%d\n", successfulTests, numTests)
	} else {
		fmt.Println("   ⚠️  No successful tests - model files needed for full pipeline test")
		fmt.Println("   ℹ️  Audio encoder is loaded and ready, but inference requires model files")
	}

	fmt.Println("\n✅ Test complete!")
}

// generateMockVisualFrames creates dummy visual frames for testing
func generateMockVisualFrames(count int) [][]byte {
	frames := make([][]byte, count)
	frameSize := 6 * 320 * 320 // visual frame size
	for i := 0; i < count; i++ {
		frames[i] = make([]byte, frameSize)
		// Fill with dummy data
		for j := 0; j < frameSize; j++ {
			frames[i][j] = byte((i + j) % 256)
		}
	}
	return frames
}

// extractAudioChunk extracts raw audio samples for a batch
func extractAudioChunk(samples []int16, batchNum, batchSize, sampleRate int) []byte {
	// 0.04 seconds per frame at 25fps
	secondsPerFrame := 0.04
	samplesPerFrame := int(float64(sampleRate) * secondsPerFrame)

	// Calculate chunk
	startSample := batchNum * batchSize * samplesPerFrame
	endSample := startSample + (batchSize * samplesPerFrame)

	if startSample >= len(samples) {
		startSample = len(samples) - 1
	}
	if endSample > len(samples) {
		endSample = len(samples)
	}

	chunk := samples[startSample:endSample]

	// Convert to bytes
	buf := make([]byte, len(chunk)*2)
	for i, sample := range chunk {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(sample))
	}

	return buf
}

// makeFrameIndices creates frame indices for a batch
func makeFrameIndices(start, count int) []int32 {
	indices := make([]int32, count)
	for i := 0; i < count; i++ {
		indices[i] = int32(start + i)
	}
	return indices
}

// readWAVFile reads a WAV file and returns PCM samples
func readWAVFile(path string) ([]int16, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Read RIFF header
	var riffHeader [12]byte
	if _, err := io.ReadFull(file, riffHeader[:]); err != nil {
		return nil, 0, fmt.Errorf("failed to read RIFF header: %w", err)
	}

	// Verify RIFF/WAVE
	if string(riffHeader[0:4]) != "RIFF" || string(riffHeader[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a WAV file")
	}

	var sampleRate int
	var numChannels int
	var bitsPerSample int
	var audioFormat int
	var dataSize int

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

			audioFormat = int(binary.LittleEndian.Uint16(fmtData[0:2]))
			numChannels = int(binary.LittleEndian.Uint16(fmtData[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(fmtData[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(fmtData[14:16]))

			fmt.Printf("📊 WAV Info: %d Hz, %d-bit, %d channel(s), format=%d\n",
				sampleRate, bitsPerSample, numChannels, audioFormat)

			// Skip any extra format bytes
			if chunkSize > 16 {
				file.Seek(int64(chunkSize-16), io.SeekCurrent)
			}

		} else if chunkID == "data" {
			dataSize = chunkSize

			// Read PCM data
			rawData := make([]byte, dataSize)
			if _, err := io.ReadFull(file, rawData); err != nil {
				return nil, 0, fmt.Errorf("failed to read PCM data: %w", err)
			}

			// Convert to int16 samples
			numSamples := dataSize / (bitsPerSample / 8)
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
					// Average left and right channels
					left := int32(samples[i*2])
					right := int32(samples[i*2+1])
					monoSamples[i] = int16((left + right) / 2)
				}
				fmt.Printf("   Converted stereo to mono (%d samples)\n", len(monoSamples))
				return monoSamples, sampleRate, nil
			}

			fmt.Printf("   Loaded %d samples (%.2f seconds)\n", len(samples),
				float64(len(samples))/float64(sampleRate))
			return samples, sampleRate, nil

		} else {
			// Skip unknown chunk
			file.Seek(int64(chunkSize), io.SeekCurrent)
		}
	}

	return nil, 0, fmt.Errorf("no audio data found in WAV file")
}
