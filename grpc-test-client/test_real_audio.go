package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"time"

	pb "github.com/cvoalex/lipsync-proxy/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// WAV file header structure
type WAVHeader struct {
	ChunkID       [4]byte
	ChunkSize     uint32
	Format        [4]byte
	Subchunk1ID   [4]byte
	Subchunk1Size uint32
	AudioFormat   uint16
	NumChannels   uint16
	SampleRate    uint32
	ByteRate      uint32
	BlockAlign    uint16
	BitsPerSample uint16
	Subchunk2ID   [4]byte
	Subchunk2Size uint32
}

func loadWAVFile(filename string) ([]byte, *WAVHeader, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open WAV file: %v", err)
	}
	defer file.Close()

	// Read entire file
	data, err := io.ReadAll(file)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read WAV file: %v", err)
	}

	// Validate RIFF header
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, nil, fmt.Errorf("not a valid WAVE file")
	}

	// Parse fmt chunk to get audio format info
	var header WAVHeader
	copy(header.ChunkID[:], data[0:4])
	header.ChunkSize = binary.LittleEndian.Uint32(data[4:8])
	copy(header.Format[:], data[8:12])

	// Find fmt chunk
	pos := 12
	for pos < len(data)-8 {
		chunkID := string(data[pos : pos+4])
		chunkSize := binary.LittleEndian.Uint32(data[pos+4 : pos+8])

		if chunkID == "fmt " {
			if pos+8+int(chunkSize) > len(data) {
				return nil, nil, fmt.Errorf("invalid fmt chunk size")
			}
			header.AudioFormat = binary.LittleEndian.Uint16(data[pos+8 : pos+10])
			header.NumChannels = binary.LittleEndian.Uint16(data[pos+10 : pos+12])
			header.SampleRate = binary.LittleEndian.Uint32(data[pos+12 : pos+16])
			header.ByteRate = binary.LittleEndian.Uint32(data[pos+16 : pos+20])
			header.BlockAlign = binary.LittleEndian.Uint16(data[pos+20 : pos+22])
			header.BitsPerSample = binary.LittleEndian.Uint16(data[pos+22 : pos+24])
			break
		}
		pos += 8 + int(chunkSize)
	}

	// Find data chunk
	pos = 12
	var audioData []byte
	for pos < len(data)-8 {
		chunkID := string(data[pos : pos+4])
		chunkSize := binary.LittleEndian.Uint32(data[pos+4 : pos+8])

		if chunkID == "data" {
			header.Subchunk2Size = chunkSize
			if pos+8+int(chunkSize) > len(data) {
				// Use remaining data
				audioData = data[pos+8:]
			} else {
				audioData = data[pos+8 : pos+8+int(chunkSize)]
			}
			break
		}
		pos += 8 + int(chunkSize)
	}

	if audioData == nil {
		return nil, nil, fmt.Errorf("data chunk not found")
	}

	return audioData, &header, nil
}

func splitIntoChunks(audioData []byte, sampleRate uint32, chunkDurationMs int, channels int, bytesPerSample int) [][]byte {
	// Calculate bytes per chunk (40ms of audio)
	samplesPerChunk := int(sampleRate) * chunkDurationMs / 1000
	bytesPerChunk := samplesPerChunk * channels * bytesPerSample

	var chunks [][]byte
	for i := 0; i < len(audioData); i += bytesPerChunk {
		end := i + bytesPerChunk
		if end > len(audioData) {
			// Pad last chunk if needed
			chunk := make([]byte, bytesPerChunk)
			copy(chunk, audioData[i:])
			chunks = append(chunks, chunk)
			break
		}
		chunk := make([]byte, bytesPerChunk)
		copy(chunk, audioData[i:end])
		chunks = append(chunks, chunk)
	}

	return chunks
}

func testRealAudioBatch() {
	// Command line flags
	server := flag.String("server", "localhost:50051", "gRPC server address")
	model := flag.String("model", "sanders", "Model name")
	audioFile := flag.String("audio", "../aud.wav", "Path to WAV audio file")
	startFrame := flag.Int("start", 0, "Starting frame ID")
	maxFrames := flag.Int("max", 999999, "Maximum frames to generate (default: all)")
	flag.Parse()

	fmt.Println("\n" + "=================================================================")
	fmt.Println("🎵 REAL AUDIO BATCH INFERENCE TEST")
	fmt.Println("=================================================================")
	fmt.Printf("🔌 Server: %s\n", *server)
	fmt.Printf("📦 Model: %s\n", *model)
	fmt.Printf("🎵 Audio File: %s\n", *audioFile)
	fmt.Printf("📊 Start Frame: %d\n", *startFrame)
	fmt.Printf("📊 Max Frames: %d\n", *maxFrames)
	fmt.Println()

	// Load WAV file
	fmt.Printf("📂 Loading WAV file: %s\n", *audioFile)
	audioData, header, err := loadWAVFile(*audioFile)
	if err != nil {
		log.Fatalf("❌ Failed to load WAV file: %v", err)
	}

	fmt.Printf("✅ WAV file loaded successfully!\n")
	fmt.Printf("   Sample Rate: %d Hz\n", header.SampleRate)
	fmt.Printf("   Channels: %d\n", header.NumChannels)
	fmt.Printf("   Bits Per Sample: %d\n", header.BitsPerSample)
	fmt.Printf("   Audio Data Size: %d bytes (%.2f MB)\n", len(audioData), float64(len(audioData))/(1024*1024))

	audioDurationSec := float64(len(audioData)) / float64(header.ByteRate)
	fmt.Printf("   Duration: %.2f seconds\n", audioDurationSec)

	// Split into 40ms chunks
	fmt.Printf("\n🎵 Splitting audio into 40ms chunks...\n")
	bytesPerSample := int(header.BitsPerSample) / 8
	chunks := splitIntoChunks(audioData, header.SampleRate, 40, int(header.NumChannels), bytesPerSample)

	fmt.Printf("✅ Created %d audio chunks (40ms each)\n", len(chunks))
	fmt.Printf("   Chunk size: %d bytes each\n", len(chunks[0]))
	fmt.Printf("   Total chunks duration: %.2f seconds\n", float64(len(chunks))*0.04)

	// Calculate how many frames we can generate
	// With padding, we can generate as many frames as we have audio chunks!
	maxPossibleFrames := len(chunks)

	// Limit to user's max or what's possible
	frameCount := *maxFrames
	if frameCount > maxPossibleFrames {
		frameCount = maxPossibleFrames
		fmt.Printf("⚠️  Limiting to %d frames (audio duration constraint)\n", frameCount)
	}

	// Pad audio chunks at the boundaries
	// For each frame, we need 16 chunks: 8 before + current + 7 after
	// Total chunks needed: frameCount + 15
	audioChunksNeeded := frameCount + 15

	// Create padded chunks array
	paddedChunks := make([][]byte, audioChunksNeeded)

	// Fill with repeated edge chunks at boundaries
	for i := 0; i < audioChunksNeeded; i++ {
		// Map to actual chunk index
		chunkIdx := i - 8 + *startFrame

		if chunkIdx < 0 {
			// Repeat first chunk for frames at the beginning
			paddedChunks[i] = chunks[0]
		} else if chunkIdx >= len(chunks) {
			// Repeat last chunk for frames at the end
			paddedChunks[i] = chunks[len(chunks)-1]
		} else {
			paddedChunks[i] = chunks[chunkIdx]
		}
	}

	audioChunksForRequest := paddedChunks

	oldMethod := frameCount * 16
	savings := float64(oldMethod-audioChunksNeeded) / float64(oldMethod) * 100

	fmt.Printf("\n📊 Batch Configuration:\n")
	fmt.Printf("   Frames to generate: %d (frames %d-%d)\n", frameCount, *startFrame, *startFrame+frameCount-1)
	fmt.Printf("   Audio chunks needed: %d\n", audioChunksNeeded)
	fmt.Printf("   Old method: %d chunks\n", oldMethod)
	fmt.Printf("   Bandwidth savings: %.1f%%\n", savings)

	totalAudioSize := len(audioChunksForRequest) * len(audioChunksForRequest[0])
	fmt.Printf("   Total audio size: %.2f MB\n", float64(totalAudioSize)/(1024*1024))

	// Connect to gRPC server
	fmt.Printf("\n🔌 Connecting to gRPC server at %s...\n", *server)
	conn, err := grpc.Dial(*server,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(50*1024*1024)),
	)
	if err != nil {
		log.Fatalf("❌ Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewOptimizedLipSyncServiceClient(conn)
	fmt.Println("✅ Connected!")

	// Convert [][]byte to [][]byte for protobuf
	audioChunksBytes := make([][]byte, len(audioChunksForRequest))
	for i, chunk := range audioChunksForRequest {
		audioChunksBytes[i] = chunk
	}

	fmt.Printf("\n🎯 Sending audio batch request with REAL AUDIO...\n\n")

	// Send audio batch request
	batchStart := time.Now()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	req := &pb.BatchInferenceWithAudioRequest{
		ModelName:    *model,
		StartFrameId: int32(*startFrame),
		FrameCount:   int32(frameCount),
		AudioChunks:  audioChunksBytes,
	}

	resp, err := client.GenerateBatchWithAudio(ctx, req)
	if err != nil {
		log.Fatalf("❌ Audio batch inference failed: %v", err)
	}

	batchDuration := time.Since(batchStart)

	// Print results
	fmt.Println("======================================================================")
	fmt.Println("📊 REAL AUDIO BATCH RESULTS")
	fmt.Println("======================================================================")
	fmt.Printf("\n✅ Received %d responses\n\n", len(resp.Responses))

	totalSize := 0
	successCount := 0

	for i, r := range resp.Responses {
		frameId := *startFrame + i
		if r.Success {
			successCount++
			size := len(r.PredictionData)
			totalSize += size

			fmt.Printf("  ✅ Frame %d: %dms (%.2fms inference) - %d bytes (%.2f KB)\n",
				frameId,
				r.ProcessingTimeMs,
				r.InferenceTimeMs,
				size,
				float64(size)/1024.0)

			// Save frame to file
			filename := fmt.Sprintf("real_audio_frame_%d.jpg", frameId)
			err := os.WriteFile(filename, r.PredictionData, 0644)
			if err != nil {
				fmt.Printf("     ⚠️  Failed to save: %v\n", err)
			} else {
				fmt.Printf("     💾 Saved: %s\n", filename)
			}
		} else {
			errorMsg := "unknown error"
			if r.Error != nil {
				errorMsg = *r.Error
			}
			fmt.Printf("  ❌ Frame %d: ERROR - %s\n", frameId, errorMsg)
		}
	}

	fmt.Println("\n======================================================================")
	fmt.Println("📈 PERFORMANCE SUMMARY")
	fmt.Println("======================================================================")
	fmt.Printf("\n🎯 Batch Stats:\n")
	fmt.Printf("   Total Time: %.2fms\n", float64(batchDuration.Milliseconds()))
	fmt.Printf("   Server Total: %dms\n", resp.TotalProcessingTimeMs)
	fmt.Printf("   Server Avg: %.2fms per frame\n", resp.AvgFrameTimeMs)
	fmt.Printf("   Success Rate: %d/%d frames\n", successCount, frameCount)

	if batchDuration.Seconds() > 0 {
		fps := float64(frameCount) / batchDuration.Seconds()
		fmt.Printf("   Throughput: %.2f FPS\n", fps)
	}

	if totalSize > 0 {
		fmt.Printf("   Frame Data: %d bytes (%.2f MB)\n", totalSize, float64(totalSize)/(1024*1024))
		dataRate := float64(totalSize) / (1024 * 1024) / batchDuration.Seconds()
		fmt.Printf("   Data Rate: %.2f MB/s\n", dataRate)
	}

	// Bandwidth comparison
	fmt.Println("\n======================================================================")
	fmt.Println("📊 BANDWIDTH ANALYSIS")
	fmt.Println("======================================================================")
	fmt.Printf("\n🎵 Audio Transfer:\n")
	fmt.Printf("   Chunks Sent: %d\n", audioChunksNeeded)
	fmt.Printf("   Old Method: %d chunks\n", oldMethod)
	fmt.Printf("   Savings: %d chunks (%.1f%%)\n", oldMethod-audioChunksNeeded, savings)
	fmt.Printf("   Audio Size: %.2f MB\n", float64(totalAudioSize)/(1024*1024))

	oldAudioSize := oldMethod * len(chunks[0])
	fmt.Printf("   Old Method Size: %.2f MB\n", float64(oldAudioSize)/(1024*1024))
	fmt.Printf("   Bandwidth Saved: %.2f MB (%.1f%%)\n",
		float64(oldAudioSize-totalAudioSize)/(1024*1024), savings)

	fmt.Println("\n======================================================================")
	fmt.Printf("\n🎉 SUCCESS! Generated %d lip-sync frames from real audio!\n", successCount)
	fmt.Println("======================================================================")
	fmt.Println()

	// Assemble video from frames
	if successCount > 0 {
		fmt.Println("\n🎬 Assembling video from frames...")
		videoFile := "lipsync_output.mp4"
		err := assembleVideo(*audioFile, videoFile, *startFrame, frameCount)
		if err != nil {
			fmt.Printf("⚠️  Video assembly failed: %v\n", err)
			fmt.Println("   (Frames are still saved individually)")
		} else {
			fmt.Printf("✅ Video saved: %s\n", videoFile)
		}
	}
}

func assembleVideo(audioFile string, outputFile string, startFrame int, frameCount int) error {
	// Create a file list for ffmpeg
	listFile := "frames_list.txt"
	f, err := os.Create(listFile)
	if err != nil {
		return fmt.Errorf("failed to create frame list: %w", err)
	}

	// Write frame paths
	for i := startFrame; i < startFrame+frameCount; i++ {
		framePath := fmt.Sprintf("real_audio_frame_%d.jpg", i)
		// Each frame is 40ms (25 fps)
		fmt.Fprintf(f, "file '%s'\n", framePath)
		fmt.Fprintf(f, "duration 0.04\n")
	}
	// Add last frame again without duration for ffmpeg
	lastFramePath := fmt.Sprintf("real_audio_frame_%d.jpg", startFrame+frameCount-1)
	fmt.Fprintf(f, "file '%s'\n", lastFramePath)
	f.Close()
	defer os.Remove(listFile)

	// Use ffmpeg to combine frames and audio
	// ffmpeg -f concat -safe 0 -i frames_list.txt -i audio.wav -c:v libx264 -pix_fmt yuv420p -c:a aac output.mp4
	fmt.Println("   Running FFmpeg...")

	cmd := fmt.Sprintf("ffmpeg -y -f concat -safe 0 -i %s -i %s -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -c:a aac -b:a 128k %s",
		listFile, audioFile, outputFile)

	// Execute ffmpeg
	result := exec.Command("cmd", "/C", cmd)
	output, err := result.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ffmpeg failed: %w\nOutput: %s", err, string(output))
	}

	return nil
}

func main() {
	testRealAudioBatch()
}
