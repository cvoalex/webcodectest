package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	visualFrameSize = 6 * 320 * 320
	audioFrameSize  = 32 * 16 * 16
	outputFrameSize = 3 * 320 * 320
)

type BatchRequest struct {
	VisualFrames  []float32 `json:"visual_frames"`
	AudioFeatures []float32 `json:"audio_features"`
	BatchSize     int       `json:"batch_size"`
	StartFrameIdx int       `json:"start_frame_idx"`
}

type BatchResponse struct {
	OutputFrames    []float32 `json:"output_frames"`
	InferenceTimeMs float64   `json:"inference_time_ms"`
	Success         bool      `json:"success"`
	Error           string    `json:"error,omitempty"`
}

func loadBinaryFile(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	floatData := make([]float32, len(data)/4)
	for i := 0; i < len(floatData); i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		floatData[i] = math.Float32frombits(bits)
	}

	return floatData, nil
}

func testBatch(serverURL string, batchSize int, visualData, audioData []float32) error {
	// Extract frames for this batch
	batchVisual := visualData[:batchSize*visualFrameSize]

	req := BatchRequest{
		VisualFrames:  batchVisual,
		AudioFeatures: audioData[:audioFrameSize],
		BatchSize:     batchSize,
		StartFrameIdx: 0,
	}

	// Serialize to JSON
	reqBody, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Send request
	start := time.Now()
	resp, err := http.Post(serverURL+"/batch", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	totalTime := time.Since(start)

	// Parse response
	var batchResp BatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&batchResp); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	if !batchResp.Success {
		return fmt.Errorf("server error: %s", batchResp.Error)
	}

	// Validate output size
	expectedSize := batchSize * outputFrameSize
	if len(batchResp.OutputFrames) != expectedSize {
		return fmt.Errorf("unexpected output size: got %d, expected %d", len(batchResp.OutputFrames), expectedSize)
	}

	fmt.Printf("✅ Batch size %d: %.2fms inference, %.2fms total (%.2f FPS)\n",
		batchSize,
		batchResp.InferenceTimeMs,
		float64(totalTime.Milliseconds()),
		float64(batchSize*1000)/batchResp.InferenceTimeMs)

	return nil
}

func main() {
	serverURL := "http://localhost:8080"
	dataDir := "d:/Projects/webcodecstest/test_data_sanders_for_go"

	fmt.Println("================================================================================")
	fmt.Println("🧪 LIP SYNC SERVER TEST CLIENT")
	fmt.Println("================================================================================")
	fmt.Printf("   Server: %s\n", serverURL)
	fmt.Printf("   Data: %s\n\n", dataDir)

	// Load test data
	fmt.Println("💾 Loading test data...")
	visualPath := filepath.Join(dataDir, "visual_input.bin")
	audioPath := filepath.Join(dataDir, "audio_input.bin")

	visualData, err := loadBinaryFile(visualPath)
	if err != nil {
		fmt.Printf("❌ Failed to load visual data: %v\n", err)
		os.Exit(1)
	}

	audioData, err := loadBinaryFile(audioPath)
	if err != nil {
		fmt.Printf("❌ Failed to load audio data: %v\n", err)
		os.Exit(1)
	}

	numFrames := len(visualData) / visualFrameSize
	fmt.Printf("✅ Loaded %d frames\n\n", numFrames)

	// Test health endpoint
	fmt.Println("🏥 Testing health endpoint...")
	resp, err := http.Get(serverURL + "/health")
	if err != nil {
		fmt.Printf("❌ Server not responding: %v\n", err)
		os.Exit(1)
	}
	resp.Body.Close()
	fmt.Println("✅ Server is healthy\n")

	// Test different batch sizes
	fmt.Println("🚀 Testing different batch sizes...")
	fmt.Println("─────────────────────────────────────────────────────────")

	batchSizes := []int{1, 5, 10, 15, 20, 25}
	for _, size := range batchSizes {
		if size > numFrames {
			break
		}
		if err := testBatch(serverURL, size, visualData, audioData); err != nil {
			fmt.Printf("❌ Batch size %d failed: %v\n", size, err)
		}
		time.Sleep(100 * time.Millisecond)
	}

	fmt.Println("─────────────────────────────────────────────────────────")
	fmt.Println("\n✅ All tests completed!")
}
