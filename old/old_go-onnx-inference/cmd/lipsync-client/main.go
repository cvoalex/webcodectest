package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type BatchRequest struct {
	VisualFrames  []float32 `json:"visual_frames"`   // Flattened: [batch_size * 6*320*320]
	AudioFeatures []float32 `json:"audio_features"`  // [32*16*16]
	BatchSize     int       `json:"batch_size"`      // 1-25
	StartFrameIdx int       `json:"start_frame_idx"` // For crop rectangle lookup
}

type BatchResponse struct {
	OutputFrames    []float32 `json:"output_frames"` // Flattened: [batch_size * 3*320*320]
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

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🧪 LipSync Server Test Client")
	fmt.Println("================================================================================")

	// Load test data
	dataDir := "../../../test_data_sanders_for_go"

	fmt.Printf("\n📦 Loading test data from: %s\n", dataDir)

	visualData, err := loadBinaryFile(filepath.Join(dataDir, "visual_input.bin"))
	if err != nil {
		fmt.Printf("❌ Error loading visual data: %v\n", err)
		return
	}

	audioData, err := loadBinaryFile(filepath.Join(dataDir, "audio_input.bin"))
	if err != nil {
		fmt.Printf("❌ Error loading audio data: %v\n", err)
		return
	}

	visualFrameSize := 6 * 320 * 320
	audioFrameSize := 32 * 16 * 16
	numFrames := len(visualData) / visualFrameSize

	fmt.Printf("✅ Loaded %d frames\n", numFrames)
	fmt.Printf("   Visual: %.2f MB\n", float64(len(visualData)*4)/(1024*1024))
	fmt.Printf("   Audio: %.2f MB\n", float64(len(audioData)*4)/(1024*1024))

	// Test different batch sizes
	testBatchSizes := []int{1, 5, 10, 25}

	for _, batchSize := range testBatchSizes {
		if batchSize > numFrames {
			continue
		}

		fmt.Printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("🔬 Testing batch size: %d\n", batchSize)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

		// Prepare flattened batch
		visualFramesBatch := make([]float32, batchSize*visualFrameSize)
		for i := 0; i < batchSize; i++ {
			start := i * visualFrameSize
			copy(visualFramesBatch[i*visualFrameSize:], visualData[start:start+visualFrameSize])
		}

		// Use the first audio window (in real app, client selects appropriate window)
		audioWindow := audioData[:audioFrameSize]

		request := BatchRequest{
			VisualFrames:  visualFramesBatch,
			AudioFeatures: audioWindow,
			BatchSize:     batchSize,
			StartFrameIdx: 0,
		}

		// Send request
		requestStart := time.Now()
		jsonData, err := json.Marshal(request)
		if err != nil {
			fmt.Printf("❌ Error marshaling request: %v\n", err)
			continue
		}

		fmt.Printf("   📤 Sending request (%.2f MB)...\n", float64(len(jsonData))/(1024*1024))

		resp, err := http.Post("http://localhost:8080/batch", "application/json", bytes.NewBuffer(jsonData))
		if err != nil {
			fmt.Printf("❌ Error sending request: %v\n", err)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			fmt.Printf("❌ Error reading response: %v\n", err)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			fmt.Printf("❌ Server returned status %d: %s\n", resp.StatusCode, string(body))
			continue
		}

		var response BatchResponse
		if err := json.Unmarshal(body, &response); err != nil {
			fmt.Printf("❌ Error unmarshaling response: %v\n", err)
			fmt.Printf("Response body: %s\n", string(body))
			continue
		}

		requestTime := time.Since(requestStart)

		if !response.Success || response.Error != "" {
			fmt.Printf("❌ Server error: %s\n", response.Error)
			continue
		}

		// Print results
		outputFrameSize := 3 * 320 * 320
		numOutputFrames := len(response.OutputFrames) / outputFrameSize

		fmt.Printf("   ✅ Inference successful\n")
		fmt.Printf("      Batch size: %d\n", batchSize)
		fmt.Printf("      Server inference time: %.2fms (%.2fms/frame)\n",
			response.InferenceTimeMs, response.InferenceTimeMs/float64(batchSize))
		fmt.Printf("      Round-trip time: %.2fms\n", float64(requestTime.Milliseconds()))
		fmt.Printf("      Network overhead: %.2fms\n", float64(requestTime.Milliseconds())-response.InferenceTimeMs)
		fmt.Printf("      Output frames: %d\n", numOutputFrames)
		fmt.Printf("      Output data size: %.2f MB\n", float64(len(response.OutputFrames)*4)/(1024*1024))

		fps := float64(batchSize) / (response.InferenceTimeMs / 1000.0)
		fpsWithNetwork := float64(batchSize) / (float64(requestTime.Milliseconds()) / 1000.0)
		fmt.Printf("      📊 Server FPS: %.2f\n", fps)
		fmt.Printf("      📊 End-to-end FPS: %.2f (including network)\n", fpsWithNetwork)
	}

	fmt.Println("\n✅ All tests completed!")
}
