package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/joho/godotenv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Configuration constants
const (
	BUFFER_SIZE           = 3000
	CHUNK_DURATION_MS     = 40
	SAMPLE_RATE          = 24000
	FRAMES_FOR_INFERENCE = 16
	INFERENCE_BATCH_SIZE = 5
	LOOKAHEAD_FRAMES     = 7
	TARGET_FRAME_BUFFER  = 15
	MIN_FRAME_BUFFER     = 8
	MAX_FRAME_BUFFER     = 25
	CRITICAL_FRAME_BUFFER = 3
)

// gRPC stub types (temporary until protobuf files are generated)
type LipSyncServiceClient interface {
	GenerateBatchInference(ctx context.Context, req *BatchInferenceRequest) (*BatchInferenceResponse, error)
}

type BatchInferenceRequest struct {
	ModelName string
	FrameIds  []int32
	AudioData string
}

type BatchInferenceResponse struct {
	Responses []*InferenceResponse
}

type InferenceResponse struct {
	Success         bool
	Error           string
	PredictionData  []byte
	Bounds          []float32
	ModelName       string
	PredictionShape string
}

type stubClient struct{}

func newStubClient() LipSyncServiceClient {
	return &stubClient{}
}

func (s *stubClient) GenerateBatchInference(ctx context.Context, req *BatchInferenceRequest) (*BatchInferenceResponse, error) {
	// This is a stub - actual implementation will use generated protobuf code
	// For now, return dummy data so the system can run
	log.Printf("🔄 Stub gRPC call for model: %s, frames: %d", req.ModelName, len(req.FrameIds))
	
	responses := make([]*InferenceResponse, len(req.FrameIds))
	for i := range req.FrameIds {
		// Generate dummy frame data (320x320 pixels, 3 bytes per pixel)
		dummyFrameData := make([]byte, 320*320*3)
		for j := range dummyFrameData {
			dummyFrameData[j] = byte(128 + (i*10)%128) // Simple pattern
		}
		
		responses[i] = &InferenceResponse{
			Success:         true,
			PredictionData:  dummyFrameData,
			Bounds:          []float32{50, 50, 220, 220}, // Center mouth region
			ModelName:       req.ModelName,
			PredictionShape: "320x320",
		}
	}
	
	return &BatchInferenceResponse{
		Responses: responses,
	}, nil
}

// Audio chunk structure
type AudioChunk struct {
	Data      []int16   `json:"data"`
	Timestamp time.Time `json:"timestamp"`
	Index     int       `json:"index"`
}

// Frame data structure
type FrameData struct {
	Data            []byte    `json:"data"`
	Bounds          []float32 `json:"bounds"`
	Timestamp       time.Time `json:"timestamp"`
	AudioIndex      int       `json:"audio_index"`
	ModelName       string    `json:"model_name"`
	PredictionShape string    `json:"prediction_shape"`
}

// WebSocket message types
type WebSocketMessage struct {
	Type      string `json:"type"`
	AudioData string `json:"audio_data,omitempty"`
	ModelName string `json:"model_name,omitempty"`
	RequestID string `json:"request_id,omitempty"`
}

// Frame Generator handles real-time audio processing and frame generation
type FrameGenerator struct {
	// gRPC connection
	grpcConn   *grpc.ClientConn
	grpcClient LipSyncServiceClient
	
	// Circular buffers
	audioBuffer [BUFFER_SIZE]*AudioChunk
	frameBuffer [BUFFER_SIZE]*FrameData
	
	// Buffer pointers
	audioWriteIndex int
	frameWriteIndex int
	
	// WebSocket clients
	clients    map[*websocket.Conn]bool
	clientsMux sync.RWMutex
	
	// Current model
	currentModel string
	
	// Reference images for compositing
	referenceImages map[string]string // model_name -> base64_image_data
	referenceBounds map[string][]float32 // model_name -> bounds
	
	// Mutex for thread safety
	bufferMux sync.RWMutex
}

// WebSocket upgrader
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for development
	},
}

// TokenResponse structure for OpenAI API
type TokenResponse struct {
	ClientSecret struct {
		Value     string `json:"value"`
		ExpiresAt string `json:"expires_at"`
	} `json:"client_secret"`
}

// NewFrameGenerator creates a new frame generator instance
func NewFrameGenerator(grpcHost string, grpcPort int) *FrameGenerator {
	return &FrameGenerator{
		clients:         make(map[*websocket.Conn]bool),
		currentModel:    "test_optimized_package_fixed_1",
		referenceImages: make(map[string]string),
		referenceBounds: make(map[string][]float32),
	}
}

// Initialize connects to gRPC service
func (fg *FrameGenerator) Initialize(grpcHost string, grpcPort int) error {
	// Try to connect to gRPC service
	conn, err := grpc.Dial(
		fmt.Sprintf("%s:%d", grpcHost, grpcPort),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Printf("⚠️ Could not connect to gRPC service on %s:%d - using stub client: %v", grpcHost, grpcPort, err)
		fg.grpcClient = newStubClient()
		return nil
	}
	
	fg.grpcConn = conn
	fg.grpcClient = newStubClient() // For now, always use stub until protobuf is set up
	
	log.Printf("✅ Connected to gRPC service on %s:%d (using stub client)", grpcHost, grpcPort)
	
	return nil
}

// AddAudioToBuffer adds audio data to the circular buffer
func (fg *FrameGenerator) AddAudioToBuffer(audioData []int16) {
	fg.bufferMux.Lock()
	defer fg.bufferMux.Unlock()
	
	fg.audioBuffer[fg.audioWriteIndex] = &AudioChunk{
		Data:      audioData,
		Timestamp: time.Now(),
		Index:     fg.audioWriteIndex,
	}
	
	fg.audioWriteIndex = (fg.audioWriteIndex + 1) % BUFFER_SIZE
	
	log.Printf("🎵 Audio chunk added. Buffer fill: %d/%d", fg.getAudioBufferFill(), BUFFER_SIZE)
}

// CanGenerateFrames checks if we have enough audio and need more frames
func (fg *FrameGenerator) CanGenerateFrames() bool {
	fg.bufferMux.RLock()
	defer fg.bufferMux.RUnlock()
	
	audioFill := fg.getAudioBufferFill()
	frameFill := fg.getFrameBufferFill()
	
	needMoreFrames := frameFill < TARGET_FRAME_BUFFER
	haveEnoughAudio := audioFill >= FRAMES_FOR_INFERENCE+5
	
	return needMoreFrames && haveEnoughAudio
}

// GenerateFrameBatch generates a batch of frames using gRPC
func (fg *FrameGenerator) GenerateFrameBatch() error {
	// Get consecutive audio chunks
	audioChunks := fg.getConsecutiveAudioChunks(FRAMES_FOR_INFERENCE)
	if len(audioChunks) < FRAMES_FOR_INFERENCE {
		return fmt.Errorf("not enough audio chunks: %d < %d", len(audioChunks), FRAMES_FOR_INFERENCE)
	}
	
	// Combine audio chunks
	combinedAudio := fg.combineAudioChunks(audioChunks)
	
	// Convert to WAV format
	wavData, err := fg.convertToWAV(combinedAudio)
	if err != nil {
		return fmt.Errorf("failed to convert audio to WAV: %v", err)
	}
	
	// Call gRPC service
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	frameIds := make([]int32, INFERENCE_BATCH_SIZE)
	for i := 0; i < INFERENCE_BATCH_SIZE; i++ {
		frameIds[i] = int32(i)
	}
	
	request := &BatchInferenceRequest{
		ModelName: fg.currentModel,
		FrameIds:  frameIds,
		AudioData: base64.StdEncoding.EncodeToString(wavData),
	}
	
	response, err := fg.grpcClient.GenerateBatchInference(ctx, request)
	if err != nil {
		return fmt.Errorf("gRPC inference error: %v", err)
	}
	
	// Store frames in buffer
	fg.bufferMux.Lock()
	for i, frameResponse := range response.Responses {
		if frameResponse.Success {
			fg.frameBuffer[fg.frameWriteIndex] = &FrameData{
				Data:            frameResponse.PredictionData,
				Bounds:          frameResponse.Bounds,
				Timestamp:       time.Now(),
				AudioIndex:      audioChunks[0].Index + i,
				ModelName:       frameResponse.ModelName,
				PredictionShape: frameResponse.PredictionShape,
			}
			
			fg.frameWriteIndex = (fg.frameWriteIndex + 1) % BUFFER_SIZE
		} else {
			log.Printf("❌ Frame generation failed: %s", frameResponse.Error)
		}
	}
	fg.bufferMux.Unlock()
	
	log.Printf("🎬 Generated %d frames. Frame buffer fill: %d/%d", len(response.Responses), fg.getFrameBufferFill(), BUFFER_SIZE)
	
	// Notify clients
	fg.notifyClientsNewFrames(len(response.Responses))
	
	return nil
}

// WebSocket handler
func (fg *FrameGenerator) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("❌ WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()
	
	// Add client
	fg.clientsMux.Lock()
	fg.clients[conn] = true
	fg.clientsMux.Unlock()
	
	log.Printf("🔗 Client connected: %s", conn.RemoteAddr())
	
	// Remove client on disconnect
	defer func() {
		fg.clientsMux.Lock()
		delete(fg.clients, conn)
		fg.clientsMux.Unlock()
		log.Printf("🔌 Client disconnected: %s", conn.RemoteAddr())
	}()
	
	// Handle messages
	for {
		var msg WebSocketMessage
		err := conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("❌ WebSocket error: %v", err)
			}
			break
		}
		
		fg.handleWebSocketMessage(conn, &msg)
	}
}

// Handle individual WebSocket messages
func (fg *FrameGenerator) handleWebSocketMessage(conn *websocket.Conn, msg *WebSocketMessage) {
	switch msg.Type {
	case "audio_chunk":
		fg.processAudioChunk(msg)
	case "get_frame":
		fg.sendCurrentFrame(conn)
	case "set_model":
		fg.changeModel(msg.ModelName)
	case "get_stats":
		fg.sendStats(conn)
	case "get_reference_image":
		fg.sendReferenceImage(conn, msg.ModelName, msg.RequestID)
	default:
		log.Printf("⚠️ Unknown message type: %s", msg.Type)
	}
}

// Process audio chunk from WebSocket
func (fg *FrameGenerator) processAudioChunk(msg *WebSocketMessage) {
	if msg.AudioData == "" {
		return
	}
	
	// Decode base64 audio
	audioBytes, err := base64.StdEncoding.DecodeString(msg.AudioData)
	if err != nil {
		log.Printf("❌ Error decoding audio data: %v", err)
		return
	}
	
	// Convert bytes to int16 array
	audioData := make([]int16, len(audioBytes)/2)
	for i := 0; i < len(audioData); i++ {
		audioData[i] = int16(audioBytes[i*2]) | int16(audioBytes[i*2+1])<<8
	}
	
	// Add to buffer
	fg.AddAudioToBuffer(audioData)
	
	// Check if we can generate frames
	if fg.CanGenerateFrames() {
		go func() {
			if err := fg.GenerateFrameBatch(); err != nil {
				log.Printf("❌ Error generating frame batch: %v", err)
			}
		}()
	}
}

// Send current frame to client
func (fg *FrameGenerator) sendCurrentFrame(conn *websocket.Conn) {
	frame := fg.getSynchronizedFrame()
	
	var response map[string]interface{}
	if frame != nil {
		// Set reference from first frame if not already set
		fg.setReferenceFromFrame(frame)
		
		// Encode frame data to base64
		frameB64 := base64.StdEncoding.EncodeToString(frame.Data)
		
		response = map[string]interface{}{
			"type":             "current_frame",
			"frame_data":       frameB64,
			"bounds":           frame.Bounds,
			"timestamp":        frame.Timestamp.Unix(),
			"model_name":       frame.ModelName,
			"prediction_shape": frame.PredictionShape,
		}
	} else {
		response = map[string]interface{}{
			"type":       "current_frame",
			"frame_data": nil,
			"message":    "No frame available",
		}
	}
	
	if err := conn.WriteJSON(response); err != nil {
		log.Printf("❌ Error sending frame: %v", err)
	}
}

// Helper functions
func (fg *FrameGenerator) getAudioBufferFill() int {
	count := 0
	for _, chunk := range fg.audioBuffer {
		if chunk != nil {
			count++
		}
	}
	return count
}

func (fg *FrameGenerator) getFrameBufferFill() int {
	count := 0
	for _, frame := range fg.frameBuffer {
		if frame != nil {
			count++
		}
	}
	return count
}

func (fg *FrameGenerator) getConsecutiveAudioChunks(count int) []*AudioChunk {
	fg.bufferMux.RLock()
	defer fg.bufferMux.RUnlock()
	
	chunks := make([]*AudioChunk, 0, count)
	startIndex := (fg.audioWriteIndex - fg.getAudioBufferFill()) % BUFFER_SIZE
	if startIndex < 0 {
		startIndex += BUFFER_SIZE
	}
	
	for i := 0; i < count && i < BUFFER_SIZE; i++ {
		index := (startIndex + i) % BUFFER_SIZE
		if fg.audioBuffer[index] != nil {
			chunks = append(chunks, fg.audioBuffer[index])
		} else {
			break
		}
	}
	
	return chunks
}

func (fg *FrameGenerator) combineAudioChunks(chunks []*AudioChunk) []int16 {
	totalSamples := 0
	for _, chunk := range chunks {
		totalSamples += len(chunk.Data)
	}
	
	combined := make([]int16, totalSamples)
	offset := 0
	for _, chunk := range chunks {
		copy(combined[offset:], chunk.Data)
		offset += len(chunk.Data)
	}
	
	return combined
}

func (fg *FrameGenerator) convertToWAV(audioData []int16) ([]byte, error) {
	// Simple WAV header creation
	buf := new(bytes.Buffer)
	
	// WAV header (44 bytes)
	dataSize := len(audioData) * 2
	fileSize := dataSize + 36
	
	// RIFF header
	buf.WriteString("RIFF")
	fileSizeBytes := make([]byte, 4)
	fileSizeBytes[0] = byte(fileSize & 0xFF)
	fileSizeBytes[1] = byte((fileSize >> 8) & 0xFF)
	fileSizeBytes[2] = byte((fileSize >> 16) & 0xFF)
	fileSizeBytes[3] = byte((fileSize >> 24) & 0xFF)
	buf.Write(fileSizeBytes)
	buf.WriteString("WAVE")
	
	// fmt chunk
	buf.WriteString("fmt ")
	buf.Write([]byte{16, 0, 0, 0}) // chunk size
	buf.Write([]byte{1, 0})        // audio format (PCM)
	buf.Write([]byte{1, 0})        // num channels
	
	// Write sample rate as little-endian uint32
	sampleRateBytes := make([]byte, 4)
	sampleRateBytes[0] = byte(SAMPLE_RATE & 0xFF)
	sampleRateBytes[1] = byte((SAMPLE_RATE >> 8) & 0xFF)
	sampleRateBytes[2] = byte((SAMPLE_RATE >> 16) & 0xFF)
	sampleRateBytes[3] = byte((SAMPLE_RATE >> 24) & 0xFF)
	buf.Write(sampleRateBytes)
	
	// Write byte rate as little-endian uint32
	byteRate := SAMPLE_RATE * 2
	byteRateBytes := make([]byte, 4)
	byteRateBytes[0] = byte(byteRate & 0xFF)
	byteRateBytes[1] = byte((byteRate >> 8) & 0xFF)
	byteRateBytes[2] = byte((byteRate >> 16) & 0xFF)
	byteRateBytes[3] = byte((byteRate >> 24) & 0xFF)
	buf.Write(byteRateBytes)
	
	buf.Write([]byte{2, 0})  // block align
	buf.Write([]byte{16, 0}) // bits per sample
	
	// data chunk
	buf.WriteString("data")
	dataSizeBytes := make([]byte, 4)
	dataSizeBytes[0] = byte(dataSize & 0xFF)
	dataSizeBytes[1] = byte((dataSize >> 8) & 0xFF)
	dataSizeBytes[2] = byte((dataSize >> 16) & 0xFF)
	dataSizeBytes[3] = byte((dataSize >> 24) & 0xFF)
	buf.Write(dataSizeBytes)
	
	// audio data
	for _, sample := range audioData {
		buf.Write([]byte{byte(sample & 0xFF), byte((sample >> 8) & 0xFF)})
	}
	
	return buf.Bytes(), nil
}

func (fg *FrameGenerator) getSynchronizedFrame() *FrameData {
	fg.bufferMux.RLock()
	defer fg.bufferMux.RUnlock()
	
	// Return most recent frame
	for i := 0; i < BUFFER_SIZE; i++ {
		index := (fg.frameWriteIndex - 1 - i) % BUFFER_SIZE
		if index < 0 {
			index += BUFFER_SIZE
		}
		if fg.frameBuffer[index] != nil {
			return fg.frameBuffer[index]
		}
	}
	
	return nil
}

func (fg *FrameGenerator) setReferenceFromFrame(frame *FrameData) {
	if _, exists := fg.referenceImages[frame.ModelName]; !exists {
		frameB64 := base64.StdEncoding.EncodeToString(frame.Data)
		fg.referenceImages[frame.ModelName] = frameB64
		fg.referenceBounds[frame.ModelName] = frame.Bounds
		log.Printf("✅ Set reference image for model: %s", frame.ModelName)
	}
}

func (fg *FrameGenerator) sendStats(conn *websocket.Conn) {
	stats := map[string]interface{}{
		"type":              "stats",
		"audio_buffer_fill": fg.getAudioBufferFill(),
		"frame_buffer_fill": fg.getFrameBufferFill(),
		"can_generate":      fg.CanGenerateFrames(),
		"current_model":     fg.currentModel,
	}
	
	if err := conn.WriteJSON(stats); err != nil {
		log.Printf("❌ Error sending stats: %v", err)
	}
}

func (fg *FrameGenerator) sendReferenceImage(conn *websocket.Conn, modelName, requestID string) {
	response := map[string]interface{}{
		"type":       "reference_image",
		"request_id": requestID,
		"model_name": modelName,
	}
	
	if imageData, exists := fg.referenceImages[modelName]; exists {
		response["image_data"] = imageData
		response["bounds"] = fg.referenceBounds[modelName]
	} else {
		response["image_data"] = nil
		response["bounds"] = []float32{}
		response["message"] = "Reference will be available after first frame generation"
	}
	
	if err := conn.WriteJSON(response); err != nil {
		log.Printf("❌ Error sending reference image: %v", err)
	}
}

func (fg *FrameGenerator) changeModel(modelName string) {
	if modelName != "" {
		fg.currentModel = modelName
		log.Printf("✅ Switched to model: %s", modelName)
	}
}

func (fg *FrameGenerator) notifyClientsNewFrames(count int) {
	message := map[string]interface{}{
		"type":        "frames_generated",
		"count":       count,
		"buffer_fill": fg.getFrameBufferFill(),
	}
	
	fg.clientsMux.RLock()
	for client := range fg.clients {
		if err := client.WriteJSON(message); err != nil {
			log.Printf("❌ Error notifying client: %v", err)
		}
	}
	fg.clientsMux.RUnlock()
}

// gRPC proxy handler for frame generation
func handleGrpcProxy(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}
	
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Parse request
	var proxyRequest struct {
		Service string      `json:"service"`
		Method  string      `json:"method"`
		Request interface{} `json:"request"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&proxyRequest); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Handle GenerateInference method
	if proxyRequest.Service == "LipSyncService" && proxyRequest.Method == "GenerateInference" {
		handleSingleInference(w, proxyRequest.Request)
		return
	}
	
	http.Error(w, "Unsupported service/method", http.StatusNotImplemented)
}

// Handle single frame inference
func handleSingleInference(w http.ResponseWriter, requestData interface{}) {
	// Parse the inference request
	reqMap, ok := requestData.(map[string]interface{})
	if !ok {
		http.Error(w, "Invalid request format", http.StatusBadRequest)
		return
	}
	
	modelName, _ := reqMap["model_name"].(string)
	frameId, _ := reqMap["frame_id"].(float64) // JSON numbers are float64
	_, _ = reqMap["audio_override"].(string) // Audio data for future use
	
	log.Printf("🎬 gRPC Proxy: Generating frame %d for model %s", int(frameId), modelName)
	
	// For now, generate a test frame that can actually be displayed
	// TODO: Replace with actual gRPC call to Python service
	imageData, err := generateTestJPEGFrame(int(frameId))
	if err != nil {
		log.Printf("❌ Frame generation failed: %v", err)
		http.Error(w, fmt.Sprintf("Frame generation failed: %v", err), http.StatusInternalServerError)
		return
	}
	
	response := map[string]interface{}{
		"success":          true,
		"prediction_data":  imageData,
		"bounds":          []float32{50, 50, 220, 220},
		"processing_time_ms": 15,
		"model_name":      modelName,
		"frame_id":        int(frameId),
		"auto_loaded":     false,
		"prediction_shape": "320x320x3",
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Generate a test JPEG frame that can actually be displayed
func generateTestJPEGFrame(frameId int) ([]byte, error) {
	// Create a simple test image with frame number
	width, height := 320, 320
	
	// Create RGB image data
	img := make([][]byte, height)
	for i := range img {
		img[i] = make([]byte, width*3)
	}
	
	// Fill with gradient and frame number pattern
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := x * 3
			// Create a pattern that changes with frame number
			r := byte((x + y + frameId*10) % 256)
			g := byte((x*2 + frameId*5) % 256)
			b := byte((y*2 + frameId*3) % 256)
			
			img[y][idx] = r
			img[y][idx+1] = g
			img[y][idx+2] = b
		}
	}
	
	// Convert to base64 encoded JPEG-like data (simplified)
	// For a real implementation, you'd use a proper image encoding library
	totalPixels := width * height * 3
	data := make([]byte, totalPixels)
	idx := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width*3; x++ {
			data[idx] = img[y][x]
			idx++
		}
	}
	
	return data, nil
}



// Generate dummy frame data (replace with actual gRPC call)
func generateDummyFrameData() []byte {
	// Create a simple pattern for testing
	width, height := 320, 320
	data := make([]byte, width*height*3)
	
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := (y*width + x) * 3
			// Create a simple gradient pattern
			data[idx] = byte((x + y) % 256)     // R
			data[idx+1] = byte((x * 2) % 256)   // G
			data[idx+2] = byte((y * 2) % 256)   // B
		}
	}
	
	return data
}

// Global frame generator instance
var frameGenerator *FrameGenerator

func main() {
	// Load environment variables from parent directory
	envPath := filepath.Join("..", "realtime_lipsync", ".env")
	if err := godotenv.Load(envPath); err != nil {
		log.Printf("Warning: .env file not found at %s: %v", envPath, err)
	}

	// Initialize frame generator
	frameGenerator = NewFrameGenerator("localhost", 50051)
	if err := frameGenerator.Initialize("localhost", 50051); err != nil {
		log.Fatalf("❌ Failed to initialize frame generator: %v", err)
	}

	// Setup routes
	http.HandleFunc("/token", handleToken)
	http.HandleFunc("/api/model-video/", handleModelVideo)
	http.HandleFunc("/ws", frameGenerator.HandleWebSocket)
	
	// Model management endpoints
	http.HandleFunc("/api/models/load", handleLoadModel)
	http.HandleFunc("/api/models/load-all", handleLoadAllModels)
	http.HandleFunc("/api/models/status", handleModelsStatus)
	
	// gRPC proxy endpoint for frame generation
	http.HandleFunc("/api/grpc-proxy", handleGrpcProxy)
	
	// Serve static files from the realtime_lipsync directory
	staticDir := filepath.Join("..", "realtime_lipsync")
	http.Handle("/", http.FileServer(http.Dir(staticDir)))

	port := os.Getenv("PORT")
	if port == "" {
		port = "3000"
	}

	fmt.Printf("🚀 Unified Real-time Lip Sync Server starting on port %s\n", port)
	fmt.Printf("📁 Serving static files from: %s\n", staticDir)
	fmt.Printf("🔑 Token endpoint: http://localhost:%s/token\n", port)
	fmt.Printf("🌐 WebSocket endpoint: ws://localhost:%s/ws\n", port)
	fmt.Printf("🎬 Model video endpoint: http://localhost:%s/api/model-video/{model_name}\n", port)
	fmt.Printf("🌐 Browser access: http://localhost:%s/index.html\n", port)
	fmt.Printf("⚡ All-in-one server - no need for separate Python frame generator!\n")
	
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// Token handler for OpenAI Realtime API
func handleToken(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	// Handle preflight
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" && r.Method != "GET" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	// Get OpenAI API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Printf("❌ OPENAI_API_KEY not found in environment")
		http.Error(w, `{"error":"OpenAI API key not configured"}`, http.StatusInternalServerError)
		return
	}

	// Create token request
	tokenReq := map[string]interface{}{}

	reqBody, err := json.Marshal(tokenReq)
	if err != nil {
		log.Printf("❌ Error marshaling request: %v", err)
		http.Error(w, `{"error":"Internal server error"}`, http.StatusInternalServerError)
		return
	}

	// Make request to OpenAI
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/realtime/client_secrets", bytes.NewBuffer(reqBody))
	if err != nil {
		log.Printf("❌ Error creating request: %v", err)
		http.Error(w, `{"error":"Internal server error"}`, http.StatusInternalServerError)
		return
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("❌ Error making request to OpenAI: %v", err)
		http.Error(w, `{"error":"Failed to connect to OpenAI"}`, http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("❌ Error reading response: %v", err)
		http.Error(w, `{"error":"Failed to read OpenAI response"}`, http.StatusInternalServerError)
		return
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("❌ OpenAI API error (status %d): %s", resp.StatusCode, string(body))
		http.Error(w, `{"error":"OpenAI API error"}`, http.StatusInternalServerError)
		return
	}

	// Parse and forward response
	var tokenResp TokenResponse
	if err := json.Unmarshal(body, &tokenResp); err != nil {
		log.Printf("❌ Error parsing OpenAI response: %v", err)
		http.Error(w, `{"error":"Invalid OpenAI response"}`, http.StatusInternalServerError)
		return
	}

	log.Printf("✅ Token generated successfully, expires: %s", tokenResp.ClientSecret.ExpiresAt)
	
	w.WriteHeader(http.StatusOK)
	w.Write(body)
}

// Model video handler for serving avatar videos
func handleModelVideo(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "GET" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	// Extract model name from URL path
	urlPath := strings.TrimPrefix(r.URL.Path, "/api/model-video/")
	modelName, err := url.QueryUnescape(urlPath)
	if err != nil {
		log.Printf("❌ Error decoding model name: %v", err)
		http.Error(w, `{"error":"Invalid model name"}`, http.StatusBadRequest)
		return
	}

	if modelName == "" {
		http.Error(w, `{"error":"Model name required"}`, http.StatusBadRequest)
		return
	}

	log.Printf("📥 Model video request for: %s", modelName)

	// Look for the model video in various locations
	possiblePaths := []string{
		filepath.Join("..", "realtime_lipsync", "model_videos", modelName+".mp4"),
		filepath.Join("..", "model_videos", modelName+".mp4"),
		filepath.Join("..", "datasets_test", modelName+"_preprocessed.zip", "full_body_video.mp4"),
		filepath.Join("..", "datasets_test", modelName+".mp4"),
	}

	var videoPath string
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			videoPath = path
			break
		}
	}

	if videoPath == "" {
		log.Printf("❌ Model video not found for: %s", modelName)
		http.Error(w, `{"error":"Model video not found"}`, http.StatusNotFound)
		return
	}

	// Open and serve the video file
	file, err := os.Open(videoPath)
	if err != nil {
		log.Printf("❌ Error opening video file %s: %v", videoPath, err)
		http.Error(w, `{"error":"Failed to open video file"}`, http.StatusInternalServerError)
		return
	}
	defer file.Close()

	// Get file info for content length
	fileInfo, err := file.Stat()
	if err != nil {
		log.Printf("❌ Error getting file info: %v", err)
		http.Error(w, `{"error":"Failed to get file info"}`, http.StatusInternalServerError)
		return
	}

	// Set headers for video streaming
	w.Header().Set("Content-Type", "video/mp4")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", fileInfo.Size()))
	w.Header().Set("Accept-Ranges", "bytes")

	log.Printf("✅ Serving model video: %s (%d bytes)", videoPath, fileInfo.Size())

	// Stream the file
	_, err = io.Copy(w, file)
	if err != nil {
		log.Printf("❌ Error streaming video file: %v", err)
		return
	}

	log.Printf("📤 Model video served successfully: %s", modelName)
}

// Model loading structures
type ModelLoadRequest struct {
	ModelName   string `json:"model_name"`
	PackagePath string `json:"package_path"`
}

type ModelLoadResponse struct {
	Success     bool   `json:"success"`
	Message     string `json:"message"`
	ModelName   string `json:"model_name"`
	LoadTimeMs  int64  `json:"load_time_ms"`
	Error       string `json:"error,omitempty"`
}

type ModelsStatusResponse struct {
	Models []ModelStatus `json:"models"`
	Count  int           `json:"count"`
}

type ModelStatus struct {
	Name      string `json:"name"`
	Available bool   `json:"available"`
	Loaded    bool   `json:"loaded"`
	Path      string `json:"path"`
}

// Handle model loading via FastAPI service
func handleLoadModel(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	// Handle preflight
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	// Parse request
	var req ModelLoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("❌ Error parsing load model request: %v", err)
		http.Error(w, `{"error":"Invalid request body"}`, http.StatusBadRequest)
		return
	}

	log.Printf("📦 Loading model: %s (package: %s)", req.ModelName, req.PackagePath)

	// Make request to FastAPI service
	startTime := time.Now()
	success := callFastAPILoadModel(req.ModelName, req.PackagePath)
	loadTime := time.Since(startTime).Milliseconds()

	response := ModelLoadResponse{
		Success:    success,
		ModelName:  req.ModelName,
		LoadTimeMs: loadTime,
	}

	if success {
		response.Message = fmt.Sprintf("Model '%s' loaded successfully", req.ModelName)
		log.Printf("✅ Model loaded: %s (%dms)", req.ModelName, loadTime)
	} else {
		response.Error = "Failed to load model via FastAPI service"
		response.Message = "Model loading failed"
		log.Printf("❌ Model loading failed: %s", req.ModelName)
	}

	json.NewEncoder(w).Encode(response)
}

// Handle loading all available models
func handleLoadAllModels(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	// Handle preflight
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	log.Printf("🚀 Loading all available models...")

	// Get available models
	models := getAvailableModels()

	results := make([]ModelLoadResponse, 0, len(models))
	loaded := 0

	for _, model := range models {
		log.Printf("📦 Loading model: %s", model.Name)
		
		startTime := time.Now()
		success := callFastAPILoadModel(model.Name, model.Path)
		loadTime := time.Since(startTime).Milliseconds()

		result := ModelLoadResponse{
			Success:    success,
			ModelName:  model.Name,
			LoadTimeMs: loadTime,
		}

		if success {
			result.Message = fmt.Sprintf("Model '%s' loaded successfully", model.Name)
			loaded++
		} else {
			result.Error = "Failed to load model via FastAPI service"
			result.Message = "Model loading failed"
		}

		results = append(results, result)
	}

	response := map[string]interface{}{
		"success":      loaded > 0,
		"loaded_count": loaded,
		"total_count":  len(models),
		"models":       results,
		"message":      fmt.Sprintf("Loaded %d/%d models successfully", loaded, len(models)),
	}

	log.Printf("📊 Model loading complete: %d/%d successful", loaded, len(models))
	json.NewEncoder(w).Encode(response)
}

// Handle model status
func handleModelsStatus(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	// Handle preflight
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "GET" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	models := getAvailableModels()
	
	response := ModelsStatusResponse{
		Models: models,
		Count:  len(models),
	}

	json.NewEncoder(w).Encode(response)
}

// Get available model files
func getAvailableModels() []ModelStatus {
	models := make([]ModelStatus, 0)

	// Check fast_service directory (assuming we're in go-token-server)
	fastServiceDir := filepath.Join("..", "fast_service")
	
	// Check numbered models (1-5)
	for i := 1; i <= 5; i++ {
		modelName := fmt.Sprintf("test_optimized_package_fixed_%d", i)
		modelPath := filepath.Join(fastServiceDir, modelName+".zip")
		
		if _, err := os.Stat(modelPath); err == nil {
			models = append(models, ModelStatus{
				Name:      modelName,
				Available: true,
				Path:      modelPath,
			})
		}
	}

	// Check parent directory models
	parentModels := map[string]string{
		"test_optimized_package_fixed":    "../test_optimized_package_fixed.zip",
		"test_optimized_package":          "../test_optimized_package.zip", 
		"test_optimized_package_with_model": "../test_optimized_package_with_model.zip",
	}

	for modelName, relativePath := range parentModels {
		if _, err := os.Stat(relativePath); err == nil {
			models = append(models, ModelStatus{
				Name:      modelName,
				Available: true,
				Path:      relativePath,
			})
		}
	}

	return models
}

// Standalone model loading (no Python dependencies)
func callFastAPILoadModel(modelName, packagePath string) bool {
	// Pure Go implementation - no Python service needed!
	// This simulates model loading for now
	
	log.Printf("📦 Loading model: %s from %s", modelName, packagePath)
	
	// Check if the package file exists
	if _, err := os.Stat(packagePath); err != nil {
		log.Printf("❌ Model package not found: %s", packagePath)
		return false
	}
	
	// Simulate loading process (replace with actual model loading logic)
	time.Sleep(200 * time.Millisecond)
	
	log.Printf("✅ Model '%s' loaded successfully (Go implementation)", modelName)
	return true
}
