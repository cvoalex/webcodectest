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

// TokenResponse structure for OpenAI API
type TokenResponse struct {
	ClientSecret struct {
		Value     string `json:"value"`
		ExpiresAt string `json:"expires_at"`
	} `json:"client_secret"`
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
	log.Printf("🔄 Stub gRPC call for model: %s, frames: %d", req.ModelName, len(req.FrameIds))
	
	responses := make([]*InferenceResponse, len(req.FrameIds))
	for i := range req.FrameIds {
		dummyFrameData := make([]byte, 320*320*3)
		for j := range dummyFrameData {
			dummyFrameData[j] = byte(128 + (i*10)%128)
		}
		
		responses[i] = &InferenceResponse{
			Success:         true,
			PredictionData:  dummyFrameData,
			Bounds:          []float32{50, 50, 220, 220},
			ModelName:       req.ModelName,
			PredictionShape: "320x320",
		}
	}
	
	return &BatchInferenceResponse{
		Responses: responses,
	}, nil
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

func main() {
	// Load environment variables
	envPath := filepath.Join("..", "realtime_lipsync", ".env")
	if err := godotenv.Load(envPath); err != nil {
		log.Printf("Warning: .env file not found at %s: %v", envPath, err)
	}

	// Setup routes
	http.HandleFunc("/token", handleToken)
	http.HandleFunc("/api/model-video/", handleModelVideo)
	http.HandleFunc("/ws", handleWebSocket)
	
	// Model management endpoints
	http.HandleFunc("/api/models/load", handleLoadModel)
	http.HandleFunc("/api/models/load-all", handleLoadAllModels)
	http.HandleFunc("/api/models/status", handleModelsStatus)
	
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
	fmt.Printf("📦 Model loading endpoints:\n")
	fmt.Printf("   POST http://localhost:%s/api/models/load\n", port)
	fmt.Printf("   POST http://localhost:%s/api/models/load-all\n", port)
	fmt.Printf("   GET  http://localhost:%s/api/models/status\n", port)
	fmt.Printf("🌐 Browser access: http://localhost:%s/index.html\n", port)
	fmt.Printf("⚡ All-in-one server with model management!\n")
	
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// Token handler for OpenAI Realtime API
func handleToken(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" && r.Method != "GET" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Printf("❌ OPENAI_API_KEY not found in environment")
		http.Error(w, `{"error":"OpenAI API key not configured"}`, http.StatusInternalServerError)
		return
	}

	tokenReq := map[string]interface{}{}
	reqBody, err := json.Marshal(tokenReq)
	if err != nil {
		http.Error(w, `{"error":"Internal server error"}`, http.StatusInternalServerError)
		return
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/realtime/client_secrets", bytes.NewBuffer(reqBody))
	if err != nil {
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
		http.Error(w, `{"error":"Failed to read OpenAI response"}`, http.StatusInternalServerError)
		return
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("❌ OpenAI API error (status %d): %s", resp.StatusCode, string(body))
		http.Error(w, `{"error":"OpenAI API error"}`, http.StatusInternalServerError)
		return
	}

	var tokenResp TokenResponse
	if err := json.Unmarshal(body, &tokenResp); err != nil {
		http.Error(w, `{"error":"Invalid OpenAI response"}`, http.StatusInternalServerError)
		return
	}

	log.Printf("✅ Token generated successfully")
	w.WriteHeader(http.StatusOK)
	w.Write(body)
}

// Simple WebSocket handler (stub for now)
func handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("❌ WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("🔗 Client connected: %s", conn.RemoteAddr())

	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("❌ WebSocket error: %v", err)
			}
			break
		}

		// Echo back for now
		response := map[string]interface{}{
			"type":    "response",
			"message": "Received: " + fmt.Sprintf("%v", msg),
		}
		
		if err := conn.WriteJSON(response); err != nil {
			log.Printf("❌ Error sending response: %v", err)
			break
		}
	}
}

// Model video handler
func handleModelVideo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "GET" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	urlPath := strings.TrimPrefix(r.URL.Path, "/api/model-video/")
	modelName, err := url.QueryUnescape(urlPath)
	if err != nil {
		http.Error(w, `{"error":"Invalid model name"}`, http.StatusBadRequest)
		return
	}

	if modelName == "" {
		http.Error(w, `{"error":"Model name required"}`, http.StatusBadRequest)
		return
	}

	log.Printf("📥 Model video request for: %s", modelName)

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

	file, err := os.Open(videoPath)
	if err != nil {
		http.Error(w, `{"error":"Failed to open video file"}`, http.StatusInternalServerError)
		return
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		http.Error(w, `{"error":"Failed to get file info"}`, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "video/mp4")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", fileInfo.Size()))
	w.Header().Set("Accept-Ranges", "bytes")

	log.Printf("✅ Serving model video: %s (%d bytes)", videoPath, fileInfo.Size())
	
	io.Copy(w, file)
	log.Printf("📤 Model video served successfully: %s", modelName)
}

// Handle model loading via FastAPI service
func handleLoadModel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req ModelLoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("❌ Error parsing load model request: %v", err)
		http.Error(w, `{"error":"Invalid request body"}`, http.StatusBadRequest)
		return
	}

	log.Printf("📦 Loading model: %s (package: %s)", req.ModelName, req.PackagePath)

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
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	log.Printf("🚀 Loading all available models...")

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
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

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

// Call FastAPI service to load model
func callFastAPILoadModel(modelName, packagePath string) bool {
	payload := map[string]string{
		"model_name":   modelName,
		"package_path": packagePath,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		log.Printf("❌ Error marshaling payload: %v", err)
		return false
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post("http://localhost:8000/models/load", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("❌ Error calling FastAPI service: %v", err)
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return true
	}

	body, _ := io.ReadAll(resp.Body)
	log.Printf("❌ FastAPI service error (status %d): %s", resp.StatusCode, string(body))
	return false
}
