package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/cvoalex/lipsync-proxy/pb"
)

var (
	wsPort     = flag.Int("ws-port", 8086, "WebSocket server port")
	grpcAddrs  = flag.String("grpc-addrs", "localhost:50051", "Comma-separated gRPC server addresses")
	startPort  = flag.Int("start-port", 50051, "First gRPC port (generates range)")
	numServers = flag.Int("num-servers", 1, "Number of gRPC servers (for port range)")
	upgrader   = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins for development
		},
		ReadBufferSize:  1024 * 64,  // 64KB
		WriteBufferSize: 1024 * 256, // 256KB
	}
)

// Backend represents a single gRPC backend server
type Backend struct {
	address    string
	client     pb.OptimizedLipSyncServiceClient
	conn       *grpc.ClientConn
	healthy    bool
	totalReqs  int64
	errorCount int64
}

// ProxyServer handles WebSocket to gRPC proxying with load balancing
type ProxyServer struct {
	backends    []*Backend
	nextBackend int
}

// NewProxyServer creates a new proxy server with multiple backends
func NewProxyServer(addresses []string) (*ProxyServer, error) {
	log.Printf("🔌 Connecting to %d gRPC servers...", len(addresses))

	backends := make([]*Backend, 0, len(addresses))

	for i, addr := range addresses {
		log.Printf("   [%d/%d] Connecting to %s...", i+1, len(addresses), addr)

		conn, err := grpc.Dial(
			addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(
				grpc.MaxCallRecvMsgSize(50*1024*1024), // 50MB
				grpc.MaxCallSendMsgSize(50*1024*1024), // 50MB
			),
		)
		if err != nil {
			log.Printf("   ⚠️  Failed to connect to %s: %v", addr, err)
			continue
		}

		client := pb.NewOptimizedLipSyncServiceClient(conn)

		// Test connection with health check
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		healthResp, err := client.HealthCheck(ctx, &pb.HealthRequest{})
		cancel()

		healthy := err == nil
		if healthy {
			log.Printf("   ✅ %s - Status: %s, Models: %d", addr, healthResp.Status, healthResp.LoadedModels)
		} else {
			log.Printf("   ⚠️  %s - Health check failed (will retry): %v", addr, err)
		}

		backends = append(backends, &Backend{
			address: addr,
			client:  client,
			conn:    conn,
			healthy: healthy,
		})
	}

	if len(backends) == 0 {
		return nil, fmt.Errorf("no gRPC backends available")
	}

	log.Printf("✅ Connected to %d/%d gRPC servers", len(backends), len(addresses))

	return &ProxyServer{
		backends:    backends,
		nextBackend: 0,
	}, nil
}

// getNextBackend returns the next healthy backend (round-robin)
func (p *ProxyServer) getNextBackend() *Backend {
	if len(p.backends) == 0 {
		return nil
	}

	// Try up to 2x the number of backends to find a healthy one
	attempts := len(p.backends) * 2

	for i := 0; i < attempts; i++ {
		backend := p.backends[p.nextBackend]
		p.nextBackend = (p.nextBackend + 1) % len(p.backends)

		if backend.healthy {
			return backend
		}
	}

	// No healthy backends found, return the next one anyway (it will fail gracefully)
	backend := p.backends[p.nextBackend]
	p.nextBackend = (p.nextBackend + 1) % len(p.backends)
	return backend
}

// Close closes all gRPC connections
func (p *ProxyServer) Close() {
	for _, backend := range p.backends {
		if backend.conn != nil {
			backend.conn.Close()
		}
	}
}

// BinaryRequest represents a binary protocol request from browser
type BinaryRequest struct {
	ModelName string
	FrameID   int32
}

// JSONRequest represents a JSON protocol request from browser
type JSONRequest struct {
	Type      string `json:"type"`
	ModelName string `json:"model_name,omitempty"`
	FrameID   int32  `json:"frame_id,omitempty"`
	VideoType string `json:"video_type,omitempty"`
}

// JSONResponse represents a JSON protocol response to browser
type JSONResponse struct {
	Success         bool      `json:"success"`
	FrameID         int32     `json:"frame_id"`
	PredictionData  string    `json:"prediction_data,omitempty"` // base64
	PredictionShape string    `json:"prediction_shape,omitempty"`
	Bounds          []float32 `json:"bounds,omitempty"`
	ProcessingTime  int32     `json:"processing_time_ms"`
	PrepareTime     float64   `json:"prepare_time_ms,omitempty"`
	InferenceTime   float64   `json:"inference_time_ms,omitempty"`
	CompositeTime   float64   `json:"composite_time_ms,omitempty"`
	Error           string    `json:"error,omitempty"`
}

// parseBinaryRequest parses binary protocol request
func parseBinaryRequest(data []byte) (*BinaryRequest, error) {
	if len(data) < 5 {
		return nil, fmt.Errorf("invalid binary request: too short")
	}

	// Format: [model_name_len:1][model_name:N][frame_id:4]
	modelNameLen := int(data[0])
	if len(data) < 1+modelNameLen+4 {
		return nil, fmt.Errorf("invalid binary request: incomplete data")
	}

	modelName := string(data[1 : 1+modelNameLen])
	frameID := int32(binary.LittleEndian.Uint32(data[1+modelNameLen : 1+modelNameLen+4]))

	return &BinaryRequest{
		ModelName: modelName,
		FrameID:   frameID,
	}, nil
}

// handleWebSocket handles a WebSocket connection
func (p *ProxyServer) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("❌ WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	clientAddr := conn.RemoteAddr().String()
	log.Printf("🌐 Client connected: %s", clientAddr)

	for {
		messageType, data, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("❌ WebSocket error from %s: %v", clientAddr, err)
			}
			break
		}

		startTime := time.Now()

		// Determine protocol and message type
		var modelName string
		var frameID int32
		var isBinary bool

		if messageType == websocket.BinaryMessage {
			// Binary protocol - always inference
			req, err := parseBinaryRequest(data)
			if err != nil {
				log.Printf("❌ Invalid binary request from %s: %v", clientAddr, err)
				continue
			}
			modelName = req.ModelName
			frameID = req.FrameID
			isBinary = true
		} else {
			// JSON protocol - check message type
			var req JSONRequest
			if err := json.Unmarshal(data, &req); err != nil {
				log.Printf("❌ Invalid JSON request from %s: %v", clientAddr, err)
				continue
			}

			// Handle non-inference requests
			switch req.Type {
			case "list_models":
				p.handleListModels(conn)
				continue
			case "get_metadata":
				p.handleGetMetadata(conn, req.ModelName)
				continue
			case "get_video_frame":
				p.handleGetVideoFrame(conn, req.ModelName, req.FrameID, req.VideoType)
				continue
			case "inference", "":
				// Fallthrough to inference handling
				modelName = req.ModelName
				frameID = req.FrameID
				isBinary = false
			default:
				log.Printf("❌ Unknown message type: %s", req.Type)
				errorResp := map[string]interface{}{
					"type":  "error",
					"error": "Unknown message type: " + req.Type,
				}
				jsonData, _ := json.Marshal(errorResp)
				conn.WriteMessage(websocket.TextMessage, jsonData)
				continue
			}
		}

		// Get next backend (round-robin)
		backend := p.getNextBackend()
		if backend == nil {
			log.Printf("❌ No backends available")
			if isBinary {
				conn.WriteMessage(websocket.BinaryMessage, []byte{})
			} else {
				errorResp := JSONResponse{Success: false, FrameID: frameID, Error: "no backends available"}
				jsonData, _ := json.Marshal(errorResp)
				conn.WriteMessage(websocket.TextMessage, jsonData)
			}
			continue
		}

		// Forward to gRPC server
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

		grpcReq := &pb.OptimizedInferenceRequest{
			ModelName: modelName,
			FrameId:   frameID,
		}

		backend.totalReqs++
		grpcResp, err := backend.client.GenerateInference(ctx, grpcReq)
		cancel()

		totalTime := time.Since(startTime).Milliseconds()

		if err != nil {
			log.Printf("❌ gRPC error for %s frame %d from %s: %v", modelName, frameID, backend.address, err)
			backend.errorCount++
			backend.healthy = false // Mark as unhealthy temporarily

			// Send error response
			if isBinary {
				// Send empty response for binary (browser will retry)
				conn.WriteMessage(websocket.BinaryMessage, []byte{})
			} else {
				errorResp := JSONResponse{
					Success: false,
					FrameID: frameID,
					Error:   err.Error(),
				}
				jsonData, _ := json.Marshal(errorResp)
				conn.WriteMessage(websocket.TextMessage, jsonData)
			}
			continue
		}

		// Mark backend as healthy on success
		backend.healthy = true

		// Log success
		log.Printf("✅ %s frame %d [%s]: gRPC=%dms total=%dms size=%d bytes",
			modelName, frameID, backend.address, int(grpcResp.ProcessingTimeMs), totalTime, len(grpcResp.PredictionData))

		// Send response to browser
		if isBinary {
			// Binary protocol: just send JPEG bytes directly
			err = conn.WriteMessage(websocket.BinaryMessage, grpcResp.PredictionData)
			if err != nil {
				log.Printf("❌ Failed to send binary response to %s: %v", clientAddr, err)
				break
			}
		} else {
			// JSON protocol: encode as base64
			jsonResp := JSONResponse{
				Success:         grpcResp.Success,
				FrameID:         grpcResp.FrameId,
				PredictionShape: grpcResp.PredictionShape,
				Bounds:          grpcResp.Bounds,
				ProcessingTime:  grpcResp.ProcessingTimeMs,
				PrepareTime:     grpcResp.PrepareTimeMs,
				InferenceTime:   grpcResp.InferenceTimeMs,
				CompositeTime:   grpcResp.CompositeTimeMs,
			}

			if grpcResp.Success {
				// For JSON, we'd need to base64 encode the image
				// But for efficiency, recommend binary protocol
				jsonResp.PredictionData = "use_binary_protocol_for_image_data"
			} else {
				if grpcResp.Error != nil {
					jsonResp.Error = *grpcResp.Error
				}
			}

			jsonData, err := json.Marshal(jsonResp)
			if err != nil {
				log.Printf("❌ Failed to marshal JSON response: %v", err)
				continue
			}

			err = conn.WriteMessage(websocket.TextMessage, jsonData)
			if err != nil {
				log.Printf("❌ Failed to send JSON response to %s: %v", clientAddr, err)
				break
			}
		}
	}

	log.Printf("👋 Client disconnected: %s", clientAddr)
}

// healthHandler handles health check endpoint
func (p *ProxyServer) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	type BackendStatus struct {
		Address    string `json:"address"`
		Healthy    bool   `json:"healthy"`
		TotalReqs  int64  `json:"total_requests"`
		ErrorCount int64  `json:"error_count"`
	}

	backendStatuses := make([]BackendStatus, len(p.backends))
	healthyCount := 0

	for i, backend := range p.backends {
		backendStatuses[i] = BackendStatus{
			Address:    backend.address,
			Healthy:    backend.healthy,
			TotalReqs:  backend.totalReqs,
			ErrorCount: backend.errorCount,
		}
		if backend.healthy {
			healthyCount++
		}
	}

	response := map[string]interface{}{
		"healthy":        healthyCount > 0,
		"total_backends": len(p.backends),
		"healthy_count":  healthyCount,
		"backends":       backendStatuses,
	}

	if healthyCount == 0 {
		w.WriteHeader(http.StatusServiceUnavailable)
	}

	json.NewEncoder(w).Encode(response)
}

func main() {
	flag.Parse()

	fmt.Println("================================================================================")
	fmt.Println("🌉 GRPC-TO-WEBSOCKET PROXY SERVER (Multi-Backend Load Balancer)")
	fmt.Println("================================================================================")
	fmt.Println()

	// Parse backend addresses
	var addresses []string

	if *grpcAddrs != "" && *grpcAddrs != "localhost:50051" {
		// Use explicit addresses from --grpc-addrs
		addresses = parseAddresses(*grpcAddrs)
	} else if *numServers > 1 {
		// Generate port range from --start-port and --num-servers
		for i := 0; i < *numServers; i++ {
			addr := fmt.Sprintf("localhost:%d", *startPort+i)
			addresses = append(addresses, addr)
		}
	} else {
		// Default to single server
		addresses = []string{fmt.Sprintf("localhost:%d", *startPort)}
	}

	fmt.Printf("📍 Backend configuration:\n")
	for i, addr := range addresses {
		fmt.Printf("   [%d] %s\n", i+1, addr)
	}
	fmt.Println()

	// Connect to gRPC servers
	proxy, err := NewProxyServer(addresses)
	if err != nil {
		log.Fatalf("❌ Failed to create proxy: %v", err)
	}
	defer proxy.Close()

	// Setup HTTP routes
	http.HandleFunc("/ws", proxy.handleWebSocket)
	http.HandleFunc("/health", proxy.healthHandler)

	// Serve static files (HTML client)
	fs := http.FileServer(http.Dir("./static"))
	http.Handle("/", fs)

	// Start server
	addr := fmt.Sprintf(":%d", *wsPort)
	fmt.Printf("🚀 WebSocket proxy started on ws://localhost%s/ws\n", addr)
	fmt.Printf("📁 Static files served from ./static/\n")
	fmt.Printf("🏥 Health check: http://localhost%s/health\n", addr)
	fmt.Printf("⚖️  Load balancing: Round-robin across %d backends\n", len(addresses))
	fmt.Println()
	fmt.Println("Press Ctrl+C to stop")
	fmt.Println()

	log.Fatal(http.ListenAndServe(addr, nil))
}

// Handler for ListModels request
func (p *ProxyServer) handleListModels(conn *websocket.Conn) {
	backend := p.getNextBackend()
	if backend == nil {
		log.Printf("❌ No backends available for list_models")
		errorResp := map[string]interface{}{
			"type":  "model_list",
			"error": "no backends available",
		}
		jsonData, _ := json.Marshal(errorResp)
		conn.WriteMessage(websocket.TextMessage, jsonData)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	grpcResp, err := backend.client.ListModels(ctx, &pb.ListModelsRequest{})
	if err != nil {
		log.Printf("❌ ListModels gRPC error: %v", err)
		errorResp := map[string]interface{}{
			"type":  "model_list",
			"error": err.Error(),
		}
		jsonData, _ := json.Marshal(errorResp)
		conn.WriteMessage(websocket.TextMessage, jsonData)
		return
	}

	response := map[string]interface{}{
		"type":   "model_list",
		"models": grpcResp.LoadedModels,
		"count":  grpcResp.Count,
	}
	jsonData, _ := json.Marshal(response)
	conn.WriteMessage(websocket.TextMessage, jsonData)
	log.Printf("✅ ListModels: %d models", grpcResp.Count)
}

// Handler for GetModelMetadata request
func (p *ProxyServer) handleGetMetadata(conn *websocket.Conn, modelName string) {
	backend := p.getNextBackend()
	if backend == nil {
		log.Printf("❌ No backends available for get_metadata")
		errorResp := map[string]interface{}{
			"type":  "metadata",
			"error": "no backends available",
		}
		jsonData, _ := json.Marshal(errorResp)
		conn.WriteMessage(websocket.TextMessage, jsonData)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	grpcReq := &pb.GetModelMetadataRequest{
		ModelName: modelName,
	}
	grpcResp, err := backend.client.GetModelMetadata(ctx, grpcReq)
	if err != nil {
		log.Printf("❌ GetModelMetadata gRPC error for %s: %v", modelName, err)
		errorResp := map[string]interface{}{
			"type":       "metadata",
			"error":      err.Error(),
			"model_name": modelName,
		}
		jsonData, _ := json.Marshal(errorResp)
		conn.WriteMessage(websocket.TextMessage, jsonData)
		return
	}

	response := map[string]interface{}{
		"type":             "metadata",
		"success":          grpcResp.Success,
		"model_name":       grpcResp.ModelName,
		"frame_count":      grpcResp.FrameCount,
		"available_videos": grpcResp.AvailableVideos,
		"audio_path":       grpcResp.AudioPath,
		"bounds":           grpcResp.Bounds,
	}
	if grpcResp.Error != nil && *grpcResp.Error != "" {
		response["error"] = *grpcResp.Error
	}
	jsonData, _ := json.Marshal(response)
	conn.WriteMessage(websocket.TextMessage, jsonData)
	log.Printf("✅ GetModelMetadata: %s (%d frames)", modelName, grpcResp.FrameCount)
}

// Handler for GetVideoFrame request
func (p *ProxyServer) handleGetVideoFrame(conn *websocket.Conn, modelName string, frameID int32, videoType string) {
	backend := p.getNextBackend()
	if backend == nil {
		log.Printf("❌ No backends available for get_video_frame")
		errorResp := map[string]interface{}{
			"type":  "video_frame",
			"error": "no backends available",
		}
		jsonData, _ := json.Marshal(errorResp)
		conn.WriteMessage(websocket.TextMessage, jsonData)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	grpcReq := &pb.GetVideoFrameRequest{
		ModelName: modelName,
		FrameId:   frameID,
		VideoType: videoType,
	}
	grpcResp, err := backend.client.GetVideoFrame(ctx, grpcReq)
	if err != nil {
		log.Printf("❌ GetVideoFrame gRPC error for %s frame %d: %v", modelName, frameID, err)
		errorResp := map[string]interface{}{
			"type":       "video_frame",
			"error":      err.Error(),
			"model_name": modelName,
			"frame_id":   frameID,
		}
		jsonData, _ := json.Marshal(errorResp)
		conn.WriteMessage(websocket.TextMessage, jsonData)
		return
	}

	// Send binary response: [type:1][frame_id:4][video_type_len:1][video_type:N][data_len:4][jpeg_data:N]
	videoTypeBytes := []byte(grpcResp.VideoType)
	response := make([]byte, 1+4+1+len(videoTypeBytes)+4+len(grpcResp.FrameData))
	
	offset := 0
	response[offset] = 2 // Message type: video frame
	offset++
	
	// Frame ID (little endian)
	response[offset] = byte(grpcResp.FrameId)
	response[offset+1] = byte(grpcResp.FrameId >> 8)
	response[offset+2] = byte(grpcResp.FrameId >> 16)
	response[offset+3] = byte(grpcResp.FrameId >> 24)
	offset += 4
	
	// Video type length
	response[offset] = byte(len(videoTypeBytes))
	offset++
	
	// Video type string
	copy(response[offset:], videoTypeBytes)
	offset += len(videoTypeBytes)
	
	// Data length (little endian)
	dataLen := uint32(len(grpcResp.FrameData))
	response[offset] = byte(dataLen)
	response[offset+1] = byte(dataLen >> 8)
	response[offset+2] = byte(dataLen >> 16)
	response[offset+3] = byte(dataLen >> 24)
	offset += 4
	
	// JPEG data
	copy(response[offset:], grpcResp.FrameData)
	
	err = conn.WriteMessage(websocket.BinaryMessage, response)
	if err != nil {
		log.Printf("❌ Failed to send video frame to client: %v", err)
		return
	}
	
	log.Printf("✅ GetVideoFrame: %s frame %d %s (%d bytes)", modelName, frameID, videoType, len(grpcResp.FrameData))
}

func parseAddresses(addrsStr string) []string {
	var addresses []string
	for _, addr := range splitString(addrsStr, ',') {
		trimmed := trimSpace(addr)
		if trimmed != "" {
			addresses = append(addresses, trimmed)
		}
	}
	return addresses
}

func splitString(s string, sep rune) []string {
	var result []string
	var current string
	for _, c := range s {
		if c == sep {
			result = append(result, current)
			current = ""
		} else {
			current += string(c)
		}
	}
	if current != "" {
		result = append(result, current)
	}
	return result
}

func trimSpace(s string) string {
	start := 0
	end := len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n') {
		end--
	}
	return s[start:end]
}
