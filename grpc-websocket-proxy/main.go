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
	wsPort    = flag.Int("ws-port", 8086, "WebSocket server port")
	grpcAddr  = flag.String("grpc-addr", "localhost:50051", "gRPC server address")
	upgrader  = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins for development
		},
		ReadBufferSize:  1024 * 64,  // 64KB
		WriteBufferSize: 1024 * 256, // 256KB
	}
)

// ProxyServer handles WebSocket to gRPC proxying
type ProxyServer struct {
	grpcClient pb.OptimizedLipSyncServiceClient
	grpcConn   *grpc.ClientConn
}

// NewProxyServer creates a new proxy server
func NewProxyServer(grpcAddr string) (*ProxyServer, error) {
	log.Printf("🔌 Connecting to gRPC server at %s...", grpcAddr)
	
	conn, err := grpc.Dial(
		grpcAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(50*1024*1024), // 50MB
			grpc.MaxCallSendMsgSize(50*1024*1024), // 50MB
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to gRPC: %w", err)
	}

	client := pb.NewOptimizedLipSyncServiceClient(conn)
	
	// Test connection with health check
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	healthResp, err := client.HealthCheck(ctx, &pb.HealthRequest{})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("gRPC health check failed: %w", err)
	}
	
	log.Printf("✅ Connected to gRPC server!")
	log.Printf("   Status: %s", healthResp.Status)
	log.Printf("   Loaded models: %d", healthResp.LoadedModels)
	
	return &ProxyServer{
		grpcClient: client,
		grpcConn:   conn,
	}, nil
}

// Close closes the gRPC connection
func (p *ProxyServer) Close() {
	if p.grpcConn != nil {
		p.grpcConn.Close()
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
	ModelName string `json:"model_name"`
	FrameID   int32  `json:"frame_id"`
}

// JSONResponse represents a JSON protocol response to browser
type JSONResponse struct {
	Success         bool     `json:"success"`
	FrameID         int32    `json:"frame_id"`
	PredictionData  string   `json:"prediction_data,omitempty"` // base64
	PredictionShape string   `json:"prediction_shape,omitempty"`
	Bounds          []int32  `json:"bounds,omitempty"`
	ProcessingTime  float32  `json:"processing_time_ms"`
	PrepareTime     float32  `json:"prepare_time_ms,omitempty"`
	InferenceTime   float32  `json:"inference_time_ms,omitempty"`
	CompositeTime   float32  `json:"composite_time_ms,omitempty"`
	Error           string   `json:"error,omitempty"`
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
		
		// Determine protocol (binary vs JSON)
		var modelName string
		var frameID int32
		var isBinary bool
		
		if messageType == websocket.BinaryMessage {
			// Binary protocol
			req, err := parseBinaryRequest(data)
			if err != nil {
				log.Printf("❌ Invalid binary request from %s: %v", clientAddr, err)
				continue
			}
			modelName = req.ModelName
			frameID = req.FrameID
			isBinary = true
		} else {
			// JSON protocol
			var req JSONRequest
			if err := json.Unmarshal(data, &req); err != nil {
				log.Printf("❌ Invalid JSON request from %s: %v", clientAddr, err)
				continue
			}
			modelName = req.ModelName
			frameID = req.FrameID
			isBinary = false
		}
		
		// Forward to gRPC server
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		
		grpcReq := &pb.OptimizedInferenceRequest{
			ModelName: modelName,
			FrameId:   frameID,
		}
		
		grpcResp, err := p.grpcClient.GenerateInference(ctx, grpcReq)
		cancel()
		
		totalTime := time.Since(startTime).Milliseconds()
		
		if err != nil {
			log.Printf("❌ gRPC error for %s frame %d: %v", modelName, frameID, err)
			
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
		
		// Log success
		log.Printf("✅ %s frame %d: gRPC=%dms total=%dms size=%d bytes",
			modelName, frameID, int(grpcResp.ProcessingTimeMs), totalTime, len(grpcResp.PredictionData))
		
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
				jsonResp.Error = grpcResp.Error
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
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	
	resp, err := p.grpcClient.HealthCheck(ctx, &pb.HealthRequest{})
	if err != nil {
		http.Error(w, fmt.Sprintf("gRPC health check failed: %v", err), http.StatusServiceUnavailable)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":        resp.Status,
		"healthy":       resp.Healthy,
		"loaded_models": resp.LoadedModels,
		"uptime":        resp.UptimeSeconds,
	})
}

func main() {
	flag.Parse()
	
	fmt.Println("================================================================================")
	fmt.Println("🌉 GRPC-TO-WEBSOCKET PROXY SERVER")
	fmt.Println("================================================================================")
	fmt.Println()
	
	// Connect to gRPC server
	proxy, err := NewProxyServer(*grpcAddr)
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
	fmt.Printf("🔗 gRPC backend: %s\n", *grpcAddr)
	fmt.Println()
	fmt.Println("Press Ctrl+C to stop")
	fmt.Println()
	
	log.Fatal(http.ListenAndServe(addr, nil))
}
