package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go-onnx-inference/lipsyncinfer"
	pb "go-onnx-inference/proto"

	"google.golang.org/grpc"
)

const (
	visualFrameSize  = 6 * 320 * 320
	audioFrameSize   = 32 * 16 * 16
	outputFrameSize  = 3 * 320 * 320
	backgroundWidth  = 1280
	backgroundHeight = 720
)

type CropRect struct {
	Rect []int `json:"rect"` // [x1, y1, x2, y2]
}

type lipSyncCompositeServer struct {
	pb.UnimplementedLipSyncServer
	inferencer  *lipsyncinfer.Inferencer
	modelPath   string
	backgrounds []*image.RGBA // Cached background frames
	cropRects   map[int][]int // Crop rectangles per frame
	bgMutex     sync.RWMutex
}

// Convert ONNX output to Go image.Image
func outputToImage(outputData []float32) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 320, 320))

	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// BGR order from ONNX
			b := outputData[0*320*320+y*320+x]
			g := outputData[1*320*320+y*320+x]
			r := outputData[2*320*320+y*320+x]

			// Output is already in [0, 1] range, just multiply by 255
			rByte := uint8(clamp(r * 255.0))
			gByte := uint8(clamp(g * 255.0))
			bByte := uint8(clamp(b * 255.0))

			img.SetRGBA(x, y, color.RGBA{R: rByte, G: gByte, B: bByte, A: 255})
		}
	}

	return img
}

func clamp(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 255 {
		return 255
	}
	return val
}

// Bilinear resize - pure Go implementation
func resizeImage(src *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	dst := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))

	xRatio := float32(srcWidth) / float32(targetWidth)
	yRatio := float32(srcHeight) / float32(targetHeight)

	for dstY := 0; dstY < targetHeight; dstY++ {
		for dstX := 0; dstX < targetWidth; dstX++ {
			srcX := float32(dstX) * xRatio
			srcY := float32(dstY) * yRatio

			x0 := int(srcX)
			y0 := int(srcY)
			x1 := x0 + 1
			y1 := y0 + 1

			if x1 >= srcWidth {
				x1 = srcWidth - 1
			}
			if y1 >= srcHeight {
				y1 = srcHeight - 1
			}

			xWeight := srcX - float32(x0)
			yWeight := srcY - float32(y0)

			// Get four neighboring pixels
			c00 := src.RGBAAt(x0, y0)
			c10 := src.RGBAAt(x1, y0)
			c01 := src.RGBAAt(x0, y1)
			c11 := src.RGBAAt(x1, y1)

			// Bilinear interpolation
			r := bilinearInterp(c00.R, c10.R, c01.R, c11.R, xWeight, yWeight)
			g := bilinearInterp(c00.G, c10.G, c01.G, c11.G, xWeight, yWeight)
			b := bilinearInterp(c00.B, c10.B, c01.B, c11.B, xWeight, yWeight)

			dst.SetRGBA(dstX, dstY, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return dst
}

func bilinearInterp(c00, c10, c01, c11 uint8, xWeight, yWeight float32) uint8 {
	top := float32(c00)*(1-xWeight) + float32(c10)*xWeight
	bottom := float32(c01)*(1-xWeight) + float32(c11)*xWeight
	result := top*(1-yWeight) + bottom*yWeight
	return uint8(result)
}

// Composite mouth region onto background - pure Go
func compositeFrame(mouthRegion *image.RGBA, background *image.RGBA, x, y, w, h int) *image.RGBA {
	// Resize mouth region
	resized := resizeImage(mouthRegion, w, h)

	// Clone background
	result := image.NewRGBA(background.Bounds())
	draw.Draw(result, result.Bounds(), background, image.Point{}, draw.Src)

	// Paste mouth region
	dstRect := image.Rect(x, y, x+w, y+h)
	draw.Draw(result, dstRect, resized, image.Point{}, draw.Src)

	return result
}

// Load backgrounds into cache
func (s *lipSyncCompositeServer) loadBackgrounds(bgDir string, numFrames int) error {
	s.bgMutex.Lock()
	defer s.bgMutex.Unlock()

	log.Printf("Loading %d background frames into cache from %s...", numFrames, bgDir)
	s.backgrounds = make([]*image.RGBA, numFrames)

	for i := 0; i < numFrames; i++ {
		framePath := filepath.Join(bgDir, fmt.Sprintf("frame_%04d.png", i))
		file, err := os.Open(framePath)
		if err != nil {
			return fmt.Errorf("failed to load background frame %d: %v", i, err)
		}

		img, err := png.Decode(file)
		file.Close()
		if err != nil {
			return fmt.Errorf("failed to decode background frame %d: %v", i, err)
		}

		// Convert to RGBA
		rgba := image.NewRGBA(img.Bounds())
		draw.Draw(rgba, rgba.Bounds(), img, image.Point{}, draw.Src)
		s.backgrounds[i] = rgba
	}

	log.Printf("✓ Loaded %d background frames (%.2f MB)",
		numFrames, float64(numFrames*backgroundWidth*backgroundHeight*4)/(1024*1024))
	return nil
}

// Load crop rectangles from JSON
func (s *lipSyncCompositeServer) loadCropRects(jsonPath string) error {
	s.bgMutex.Lock()
	defer s.bgMutex.Unlock()

	log.Printf("Loading crop rectangles from %s...", jsonPath)

	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return fmt.Errorf("failed to read crop rectangles: %v", err)
	}

	var cropData map[string]CropRect
	if err := json.Unmarshal(data, &cropData); err != nil {
		return fmt.Errorf("failed to parse crop rectangles: %v", err)
	}

	s.cropRects = make(map[int][]int)
	for k, v := range cropData {
		var frameIdx int
		fmt.Sscanf(k, "%d", &frameIdx)
		s.cropRects[frameIdx] = v.Rect
	}

	log.Printf("✓ Loaded %d crop rectangles", len(s.cropRects))
	return nil
}

func (s *lipSyncCompositeServer) InferBatch(ctx context.Context, req *pb.BatchRequest) (*pb.BatchResponse, error) {
	totalStart := time.Now()

	// Validate batch size
	if req.BatchSize < 1 || req.BatchSize > 25 {
		return &pb.BatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid batch size: %d (must be 1-25)", req.BatchSize),
		}, nil
	}

	// Validate input sizes
	expectedVisualSize := int(req.BatchSize) * visualFrameSize
	expectedAudioSize := audioFrameSize

	if len(req.VisualFrames) != expectedVisualSize {
		return &pb.BatchResponse{
			Success: false,
			Error: fmt.Sprintf("Invalid visual frames size: got %d, expected %d",
				len(req.VisualFrames), expectedVisualSize),
		}, nil
	}

	if len(req.AudioFeatures) != expectedAudioSize {
		return &pb.BatchResponse{
			Success: false,
			Error: fmt.Sprintf("Invalid audio features size: got %d, expected %d",
				len(req.AudioFeatures), expectedAudioSize),
		}, nil
	}

	// Replicate audio window for each frame in batch
	audioForBatch := make([]float32, int(req.BatchSize)*audioFrameSize)
	for i := 0; i < int(req.BatchSize); i++ {
		copy(audioForBatch[i*audioFrameSize:], req.AudioFeatures)
	}

	// Run inference
	inferStart := time.Now()
	outputs, err := s.inferencer.InferBatch(req.VisualFrames, audioForBatch, int(req.BatchSize))
	inferTime := time.Since(inferStart)

	if err != nil {
		return &pb.BatchResponse{
			Success: false,
			Error:   fmt.Sprintf("Inference failed: %v", err),
		}, nil
	}

	// Composite each frame
	compStart := time.Now()
	compositedPNGs := make([][]byte, req.BatchSize)
	startFrameIdx := int(req.StartFrameIdx)

	for i := 0; i < int(req.BatchSize); i++ {
		frameIdx := startFrameIdx + i
		frameOutput := outputs[i*outputFrameSize : (i+1)*outputFrameSize]

		// Convert output to image
		mouthRegion := outputToImage(frameOutput)

		// Get crop rectangle for this frame
		s.bgMutex.RLock()
		cropRect, exists := s.cropRects[frameIdx]
		background := s.backgrounds[frameIdx]
		s.bgMutex.RUnlock()

		if !exists {
			return &pb.BatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Crop rectangle not found for frame %d", frameIdx),
			}, nil
		}

		x1, y1, x2, y2 := cropRect[0], cropRect[1], cropRect[2], cropRect[3]
		w := x2 - x1
		h := y2 - y1

		// Composite with background
		composited := compositeFrame(mouthRegion, background, x1, y1, w, h)

		// Encode to PNG
		var buf bytes.Buffer
		if err := png.Encode(&buf, composited); err != nil {
			return &pb.BatchResponse{
				Success: false,
				Error:   fmt.Sprintf("Failed to encode frame %d: %v", i, err),
			}, nil
		}

		compositedPNGs[i] = buf.Bytes()
	}
	compTime := time.Since(compStart)
	totalTime := time.Since(totalStart)

	// For now, return empty output_frames since we're returning PNGs
	// In a real implementation, you'd update the proto to support bytes
	return &pb.BatchResponse{
		OutputFrames:    make([]float32, 0), // Not used with compositing
		InferenceTimeMs: inferTime.Seconds() * 1000,
		Success:         true,
		Error:           fmt.Sprintf("Composite: %.2fms, Total: %.2fms, PNGs: %d", compTime.Seconds()*1000, totalTime.Seconds()*1000, len(compositedPNGs)),
	}, nil
}

func (s *lipSyncCompositeServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{
		Healthy: true,
	}, nil
}

func main() {
	modelPath := "d:/Projects/webcodecstest/minimal_server/models/sanders/sanders.onnx"
	bgDir := "d:/Projects/webcodecstest/minimal_server/models/sanders/frames"
	cropPath := "d:/Projects/webcodecstest/minimal_server/models/sanders/cache/crop_rectangles.json"
	numBackgrounds := 100 // Adjust based on your dataset

	log.Println("================================================================================")
	log.Println("🚀 gRPC LipSync Server with Compositing (Pure Go)")
	log.Println("================================================================================")
	log.Printf("   Model: %s", modelPath)
	log.Printf("   Backgrounds: %s", bgDir)
	log.Printf("   Crop Rects: %s", cropPath)
	log.Println()

	// Initialize inferencer
	log.Println("📦 Loading ONNX model...")
	inferencer, err := lipsyncinfer.NewInferencer(modelPath)
	if err != nil {
		log.Fatalf("❌ Failed to create inferencer: %v", err)
	}
	defer inferencer.Close()
	log.Println("✓ Model loaded")

	// Warmup
	log.Println("\n🔥 Warming up CUDA...")
	warmupStart := time.Now()
	dummyVisual := make([]float32, visualFrameSize)
	dummyAudio := make([]float32, audioFrameSize)
	_, err = inferencer.Infer(dummyVisual, dummyAudio)
	if err != nil {
		log.Fatalf("❌ Warmup failed: %v", err)
	}
	log.Printf("✓ Warmup complete (%.2fms)\n", time.Since(warmupStart).Seconds()*1000)

	// Create server
	server := &lipSyncCompositeServer{
		inferencer: inferencer,
		modelPath:  modelPath,
	}

	// Load backgrounds
	if err := server.loadBackgrounds(bgDir, numBackgrounds); err != nil {
		log.Fatalf("❌ Failed to load backgrounds: %v", err)
	}

	// Load crop rectangles
	if err := server.loadCropRects(cropPath); err != nil {
		log.Fatalf("❌ Failed to load crop rectangles: %v", err)
	}

	// Start gRPC server
	port := 50052 // Different port from non-composite server
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("❌ Failed to listen on port %d: %v", port, err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(100*1024*1024), // 100MB max message size
		grpc.MaxSendMsgSize(100*1024*1024),
	)
	pb.RegisterLipSyncServer(grpcServer, server)

	log.Printf("\n🚀 Server listening on port %d", port)
	log.Println("   Ready to composite 1280x720 frames!")
	log.Println("================================================================================")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("❌ Failed to serve: %v", err)
	}
}
