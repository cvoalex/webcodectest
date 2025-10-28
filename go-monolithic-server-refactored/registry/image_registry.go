package registry

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go-monolithic-server/config"
)

// ImageRegistry manages background images and crop rects for compositing
type ImageRegistry struct {
	models map[string]*ImageModelData
	mu     sync.RWMutex
	config *config.Config
}

// ImageModelData holds background images and crop rects for a single model
type ImageModelData struct {
	ModelID     string
	Backgrounds []*image.RGBA
	CropRects   []image.Rectangle
	UsageCount  int64
	LastUsed    time.Time
	MemoryBytes int64
	LoadedAt    time.Time
}

// NewImageRegistry creates a new image registry
func NewImageRegistry(cfg *config.Config) (*ImageRegistry, error) {
	reg := &ImageRegistry{
		models: make(map[string]*ImageModelData),
		config: cfg,
	}

	// Pre-extract visual frames for testing (crops and ROIs) if configured
	log.Println("🎬 Checking for visual frame extraction needs...")
	for modelID, modelCfg := range cfg.Models {
		// Extract crops frames if directory empty
		if modelCfg.CropsVideoPath != "" && modelCfg.CropsFramesDir != "" {
			if isDirectoryEmpty(modelCfg.CropsFramesDir) {
				log.Printf("📹 Extracting crops frames for '%s' from %s...", modelID, modelCfg.CropsVideoPath)
				if err := extractFramesFromVideo(modelCfg.CropsVideoPath, modelCfg.CropsFramesDir, modelCfg.NumFrames); err != nil {
					log.Printf("⚠️  Failed to extract crops frames: %v", err)
				}
			}
		}

		// Extract ROIs frames if directory empty
		if modelCfg.ROIsVideoPath != "" && modelCfg.ROIsFramesDir != "" {
			if isDirectoryEmpty(modelCfg.ROIsFramesDir) {
				log.Printf("📹 Extracting ROIs frames for '%s' from %s...", modelID, modelCfg.ROIsVideoPath)
				if err := extractFramesFromVideo(modelCfg.ROIsVideoPath, modelCfg.ROIsFramesDir, modelCfg.NumFrames); err != nil {
					log.Printf("⚠️  Failed to extract ROIs frames: %v", err)
				}
			}
		}
	}

	// Preload models if configured
	for modelID, modelCfg := range cfg.Models {
		if modelCfg.PreloadBackgrounds {
			if err := reg.LoadModel(modelID); err != nil {
				log.Printf("⚠️  Failed to preload backgrounds for '%s': %v", modelID, err)
			}
		}
	}

	return reg, nil
}

// GetModelData gets model data, loading if necessary
func (r *ImageRegistry) GetModelData(modelID string) (*ImageModelData, error) {
	r.mu.RLock()
	data, exists := r.models[modelID]
	r.mu.RUnlock()

	if exists {
		data.UsageCount++
		data.LastUsed = time.Now()
		return data, nil
	}

	// Load model
	return r.loadModel(modelID)
}

// LoadModel explicitly loads a model's backgrounds and crop rects
func (r *ImageRegistry) LoadModel(modelID string) error {
	_, err := r.loadModel(modelID)
	return err
}

func (r *ImageRegistry) loadModel(modelID string) (*ImageModelData, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check again after acquiring lock
	if data, exists := r.models[modelID]; exists {
		return data, nil
	}

	// Get model config
	modelCfg, exists := r.config.Models[modelID]
	if !exists {
		return nil, fmt.Errorf("model '%s' not configured", modelID)
	}

	log.Printf("🖼️  Loading backgrounds for model '%s'...", modelID)
	startTime := time.Now()

	// Check if we need to extract frames from source video
	if isDirectoryEmpty(modelCfg.BackgroundDir) {
		if modelCfg.SourceVideo == "" {
			return nil, fmt.Errorf("background directory '%s' is empty and no source_video configured", modelCfg.BackgroundDir)
		}

		log.Printf("📹 Background directory empty, extracting frames from source video...")
		if err := extractFramesFromVideo(modelCfg.SourceVideo, modelCfg.BackgroundDir, modelCfg.NumFrames); err != nil {
			return nil, fmt.Errorf("failed to extract frames from video: %w", err)
		}
	}

	// Load crop rects
	cropRects, err := loadCropRects(modelCfg.CropRectsPath, modelCfg.NumFrames)
	if err != nil {
		return nil, fmt.Errorf("failed to load crop rects: %w", err)
	}

	// Load backgrounds
	backgrounds, memoryBytes, err := loadBackgrounds(modelCfg.BackgroundDir, modelCfg.NumFrames)
	if err != nil {
		return nil, fmt.Errorf("failed to load backgrounds: %w", err)
	}

	data := &ImageModelData{
		ModelID:     modelID,
		Backgrounds: backgrounds,
		CropRects:   cropRects,
		LoadedAt:    time.Now(),
		LastUsed:    time.Now(),
		MemoryBytes: memoryBytes,
	}

	r.models[modelID] = data

	loadTime := time.Since(startTime)
	log.Printf("✅ Loaded backgrounds for '%s' in %.2fs (%d frames, %.2f MB)",
		modelID, loadTime.Seconds(), len(backgrounds),
		float64(memoryBytes)/(1024*1024))

	return data, nil
}

// UnloadModel unloads a model's backgrounds
func (r *ImageRegistry) UnloadModel(modelID string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.models, modelID)
	log.Printf("🗑️  Unloaded backgrounds for model '%s'", modelID)
}

// GetLoadedCount returns the number of loaded models
func (r *ImageRegistry) GetLoadedCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.models)
}

// Close releases all resources
func (r *ImageRegistry) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.models = make(map[string]*ImageModelData)
}

// loadCropRects loads crop rectangles from JSON file
func loadCropRects(path string, numFrames int) ([]image.Rectangle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read crop rects file: %w", err)
	}

	var rects [][4]int
	if err := json.Unmarshal(data, &rects); err != nil {
		return nil, fmt.Errorf("failed to parse crop rects JSON: %w", err)
	}

	// Convert to image.Rectangle
	cropRects := make([]image.Rectangle, len(rects))
	for i, r := range rects {
		cropRects[i] = image.Rect(r[0], r[1], r[2], r[3])
	}

	return cropRects, nil
}

// loadBackgrounds loads all background images from a directory
func loadBackgrounds(dir string, numFrames int) ([]*image.RGBA, int64, error) {
	backgrounds := make([]*image.RGBA, 0, numFrames)
	var totalMemory int64

	for i := 0; i < numFrames; i++ {
		// Try different file extensions and naming conventions
		var imgPath string
		var found bool

		for _, ext := range []string{".png", ".jpg", ".jpeg"} {
			// Try with zero-padding (frame_0000.png)
			path := filepath.Join(dir, fmt.Sprintf("frame_%04d%s", i, ext))
			if _, err := os.Stat(path); err == nil {
				imgPath = path
				found = true
				break
			}
			// Try without padding (frame_0.png)
			path = filepath.Join(dir, fmt.Sprintf("frame_%d%s", i, ext))
			if _, err := os.Stat(path); err == nil {
				imgPath = path
				found = true
				break
			}
		}

		if !found {
			return nil, 0, fmt.Errorf("background frame %d not found in %s", i, dir)
		}

		// Load image
		file, err := os.Open(imgPath)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to open frame %d: %w", i, err)
		}

		var img image.Image
		ext := filepath.Ext(imgPath)
		if ext == ".png" {
			img, err = png.Decode(file)
		} else {
			img, err = jpeg.Decode(file)
		}
		file.Close()

		if err != nil {
			return nil, 0, fmt.Errorf("failed to decode frame %d: %w", i, err)
		}

		// Convert to RGBA
		rgba := image.NewRGBA(img.Bounds())
		for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
			for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
				rgba.Set(x, y, color.RGBAModel.Convert(img.At(x, y)))
			}
		}

		backgrounds = append(backgrounds, rgba)
		totalMemory += int64(len(rgba.Pix))
	}

	return backgrounds, totalMemory, nil
}
