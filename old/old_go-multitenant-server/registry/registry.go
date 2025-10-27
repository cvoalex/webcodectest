package registry

import (
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"go-multitenant-server/cache"
	"go-multitenant-server/config"
	"go-multitenant-server/lipsyncinfer"
)

// ModelInstance represents a loaded model with all its resources
type ModelInstance struct {
	ID              string
	Inferencer      *lipsyncinfer.Inferencer
	BackgroundCache *cache.BackgroundCache // Lazy-loading cache
	CropRects       map[int][]int
	Config          config.ModelConfig

	// Usage statistics
	UsageCount       int64
	LastUsed         time.Time
	LoadedAt         time.Time
	TotalInferenceMs float64
	MemoryBytes      int64

	Mu sync.RWMutex // Protects stats (exported for external access)
}

// ModelRegistry manages multiple models with automatic loading/unloading
type ModelRegistry struct {
	models          map[string]*ModelInstance
	cfg             *config.Config
	Mu              sync.RWMutex // Exported for external access
	onnxLibraryPath string
}

type CropRect struct {
	Rect []int `json:"rect"` // [x1, y1, x2, y2]
}

// NewModelRegistry creates a new model registry
func NewModelRegistry(cfg *config.Config) (*ModelRegistry, error) {
	registry := &ModelRegistry{
		models:          make(map[string]*ModelInstance),
		cfg:             cfg,
		onnxLibraryPath: cfg.ONNX.LibraryPath,
	}

	// Preload models marked for preloading
	for modelID, modelCfg := range cfg.Models {
		if modelCfg.Preload {
			log.Printf("📦 Preloading model '%s'...", modelID)
			if err := registry.LoadModel(modelID); err != nil {
				return nil, fmt.Errorf("failed to preload model '%s': %w", modelID, err)
			}
		}
	}

	// Start stats reporter if configured
	if cfg.Logging.StatsReportMinutes > 0 {
		go registry.periodicStatsReporter(time.Duration(cfg.Logging.StatsReportMinutes) * time.Minute)
	}

	// Start idle unloader if configured
	if cfg.Capacity.IdleUnloadMinutes > 0 {
		go registry.idleUnloader(time.Duration(cfg.Capacity.IdleUnloadMinutes) * time.Minute)
	}

	return registry, nil
}

// LoadModel loads a model and all its resources
func (r *ModelRegistry) LoadModel(modelID string) error {
	r.Mu.Lock()
	defer r.Mu.Unlock()

	// Check if already loaded
	if _, exists := r.models[modelID]; exists {
		return nil // Already loaded
	}

	// Get model config
	modelCfg, exists := r.cfg.Models[modelID]
	if !exists {
		return fmt.Errorf("model '%s' not found in configuration", modelID)
	}

	// Check capacity before loading
	if len(r.models) >= r.cfg.Capacity.MaxModels {
		if err := r.evictLeastUsedModel(); err != nil {
			return fmt.Errorf("capacity full and eviction failed: %w", err)
		}
	}

	startLoad := time.Now()

	// Create inferencer
	log.Printf("   Loading ONNX model: %s", modelCfg.ModelPath)
	inferencer, err := lipsyncinfer.NewInferencer(modelCfg.ModelPath, r.onnxLibraryPath)
	if err != nil {
		return fmt.Errorf("failed to create inferencer: %w", err)
	}

	// Create background cache (lazy loading)
	cacheSize := r.cfg.Capacity.BackgroundCacheFrames
	if cacheSize < 10 {
		cacheSize = 50 // Default
	}
	log.Printf("   Creating background cache (capacity: %d frames): %s", cacheSize, modelCfg.BackgroundDir)
	backgroundCache := cache.NewBackgroundCache(modelCfg.BackgroundDir, cacheSize)

	// Preload first N frames into cache
	preloadCount := 10
	if preloadCount > cacheSize {
		preloadCount = cacheSize
	}
	if err := backgroundCache.Preload(0, preloadCount); err != nil {
		log.Printf("   Warning: Failed to preload backgrounds: %v", err)
	} else {
		log.Printf("   Preloaded %d background frames into cache", preloadCount)
	}

	// Load crop rectangles
	log.Printf("   Loading crop rectangles: %s", modelCfg.CropRectsPath)
	cropRects, err := r.loadCropRects(modelCfg.CropRectsPath)
	if err != nil {
		inferencer.Close()
		return fmt.Errorf("failed to load crop rectangles: %w", err)
	}

	// Perform warmup
	log.Printf("   Warming up CUDA...")
	visualDummy := make([]float32, 6*320*320)
	audioDummy := make([]float32, 32*16*16)
	_, err = inferencer.Infer(visualDummy, audioDummy)
	if err != nil {
		inferencer.Close()
		return fmt.Errorf("warmup failed: %w", err)
	}

	loadTime := time.Since(startLoad)

	// Estimate memory usage (reduced with lazy loading)
	memoryBytes := int64(modelCfg.MemoryEstimateMB * 1024 * 1024) // ONNX model
	memoryBytes += int64(cacheSize * 1280 * 720 * 4)              // Cached backgrounds

	// Create model instance
	instance := &ModelInstance{
		ID:              modelID,
		Inferencer:      inferencer,
		BackgroundCache: backgroundCache,
		CropRects:       cropRects,
		Config:          modelCfg,
		LoadedAt:        time.Now(),
		LastUsed:        time.Now(),
		MemoryBytes:     memoryBytes,
	}

	r.models[modelID] = instance

	log.Printf("✅ Model '%s' loaded in %.2fs (memory: %.2f MB)",
		modelID, loadTime.Seconds(), float64(memoryBytes)/(1024*1024))

	return nil
}

// UnloadModel unloads a model and frees resources
func (r *ModelRegistry) UnloadModel(modelID string) error {
	r.Mu.Lock()
	defer r.Mu.Unlock()

	instance, exists := r.models[modelID]
	if !exists {
		return fmt.Errorf("model '%s' not loaded", modelID)
	}

	// Close inferencer
	if err := instance.Inferencer.Close(); err != nil {
		log.Printf("Warning: Error closing inferencer for '%s': %v", modelID, err)
	}

	// Clear references
	delete(r.models, modelID)

	log.Printf("🗑️  Model '%s' unloaded (was active for %s, used %d times)",
		modelID, time.Since(instance.LoadedAt).Round(time.Second), instance.UsageCount)

	return nil
}

// GetOrLoadModel gets a model instance, loading it if necessary
func (r *ModelRegistry) GetOrLoadModel(modelID string) (*ModelInstance, error) {
	// Try read lock first (fast path)
	r.Mu.RLock()
	instance, exists := r.models[modelID]
	r.Mu.RUnlock()

	if exists {
		// Update last used time
		instance.Mu.Lock()
		instance.LastUsed = time.Now()
		instance.Mu.Unlock()
		return instance, nil
	}

	// Need to load (slow path)
	if err := r.LoadModel(modelID); err != nil {
		return nil, err
	}

	// Get the newly loaded instance
	r.Mu.RLock()
	instance = r.models[modelID]
	r.Mu.RUnlock()

	return instance, nil
}

// RecordInference records usage statistics for an inference
func (r *ModelRegistry) RecordInference(modelID string, inferenceTimeMs float64) {
	r.Mu.RLock()
	instance, exists := r.models[modelID]
	r.Mu.RUnlock()

	if !exists {
		return
	}

	instance.Mu.Lock()
	instance.UsageCount++
	instance.TotalInferenceMs += inferenceTimeMs
	instance.LastUsed = time.Now()
	instance.Mu.Unlock()
}

// ListModels returns information about all configured models
func (r *ModelRegistry) ListModels() []*ModelInstance {
	r.Mu.RLock()
	defer r.Mu.RUnlock()

	instances := make([]*ModelInstance, 0, len(r.models))
	for _, instance := range r.models {
		instances = append(instances, instance)
	}

	return instances
}

// GetStats returns statistics for a specific model or all models
func (r *ModelRegistry) GetStats(modelID string) ([]*ModelInstance, error) {
	if modelID == "" {
		// Return all models
		return r.ListModels(), nil
	}

	r.Mu.RLock()
	instance, exists := r.models[modelID]
	r.Mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model '%s' not loaded", modelID)
	}

	return []*ModelInstance{instance}, nil
}

// GetLoadedCount returns the number of currently loaded models
func (r *ModelRegistry) GetLoadedCount() int {
	r.Mu.RLock()
	defer r.Mu.RUnlock()
	return len(r.models)
}

// GetTotalMemory returns the total estimated memory usage
func (r *ModelRegistry) GetTotalMemory() int64 {
	r.Mu.RLock()
	defer r.Mu.RUnlock()

	var total int64
	for _, instance := range r.models {
		instance.Mu.RLock()
		total += instance.MemoryBytes
		instance.Mu.RUnlock()
	}

	return total
}

// evictLeastUsedModel unloads the least recently used model (LRU)
// or least frequently used model (LFU) based on configuration
// Must be called with write lock held!
func (r *ModelRegistry) evictLeastUsedModel() error {
	if len(r.models) == 0 {
		return fmt.Errorf("no models to evict")
	}

	var candidates []*ModelInstance
	for _, instance := range r.models {
		candidates = append(candidates, instance)
	}

	// Sort by eviction policy
	if r.cfg.Capacity.EvictionPolicy == "lru" {
		// Least Recently Used
		sort.Slice(candidates, func(i, j int) bool {
			candidates[i].Mu.RLock()
			candidates[j].Mu.RLock()
			defer candidates[i].Mu.RUnlock()
			defer candidates[j].Mu.RUnlock()
			return candidates[i].LastUsed.Before(candidates[j].LastUsed)
		})
	} else {
		// Least Frequently Used
		sort.Slice(candidates, func(i, j int) bool {
			candidates[i].Mu.RLock()
			candidates[j].Mu.RLock()
			defer candidates[i].Mu.RUnlock()
			defer candidates[j].Mu.RUnlock()
			return candidates[i].UsageCount < candidates[j].UsageCount
		})
	}

	// Evict the first candidate
	toEvict := candidates[0]
	log.Printf("⚠️  Evicting model '%s' (policy: %s, usage: %d, last used: %s)",
		toEvict.ID, r.cfg.Capacity.EvictionPolicy, toEvict.UsageCount,
		time.Since(toEvict.LastUsed).Round(time.Second))

	// Close and remove
	if err := toEvict.Inferencer.Close(); err != nil {
		log.Printf("Warning: Error closing inferencer during eviction: %v", err)
	}

	delete(r.models, toEvict.ID)

	return nil
}

// periodicStatsReporter logs model statistics periodically
func (r *ModelRegistry) periodicStatsReporter(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		r.Mu.RLock()
		if len(r.models) == 0 {
			r.Mu.RUnlock()
			continue
		}

		log.Println("📊 Model Statistics Report:")
		log.Printf("   Loaded models: %d/%d", len(r.models), r.cfg.Capacity.MaxModels)
		log.Printf("   Total memory: %.2f MB", float64(r.GetTotalMemory())/(1024*1024))

		for modelID, instance := range r.models {
			instance.Mu.RLock()
			log.Printf("   • %s: usage=%d, last=%s, avg_inference=%.2fms",
				modelID,
				instance.UsageCount,
				time.Since(instance.LastUsed).Round(time.Second),
				instance.TotalInferenceMs/float64(instance.UsageCount))
			instance.Mu.RUnlock()
		}
		r.Mu.RUnlock()
	}
}

// idleUnloader automatically unloads models that haven't been used
func (r *ModelRegistry) idleUnloader(idleTimeout time.Duration) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		now := time.Now()
		r.Mu.Lock()

		for modelID, instance := range r.models {
			instance.Mu.RLock()
			idleTime := now.Sub(instance.LastUsed)
			instance.Mu.RUnlock()

			if idleTime > idleTimeout {
				log.Printf("⏰ Auto-unloading idle model '%s' (idle for %s)",
					modelID, idleTime.Round(time.Second))

				if err := instance.Inferencer.Close(); err != nil {
					log.Printf("Warning: Error closing idle model: %v", err)
				}

				delete(r.models, modelID)
			}
		}

		r.Mu.Unlock()
	}
}

// loadBackgrounds loads background frames from directory
func (r *ModelRegistry) loadBackgrounds(bgDir string, numFrames int) ([]*image.RGBA, error) {
	backgrounds := make([]*image.RGBA, numFrames)

	for i := 0; i < numFrames; i++ {
		framePath := filepath.Join(bgDir, fmt.Sprintf("frame_%04d.png", i))
		file, err := os.Open(framePath)
		if err != nil {
			return nil, fmt.Errorf("failed to load background frame %d: %w", i, err)
		}

		img, err := png.Decode(file)
		file.Close()
		if err != nil {
			return nil, fmt.Errorf("failed to decode background frame %d: %w", i, err)
		}

		// Convert to RGBA
		rgba := image.NewRGBA(img.Bounds())
		draw.Draw(rgba, rgba.Bounds(), img, image.Point{}, draw.Src)
		backgrounds[i] = rgba
	}

	return backgrounds, nil
}

// loadCropRects loads crop rectangles from JSON
func (r *ModelRegistry) loadCropRects(jsonPath string) (map[int][]int, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read crop rectangles: %w", err)
	}

	var cropData map[string]CropRect
	if err := json.Unmarshal(data, &cropData); err != nil {
		return nil, fmt.Errorf("failed to parse crop rectangles: %w", err)
	}

	cropRects := make(map[int][]int)
	for k, v := range cropData {
		var frameIdx int
		fmt.Sscanf(k, "%d", &frameIdx)
		cropRects[frameIdx] = v.Rect
	}

	return cropRects, nil
}

// Close closes all loaded models
func (r *ModelRegistry) Close() error {
	r.Mu.Lock()
	defer r.Mu.Unlock()

	for modelID, instance := range r.models {
		log.Printf("Closing model '%s'...", modelID)
		if err := instance.Inferencer.Close(); err != nil {
			log.Printf("Warning: Error closing model '%s': %v", modelID, err)
		}
	}

	r.models = make(map[string]*ModelInstance)
	return nil
}
