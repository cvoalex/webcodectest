package registry

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go-compositing-server/cache"
	"go-compositing-server/config"
)

// ModelInstance represents a model's compositing resources (backgrounds, crop rects)
type ModelInstance struct {
	ID               string
	Config           config.ModelConfig
	BackgroundCache  *cache.BackgroundCache
	CropRects        map[int][4]int // frameIdx -> [x1, y1, x2, y2]
	UsageCount       int64
	LastUsed         time.Time
	TotalCompositeMs float64
	MemoryBytes      int64 // Background cache memory
	LoadedAt         time.Time
	Mu               sync.RWMutex
}

// ModelRegistry manages model compositing resources
type ModelRegistry struct {
	models         map[string]*ModelInstance
	mu             sync.RWMutex
	config         *config.Config
	maxModels      int
	evictionPolicy string
}

// NewModelRegistry creates a new model registry
func NewModelRegistry(cfg *config.Config) (*ModelRegistry, error) {
	reg := &ModelRegistry{
		models:         make(map[string]*ModelInstance),
		config:         cfg,
		maxModels:      cfg.Capacity.MaxModels,
		evictionPolicy: cfg.Capacity.EvictionPolicy,
	}

	return reg, nil
}

// LoadModel loads model resources (backgrounds, crop rects)
func (r *ModelRegistry) LoadModel(modelID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if already loaded
	if _, exists := r.models[modelID]; exists {
		return nil // Already loaded
	}

	// Get model config
	modelCfg, exists := r.config.Models[modelID]
	if !exists {
		return fmt.Errorf("model '%s' not configured", modelID)
	}

	// Check capacity
	if len(r.models) >= r.maxModels {
		// Evict least used model
		if err := r.evictLeastUsedLocked(); err != nil {
			return fmt.Errorf("failed to evict model: %w", err)
		}
	}

	log.Printf("🔄 Loading compositing resources for model '%s'...", modelID)
	startTime := time.Now()

	// Resolve paths relative to models_root if configured
	cropRectsPath := modelCfg.CropRectsPath
	if !filepath.IsAbs(cropRectsPath) && r.config.ModelsRoot != "" {
		cropRectsPath = filepath.Join(r.config.ModelsRoot, cropRectsPath)
	}
	log.Printf("   Crop rects path: %s", cropRectsPath)

	backgroundDir := modelCfg.BackgroundDir
	if !filepath.IsAbs(backgroundDir) && r.config.ModelsRoot != "" {
		backgroundDir = filepath.Join(r.config.ModelsRoot, backgroundDir)
	}
	log.Printf("   Background dir: %s", backgroundDir)

	// Load crop rectangles
	cropRects, err := r.loadCropRects(cropRectsPath)
	if err != nil {
		return fmt.Errorf("failed to load crop rects: %w", err)
	}

	// Create background cache
	backgroundCache := cache.NewBackgroundCache(
		backgroundDir,
		r.config.Capacity.BackgroundCacheFrames,
	)

	// Preload backgrounds if configured
	if modelCfg.PreloadBackgrounds {
		numFrames := modelCfg.NumFrames
		if numFrames <= 0 {
			numFrames = 1000 // Default
		}
		log.Printf("   Preloading %d background frames into memory...", numFrames)
		frameIndices := make([]int, numFrames)
		for i := 0; i < numFrames; i++ {
			frameIndices[i] = i
		}
		if err := backgroundCache.Preload(frameIndices); err != nil {
			return fmt.Errorf("failed to preload backgrounds: %w", err)
		}
		log.Printf("   ✅ Preloaded %d frames", numFrames)
	}

	loadTime := time.Since(startTime)

	// Estimate memory usage (cache capacity × frame size)
	memoryBytes := int64(r.config.Capacity.BackgroundCacheFrames) * 1280 * 720 * 4 // RGBA

	instance := &ModelInstance{
		ID:              modelID,
		Config:          modelCfg,
		BackgroundCache: backgroundCache,
		CropRects:       cropRects,
		UsageCount:      0,
		LastUsed:        time.Now(),
		MemoryBytes:     memoryBytes,
		LoadedAt:        time.Now(),
	}

	r.models[modelID] = instance

	log.Printf("✅ Model '%s' compositing resources loaded in %.2fs (memory: %d MB)",
		modelID, loadTime.Seconds(), memoryBytes/(1024*1024))

	return nil
}

// loadCropRects loads crop rectangles from JSON file
func (r *ModelRegistry) loadCropRects(path string) (map[int][4]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read crop rects file: %w", err)
	}

	var cropRectsList [][4]int
	if err := json.Unmarshal(data, &cropRectsList); err != nil {
		return nil, fmt.Errorf("failed to parse crop rects: %w", err)
	}

	// Convert to map
	cropRects := make(map[int][4]int)
	for i, rect := range cropRectsList {
		cropRects[i] = rect
	}

	return cropRects, nil
}

// GetOrLoadModel gets a model, loading it if necessary
func (r *ModelRegistry) GetOrLoadModel(modelID string) (*ModelInstance, error) {
	// Try to get existing model (read lock)
	r.mu.RLock()
	instance, exists := r.models[modelID]
	r.mu.RUnlock()

	if exists {
		instance.Mu.Lock()
		instance.LastUsed = time.Now()
		instance.Mu.Unlock()
		return instance, nil
	}

	// Need to load model (write lock)
	if err := r.LoadModel(modelID); err != nil {
		return nil, err
	}

	// Get the loaded model
	r.mu.RLock()
	instance = r.models[modelID]
	r.mu.RUnlock()

	return instance, nil
}

// UnloadModel unloads a model from memory
func (r *ModelRegistry) UnloadModel(modelID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	instance, exists := r.models[modelID]
	if !exists {
		return fmt.Errorf("model '%s' not loaded", modelID)
	}

	// Clear background cache
	if instance.BackgroundCache != nil {
		instance.BackgroundCache.Clear()
	}

	delete(r.models, modelID)
	log.Printf("📤 Model '%s' compositing resources unloaded", modelID)

	return nil
}

// evictLeastUsedLocked evicts the least used model (caller must hold write lock)
func (r *ModelRegistry) evictLeastUsedLocked() error {
	if len(r.models) == 0 {
		return fmt.Errorf("no models to evict")
	}

	var victimID string
	var oldestTime time.Time
	var lowestCount int64 = -1

	for id, instance := range r.models {
		instance.Mu.RLock()
		if r.evictionPolicy == "lru" {
			// Least Recently Used
			if victimID == "" || instance.LastUsed.Before(oldestTime) {
				victimID = id
				oldestTime = instance.LastUsed
			}
		} else {
			// Least Frequently Used (default)
			if lowestCount == -1 || instance.UsageCount < lowestCount {
				victimID = id
				lowestCount = instance.UsageCount
			}
		}
		instance.Mu.RUnlock()
	}

	if victimID == "" {
		return fmt.Errorf("failed to select victim for eviction")
	}

	log.Printf("⚠️  Evicting model '%s' (%s policy)", victimID, r.evictionPolicy)
	instance := r.models[victimID]
	if instance.BackgroundCache != nil {
		instance.BackgroundCache.Clear()
	}
	delete(r.models, victimID)

	return nil
}

// RecordCompositing records compositing statistics
func (r *ModelRegistry) RecordCompositing(modelID string, compositeTimeMs float64) {
	r.mu.RLock()
	instance, exists := r.models[modelID]
	r.mu.RUnlock()

	if exists {
		instance.Mu.Lock()
		instance.UsageCount++
		instance.TotalCompositeMs += compositeTimeMs
		instance.LastUsed = time.Now()
		instance.Mu.Unlock()
	}
}

// ListModels returns all loaded models
func (r *ModelRegistry) ListModels() []*ModelInstance {
	r.mu.RLock()
	defer r.mu.RUnlock()

	instances := make([]*ModelInstance, 0, len(r.models))
	for _, instance := range r.models {
		instances = append(instances, instance)
	}
	return instances
}

// GetStats returns statistics for a specific model or all models
func (r *ModelRegistry) GetStats(modelID string) ([]*ModelInstance, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if modelID == "" {
		// Return all models
		instances := make([]*ModelInstance, 0, len(r.models))
		for _, instance := range r.models {
			instances = append(instances, instance)
		}
		return instances, nil
	}

	// Return specific model
	instance, exists := r.models[modelID]
	if !exists {
		return nil, fmt.Errorf("model '%s' not loaded", modelID)
	}

	return []*ModelInstance{instance}, nil
}

// GetLoadedCount returns the number of loaded models
func (r *ModelRegistry) GetLoadedCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.models)
}

// GetTotalMemory returns total memory used by loaded models
func (r *ModelRegistry) GetTotalMemory() int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var total int64
	for _, instance := range r.models {
		instance.Mu.RLock()
		total += instance.MemoryBytes
		instance.Mu.RUnlock()
	}
	return total
}

// Close releases all resources
func (r *ModelRegistry) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, instance := range r.models {
		if instance.BackgroundCache != nil {
			instance.BackgroundCache.Clear()
		}
	}
	r.models = make(map[string]*ModelInstance)
}
