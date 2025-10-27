package cache

import (
	"container/list"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"sync"
)

// BackgroundCache is an LRU cache for background frames
type BackgroundCache struct {
	capacity      int
	backgroundDir string

	cache   map[int]*list.Element
	lruList *list.List
	mu      sync.RWMutex

	// Statistics
	hits      int64
	misses    int64
	evictions int64
}

type cacheEntry struct {
	frameIdx int
	frame    *image.RGBA
}

// NewBackgroundCache creates a new LRU cache for background frames
func NewBackgroundCache(backgroundDir string, capacity int) *BackgroundCache {
	if capacity < 10 {
		capacity = 50 // Default to 50 frames
	}

	return &BackgroundCache{
		capacity:      capacity,
		backgroundDir: backgroundDir,
		cache:         make(map[int]*list.Element),
		lruList:       list.New(),
	}
}

// Get retrieves a background frame, loading from disk if needed
func (c *BackgroundCache) Get(frameIdx int) (*image.RGBA, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check cache
	if elem, exists := c.cache[frameIdx]; exists {
		// Move to front (most recently used)
		c.lruList.MoveToFront(elem)
		c.hits++
		return elem.Value.(*cacheEntry).frame, nil
	}

	// Cache miss - load from disk
	c.misses++

	frame, err := c.loadFrameFromDisk(frameIdx)
	if err != nil {
		return nil, err
	}

	// Add to cache
	c.add(frameIdx, frame)

	return frame, nil
}

// loadFrameFromDisk loads a frame from disk
func (c *BackgroundCache) loadFrameFromDisk(frameIdx int) (*image.RGBA, error) {
	framePath := filepath.Join(c.backgroundDir, fmt.Sprintf("frame_%04d.png", frameIdx))

	file, err := os.Open(framePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open frame %d: %w", frameIdx, err)
	}
	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode frame %d: %w", frameIdx, err)
	}

	// Convert to RGBA
	rgba := image.NewRGBA(img.Bounds())
	draw.Draw(rgba, rgba.Bounds(), img, image.Point{}, draw.Src)

	return rgba, nil
}

// add adds a frame to the cache, evicting LRU if necessary
func (c *BackgroundCache) add(frameIdx int, frame *image.RGBA) {
	// Check if cache is full
	if c.lruList.Len() >= c.capacity {
		// Evict least recently used
		oldest := c.lruList.Back()
		if oldest != nil {
			c.lruList.Remove(oldest)
			oldEntry := oldest.Value.(*cacheEntry)
			delete(c.cache, oldEntry.frameIdx)
			c.evictions++
		}
	}

	// Add new entry to front
	entry := &cacheEntry{
		frameIdx: frameIdx,
		frame:    frame,
	}
	elem := c.lruList.PushFront(entry)
	c.cache[frameIdx] = elem
}

// Preload preloads a range of frames into the cache
func (c *BackgroundCache) Preload(startIdx, count int) error {
	for i := 0; i < count; i++ {
		frameIdx := startIdx + i
		_, err := c.Get(frameIdx)
		if err != nil {
			return fmt.Errorf("failed to preload frame %d: %w", frameIdx, err)
		}
	}
	return nil
}

// GetStats returns cache statistics
func (c *BackgroundCache) GetStats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(c.hits) / float64(total) * 100
	}

	return map[string]interface{}{
		"capacity":  c.capacity,
		"size":      c.lruList.Len(),
		"hits":      c.hits,
		"misses":    c.misses,
		"evictions": c.evictions,
		"hit_rate":  hitRate,
	}
}

// Clear clears the cache
func (c *BackgroundCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.cache = make(map[int]*list.Element)
	c.lruList = list.New()
	c.hits = 0
	c.misses = 0
	c.evictions = 0
}

// Resize changes the cache capacity
func (c *BackgroundCache) Resize(newCapacity int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.capacity = newCapacity

	// Evict oldest entries if over capacity
	for c.lruList.Len() > newCapacity {
		oldest := c.lruList.Back()
		if oldest != nil {
			c.lruList.Remove(oldest)
			oldEntry := oldest.Value.(*cacheEntry)
			delete(c.cache, oldEntry.frameIdx)
			c.evictions++
		}
	}
}
