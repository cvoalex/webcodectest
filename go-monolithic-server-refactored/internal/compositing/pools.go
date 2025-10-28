package compositing

import (
	"bytes"
	"image"
	"sync"
)

// Memory pools for efficient buffer reuse
// These pools eliminate repeated allocations and reduce GC pressure

// BufferPool provides reusable byte buffers for JPEG encoding
var BufferPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

// RGBA320Pool provides reusable 320x320 RGBA images for compositing
var RGBA320Pool = sync.Pool{
	New: func() interface{} {
		return image.NewRGBA(image.Rect(0, 0, 320, 320))
	},
}

// RGBAFullHDPool provides reusable 1920x1080 RGBA images for background frames
var RGBAFullHDPool = sync.Pool{
	New: func() interface{} {
		return image.NewRGBA(image.Rect(0, 0, 1920, 1080))
	},
}

// RGBAResizePool provides reusable images for resize operations
// Max size 400x400 to cover most crop rectangles
var RGBAResizePool = sync.Pool{
	New: func() interface{} {
		return image.NewRGBA(image.Rect(0, 0, 400, 400))
	},
}

// MelWindowPool provides reusable [80][16] float32 arrays for mel-spectrogram windows
var MelWindowPool = sync.Pool{
	New: func() interface{} {
		window := make([][]float32, 80)
		for i := 0; i < 80; i++ {
			window[i] = make([]float32, 16)
		}
		return window
	},
}

// GetPooledImageForSize returns an appropriate pooled image for the given dimensions
func GetPooledImageForSize(width, height int) *image.RGBA {
	if width == 320 && height == 320 {
		return RGBA320Pool.Get().(*image.RGBA)
	} else if width == 1920 && height == 1080 {
		return RGBAFullHDPool.Get().(*image.RGBA)
	} else if width <= 400 && height <= 400 {
		return RGBAResizePool.Get().(*image.RGBA)
	}
	// Fallback: allocate directly for unusual sizes
	return image.NewRGBA(image.Rect(0, 0, width, height))
}

// ReturnPooledImageForSize returns an image to the appropriate pool
func ReturnPooledImageForSize(img *image.RGBA, width, height int) {
	if width == 320 && height == 320 {
		RGBA320Pool.Put(img)
	} else if width == 1920 && height == 1080 {
		RGBAFullHDPool.Put(img)
	} else if width <= 400 && height <= 400 {
		RGBAResizePool.Put(img)
	}
	// Non-pooled images are simply garbage collected
}
