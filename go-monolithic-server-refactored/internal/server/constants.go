package server

import (
	"bytes"
	"image"
	"sync"
)

// Frame size constants (EXACT values from original main.go)
const (
	visualFrameSize = 6 * 320 * 320 // BGR CHW format (6 channels × 320 × 320 for double-sided faces)
	audioFrameSize  = 32 * 16 * 16  // Reshaped audio features (32 × 16 × 16 = 8192)
	outputFrameSize = 3 * 320 * 320 // Output mouth region (BGR CHW format)
)

// Memory pools for image and buffer reuse (EXACT from original main.go)
var (
	// Buffer pool for JPEG encoding - reuse byte buffers
	bufferPool = sync.Pool{
		New: func() interface{} {
			return new(bytes.Buffer)
		},
	}

	// RGBA image pool for compositing - reuse images instead of allocating per frame
	rgbaPool320 = sync.Pool{
		New: func() interface{} {
			return image.NewRGBA(image.Rect(0, 0, 320, 320))
		},
	}

	// Pool for full HD background images (1920x1080)
	rgbaPoolFullHD = sync.Pool{
		New: func() interface{} {
			return image.NewRGBA(image.Rect(0, 0, 1920, 1080))
		},
	}

	// Pool for common resize dimensions (we'll use a max size and subset it)
	rgbaPoolResize = sync.Pool{
		New: func() interface{} {
			// Allocate max size (400x400 should cover most crop rects)
			return image.NewRGBA(image.Rect(0, 0, 400, 400))
		},
	}

	// Pool for mel windows [80][16] used in audio processing
	melWindowPool = sync.Pool{
		New: func() interface{} {
			window := make([][]float32, 80)
			for i := 0; i < 80; i++ {
				window[i] = make([]float32, 16)
			}
			return window
		},
	}
)
