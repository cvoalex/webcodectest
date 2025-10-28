package compositing

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
)

// Compositor handles the compositing of generated mouth regions onto background frames
type Compositor struct {
	jpegQuality int
}

// NewCompositor creates a new compositor with the specified JPEG quality
func NewCompositor(jpegQuality int) *Compositor {
	return &Compositor{
		jpegQuality: jpegQuality,
	}
}

// CompositeFrame composites a mouth region onto a background frame and encodes to JPEG
// Parameters:
//   - background: Full HD background frame (1920x1080)
//   - mouthRegion: Generated mouth output data [3, 320, 320] in CHW format
//   - cropRect: Rectangle defining where to place the mouth on the background
//
// Returns JPEG-encoded frame bytes
func (c *Compositor) CompositeFrame(background *image.RGBA, mouthRegion []float32, cropRect image.Rectangle) ([]byte, error) {
	// Convert mouth region output to RGBA image
	mouthImg := OutputToImage(mouthRegion)
	defer RGBA320Pool.Put(mouthImg) // Return to pool when done

	// Composite mouth onto background
	CompositeROI(background, mouthImg, cropRect)

	// Encode to JPEG
	return c.encodeJPEG(background)
}

// EncodeJPEG encodes an RGBA image to JPEG format
func (c *Compositor) encodeJPEG(img *image.RGBA) ([]byte, error) {
	// Get pooled buffer
	buf := BufferPool.Get().(*bytes.Buffer)
	defer BufferPool.Put(buf)
	buf.Reset()

	// Encode as JPEG
	opts := &jpeg.Options{Quality: c.jpegQuality}
	if err := jpeg.Encode(buf, img, opts); err != nil {
		return nil, fmt.Errorf("JPEG encoding failed: %w", err)
	}

	// Copy to new slice (buf will be returned to pool)
	result := make([]byte, buf.Len())
	copy(result, buf.Bytes())

	return result, nil
}
