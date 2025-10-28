package server

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"log"
	"unsafe"
)

// compositeFrame composites a mouth region onto a background image at the crop rect
// EXACT copy from original main.go - NO changes to algorithm
func compositeFrame(background *image.RGBA, mouthRegion []float32, cropRect image.Rectangle, jpegQuality int) ([]byte, error) {
	// Convert float32 output to RGBA image (uses pooled 320x320 image)
	mouthImg := outputToImage(mouthRegion)
	defer rgbaPool320.Put(mouthImg) // Return to pool when done

	// Get crop rect dimensions
	x := cropRect.Min.X
	y := cropRect.Min.Y
	w := cropRect.Dx()
	h := cropRect.Dy()

	// Resize mouth region to match crop rect (uses pooled image)
	resized := resizeImagePooled(mouthImg, w, h)
	defer rgbaPoolResize.Put(resized) // Return to pool when done

	// Clone background using pooled image
	bgBounds := background.Bounds()
	result := getPooledImageForSize(bgBounds.Dx(), bgBounds.Dy())
	defer returnPooledImageForSize(result, bgBounds.Dx(), bgBounds.Dy())

	draw.Draw(result, result.Bounds(), background, image.Point{}, draw.Src)

	// Paste resized mouth region onto background
	dstRect := image.Rect(x, y, x+w, y+h)
	// Create a sub-image view to only draw the needed portion
	resizedView := resized.SubImage(image.Rect(0, 0, w, h)).(*image.RGBA)
	draw.Draw(result, dstRect, resizedView, image.Point{}, draw.Src)

	// Encode to JPEG using pooled buffer
	buf := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufferPool.Put(buf)

	err := jpeg.Encode(buf, result, &jpeg.Options{Quality: jpegQuality})
	if err != nil {
		return nil, err
	}

	// Copy to new slice (buf will be returned to pool)
	jpegData := make([]byte, buf.Len())
	copy(jpegData, buf.Bytes())

	return jpegData, nil
}

// outputToImage converts model output float32 data to RGBA image
func outputToImage(outputData []float32) *image.RGBA {
	// Get image from pool
	img := rgbaPool320.Get().(*image.RGBA)

	// Convert BGR float32 [0,1] to RGB bytes [0,255]
	// Model outputs BGR in [C, H, W] format: [3, 320, 320] (same as OpenCV input)
	for y := 0; y < 320; y++ {
		for x := 0; x < 320; x++ {
			// BGR order from ONNX model (OpenCV convention)
			b := outputData[0*320*320+y*320+x]
			g := outputData[1*320*320+y*320+x]
			r := outputData[2*320*320+y*320+x]

			// Clamp and convert to bytes
			rByte := uint8(clampFloat(r * 255.0))
			gByte := uint8(clampFloat(g * 255.0))
			bByte := uint8(clampFloat(b * 255.0))

			img.SetRGBA(x, y, color.RGBA{R: rByte, G: gByte, B: bByte, A: 255})
		}
	}

	return img
}

// resizeImage resizes an image using bilinear interpolation
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

			c00 := src.RGBAAt(x0, y0)
			c10 := src.RGBAAt(x1, y0)
			c01 := src.RGBAAt(x0, y1)
			c11 := src.RGBAAt(x1, y1)

			r := bilinearInterp(c00.R, c10.R, c01.R, c11.R, xWeight, yWeight)
			g := bilinearInterp(c00.G, c10.G, c01.G, c11.G, xWeight, yWeight)
			b := bilinearInterp(c00.B, c10.B, c01.B, c11.B, xWeight, yWeight)

			dst.SetRGBA(dstX, dstY, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return dst
}

// resizeImagePooled resizes an image using bilinear interpolation with pooled destination
func resizeImagePooled(src *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	// Get pooled image (400x400 max)
	dst := rgbaPoolResize.Get().(*image.RGBA)

	// Verify the pooled image is large enough
	if targetWidth > 400 || targetHeight > 400 {
		// Fallback to regular allocation for oversized requests
		log.Printf("⚠️  Resize target (%dx%d) exceeds pool size (400x400), allocating", targetWidth, targetHeight)
		return resizeImage(src, targetWidth, targetHeight)
	}

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

			c00 := src.RGBAAt(x0, y0)
			c10 := src.RGBAAt(x1, y0)
			c01 := src.RGBAAt(x0, y1)
			c11 := src.RGBAAt(x1, y1)

			r := bilinearInterp(c00.R, c10.R, c01.R, c11.R, xWeight, yWeight)
			g := bilinearInterp(c00.G, c10.G, c01.G, c11.G, xWeight, yWeight)
			b := bilinearInterp(c00.B, c10.B, c01.B, c11.B, xWeight, yWeight)

			dst.SetRGBA(dstX, dstY, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return dst
}

// getPooledImageForSize returns a pooled image appropriate for the given size
func getPooledImageForSize(width, height int) *image.RGBA {
	// Full HD (1920x1080) - most common for backgrounds
	if width == 1920 && height == 1080 {
		return rgbaPoolFullHD.Get().(*image.RGBA)
	}

	// For other sizes, allocate (could add more pools for common sizes)
	// In production you might want to add pools for 1280x720, 2560x1440, etc.
	return image.NewRGBA(image.Rect(0, 0, width, height))
}

// returnPooledImageForSize returns an image to the appropriate pool
func returnPooledImageForSize(img *image.RGBA, width, height int) {
	// Full HD (1920x1080)
	if width == 1920 && height == 1080 {
		rgbaPoolFullHD.Put(img)
	}
	// For other sizes, let GC handle it (no pool available)
}

// bilinearInterp performs bilinear interpolation for a single channel
func bilinearInterp(c00, c10, c01, c11 uint8, xWeight, yWeight float32) uint8 {
	top := float32(c00)*(1-xWeight) + float32(c10)*xWeight
	bottom := float32(c01)*(1-xWeight) + float32(c11)*xWeight
	result := top*(1-yWeight) + bottom*yWeight
	return uint8(result)
}

// clampFloat clamps a float32 value between 0 and 255
func clampFloat(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 255 {
		return 255
	}
	return val
}

// bytesToFloat32 converts a byte slice to float32 slice (zero-copy using unsafe)
func bytesToFloat32(b []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}
