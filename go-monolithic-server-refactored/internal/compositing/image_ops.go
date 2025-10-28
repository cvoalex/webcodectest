package compositing

import (
	"image"
	"image/color"
	"image/draw"
)

// OutputToImage converts float32 output data [3, 320, 320] (CHW format) to RGBA image
// Data is in range [0, 1] and needs to be scaled to [0, 255]
func OutputToImage(outputData []float32) *image.RGBA {
	const width = 320
	const height = 320
	const channels = 3

	// Get pooled image
	img := RGBA320Pool.Get().(*image.RGBA)

	// Convert from CHW (Channel, Height, Width) to HWC (Height, Width, Channel)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Calculate indices for CHW format
			rIdx := 0*width*height + y*width + x
			gIdx := 1*width*height + y*width + x
			bIdx := 2*width*height + y*width + x

			// Clamp and convert to uint8
			r := clampFloat(outputData[rIdx])
			g := clampFloat(outputData[gIdx])
			b := clampFloat(outputData[bIdx])

			// Set pixel
			img.SetRGBA(x, y, color.RGBA{
				R: uint8(r * 255),
				G: uint8(g * 255),
				B: uint8(b * 255),
				A: 255,
			})
		}
	}

	return img
}

// ResizeImage resizes an RGBA image to target dimensions using bilinear interpolation
// Uses pooled buffers for efficiency
func ResizeImage(src *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	// Get pooled destination image
	dst := GetPooledImageForSize(targetWidth, targetHeight)

	// Bilinear interpolation
	xRatio := float32(srcWidth) / float32(targetWidth)
	yRatio := float32(srcHeight) / float32(targetHeight)

	for dstY := 0; dstY < targetHeight; dstY++ {
		for dstX := 0; dstX < targetWidth; dstX++ {
			// Calculate source coordinates
			srcXf := float32(dstX) * xRatio
			srcYf := float32(dstY) * yRatio

			srcX0 := int(srcXf)
			srcY0 := int(srcYf)
			srcX1 := srcX0 + 1
			srcY1 := srcY0 + 1

			// Clamp to source bounds
			if srcX1 >= srcWidth {
				srcX1 = srcWidth - 1
			}
			if srcY1 >= srcHeight {
				srcY1 = srcHeight - 1
			}

			// Interpolation weights
			xWeight := srcXf - float32(srcX0)
			yWeight := srcYf - float32(srcY0)

			// Get four neighboring pixels
			c00 := src.RGBAAt(srcX0, srcY0)
			c10 := src.RGBAAt(srcX1, srcY0)
			c01 := src.RGBAAt(srcX0, srcY1)
			c11 := src.RGBAAt(srcX1, srcY1)

			// Interpolate each channel
			r := bilinearInterp(c00.R, c10.R, c01.R, c11.R, xWeight, yWeight)
			g := bilinearInterp(c00.G, c10.G, c01.G, c11.G, xWeight, yWeight)
			b := bilinearInterp(c00.B, c10.B, c01.B, c11.B, xWeight, yWeight)
			a := bilinearInterp(c00.A, c10.A, c01.A, c11.A, xWeight, yWeight)

			dst.SetRGBA(dstX, dstY, color.RGBA{R: r, G: g, B: b, A: a})
		}
	}

	return dst
}

// ResizeImageSimple is a faster but lower-quality resize using nearest neighbor
// Good for non-critical resizing operations
func ResizeImageSimple(src *image.RGBA, targetWidth, targetHeight int) *image.RGBA {
	dst := GetPooledImageForSize(targetWidth, targetHeight)
	dstRect := image.Rect(0, 0, targetWidth, targetHeight)

	// Simple nearest neighbor scaling
	draw.Draw(dst, dstRect, src, src.Bounds().Min, draw.Src)

	return dst
}

// CompositeROI composites a mouth region onto a background at the specified crop rectangle
// Returns the composited background (caller should return it to pool when done)
func CompositeROI(background *image.RGBA, mouthImg *image.RGBA, cropRect image.Rectangle) {
	// Resize mouth to match crop rectangle dimensions
	cropWidth := cropRect.Dx()
	cropHeight := cropRect.Dy()

	resizedMouth := ResizeImage(mouthImg, cropWidth, cropHeight)
	defer ReturnPooledImageForSize(resizedMouth, cropWidth, cropHeight)

	// Draw resized mouth onto background at crop rectangle position
	draw.Draw(background, cropRect, resizedMouth, image.Point{0, 0}, draw.Over)
}

// BilinearInterp performs bilinear interpolation between four values
func bilinearInterp(c00, c10, c01, c11 uint8, xWeight, yWeight float32) uint8 {
	// Interpolate top edge
	top := float32(c00)*(1-xWeight) + float32(c10)*xWeight

	// Interpolate bottom edge
	bottom := float32(c01)*(1-xWeight) + float32(c11)*xWeight

	// Interpolate between top and bottom
	result := top*(1-yWeight) + bottom*yWeight

	// Clamp and convert
	if result < 0 {
		return 0
	}
	if result > 255 {
		return 255
	}
	return uint8(result)
}

// ClampFloat clamps a float32 value to [0, 1] range
func clampFloat(val float32) float32 {
	if val < 0 {
		return 0
	}
	if val > 1 {
		return 1
	}
	return val
}
