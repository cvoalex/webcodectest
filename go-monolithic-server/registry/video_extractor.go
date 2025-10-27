package registry

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

// extractFramesFromVideo extracts frames from a source video to the background directory
func extractFramesFromVideo(sourceVideo string, outputDir string, numFrames int) error {
	log.Printf("📹 Extracting %d frames from video: %s", numFrames, sourceVideo)
	startTime := time.Now()

	// Verify source video exists
	if _, err := os.Stat(sourceVideo); err != nil {
		return fmt.Errorf("source video not found: %s", sourceVideo)
	}

	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Try to use FFmpeg first (faster), fall back to Python/OpenCV
	if err := extractWithFFmpeg(sourceVideo, outputDir, numFrames); err == nil {
		extractTime := time.Since(startTime)
		log.Printf("✅ Extracted %d frames using FFmpeg in %.2fs", numFrames, extractTime.Seconds())
		return nil
	}

	// Fall back to Python script
	log.Printf("⚠️  FFmpeg not available, trying Python/OpenCV...")
	if err := extractWithPython(sourceVideo, outputDir, numFrames); err != nil {
		return fmt.Errorf("failed to extract frames: %w", err)
	}

	extractTime := time.Since(startTime)
	log.Printf("✅ Extracted %d frames using Python in %.2fs", numFrames, extractTime.Seconds())
	return nil
}

// extractWithFFmpeg uses FFmpeg to extract frames (fastest method)
func extractWithFFmpeg(sourceVideo string, outputDir string, numFrames int) error {
	// Try to find ffmpeg
	ffmpegCmd := "ffmpeg"
	if runtime.GOOS == "windows" {
		// Check if ffmpeg.exe exists in PATH
		if _, err := exec.LookPath("ffmpeg.exe"); err == nil {
			ffmpegCmd = "ffmpeg.exe"
		} else {
			return fmt.Errorf("ffmpeg not found in PATH")
		}
	}

	// Build FFmpeg command
	// Extract frames as JPEG with 95% quality, numbered sequentially
	outputPattern := filepath.Join(outputDir, "frame_%06d.jpg")
	args := []string{
		"-i", sourceVideo,
		"-vf", fmt.Sprintf("select='lt(n,%d)'", numFrames), // Select first N frames
		"-vsync", "0", // Don't drop frames
		"-q:v", "2", // JPEG quality (2 = ~95%)
		"-start_number", "0",
		outputPattern,
	}

	cmd := exec.Command(ffmpegCmd, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ffmpeg failed: %w (output: %s)", err, string(output))
	}

	return nil
}

// extractWithPython uses Python/OpenCV to extract frames
func extractWithPython(sourceVideo string, outputDir string, numFrames int) error {
	// Create inline Python script
	pythonScript := fmt.Sprintf(`
import cv2
import os
import sys

video_path = r"%s"
output_dir = r"%s"
max_frames = %d

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Failed to open video", file=sys.stderr)
    sys.exit(1)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Extract frames
frame_count = 0
while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame with 6-digit zero padding
    filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames")
`, sourceVideo, outputDir, numFrames)

	// Find Python executable
	pythonCmd := "python"
	if runtime.GOOS == "windows" {
		if _, err := exec.LookPath("python.exe"); err == nil {
			pythonCmd = "python.exe"
		} else if _, err := exec.LookPath("python3.exe"); err == nil {
			pythonCmd = "python3.exe"
		}
	} else {
		if _, err := exec.LookPath("python3"); err == nil {
			pythonCmd = "python3"
		}
	}

	// Execute Python script
	cmd := exec.Command(pythonCmd, "-c", pythonScript)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("python extraction failed: %w (output: %s)", err, string(output))
	}

	return nil
}

// isDirectoryEmpty checks if a directory is empty or doesn't exist
func isDirectoryEmpty(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		// Directory doesn't exist or can't be read
		return true
	}

	// Count actual image files (not hidden files or directories)
	imageCount := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		ext := filepath.Ext(name)
		if ext == ".jpg" || ext == ".jpeg" || ext == ".png" {
			imageCount++
		}
	}

	return imageCount == 0
}
