package audio

import (
	"fmt"
)

// ExtractMelWindows extracts sliding 16-frame windows from a mel-spectrogram
// for audio feature encoding. Each window corresponds to one output frame.
//
// Input: melSpec [numFrames, numMelBins] - full mel-spectrogram
// Output: windows [numWindows][80][16] - array of mel windows
//
// The window extraction uses a sliding window approach where:
// - Window 0 uses frames 0-15
// - Window 1 uses frames 1-16
// - Window i uses frames i to i+15
//
// For the last few frames that don't have 16 frames ahead, we:
// - Pad with the last frame repeated
// - Or return fewer windows (if strict=true)
func ExtractMelWindows(melSpec [][]float32, windowSize int, strict bool) ([][][]float32, error) {
	if len(melSpec) == 0 {
		return nil, fmt.Errorf("empty mel-spectrogram")
	}

	numFrames := len(melSpec)
	numMelBins := len(melSpec[0])

	if numMelBins != 80 {
		return nil, fmt.Errorf("expected 80 mel bins, got %d", numMelBins)
	}

	if windowSize <= 0 {
		windowSize = 16 // Default to 16 frames
	}

	// Calculate number of windows
	var numWindows int
	if strict {
		// Only create windows where we have enough frames
		numWindows = numFrames - windowSize + 1
		if numWindows < 0 {
			return nil, fmt.Errorf("not enough frames: have %d, need %d", numFrames, windowSize)
		}
	} else {
		// Create one window per frame (padding at the end)
		numWindows = numFrames
	}

	windows := make([][][]float32, numWindows)

	for w := 0; w < numWindows; w++ {
		// Create window [80][16] (transposed from [16][80])
		window := make([][]float32, 80)
		for i := 0; i < 80; i++ {
			window[i] = make([]float32, windowSize)
		}

		// Fill window with frames
		for f := 0; f < windowSize; f++ {
			frameIdx := w + f

			// Handle padding if we run out of frames
			if frameIdx >= numFrames {
				frameIdx = numFrames - 1 // Repeat last frame
			}

			// Copy frame into window (transpose: [frame][mel] -> [mel][frame])
			for m := 0; m < 80; m++ {
				window[m][f] = melSpec[frameIdx][m]
			}
		}

		windows[w] = window
	}

	return windows, nil
}

// ExtractMelWindowsForBatch extracts mel windows aligned with a specific batch size
// This ensures we have exactly batchSize windows for processing.
//
// Input: melSpec [numFrames, 80] - full mel-spectrogram
//
//	batchSize - number of windows to extract
//	startFrame - which frame to start from (for chunked processing)
//
// Output: windows [batchSize][80][16] - array of mel windows
func ExtractMelWindowsForBatch(melSpec [][]float32, batchSize int, startFrame int) ([][][]float32, error) {
	if len(melSpec) == 0 {
		return nil, fmt.Errorf("empty mel-spectrogram")
	}

	numFrames := len(melSpec)
	numMelBins := len(melSpec[0])

	if numMelBins != 80 {
		return nil, fmt.Errorf("expected 80 mel bins, got %d", numMelBins)
	}

	if startFrame < 0 || startFrame >= numFrames {
		return nil, fmt.Errorf("invalid start frame: %d (total frames: %d)", startFrame, numFrames)
	}

	windows := make([][][]float32, batchSize)

	for w := 0; w < batchSize; w++ {
		// Create window [80][16]
		window := make([][]float32, 80)
		for i := 0; i < 80; i++ {
			window[i] = make([]float32, 16)
		}

		// Fill window with 16 frames starting from (startFrame + w)
		for f := 0; f < 16; f++ {
			frameIdx := startFrame + w + f

			// Handle padding if we run out of frames
			if frameIdx >= numFrames {
				frameIdx = numFrames - 1
			}

			// Transpose: [frame][mel] -> [mel][frame]
			for m := 0; m < 80; m++ {
				window[m][f] = melSpec[frameIdx][m]
			}
		}

		windows[w] = window
	}

	return windows, nil
}

// CalculateExpectedFrames calculates how many mel-spectrogram frames
// will be produced from a given number of audio samples
func CalculateExpectedFrames(numSamples, hopLength, windowSize int) int {
	// Account for padding (n_fft/2 on each side)
	paddedLength := numSamples + windowSize // +windowSize because we add windowSize/2 on each side

	// Calculate number of frames
	// Formula: 1 + (length - windowSize) / hopLength
	numFrames := 1 + (paddedLength-windowSize)/hopLength

	return numFrames
}

// ValidateAudioLength validates that the audio length will produce
// enough frames for the requested batch size
func ValidateAudioLength(numSamples, batchSize int, cfg *ProcessorConfig) error {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	expectedFrames := CalculateExpectedFrames(numSamples, cfg.HopLength, cfg.WindowSize)
	requiredFrames := batchSize + 15 // Need batchSize windows, each 16 frames wide

	if expectedFrames < requiredFrames {
		return fmt.Errorf("audio too short: will produce %d frames, need %d for batch size %d",
			expectedFrames, requiredFrames, batchSize)
	}

	return nil
}
