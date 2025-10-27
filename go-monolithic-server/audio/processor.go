package audio

import (
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"os"

	"gonum.org/v1/gonum/dsp/fourier"
)

// ProcessorConfig holds audio processing parameters
type ProcessorConfig struct {
	SampleRate      int     // 16000 Hz
	WindowSize      int     // 800 samples (50ms at 16kHz)
	HopLength       int     // 200 samples (12.5ms at 16kHz)
	NumMelBins      int     // 80 mel frequency bins
	NumFFT          int     // 800 FFT size
	PreEmphasis     float64 // 0.97
	RefLevelDB      float64 // 20.0
	MinLevelDB      float64 // -100.0
	TargetMelFrames int     // 16 frames for audio encoder
}

// DefaultConfig returns the default audio processing configuration
// matching the Python implementation
func DefaultConfig() *ProcessorConfig {
	return &ProcessorConfig{
		SampleRate:      16000,
		WindowSize:      800,
		HopLength:       200,
		NumMelBins:      80,
		NumFFT:          800,
		PreEmphasis:     0.97,
		RefLevelDB:      20.0,
		MinLevelDB:      -100.0,
		TargetMelFrames: 16,
	}
}

// Processor handles audio to mel-spectrogram conversion
type Processor struct {
	config     *ProcessorConfig
	melFilters [][]float32  // Pre-computed mel filterbank
	window     []float32    // Hanning window
	fftObj     *fourier.FFT // Fast Fourier Transform
}

// NewProcessor creates a new audio processor
func NewProcessor(config *ProcessorConfig) *Processor {
	if config == nil {
		config = DefaultConfig()
	}

	p := &Processor{
		config: config,
	}

	// Load pre-computed mel filterbank from JSON
	var err error
	p.melFilters, err = p.loadMelFilterbank()
	if err != nil {
		// This is a fatal error during initialization.
		// Panicking is appropriate as the processor cannot function.
		panic(fmt.Sprintf("FATAL: failed to load mel filterbank: %v", err))
	}

	// Pre-compute Hanning window
	p.window = createHanningWindow(config.WindowSize)

	// Initialize FFT
	p.fftObj = fourier.NewFFT(config.NumFFT)

	return p
}

// ProcessAudio converts raw PCM audio to mel-spectrogram
// Input: raw audio samples (float32 or int16 converted to float32)
// Output: mel-spectrogram [numFrames, numMelBins]
func (p *Processor) ProcessAudio(audioSamples []float32) ([][]float32, error) {
	debugMode := false // Disable debug for production

	// 1. Pre-emphasis
	emphasized := p.preEmphasis(audioSamples)

	if debugMode {
		fmt.Printf("\n[DEBUG] Step 1: Pre-emphasis\n")
		fmt.Printf("  Input samples: %d\n", len(audioSamples))
		fmt.Printf("  First 5 values: %v\n", audioSamples[:5])
		fmt.Printf("  After pre-emphasis first 5: %v\n", emphasized[:5])
		mean := float32(0)
		for _, v := range emphasized[:min(10000, len(emphasized))] {
			mean += v
		}
		mean /= float32(min(10000, len(emphasized)))
		fmt.Printf("  Mean (first 10k): %.8f\n", mean)
	}

	// 2. Pad audio to match librosa's center=True behavior
	// Add n_fft/2 zeros on each side
	padSize := p.config.NumFFT / 2
	paddedLen := len(emphasized) + 2*padSize
	padded := make([]float32, paddedLen)
	copy(padded[padSize:], emphasized)

	if debugMode {
		fmt.Printf("\n[DEBUG] Step 2: Padding\n")
		fmt.Printf("  Pad size: %d on each side\n", padSize)
		fmt.Printf("  Padded length: %d (was %d)\n", len(padded), len(emphasized))
		fmt.Printf("  First 5 (should be 0): %v\n", padded[:5])
		fmt.Printf("  At boundary: %v\n", padded[padSize:padSize+5])
	}

	// 3. STFT (Short-Time Fourier Transform)
	spectrogram := p.stft(padded)

	if debugMode && len(spectrogram) > 0 {
		fmt.Printf("\n[DEBUG] Step 3: STFT\n")
		fmt.Printf("  Spectrogram shape: [%d, %d] (frames, freq_bins)\n",
			len(spectrogram), len(spectrogram[0]))
		mean := float32(0)
		max := float32(0)
		for i := 0; i < min(10, len(spectrogram)); i++ {
			for j := 0; j < len(spectrogram[i]); j++ {
				mean += spectrogram[i][j]
				if spectrogram[i][j] > max {
					max = spectrogram[i][j]
				}
			}
		}
		mean /= float32(min(10, len(spectrogram)) * len(spectrogram[0]))
		fmt.Printf("  Magnitude (first 10 frames) mean: %.8f, max: %.8f\n", mean, max)
		fmt.Printf("  First frame first 5: %v\n", spectrogram[0][:5])
	}

	// 4. Convert to mel scale
	melSpec := p.linearToMel(spectrogram)

	if debugMode && len(melSpec) > 0 {
		fmt.Printf("\n[DEBUG] Step 4: Linear to Mel\n")
		fmt.Printf("  Mel-spec shape: [%d, %d]\n", len(melSpec), len(melSpec[0]))
		mean := float32(0)
		max := float32(0)
		for i := 0; i < min(10, len(melSpec)); i++ {
			for j := 0; j < len(melSpec[i]); j++ {
				mean += melSpec[i][j]
				if melSpec[i][j] > max {
					max = melSpec[i][j]
				}
			}
		}
		mean /= float32(min(10, len(melSpec)) * len(melSpec[0]))
		fmt.Printf("  Linear mel (first 10 frames) mean: %.8f, max: %.8f\n", mean, max)
		fmt.Printf("  First frame first 5: %v\n", melSpec[0][:5])
	}

	// 5. Convert to dB scale
	melSpecDB := p.ampToDB(melSpec)

	if debugMode && len(melSpecDB) > 0 {
		fmt.Printf("\n[DEBUG] Step 5: Amp to dB\n")
		mean := float32(0)
		min_val := melSpecDB[0][0]
		max := melSpecDB[0][0]
		for i := 0; i < len(melSpecDB); i++ {
			for j := 0; j < len(melSpecDB[i]); j++ {
				v := melSpecDB[i][j]
				mean += v
				if v < min_val {
					min_val = v
				}
				if v > max {
					max = v
				}
			}
		}
		mean /= float32(len(melSpecDB) * len(melSpecDB[0]))
		fmt.Printf("  dB mean: %.6f, min: %.6f, max: %.6f\n", mean, min_val, max)
		fmt.Printf("  First frame first 5: %v\n", melSpecDB[0][:5])
	}

	// 6. Normalize
	normalized := p.normalize(melSpecDB)

	if debugMode && len(normalized) > 0 {
		fmt.Printf("\n[DEBUG] Step 6: Normalize\n")
		mean := float32(0)
		min_val := normalized[0][0]
		max := normalized[0][0]
		count_neg4 := 0
		count_pos4 := 0
		for i := 0; i < len(normalized); i++ {
			for j := 0; j < len(normalized[i]); j++ {
				v := normalized[i][j]
				mean += v
				if v < min_val {
					min_val = v
				}
				if v > max {
					max = v
				}
				if v == -4.0 {
					count_neg4++
				}
				if v == 4.0 {
					count_pos4++
				}
			}
		}
		mean /= float32(len(normalized) * len(normalized[0]))
		fmt.Printf("  Normalized mean: %.6f, min: %.6f, max: %.6f\n", mean, min_val, max)
		fmt.Printf("  Count at -4.0: %d, at +4.0: %d\n", count_neg4, count_pos4)
		fmt.Printf("  First frame first 10: %v\n", normalized[0][:10])
	}

	return normalized, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// preEmphasis applies pre-emphasis filter
func (p *Processor) preEmphasis(samples []float32) []float32 {
	if p.config.PreEmphasis == 0 {
		return samples
	}

	result := make([]float32, len(samples))
	result[0] = samples[0]

	for i := 1; i < len(samples); i++ {
		result[i] = samples[i] - float32(p.config.PreEmphasis)*samples[i-1]
	}

	return result
}

// stft performs Short-Time Fourier Transform
// Matches librosa.stft behavior exactly
func (p *Processor) stft(samples []float32) [][]float32 {
	// Calculate number of frames
	// librosa: num_frames = 1 + (len - win_length) // hop_length
	numFrames := 1 + (len(samples)-p.config.WindowSize)/p.config.HopLength
	numFreqBins := p.config.NumFFT/2 + 1

	result := make([][]float32, numFrames)

	for i := 0; i < numFrames; i++ {
		// Extract frame
		start := i * p.config.HopLength
		end := start + p.config.WindowSize

		// Safety check
		if end > len(samples) {
			break
		}

		// Get frame slice
		frame := samples[start:end]

		// Apply Hanning window
		windowed := make([]float64, p.config.WindowSize)
		for j := 0; j < p.config.WindowSize; j++ {
			windowed[j] = float64(frame[j]) * float64(p.window[j])
		}

		// Zero-pad to n_fft if needed (in our case WindowSize == NumFFT)
		fftInput := make([]float64, p.config.NumFFT)
		copy(fftInput, windowed)

		// Compute FFT using gonum
		fftResult := p.fftObj.Coefficients(nil, fftInput)

		// Compute magnitude spectrum (only positive frequencies)
		magnitudes := make([]float32, numFreqBins)
		for j := 0; j < numFreqBins; j++ {
			magnitudes[j] = float32(cmplx.Abs(fftResult[j]))
		}

		result[i] = magnitudes
	}

	return result
}

// fft computes Fast Fourier Transform using gonum
func (p *Processor) fft(samples []float32) []complex128 {
	n := p.config.NumFFT

	// Convert to float64 for gonum
	input := make([]float64, n)
	for i := 0; i < len(samples) && i < n; i++ {
		input[i] = float64(samples[i])
	}

	// Compute FFT using gonum (O(n log n))
	coeffs := p.fftObj.Coefficients(nil, input)

	return coeffs
}

// linearToMel converts linear spectrogram to mel scale
func (p *Processor) linearToMel(spectrogram [][]float32) [][]float32 {
	numFrames := len(spectrogram)
	result := make([][]float32, numFrames)

	for i := 0; i < numFrames; i++ {
		melFrame := make([]float32, p.config.NumMelBins)

		// Apply mel filterbank
		for j := 0; j < p.config.NumMelBins; j++ {
			sum := float32(0.0)
			for k := 0; k < len(spectrogram[i]); k++ {
				sum += spectrogram[i][k] * p.melFilters[j][k]
			}
			melFrame[j] = sum
		}

		result[i] = melFrame
	}

	return result
}

// ampToDB converts amplitude to decibels
// Uses 20 * log10(amplitude) to match librosa
func (p *Processor) ampToDB(melSpec [][]float32) [][]float32 {
	result := make([][]float32, len(melSpec))

	// Minimum level to avoid log(0)
	// min_level = exp(min_level_db / 20 * log(10))
	minLevel := float32(math.Exp(float64(p.config.MinLevelDB) / 20.0 * math.Log(10.0)))

	for i := 0; i < len(melSpec); i++ {
		frame := make([]float32, len(melSpec[i]))
		for j := 0; j < len(melSpec[i]); j++ {
			// Clamp to minimum level
			amp := melSpec[i][j]
			if amp < minLevel {
				amp = minLevel
			}

			// Convert amplitude to dB: 20 * log10(amp)
			db := float32(20.0 * math.Log10(float64(amp)))
			db -= float32(p.config.RefLevelDB)

			frame[j] = db
		}
		result[i] = frame
	}

	return result
}

// normalize normalizes the mel-spectrogram to match original Python code
// Uses symmetric normalization: range [-4, +4]
func (p *Processor) normalize(melSpecDB [][]float32) [][]float32 {
	result := make([][]float32, len(melSpecDB))

	// Parameters from hparams.py:
	// symmetric_mels=True, max_abs_value=4.0, allow_clipping=True
	// Formula: clip(8 * ((S + 100) / 100) - 4, -4, 4)
	//        = clip(0.08 * S + 4, -4, 4)

	const maxAbsValue = 4.0

	for i := 0; i < len(melSpecDB); i++ {
		frame := make([]float32, len(melSpecDB[i]))
		for j := 0; j < len(melSpecDB[i]); j++ {
			val := melSpecDB[i][j]

			// Symmetric normalization with clipping
			// normalized = (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value
			normalized := (2.0*maxAbsValue)*((val-float32(p.config.MinLevelDB))/-float32(p.config.MinLevelDB)) - maxAbsValue

			// Clip to [-4, +4]
			if normalized < -maxAbsValue {
				normalized = -maxAbsValue
			} else if normalized > maxAbsValue {
				normalized = maxAbsValue
			}

			frame[j] = normalized
		}
		result[i] = frame
	}

	return result
}

// loadMelFilterbank loads the pre-computed mel filterbank from a JSON file.
func (p *Processor) loadMelFilterbank() ([][]float32, error) {
	// Define possible paths to find the JSON file
	possiblePaths := []string{
		"../audio_test_data/mel_filters.json",
		"../../audio_test_data/mel_filters.json",
		"audio_test_data/mel_filters.json",
		"mel_filters.json",
	}

	var data []byte
	var err error
	var foundPath string

	for _, path := range possiblePaths {
		data, err = os.ReadFile(path)
		if err == nil {
			foundPath = path
			break
		}
	}

	if err != nil {
		return nil, fmt.Errorf("could not find mel_filters.json in any of the expected paths: %v", possiblePaths)
	}
	fmt.Printf("Loaded mel filterbank from: %s\n", foundPath)

	var filtersData struct {
		Filters [][]float32 `json:"filters"`
	}

	if err := json.Unmarshal(data, &filtersData); err != nil {
		return nil, fmt.Errorf("failed to parse mel_filters.json from %s: %w", foundPath, err)
	}

	// Basic validation
	if len(filtersData.Filters) != p.config.NumMelBins {
		return nil, fmt.Errorf("mel filterbank has wrong number of mel bins: got %d, want %d", len(filtersData.Filters), p.config.NumMelBins)
	}
	if len(filtersData.Filters) > 0 && len(filtersData.Filters[0]) != (p.config.NumFFT/2+1) {
		return nil, fmt.Errorf("mel filterbank has wrong number of frequency bins: got %d, want %d", len(filtersData.Filters[0]), (p.config.NumFFT/2 + 1))
	}

	return filtersData.Filters, nil
}

// createHanningWindow creates a Hanning window
func createHanningWindow(size int) []float32 {
	window := make([]float32, size)
	for i := 0; i < size; i++ {
		window[i] = float32(0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(size-1))))
	}
	return window
}

// ExtractMelWindow extracts a specific time window from mel-spectrogram
// Used for sliding window processing
func ExtractMelWindow(melSpec [][]float32, startFrame, numFrames int) ([][]float32, error) {
	if startFrame < 0 || startFrame+numFrames > len(melSpec) {
		return nil, fmt.Errorf("window out of bounds: start=%d, numFrames=%d, total=%d",
			startFrame, numFrames, len(melSpec))
	}

	window := make([][]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		window[i] = melSpec[startFrame+i]
	}

	return window, nil
}

// TransposeForEncoder transposes mel-spectrogram for audio encoder
// From [numFrames, numMelBins] to [numMelBins, numFrames]
func TransposeForEncoder(melWindow [][]float32) [][]float32 {
	numFrames := len(melWindow)
	if numFrames == 0 {
		return nil
	}

	numMelBins := len(melWindow[0])
	transposed := make([][]float32, numMelBins)

	for i := 0; i < numMelBins; i++ {
		transposed[i] = make([]float32, numFrames)
		for j := 0; j < numFrames; j++ {
			transposed[i][j] = melWindow[j][i]
		}
	}

	return transposed
}
