package audio

import (
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"sync"

	"gonum.org/v1/gonum/dsp/fourier"
)

// Buffer pools for STFT processing to avoid repeated allocations
var stftBufferPool = sync.Pool{
	New: func() interface{} {
		return &stftBuffers{
			windowed:   make([]float64, 640),
			fftInput:   make([]float64, 1024),
			magnitudes: make([]float32, 513),
		}
	},
}

// Pool for transposed mel matrices [80][16] used in TransposeForEncoder
var transposedMelPool = sync.Pool{
	New: func() interface{} {
		transposed := make([][]float32, 80)
		for i := 0; i < 80; i++ {
			transposed[i] = make([]float32, 16)
		}
		return transposed
	},
}

// Pool for padded audio buffers (~26K floats)
var paddedAudioPool = sync.Pool{
	New: func() interface{} {
		// Max size for 25fps video: ~1040 samples/frame * 25 frames + padding
		// Actually for our use case: 16000 samples + 2*512 padding = ~17024
		// Let's use 32K to be safe for all cases
		return make([]float32, 32768)
	},
}

type stftBuffers struct {
	windowed   []float64
	fftInput   []float64
	magnitudes []float32
}

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

	// Get pooled buffer (might be larger than needed)
	padded := paddedAudioPool.Get().([]float32)
	defer paddedAudioPool.Put(padded)

	// Ensure buffer is large enough
	if len(padded) < paddedLen {
		// Buffer too small, allocate a larger one and update pool
		padded = make([]float32, paddedLen)
	} else {
		// Use only the slice we need and zero it
		padded = padded[:paddedLen]
		for i := range padded {
			padded[i] = 0
		}
	}

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

// preEmphasis applies pre-emphasis filter in-place
func (p *Processor) preEmphasis(samples []float32) []float32 {
	if p.config.PreEmphasis == 0 {
		return samples
	}

	// Apply filter in reverse to avoid needing a temporary buffer
	// result[i] = samples[i] - coef * samples[i-1]
	// We can do this in-place by working backwards, since each position
	// only depends on the previous (unmodified) value

	// Actually, we need to work FORWARD but save the previous value
	prev := samples[0]
	for i := 1; i < len(samples); i++ {
		current := samples[i]
		samples[i] = current - float32(p.config.PreEmphasis)*prev
		prev = current
	}

	return samples
}

// stft performs Short-Time Fourier Transform with parallel processing
// Matches librosa.stft behavior exactly
func (p *Processor) stft(samples []float32) [][]float32 {
	// Calculate number of frames
	// librosa: num_frames = 1 + (len - win_length) // hop_length
	numFrames := 1 + (len(samples)-p.config.WindowSize)/p.config.HopLength
	numFreqBins := p.config.NumFFT/2 + 1

	result := make([][]float32, numFrames)

	// Parallelize STFT computation across frames
	// Use worker pool to limit concurrency and buffer reuse
	const maxWorkers = 8
	numWorkers := numFrames
	if numWorkers > maxWorkers {
		numWorkers = maxWorkers
	}

	var wg sync.WaitGroup
	frameChan := make(chan int, numFrames)

	// Spawn workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Each worker gets its own buffer
			buffers := stftBufferPool.Get().(*stftBuffers)
			defer stftBufferPool.Put(buffers)

			// Create worker-specific FFT object (gonum FFT is not thread-safe)
			fftObj := fourier.NewFFT(p.config.NumFFT)

			for i := range frameChan {
				// Extract frame
				start := i * p.config.HopLength
				end := start + p.config.WindowSize

				// Safety check
				if end > len(samples) {
					continue
				}

				// Get frame slice
				frame := samples[start:end]

				// Safety check: ensure we don't exceed buffer bounds
				windowLen := len(frame)
				if windowLen > len(buffers.windowed) {
					windowLen = len(buffers.windowed)
				}
				if windowLen > len(p.window) {
					windowLen = len(p.window)
				}

				// Apply Hanning window using pooled buffer
				for j := 0; j < windowLen; j++ {
					buffers.windowed[j] = float64(frame[j]) * float64(p.window[j])
				}
				// Zero out remaining if window is shorter than buffer
				maxZeroIdx := p.config.WindowSize
				if maxZeroIdx > len(buffers.windowed) {
					maxZeroIdx = len(buffers.windowed)
				}
				for j := windowLen; j < maxZeroIdx; j++ {
					buffers.windowed[j] = 0
				}

				// Zero-pad to n_fft if needed
				copyLen := len(buffers.windowed)
				if copyLen > len(buffers.fftInput) {
					copyLen = len(buffers.fftInput)
				}
				copy(buffers.fftInput[:copyLen], buffers.windowed[:copyLen])
				maxFFTIdx := p.config.NumFFT
				if maxFFTIdx > len(buffers.fftInput) {
					maxFFTIdx = len(buffers.fftInput)
				}
				startIdx := p.config.WindowSize
				if startIdx > len(buffers.fftInput) {
					startIdx = len(buffers.fftInput)
				}
				for j := startIdx; j < maxFFTIdx; j++ {
					buffers.fftInput[j] = 0
				}

				// Compute FFT - pass only the required length
				fftInputSlice := buffers.fftInput[:p.config.NumFFT]
				fftResult := fftObj.Coefficients(nil, fftInputSlice)

				// Compute magnitude spectrum using pooled buffer
				for j := 0; j < numFreqBins; j++ {
					buffers.magnitudes[j] = float32(cmplx.Abs(fftResult[j]))
				}

				// Allocate and copy magnitudes for this frame
				result[i] = make([]float32, numFreqBins)
				copy(result[i], buffers.magnitudes[:numFreqBins])
			}
		}()
	}

	// Feed frames to workers
	for i := 0; i < numFrames; i++ {
		frameChan <- i
	}
	close(frameChan)

	wg.Wait()

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

// linearToMel converts linear spectrogram to mel scale with parallel processing
func (p *Processor) linearToMel(spectrogram [][]float32) [][]float32 {
	numFrames := len(spectrogram)

	// Pre-allocate entire result matrix
	result := make([][]float32, numFrames)

	// Parallelize mel filterbank application
	const maxWorkers = 8
	numWorkers := numFrames
	if numWorkers > maxWorkers {
		numWorkers = maxWorkers
	}
	if numWorkers < 1 {
		numWorkers = 1
	}

	var wg sync.WaitGroup
	frameChan := make(chan int, numFrames)

	// Spawn workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for i := range frameChan {
				result[i] = make([]float32, p.config.NumMelBins)

				// Apply mel filterbank
				for j := 0; j < p.config.NumMelBins; j++ {
					sum := float32(0.0)
					for k := 0; k < len(spectrogram[i]); k++ {
						sum += spectrogram[i][k] * p.melFilters[j][k]
					}
					result[i][j] = sum
				}
			}
		}()
	}

	// Feed frames to workers
	for i := 0; i < numFrames; i++ {
		frameChan <- i
	}
	close(frameChan)

	wg.Wait()

	return result
}

// ampToDB converts amplitude to decibels
// Uses 20 * log10(amplitude) to match librosa
// Operates in-place for memory efficiency
func (p *Processor) ampToDB(melSpec [][]float32) [][]float32 {
	// Minimum level to avoid log(0)
	// min_level = exp(min_level_db / 20 * log(10))
	minLevel := float32(math.Exp(float64(p.config.MinLevelDB) / 20.0 * math.Log(10.0)))
	refLevel := float32(p.config.RefLevelDB)

	for i := 0; i < len(melSpec); i++ {
		for j := 0; j < len(melSpec[i]); j++ {
			// Clamp to minimum level
			amp := melSpec[i][j]
			if amp < minLevel {
				amp = minLevel
			}

			// Convert amplitude to dB: 20 * log10(amp) - ref_level
			// Operate in-place
			melSpec[i][j] = float32(20.0*math.Log10(float64(amp))) - refLevel
		}
	}

	return melSpec
}

// normalize normalizes the mel-spectrogram to match original Python code
// Uses symmetric normalization: range [-4, +4]
// Operates in-place for memory efficiency
func (p *Processor) normalize(melSpecDB [][]float32) [][]float32 {
	// Parameters from hparams.py:
	// symmetric_mels=True, max_abs_value=4.0, allow_clipping=True
	// Formula: clip(8 * ((S + 100) / 100) - 4, -4, 4)
	//        = clip(0.08 * S + 4, -4, 4)

	const maxAbsValue = 4.0
	minLevelDB := float32(p.config.MinLevelDB)
	negMinLevelDB := -minLevelDB

	for i := 0; i < len(melSpecDB); i++ {
		for j := 0; j < len(melSpecDB[i]); j++ {
			val := melSpecDB[i][j]

			// Symmetric normalization with clipping
			// normalized = (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value
			normalized := (2.0*maxAbsValue)*((val-minLevelDB)/negMinLevelDB) - maxAbsValue

			// Clip to [-4, +4] and store in-place
			if normalized < -maxAbsValue {
				normalized = -maxAbsValue
			} else if normalized > maxAbsValue {
				normalized = maxAbsValue
			}

			melSpecDB[i][j] = normalized
		}
	}

	return melSpecDB
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
// NOTE: Caller should NOT return the result to pool - it's managed internally by the encoder
func TransposeForEncoder(melWindow [][]float32) [][]float32 {
	numFrames := len(melWindow)
	if numFrames == 0 {
		return nil
	}

	numMelBins := len(melWindow[0])

	// Get pooled transpose matrix
	transposed := transposedMelPool.Get().([][]float32)

	// Fill it (assuming standard size of [80][16])
	for i := 0; i < numMelBins && i < 80; i++ {
		for j := 0; j < numFrames && j < 16; j++ {
			transposed[i][j] = melWindow[j][i]
		}
	}

	return transposed
}

// ReturnTransposedMel returns a transposed mel matrix back to the pool
func ReturnTransposedMel(transposed [][]float32) {
	transposedMelPool.Put(transposed)
}
