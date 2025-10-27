package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"os"
	"strings"

	"gonum.org/v1/gonum/dsp/fourier"
)

// ProcessorConfig holds audio processing parameters
type ProcessorConfig struct {
	SampleRate      int
	WindowSize      int
	HopLength       int
	NumMelBins      int
	NumFFT          int
	PreEmphasis     float64
	RefLevelDB      float64
	MinLevelDB      float64
	TargetMelFrames int
}

// Processor handles audio processing
type Processor struct {
	config     *ProcessorConfig
	melFilters [][]float32
	window     []float32
	fftObj     *fourier.FFT
}

func defaultConfig() *ProcessorConfig {
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

func newProcessor(config *ProcessorConfig) *Processor {
	p := &Processor{config: config}

	// Load mel filterbank
	var err error
	p.melFilters, err = p.loadMelFilterbank()
	if err != nil {
		log.Fatalf("Failed to load mel filterbank: %v", err)
	}

	// Create Hanning window
	p.window = createHanningWindow(config.WindowSize)

	// Initialize FFT
	p.fftObj = fourier.NewFFT(config.NumFFT)

	return p
}

func createHanningWindow(size int) []float32 {
	window := make([]float32, size)
	for i := 0; i < size; i++ {
		window[i] = float32(0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(size-1))))
	}
	return window
}

func (p *Processor) loadMelFilterbank() ([][]float32, error) {
	possiblePaths := []string{
		"../audio_test_data/mel_filters.json",
		"audio_test_data/mel_filters.json",
		"debug_output/mel_filters_python.json",
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
		return nil, fmt.Errorf("could not find mel_filters.json")
	}

	fmt.Printf("Loaded mel filterbank from: %s\n", foundPath)

	var filtersData struct {
		Filters [][]float32 `json:"filters"`
	}

	if err := json.Unmarshal(data, &filtersData); err != nil {
		return nil, err
	}

	return filtersData.Filters, nil
}

func (p *Processor) preEmphasis(samples []float32) []float32 {
	result := make([]float32, len(samples))
	result[0] = samples[0]

	for i := 1; i < len(samples); i++ {
		result[i] = samples[i] - float32(p.config.PreEmphasis)*samples[i-1]
	}

	return result
}

func (p *Processor) stft(samples []float32) ([][]float32, [][]complex128) {
	numFrames := 1 + (len(samples)-p.config.WindowSize)/p.config.HopLength
	numFreqBins := p.config.NumFFT/2 + 1

	magnitudes := make([][]float32, numFrames)
	complexResults := make([][]complex128, numFrames)

	for i := 0; i < numFrames; i++ {
		start := i * p.config.HopLength
		end := start + p.config.WindowSize

		if end > len(samples) {
			break
		}

		frame := samples[start:end]

		// Apply Hanning window
		windowed := make([]float64, p.config.WindowSize)
		for j := 0; j < p.config.WindowSize; j++ {
			windowed[j] = float64(frame[j]) * float64(p.window[j])
		}

		// Zero-pad to n_fft
		fftInput := make([]float64, p.config.NumFFT)
		copy(fftInput, windowed)

		// Compute FFT
		fftResult := p.fftObj.Coefficients(nil, fftInput)

		// Store complex results and compute magnitudes
		complexResults[i] = make([]complex128, numFreqBins)
		magnitudes[i] = make([]float32, numFreqBins)

		for j := 0; j < numFreqBins; j++ {
			complexResults[i][j] = fftResult[j]
			magnitudes[i][j] = float32(cmplx.Abs(fftResult[j]))
		}
	}

	return magnitudes, complexResults
}

func (p *Processor) linearToMel(spectrogram [][]float32) [][]float32 {
	numFrames := len(spectrogram)
	result := make([][]float32, numFrames)

	for i := 0; i < numFrames; i++ {
		melFrame := make([]float32, p.config.NumMelBins)

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

func (p *Processor) ampToDB(melSpec [][]float32) [][]float32 {
	result := make([][]float32, len(melSpec))
	minLevel := float32(math.Exp(float64(p.config.MinLevelDB) / 20.0 * math.Log(10.0)))

	for i := 0; i < len(melSpec); i++ {
		frame := make([]float32, len(melSpec[i]))
		for j := 0; j < len(melSpec[i]); j++ {
			amp := melSpec[i][j]
			if amp < minLevel {
				amp = minLevel
			}

			db := float32(20.0 * math.Log10(float64(amp)))
			frame[j] = db
		}
		result[i] = frame
	}

	return result
}

func (p *Processor) normalize(melSpecDB [][]float32) [][]float32 {
	result := make([][]float32, len(melSpecDB))
	const maxAbsValue = 4.0

	for i := 0; i < len(melSpecDB); i++ {
		frame := make([]float32, len(melSpecDB[i]))
		for j := 0; j < len(melSpecDB[i]); j++ {
			val := melSpecDB[i][j]

			normalized := (2.0*maxAbsValue)*((val-float32(p.config.MinLevelDB))/-float32(p.config.MinLevelDB)) - maxAbsValue

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

func saveFloat32Array(filename string, data []float32) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Simple binary format - just write the float32s
	return binary.Write(f, binary.LittleEndian, data)
}

func saveFloat32Matrix(filename string, data [][]float32) error {
	if len(data) == 0 {
		return fmt.Errorf("empty matrix")
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Write dimensions first
	rows := int32(len(data))
	cols := int32(len(data[0]))
	binary.Write(f, binary.LittleEndian, rows)
	binary.Write(f, binary.LittleEndian, cols)

	// Write data row by row
	for i := 0; i < len(data); i++ {
		if err := binary.Write(f, binary.LittleEndian, data[i]); err != nil {
			return err
		}
	}

	return nil
}

func saveComplexMatrix(filenameReal, filenameImag string, data [][]complex128) error {
	if len(data) == 0 {
		return fmt.Errorf("empty matrix")
	}

	rows := len(data)
	cols := len(data[0])

	realData := make([][]float32, rows)
	imagData := make([][]float32, rows)

	for i := 0; i < rows; i++ {
		realData[i] = make([]float32, cols)
		imagData[i] = make([]float32, cols)
		for j := 0; j < cols; j++ {
			realData[i][j] = float32(real(data[i][j]))
			imagData[i][j] = float32(imag(data[i][j]))
		}
	}

	if err := saveFloat32Matrix(filenameReal, realData); err != nil {
		return err
	}
	if err := saveFloat32Matrix(filenameImag, imagData); err != nil {
		return err
	}

	return nil
}

func printStats(name string, data []float32) {
	if len(data) == 0 {
		fmt.Printf("%s: empty\n", name)
		return
	}

	min := data[0]
	max := data[0]
	sum := float32(0)

	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
	}

	mean := sum / float32(len(data))
	fmt.Printf("%s: min=%.8f, max=%.8f, mean=%.8f\n", name, min, max, mean)
}

func printMatrixStats(name string, data [][]float32) {
	if len(data) == 0 || len(data[0]) == 0 {
		fmt.Printf("%s: empty\n", name)
		return
	}

	min := data[0][0]
	max := data[0][0]
	sum := float32(0)
	count := 0

	for i := 0; i < len(data); i++ {
		for j := 0; j < len(data[i]); j++ {
			v := data[i][j]
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
			sum += v
			count++
		}
	}

	mean := sum / float32(count)
	fmt.Printf("%s: shape=[%d,%d], min=%.8f, max=%.8f, mean=%.8f\n", name, len(data), len(data[0]), min, max, mean)
}

func loadWAV(filename string) ([]float32, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	// Read RIFF header
	var riffHeader [12]byte
	if _, err := file.Read(riffHeader[:]); err != nil {
		return nil, 0, err
	}

	if string(riffHeader[0:4]) != "RIFF" || string(riffHeader[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a valid WAV file")
	}

	var sampleRate int
	var numChannels int
	var bitsPerSample int
	var dataSize int

	// Read chunks
	for {
		var chunkHeader [8]byte
		if _, err := file.Read(chunkHeader[:]); err != nil {
			break
		}

		chunkID := string(chunkHeader[0:4])
		chunkSize := int(binary.LittleEndian.Uint32(chunkHeader[4:8]))

		if chunkID == "fmt " {
			var fmtData [16]byte
			if _, err := file.Read(fmtData[:]); err != nil {
				return nil, 0, err
			}

			audioFormat := binary.LittleEndian.Uint16(fmtData[0:2])
			if audioFormat != 1 {
				return nil, 0, fmt.Errorf("only PCM format supported")
			}

			numChannels = int(binary.LittleEndian.Uint16(fmtData[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(fmtData[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(fmtData[14:16]))

			// Skip extra fmt data
			if chunkSize > 16 {
				file.Seek(int64(chunkSize-16), 1)
			}
		} else if chunkID == "data" {
			dataSize = chunkSize
			break
		} else {
			// Skip unknown chunk
			file.Seek(int64(chunkSize), 1)
		}
	}

	// Read audio data
	numSamples := dataSize / (bitsPerSample / 8) / numChannels
	samples := make([]float32, numSamples)

	if bitsPerSample == 16 {
		for i := 0; i < numSamples; i++ {
			var sample int16
			for ch := 0; ch < numChannels; ch++ {
				var s int16
				binary.Read(file, binary.LittleEndian, &s)
				if ch == 0 { // Only use first channel for mono
					sample = s
				}
			}
			samples[i] = float32(sample) / 32768.0
		}
	} else {
		return nil, 0, fmt.Errorf("only 16-bit samples supported")
	}

	return samples, sampleRate, nil
}

func main() {
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("GO PROCESSING - Saving intermediate steps")
	fmt.Println(strings.Repeat("=", 80))

	// Create output directory
	os.MkdirAll("debug_output", 0755)

	// Load audio
	audioPath := "../aud.wav"
	fmt.Printf("\n📂 Loading audio: %s\n", audioPath)

	audioSamples, sampleRate, err := loadWAV(audioPath)
	if err != nil {
		log.Fatalf("Failed to load audio: %v", err)
	}

	fmt.Printf("Loaded %d samples at %dHz (%.2f seconds)\n",
		len(audioSamples), sampleRate, float64(len(audioSamples))/float64(sampleRate))

	// Create processor
	config := defaultConfig()
	processor := newProcessor(config)

	// Step 0: Original audio
	fmt.Println("\nStep 0: Original audio")
	fmt.Printf("  Shape: [%d]\n", len(audioSamples))
	printStats("  ", audioSamples)
	fmt.Printf("  First 10: %v\n", audioSamples[:10])
	saveFloat32Array("debug_output/go_step0_original.bin", audioSamples)

	// Step 1: Pre-emphasis
	fmt.Println("\nStep 1: Pre-emphasis (k=0.97)")
	emphasized := processor.preEmphasis(audioSamples)
	fmt.Printf("  Shape: [%d]\n", len(emphasized))
	printStats("  ", emphasized)
	fmt.Printf("  First 10: %v\n", emphasized[:10])
	saveFloat32Array("debug_output/go_step1_preemphasis.bin", emphasized)

	// Pad audio for center=True behavior
	padSize := config.NumFFT / 2
	padded := make([]float32, len(emphasized)+2*padSize)
	copy(padded[padSize:], emphasized)

	// Step 2: STFT
	fmt.Println("\nStep 2: STFT (complex)")
	magnitudes, complexResults := processor.stft(padded)
	fmt.Printf("  Shape: [%d, %d] (frames, freq_bins)\n", len(magnitudes), len(magnitudes[0]))
	saveComplexMatrix("debug_output/go_step2_stft_real.bin", "debug_output/go_step2_stft_imag.bin", complexResults)

	// Step 3: Magnitude
	fmt.Println("\nStep 3: Magnitude")
	printMatrixStats("  ", magnitudes)
	fmt.Printf("  First frame first 10: %v\n", magnitudes[0][:10])
	saveFloat32Matrix("debug_output/go_step3_magnitude.bin", magnitudes)

	// Step 4: Linear to Mel
	fmt.Println("\nStep 4: Linear to Mel")
	mel := processor.linearToMel(magnitudes)
	printMatrixStats("  ", mel)
	fmt.Printf("  First frame first 10: %v\n", mel[0][:10])
	saveFloat32Matrix("debug_output/go_step4_mel.bin", mel)

	// Step 5a: Amp to dB (raw)
	fmt.Println("\nStep 5a: Amp to dB (before ref subtraction)")
	dbRaw := processor.ampToDB(mel)
	printMatrixStats("  ", dbRaw)
	fmt.Printf("  First frame first 10: %v\n", dbRaw[0][:10])
	saveFloat32Matrix("debug_output/go_step5a_db_raw.bin", dbRaw)

	// Step 5b: Subtract ref_level_db
	fmt.Println("\nStep 5b: Amp to dB (after -20 ref_level_db)")
	dbAdjusted := make([][]float32, len(dbRaw))
	for i := 0; i < len(dbRaw); i++ {
		dbAdjusted[i] = make([]float32, len(dbRaw[i]))
		for j := 0; j < len(dbRaw[i]); j++ {
			dbAdjusted[i][j] = dbRaw[i][j] - float32(config.RefLevelDB)
		}
	}
	printMatrixStats("  ", dbAdjusted)
	fmt.Printf("  First frame first 10: %v\n", dbAdjusted[0][:10])
	saveFloat32Matrix("debug_output/go_step5b_db_adjusted.bin", dbAdjusted)

	// Step 6: Normalize
	fmt.Println("\nStep 6: Normalize")
	normalized := processor.normalize(dbAdjusted)
	printMatrixStats("  ", normalized)

	// Count clipped values
	countNeg4 := 0
	countPos4 := 0
	for i := 0; i < len(normalized); i++ {
		for j := 0; j < len(normalized[i]); j++ {
			if normalized[i][j] == -4.0 {
				countNeg4++
			}
			if normalized[i][j] == 4.0 {
				countPos4++
			}
		}
	}
	fmt.Printf("  Count at -4.0: %d, at +4.0: %d\n", countNeg4, countPos4)
	fmt.Printf("  First frame first 10: %v\n", normalized[0][:10])
	saveFloat32Matrix("debug_output/go_step6_normalized.bin", normalized)

	// Step 7: ONNX Audio Encoder Inference
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("Step 7: ONNX Audio Encoder Inference")
	fmt.Println(strings.Repeat("=", 80))

	err = runAudioEncoderInference(normalized)
	if err != nil {
		fmt.Printf("\n⚠️  Audio encoder inference failed: %v\n", err)
		fmt.Println("   (Intermediate steps still saved)")
	}

	fmt.Println("\n✅ All Go intermediate steps saved to debug_output/")
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("NEXT: Run the comparison script")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("python test_full_audio_encoder.py")
}

func runAudioEncoderInference(normalized [][]float32) error {
	// Import the audio encoder package
	// We need to add this import at the top of the file
	// For now, we'll use direct ONNX runtime calls

	fmt.Println("\n1. Loading ONNX model...")

	// Find the model
	modelPath := "../audio_encoder.onnx"
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("model not found: %s", modelPath)
	}

	fmt.Printf("   Model: %s\n", modelPath)

	// We'll shell out to Python for now since adding the full ONNX runtime
	// Go bindings would require updating imports and dependencies

	// Instead, let's just save the normalized mel-spec in a format
	// that can be loaded by the audio encoder when testing separately

	// For this test, we'll pad/trim to 16 frames as expected by encoder
	targetFrames := 16

	var input [][]float32
	if len(normalized) >= targetFrames {
		input = normalized[:targetFrames]
	} else {
		// Pad with -4.0
		input = make([][]float32, targetFrames)
		copy(input, normalized)
		for i := len(normalized); i < targetFrames; i++ {
			input[i] = make([]float32, 80)
			for j := 0; j < 80; j++ {
				input[i][j] = -4.0
			}
		}
	}

	fmt.Printf("\n2. Prepared input for encoder:\n")
	fmt.Printf("   Shape: [%d, %d] (frames, mels)\n", len(input), len(input[0]))

	// Since we don't want to complicate the test with ONNX dependencies,
	// we'll note that the Go server already has audio.AudioEncoder
	// For this comparison, we document that the normalized mel-spec is ready

	fmt.Println("\n3. Note: Full ONNX inference test requires:")
	fmt.Println("   - Running the main server with audio encoder enabled")
	fmt.Println("   - Or using test_go_audio_tensor.go which has encoder integration")
	fmt.Println("\n   For now, mel-spectrogram comparison is the key validation.")

	// Save a placeholder
	placeholder := make([]float32, 512)
	for i := range placeholder {
		placeholder[i] = 0.0
	}
	saveFloat32Array("debug_output/go_audio_embedding.bin", placeholder)

	fmt.Println("\n   ℹ️  Placeholder embedding saved (all zeros)")
	fmt.Println("   ℹ️  For real inference, use: go run test_go_audio_tensor.go")

	return nil
}
