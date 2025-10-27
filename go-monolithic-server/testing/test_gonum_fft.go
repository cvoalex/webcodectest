package main

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/dsp/fourier"
)

func main() {
	// Create a simple sine wave at 440Hz
	sr := 16000
	freq := 440.0
	duration := 0.1
	n := int(float64(sr) * duration)

	signal := make([]float64, n)
	for i := 0; i < n; i++ {
		t := float64(i) / float64(sr)
		signal[i] = math.Sin(2.0 * math.Pi * freq * t)
	}

	fmt.Printf("Test Signal: %d samples\n", len(signal))
	fmt.Printf("First 5: %v\n", signal[:5])

	// Create Hanning window
	windowSize := 800
	window := make([]float64, windowSize)
	for i := 0; i < windowSize; i++ {
		window[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(windowSize-1)))
	}

	// Pad signal (center=True)
	padSize := windowSize / 2
	padded := make([]float64, len(signal)+2*padSize)
	copy(padded[padSize:], signal)

	fmt.Printf("Padded signal: %d (was %d)\n", len(padded), len(signal))

	// Extract first frame and window it
	frame := padded[0:windowSize]
	windowed := make([]float64, windowSize)
	for i := 0; i < windowSize; i++ {
		windowed[i] = frame[i] * window[i]
	}

	fmt.Printf("Windowed RMS: %.8f\n", rms(windowed))

	// Compute FFT using gonum
	fftObj := fourier.NewFFT(windowSize)
	coeffs := fftObj.Coefficients(nil, windowed)

	// Get magnitudes
	numFreqBins := windowSize/2 + 1
	magnitudes := make([]float64, numFreqBins)
	for i := 0; i < numFreqBins; i++ {
		magnitudes[i] = cmplx.Abs(coeffs[i])
	}

	fmt.Printf("\nFFT Results:\n")
	fmt.Printf("  Output length: %d\n", len(magnitudes))
	fmt.Printf("  First 5 magnitudes: %v\n", magnitudes[:5])
	fmt.Printf("  Mean magnitude: %.8f\n", mean(magnitudes))
	fmt.Printf("  Max magnitude: %.8f\n", max(magnitudes))
	fmt.Printf("  Peak bin: %d (expected ~%d)\n", argmax(magnitudes), int(freq*float64(windowSize)/float64(sr)))

	// Compare with expected from Python
	fmt.Printf("\nExpected from Python test:\n")
	fmt.Printf("  First 5: [5.7850285, 5.7850285, 5.8338656, 5.8827033, 5.9854846]\n")
	fmt.Printf("  Mean: ~1.79\n")
	fmt.Printf("  Max: ~99.64\n")
}

func rms(x []float64) float64 {
	sum := 0.0
	for _, v := range x {
		sum += v * v
	}
	return math.Sqrt(sum / float64(len(x)))
}

func mean(x []float64) float64 {
	sum := 0.0
	for _, v := range x {
		sum += v
	}
	return sum / float64(len(x))
}

func max(x []float64) float64 {
	m := x[0]
	for _, v := range x {
		if v > m {
			m = v
		}
	}
	return m
}

func argmax(x []float64) int {
	maxIdx := 0
	maxVal := x[0]
	for i, v := range x {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
