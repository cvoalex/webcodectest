# Audio Processing Implementation Notes

## Current Status

We've implemented a pure Go mel-spectrogram processor in `audio/processor.go`.

### Current Implementation

- **Pre-emphasis filter**: ✅ Implemented
- **STFT**: ✅ Basic DFT implementation (SLOW!)
- **Mel filterbank**: ✅ Implemented
- **dB conversion**: ✅ Implemented
- **Normalization**: ✅ Implemented

### Performance Optimization TODO

The current FFT implementation is a naive DFT which is **O(n²)** - very slow!

**RECOMMENDED**: Replace with `gonum` FFT library for **O(n log n)** performance:

```bash
go get gonum.org/v1/gonum/dsp/fourier
```

Then replace the `fft()` function in `processor.go` with:

```go
import "gonum.org/v1/gonum/dsp/fourier"

func (p *Processor) fft(samples []float32) []complex128 {
    // Convert to float64
    input := make([]float64, p.config.NumFFT)
    for i := 0; i < len(samples) && i < len(input); i++ {
        input[i] = float64(samples[i])
    }
    
    // Use gonum FFT
    fft := fourier.NewFFT(p.config.NumFFT)
    coeffs := fft.Coefficients(nil, input)
    
    return coeffs
}
```

### Testing

Create test cases comparing against Python librosa output to validate correctness.

### Integration

The processor is ready to be integrated into the inference server's InferBatch handler.
