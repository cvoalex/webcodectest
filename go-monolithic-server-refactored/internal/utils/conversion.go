package utils

import (
	"encoding/binary"
	"unsafe"
)

// BytesToFloat32 converts a byte slice to float32 slice (zero-copy via unsafe)
// WARNING: The input byte slice must remain valid and unchanged while the
// returned float32 slice is in use
func BytesToFloat32(b []byte) []float32 {
	// Length must be multiple of 4 (size of float32)
	if len(b)%4 != 0 {
		panic("byte slice length must be multiple of 4 for float32 conversion")
	}

	// Use unsafe pointer conversion for zero-copy
	// This is safe as long as:
	// 1. Byte slice is properly aligned (Go allocator ensures this)
	// 2. Byte slice remains valid during float32 slice lifetime
	// 3. We're on a little-endian system (most modern systems)
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// Float32ToBytes converts a float32 value to 4 bytes (little-endian)
func Float32ToBytes(f float32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, *(*uint32)(unsafe.Pointer(&f)))
	return b
}

// BytesToFloat32Copy creates a new float32 slice and copies data
// Safer than BytesToFloat32 but requires allocation
func BytesToFloat32Copy(b []byte) []float32 {
	if len(b)%4 != 0 {
		panic("byte slice length must be multiple of 4 for float32 conversion")
	}

	result := make([]float32, len(b)/4)
	for i := range result {
		bits := binary.LittleEndian.Uint32(b[i*4 : (i+1)*4])
		result[i] = *(*float32)(unsafe.Pointer(&bits))
	}
	return result
}
