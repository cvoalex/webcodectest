package logger

import (
	"bytes"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// BufferedLogger accumulates log entries in memory and flushes them asynchronously
// This allows logging without impacting request latency
type BufferedLogger struct {
	buffer     bytes.Buffer
	mu         sync.Mutex
	autoFlush  bool
	flushChan  chan struct{}
	stopChan   chan struct{}
	enabled    atomic.Bool
	requestNum atomic.Uint64
	sampleRate int // 0 = log all, N = log 1 in N requests
}

// NewBufferedLogger creates a new buffered logger
func NewBufferedLogger(autoFlush bool, sampleRate int) *BufferedLogger {
	bl := &BufferedLogger{
		autoFlush:  autoFlush,
		flushChan:  make(chan struct{}, 100),
		stopChan:   make(chan struct{}),
		sampleRate: sampleRate,
	}
	bl.enabled.Store(true)

	if autoFlush {
		// Start background flusher
		go bl.flusher()
	}

	return bl
}

// RequestLogger provides a per-request logging context
type RequestLogger struct {
	parent     *BufferedLogger
	buffer     bytes.Buffer
	shouldLog  bool
	requestNum uint64
}

// StartRequest creates a new request logger
// Returns nil if this request should not be logged (based on sampling)
func (bl *BufferedLogger) StartRequest() *RequestLogger {
	if !bl.enabled.Load() {
		return nil
	}

	requestNum := bl.requestNum.Add(1)

	// Check if we should log this request (sampling)
	shouldLog := bl.sampleRate == 0 || (requestNum%uint64(bl.sampleRate) == 0)
	if !shouldLog {
		return nil
	}

	return &RequestLogger{
		parent:     bl,
		shouldLog:  shouldLog,
		requestNum: requestNum,
	}
}

// Printf adds a formatted log entry to the request buffer
func (rl *RequestLogger) Printf(format string, args ...interface{}) {
	if rl == nil || !rl.shouldLog {
		return
	}

	timestamp := time.Now().Format("2006/01/02 15:04:05")
	msg := fmt.Sprintf(format, args...)
	fmt.Fprintf(&rl.buffer, "[%s] [Req#%d] %s\n", timestamp, rl.requestNum, msg)
}

// Commit flushes the request logs to the parent buffer
// Call this AFTER sending the response to the client
func (rl *RequestLogger) Commit() {
	if rl == nil || !rl.shouldLog || rl.buffer.Len() == 0 {
		return
	}

	// Add to parent buffer
	rl.parent.mu.Lock()
	rl.parent.buffer.Write(rl.buffer.Bytes())
	rl.parent.mu.Unlock()

	if rl.parent.autoFlush {
		// Trigger async flush
		select {
		case rl.parent.flushChan <- struct{}{}:
		default:
			// Channel full, flush will happen soon anyway
		}
	}
}

// Flush immediately writes all buffered logs to stdout
func (bl *BufferedLogger) Flush() {
	bl.mu.Lock()
	defer bl.mu.Unlock()

	if bl.buffer.Len() > 0 {
		log.Print(bl.buffer.String())
		bl.buffer.Reset()
	}
}

// flusher runs in background and periodically flushes logs
func (bl *BufferedLogger) flusher() {
	ticker := time.NewTicker(100 * time.Millisecond) // Flush every 100ms
	defer ticker.Stop()

	for {
		select {
		case <-bl.flushChan:
			bl.Flush()
		case <-ticker.C:
			bl.Flush()
		case <-bl.stopChan:
			bl.Flush() // Final flush
			return
		}
	}
}

// Stop stops the background flusher
func (bl *BufferedLogger) Stop() {
	close(bl.stopChan)
}

// Enable/Disable logging
func (bl *BufferedLogger) SetEnabled(enabled bool) {
	bl.enabled.Store(enabled)
}

func (bl *BufferedLogger) IsEnabled() bool {
	return bl.enabled.Load()
}

// SetSampleRate changes the sampling rate
// 0 = log all requests, N = log 1 in N requests
func (bl *BufferedLogger) SetSampleRate(rate int) {
	bl.sampleRate = rate
}

// GetStats returns current logging statistics
func (bl *BufferedLogger) GetStats() map[string]interface{} {
	bl.mu.Lock()
	bufferSize := bl.buffer.Len()
	bl.mu.Unlock()

	return map[string]interface{}{
		"total_requests": bl.requestNum.Load(),
		"buffer_size":    bufferSize,
		"sample_rate":    bl.sampleRate,
		"enabled":        bl.enabled.Load(),
	}
}
