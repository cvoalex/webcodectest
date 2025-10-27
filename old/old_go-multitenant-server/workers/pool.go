package workers

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"go-multitenant-server/registry"
)

// InferenceRequest represents a single inference request
type InferenceRequest struct {
	ModelID       string
	VisualFrames  []float32
	AudioFeatures []float32
	BatchSize     int
	StartFrameIdx int

	// Response channel
	ResultChan chan *InferenceResult

	// Timing
	QueuedAt time.Time
}

// InferenceResult contains the inference output
type InferenceResult struct {
	Outputs []float32
	Success bool
	Error   error

	// Timing information
	QueueTime     time.Duration
	WaitTime      time.Duration
	InferenceTime time.Duration
	WorkerID      int
}

// WorkerPool manages multiple parallel inference workers
type WorkerPool struct {
	workers      []*Worker
	requestQueue chan *InferenceRequest
	registry     *registry.ModelRegistry
	workerCount  int
	queueSize    int

	// Statistics
	mu             sync.RWMutex
	totalRequests  int64
	activeRequests int32
	queueDepth     int32

	// Shutdown
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// Worker represents a single inference worker with its own ONNX session
type Worker struct {
	id          int
	pool        *WorkerPool
	requestChan chan *InferenceRequest

	// Statistics
	mu                 sync.RWMutex
	requestsProcessed  int64
	totalInferenceTime time.Duration
	isIdle             bool
}

// NewWorkerPool creates a worker pool with parallel CUDA streams
func NewWorkerPool(registry *registry.ModelRegistry, workerCount, queueSize int) (*WorkerPool, error) {
	if workerCount < 1 {
		workerCount = 4 // Default to 4 workers
	}
	if queueSize < 100 {
		queueSize = 100
	}

	ctx, cancel := context.WithCancel(context.Background())

	pool := &WorkerPool{
		workers:      make([]*Worker, workerCount),
		requestQueue: make(chan *InferenceRequest, queueSize),
		registry:     registry,
		workerCount:  workerCount,
		queueSize:    queueSize,
		ctx:          ctx,
		cancel:       cancel,
	}

	// Create workers
	for i := 0; i < workerCount; i++ {
		worker := &Worker{
			id:          i,
			pool:        pool,
			requestChan: make(chan *InferenceRequest, 10),
			isIdle:      true,
		}
		pool.workers[i] = worker
	}

	// Start dispatcher
	pool.wg.Add(1)
	go pool.dispatcher()

	// Start workers
	for _, worker := range pool.workers {
		pool.wg.Add(1)
		go worker.run()
	}

	log.Printf("✅ Worker pool started with %d parallel workers (queue size: %d)", workerCount, queueSize)

	return pool, nil
}

// Submit submits an inference request to the pool
func (p *WorkerPool) Submit(req *InferenceRequest) error {
	req.QueuedAt = time.Now()

	select {
	case p.requestQueue <- req:
		// Update stats
		p.mu.Lock()
		p.totalRequests++
		p.queueDepth = int32(len(p.requestQueue))
		p.mu.Unlock()
		return nil
	case <-p.ctx.Done():
		return fmt.Errorf("worker pool is shutting down")
	default:
		return fmt.Errorf("request queue is full (%d/%d)", len(p.requestQueue), p.queueSize)
	}
}

// dispatcher dispatches requests to available workers (round-robin)
func (p *WorkerPool) dispatcher() {
	defer p.wg.Done()

	workerIdx := 0

	for {
		select {
		case req := <-p.requestQueue:
			// Update queue depth
			p.mu.Lock()
			p.queueDepth = int32(len(p.requestQueue))
			p.mu.Unlock()

			// Dispatch to next worker (round-robin)
			worker := p.workers[workerIdx]
			workerIdx = (workerIdx + 1) % p.workerCount

			select {
			case worker.requestChan <- req:
				// Successfully dispatched
			case <-p.ctx.Done():
				return
			}

		case <-p.ctx.Done():
			return
		}
	}
}

// run is the main worker loop
func (w *Worker) run() {
	defer w.pool.wg.Done()

	log.Printf("🔧 Worker %d started", w.id)

	for {
		select {
		case req := <-w.requestChan:
			w.processRequest(req)

		case <-w.pool.ctx.Done():
			log.Printf("🛑 Worker %d stopping", w.id)
			return
		}
	}
}

// processRequest processes a single inference request
func (w *Worker) processRequest(req *InferenceRequest) {
	w.mu.Lock()
	w.isIdle = false
	w.mu.Unlock()

	defer func() {
		w.mu.Lock()
		w.isIdle = true
		w.mu.Unlock()
	}()

	result := &InferenceResult{
		WorkerID:  w.id,
		QueueTime: time.Since(req.QueuedAt),
	}

	waitStart := time.Now()

	// Get or load model
	modelInstance, err := w.pool.registry.GetOrLoadModel(req.ModelID)
	if err != nil {
		result.Success = false
		result.Error = fmt.Errorf("failed to load model: %w", err)
		req.ResultChan <- result
		return
	}

	result.WaitTime = time.Since(waitStart)

	// Run inference
	inferStart := time.Now()

	// Replicate audio for batch
	audioFrameSize := 32 * 16 * 16
	audioForBatch := make([]float32, req.BatchSize*audioFrameSize)
	for i := 0; i < req.BatchSize; i++ {
		copy(audioForBatch[i*audioFrameSize:], req.AudioFeatures)
	}

	outputs, err := modelInstance.Inferencer.InferBatch(req.VisualFrames, audioForBatch, req.BatchSize)
	if err != nil {
		result.Success = false
		result.Error = fmt.Errorf("inference failed: %w", err)
		req.ResultChan <- result
		return
	}

	result.InferenceTime = time.Since(inferStart)
	result.Outputs = outputs
	result.Success = true

	// Update worker stats
	w.mu.Lock()
	w.requestsProcessed++
	w.totalInferenceTime += result.InferenceTime
	w.mu.Unlock()

	// Update model stats
	w.pool.registry.RecordInference(req.ModelID, result.InferenceTime.Seconds()*1000)

	// Send result
	req.ResultChan <- result
}

// GetStats returns pool statistics
func (p *WorkerPool) GetStats() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats := map[string]interface{}{
		"total_requests":  p.totalRequests,
		"active_requests": p.activeRequests,
		"queue_depth":     p.queueDepth,
		"worker_count":    p.workerCount,
		"queue_size":      p.queueSize,
	}

	// Add per-worker stats
	workerStats := make([]map[string]interface{}, p.workerCount)
	for i, worker := range p.workers {
		worker.mu.RLock()
		workerStats[i] = map[string]interface{}{
			"worker_id":          worker.id,
			"requests_processed": worker.requestsProcessed,
			"total_inference_ms": worker.totalInferenceTime.Milliseconds(),
			"is_idle":            worker.isIdle,
		}
		if worker.requestsProcessed > 0 {
			avgTime := worker.totalInferenceTime.Milliseconds() / int64(worker.requestsProcessed)
			workerStats[i]["avg_inference_ms"] = avgTime
		}
		worker.mu.RUnlock()
	}
	stats["workers"] = workerStats

	return stats
}

// Shutdown gracefully shuts down the worker pool
func (p *WorkerPool) Shutdown() {
	log.Println("🛑 Shutting down worker pool...")

	p.cancel()

	// Close request queue
	close(p.requestQueue)

	// Wait for all workers to finish
	p.wg.Wait()

	log.Println("✅ Worker pool shut down complete")
}
