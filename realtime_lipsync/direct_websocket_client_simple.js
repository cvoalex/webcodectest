/**
 * SIMPLE WORKING VERSION: Direct WebSocket Client
 * 22.6 FPS performance - no batch processing complexity
 */

class DirectWebSocketClient {
    constructor(wsEndpoint = 'ws://localhost:8082', modelName = 'default_model') {
        this.wsEndpoint = wsEndpoint;
        this.modelName = modelName;
        this.websocket = null;
        this.frameBuffer = [];
        this.currentFrameIndex = 0;
        this.isGenerating = false;
        this.isConnected = false;
        this.generationStartTime = null;
        this.requestId = 0;
        
        // Performance tracking
        this.stats = {
            totalFramesGenerated: 0,
            totalProcessingTime: 0,
            averageFrameTime: 0,
            frameRate: 0,
            lastFrameTime: 0,
            connectionLatency: 0
        };
        
        console.log(`🚀 SIMPLE Direct WebSocket Client initialized for model: ${modelName}`);
        console.log(`⚡ 22.6 FPS performance mode - direct engine access!`);
    }

    /**
     * Connect to WebSocket server
     */
    async connect() {
        return new Promise((resolve, reject) => {
            console.log(`🔌 Connecting to ${this.wsEndpoint}...`);
            
            this.websocket = new WebSocket(this.wsEndpoint);
            
            this.websocket.onopen = () => {
                this.isConnected = true;
                console.log(`✅ Direct WebSocket connected to ${this.wsEndpoint}`);
                resolve();
            };
            
            this.websocket.onmessage = (event) => {
                this.handleMessage(event.data);
            };
            
            this.websocket.onclose = () => {
                this.isConnected = false;
                console.log('🔌 Direct WebSocket disconnected');
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ Direct WebSocket error:', error);
                reject(error);
            };
            
            // Timeout after 5 seconds
            setTimeout(() => {
                if (!this.isConnected) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 5000);
        });
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(data) {
        try {
            const message = JSON.parse(data);
            
            if (message.type === 'stats') {
                // Handle server stats broadcast
                console.log(`📊 Server stats: ${message.total_requests} requests, avg ${message.average_time_ms.toFixed(1)}ms`);
                return;
            }
            
            // Handle inference response
            if (message.success) {
                const processingTime = message.processing_time_ms;
                
                // Update performance stats
                this.stats.totalFramesGenerated++;
                this.stats.totalProcessingTime += processingTime;
                this.stats.averageFrameTime = this.stats.totalProcessingTime / this.stats.totalFramesGenerated;
                this.stats.lastFrameTime = processingTime;
                
                // Calculate frame rate
                const currentTime = performance.now();
                if (this.generationStartTime) {
                    const totalElapsed = (currentTime - this.generationStartTime) / 1000;
                    this.stats.frameRate = this.stats.totalFramesGenerated / totalElapsed;
                } else {
                    this.generationStartTime = currentTime;
                }
                
                console.log(`🚀 SIMPLE Frame #${this.stats.totalFramesGenerated}: ${processingTime.toFixed(1)}ms (${this.stats.frameRate.toFixed(1)} FPS) - ZERO OVERHEAD!`);
                
                // Create frame data
                const frameData = {
                    frameId: message.frame_id,
                    imageData: message.prediction_data,
                    bounds: message.bounds,
                    timestamp: currentTime,
                    processingTime: processingTime,
                    modelName: message.model_name,
                    requestId: message.request_id
                };
                
                this.frameBuffer.push(frameData);
                
                // Keep buffer manageable
                if (this.frameBuffer.length > 10) {
                    this.frameBuffer.shift();
                }
                
                // Notify frame ready
                this.onFrameReady(frameData);
                
            } else {
                console.error('❌ Direct inference failed:', message.error);
            }
            
        } catch (error) {
            console.error('❌ Failed to parse WebSocket message:', error);
        }
        
        // Reset generation flag
        this.isGenerating = false;
    }

    /**
     * Generate frame using DIRECT WebSocket call - Ultimate performance!
     */
    async generateFrameFromAudio(audioData) {
        if (!this.isConnected || this.isGenerating) {
            if (!this.isConnected) console.log('⚠️ WebSocket not connected');
            if (this.isGenerating) console.log('⏳ Frame generation in progress, skipping...');
            return null;
        }
        
        this.isGenerating = true;
        this.requestId++;
        
        try {
            // Create request - direct JSON, no protocol wrapper
            const request = {
                model_name: this.modelName,
                frame_id: this.currentFrameIndex,
                audio_override: audioData ? this.arrayBufferToBase64(audioData) : '',
                request_id: this.requestId
            };
            
            // Send DIRECT to inference engine via WebSocket
            this.websocket.send(JSON.stringify(request));
            
            // Increment frame index
            this.currentFrameIndex = (this.currentFrameIndex + 1) % 3305;
            
        } catch (error) {
            console.error('❌ Direct WebSocket send failed:', error);
            this.isGenerating = false;
            return null;
        }
    }

    /**
     * Convert ArrayBuffer to base64
     */
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return window.btoa(binary);
    }

    /**
     * Start automatic generation from buffer
     */
    async startGeneratingFromBuffer(audioBufferManager) {
        console.log('🎬 Starting SIMPLE automatic frame generation from audio buffer');
        
        const self = this;
        audioBufferManager.onAudioChunk = function(audioChunks) {
            if (audioChunks.length > 0 && self.isConnected) {
                // Generate frame from latest audio chunk
                self.generateFrameFromAudio(audioChunks[0].data).catch(error => {
                    console.error('Frame generation error:', error);
                });
            }
        };
    }

    /**
     * Reset client state
     */
    reset() {
        this.frameBuffer = [];
        this.currentFrameIndex = 0;
        this.isGenerating = false;
        this.generationStartTime = null;
        this.requestId = 0;
        this.resetStats();
    }

    /**
     * Get current performance stats
     */
    getStats() {
        return { ...this.stats };
    }

    /**
     * Reset performance counters
     */
    resetStats() {
        this.stats = {
            totalFramesGenerated: 0,
            totalProcessingTime: 0,
            averageFrameTime: 0,
            frameRate: 0,
            lastFrameTime: 0,
            connectionLatency: 0
        };
        this.generationStartTime = null;
    }

    /**
     * Disconnect WebSocket
     */
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.isConnected = false;
        }
    }
}
