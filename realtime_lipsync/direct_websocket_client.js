/**
 * ULTIMATE PERFORMANCE: Direct WebSocket Client
 * No HTTP, no gRPC, no protocol overhead - Maximum speed!
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
        
        console.log(`🚀 Direct WebSocket Client initialized for model: ${modelName}`);
        console.log(`⚡ Ultimate performance mode - direct engine access!`);
    }
    
    /**
     * Connect to the direct WebSocket server
     */
    async connect() {
        return new Promise((resolve, reject) => {
            console.log(`🔗 Connecting to direct WebSocket: ${this.wsEndpoint}`);
            
            this.websocket = new WebSocket(this.wsEndpoint);
            
            this.websocket.onopen = () => {
                this.isConnected = true;
                console.log('✅ Direct WebSocket connected - zero overhead mode!');
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
            
            // Handle batch response
            if (message.batch) {
                return this.handleBatchResponse(message);
            }
            
            // Handle single inference response
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
                
                console.log(`🚀 DIRECT Frame #${this.stats.totalFramesGenerated}: ${processingTime.toFixed(1)}ms (${this.stats.frameRate.toFixed(1)} FPS) - ZERO OVERHEAD!`);
                
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
     * Handle batch response from server
     */
    handleBatchResponse(message) {
        const batchSize = message.batch_size;
        const totalTime = message.total_processing_time_ms;
        const avgFrameTime = message.average_frame_time_ms;
        const batchFps = message.batch_fps;
        
        console.log(`🔥 BATCH COMPLETE: ${batchSize} frames in ${totalTime.toFixed(1)}ms (${avgFrameTime.toFixed(1)}ms/frame) - ${batchFps.toFixed(1)} FPS!`);
        
        const currentTime = performance.now();
        
        // Process each frame in the batch
        message.frames.forEach((frame, index) => {
            this.stats.totalFramesGenerated++;
            
            const frameData = {
                frameId: frame.frame_id,
                imageData: frame.prediction_data,
                bounds: frame.bounds,
                timestamp: currentTime + index, // Slight offset for ordering
                processingTime: avgFrameTime,
                modelName: message.model_name,
                requestId: message.request_id,
                batchIndex: index,
                batchSize: batchSize
            };
            
            this.frameBuffer.push(frameData);
            this.onFrameReady(frameData);
        });
        
        // Update performance stats with batch data
        this.stats.totalProcessingTime += totalTime;
        this.stats.averageFrameTime = this.stats.totalProcessingTime / this.stats.totalFramesGenerated;
        
        // Calculate frame rate including batch performance
        if (this.generationStartTime) {
            const totalElapsed = (currentTime - this.generationStartTime) / 1000;
            this.stats.frameRate = this.stats.totalFramesGenerated / totalElapsed;
        } else {
            this.generationStartTime = currentTime;
        }
        
        // Keep buffer manageable
        while (this.frameBuffer.length > 10) {
            this.frameBuffer.shift();
        }
    }
    
    /**
     * Generate batch of frames using DIRECT WebSocket - SMART BATCH PROCESSING!
     */
    async generateBatchFrames(batchSize = 2, audioData = null) {
        if (!this.isConnected || this.isGenerating) {
            if (!this.isConnected) console.log('⚠️ WebSocket not connected');
            if (this.isGenerating) console.log('⏳ Frame generation in progress, skipping batch...');
            return null;
        }
        
        this.isGenerating = true;
        this.requestId++;
        
        try {
            // Create batch request
            const batchFrames = [];
            for (let i = 0; i < batchSize; i++) {
                batchFrames.push({
                    frame_id: (this.currentFrameIndex + i) % 3305
                });
            }
            
            const request = {
                model_name: this.modelName,
                batch_frames: batchFrames,
                audio_override: audioData ? this.arrayBufferToBase64(audioData) : '',
                request_id: this.requestId
            };
            
            console.log(`🔥 Sending BATCH request: ${batchSize} frames`);
            
            // Send BATCH request to inference engine via WebSocket
            this.websocket.send(JSON.stringify(request));
            
            // Update frame index for next batch
            this.currentFrameIndex = (this.currentFrameIndex + batchSize) % 3305;
            
        } catch (error) {
            console.error('❌ Direct WebSocket batch send failed:', error);
            this.isGenerating = false;
            return null;
        }
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
     * Convert ArrayBuffer to base64 string
     */
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    /**
     * Frame ready callback - override this
     */
    onFrameReady(frameData) {
        // Override this method to handle new frames
        console.log(`🎬 Frame ready: ${frameData.frameId}`);
    }
    
    /**
     * Connect to audio buffer for ultimate performance frame generation
     */
    async connectToAudioBuffer(audioBufferManager) {
        if (!audioBufferManager) {
            throw new Error("Audio buffer manager is required");
        }
        
        // Ensure WebSocket is connected first
        if (!this.isConnected) {
            await this.connect();
        }
        
        this.audioBufferManager = audioBufferManager;
        
        // Hook into audio buffer
        const originalAddAudio = audioBufferManager.addAudioToBuffer;
        const self = this;
        
        audioBufferManager.addAudioToBuffer = function(pcmData) {
            const result = originalAddAudio.call(this, pcmData);
            
            if (this.getBufferFillLevel() >= 1) {
                const audioChunks = this.getConsecutiveAudioChunks(1);
                if (audioChunks && audioChunks.length > 0) {
                    // DIRECT WebSocket call - ultimate performance!
                    self.generateFrameFromAudio(audioChunks[0].data).catch(error => {
                        console.error('❌ Direct frame generation error:', error);
                    });
                }
            }
            
            return result;
        };
        
        console.log('🔗 Connected to audio buffer - DIRECT WebSocket mode active!');
        console.log('⚡ Zero protocol overhead - maximum performance achieved!');
    }
    
    /**
     * Get the latest generated frame
     */
    getLatestFrame() {
        return this.frameBuffer.length > 0 ? this.frameBuffer[this.frameBuffer.length - 1] : null;
    }
    
    /**
     * Get performance statistics
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
