/**
 * Direct gRPC-Web Client for Maximum Performance
 * Eliminates Go middleman bottleneck - calls Python gRPC service directly
 */

class DirectGrpcClient {
    constructor(grpcEndpoint = 'http://localhost:50051', modelName = 'default_model') {
        this.grpcEndpoint = grpcEndpoint;
        this.modelName = modelName;
        this.frameBuffer = [];
        this.currentFrameIndex = 0;
        this.isGenerating = false;
        this.generationStartTime = null;
        
        // Performance tracking
        this.stats = {
            totalFramesGenerated: 0,
            totalProcessingTime: 0,
            averageFrameTime: 0,
            frameRate: 0,
            lastFrameTime: 0
        };
        
        console.log(`🚀 Direct gRPC Client initialized for model: ${modelName}`);
        console.log(`📡 Connecting directly to: ${grpcEndpoint}`);
    }
    
    /**
     * Generate frame using direct gRPC call (no Go middleman)
     */
    async generateFrameFromAudio(audioData) {
        if (this.isGenerating) {
            console.log('⏳ Frame generation in progress, skipping...');
            return null;
        }
        
        this.isGenerating = true;
        const frameStartTime = performance.now();
        
        try {
            // Prepare gRPC request data
            const requestData = {
                model_name: this.modelName,
                frame_id: this.currentFrameIndex,
                audio_override: audioData ? this.arrayBufferToBase64(audioData) : ''
            };
            
            // Make direct gRPC call using fetch with binary data
            const response = await this.makeGrpcCall(requestData);
            
            if (response.success) {
                const processingTime = performance.now() - frameStartTime;
                
                // Update performance stats
                this.stats.totalFramesGenerated++;
                this.stats.totalProcessingTime += processingTime;
                this.stats.averageFrameTime = this.stats.totalProcessingTime / this.stats.totalFramesGenerated;
                this.stats.lastFrameTime = processingTime;
                
                // Calculate frame rate
                const currentTime = performance.now();
                if (this.generationStartTime) {
                    const totalElapsed = (currentTime - this.generationStartTime) / 1000; // seconds
                    this.stats.frameRate = this.stats.totalFramesGenerated / totalElapsed;
                } else {
                    this.generationStartTime = currentTime;
                }
                
                console.log(`⚡ Direct gRPC Frame #${this.stats.totalFramesGenerated}: ${processingTime.toFixed(1)}ms (${this.stats.frameRate.toFixed(1)} FPS)`);
                
                // Increment frame index
                this.currentFrameIndex = (this.currentFrameIndex + 1) % 3305; // Cycle through available frames
                
                // Add to frame buffer
                const frameData = {
                    frameId: response.frame_id,
                    imageData: response.prediction_data, // Already base64 encoded
                    bounds: response.bounds,
                    timestamp: currentTime,
                    processingTime: processingTime,
                    modelName: response.model_name
                };
                
                this.frameBuffer.push(frameData);
                
                // Keep buffer size manageable
                if (this.frameBuffer.length > 10) {
                    this.frameBuffer.shift();
                }
                
                return frameData;
            } else {
                console.error('❌ gRPC generation failed:', response.error);
                return null;
            }
        } catch (error) {
            console.error('❌ Direct gRPC call failed:', error);
            return null;
        } finally {
            this.isGenerating = false;
        }
    }
    
    /**
     * Make direct gRPC call using binary protocol
     */
    async makeGrpcCall(requestData) {
        try {
            // Create gRPC request in protobuf format
            const grpcRequest = this.createGrpcRequest(requestData);
            
            const response = await fetch(`${this.grpcEndpoint}/LipSyncService/GenerateInference`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/grpc-web+proto',
                    'Accept': 'application/grpc-web+proto'
                },
                body: grpcRequest
            });
            
            if (!response.ok) {
                throw new Error(`gRPC call failed: ${response.status} ${response.statusText}`);
            }
            
            // Parse gRPC response
            const responseData = await response.arrayBuffer();
            return this.parseGrpcResponse(responseData);
            
        } catch (error) {
            console.error('❌ gRPC call error:', error);
            throw error;
        }
    }
    
    /**
     * Create gRPC protobuf request
     */
    createGrpcRequest(data) {
        // For now, use JSON over gRPC-Web (simpler than full protobuf)
        // In production, would use generated protobuf classes
        const jsonData = JSON.stringify(data);
        const encoder = new TextEncoder();
        return encoder.encode(jsonData);
    }
    
    /**
     * Parse gRPC protobuf response
     */
    parseGrpcResponse(responseBuffer) {
        // For now, parse as JSON (would use protobuf parser in production)
        const decoder = new TextDecoder();
        const jsonString = decoder.decode(responseBuffer);
        return JSON.parse(jsonString);
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
            lastFrameTime: 0
        };
        this.generationStartTime = null;
    }
    
    /**
     * Connect to audio buffer and start frame generation
     */
    async connectToAudioBuffer(audioBufferManager) {
        if (!audioBufferManager) {
            throw new Error("Audio buffer manager is required");
        }
        
        this.audioBufferManager = audioBufferManager;
        
        // Hook into audio buffer to trigger direct gRPC frame generation
        const originalAddAudio = audioBufferManager.addAudioToBuffer;
        const self = this;
        
        audioBufferManager.addAudioToBuffer = function(pcmData) {
            // Call original method
            const result = originalAddAudio.call(this, pcmData);
            
            // Trigger direct gRPC frame generation when we have enough audio
            if (this.getBufferFillLevel() >= 1) { // At least 1 chunk (40ms)
                // Get the latest audio chunk data
                const audioChunks = this.getConsecutiveAudioChunks(1);
                if (audioChunks && audioChunks.length > 0) {
                    // Fire and forget - don't await to avoid blocking audio processing
                    self.generateFrameFromAudio(audioChunks[0].data).catch(error => {
                        console.error('❌ Frame generation error:', error);
                    });
                }
            }
            
            return result;
        };
        
        console.log('🔗 Connected to audio buffer - direct gRPC frame generation active');
    }
}
