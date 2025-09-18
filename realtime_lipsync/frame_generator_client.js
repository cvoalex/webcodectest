/**
 * Frame Generator Client
 * Connects audio buffer to gRPC service for real-time frame generation
 */

class FrameGeneratorClient {
    constructor(grpcEndpoint = 'http://localhost:50051/direct', modelName = 'default_model') {
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
            frameRate: 0
        };
        
        console.log(`🎬 Frame Generator Client initialized for model: ${modelName}`);
    }
    
    /**
     * Connect to audio buffer and start frame generation
     */
    async connectToAudioBuffer(audioBufferManager) {
        if (!audioBufferManager) {
            throw new Error("Audio buffer manager is required");
        }
        
        this.audioBufferManager = audioBufferManager;
        
        // Hook into audio buffer to trigger frame generation
        const originalAddAudio = audioBufferManager.addAudioToBuffer;
        const self = this;
        
        audioBufferManager.addAudioToBuffer = function(pcmData) {
            // Call original method
            const result = originalAddAudio.call(this, pcmData);
            
            // Trigger frame generation when we have enough audio
            if (this.getBufferFillLevel() >= 1) { // At least 1 chunk (40ms)
                // Get the latest audio chunk data
                const audioChunks = this.getConsecutiveAudioChunks(1);
                if (audioChunks && audioChunks.length > 0) {
                    self.generateFrameFromAudio(audioChunks[0].data);
                }
            }
            
            return result;
        };
        
        console.log("🔗 Connected to audio buffer for frame generation");
    }
    
    /**
     * Generate frame from audio data via gRPC
     */
    async generateFrameFromAudio(audioData) {
        if (this.isGenerating) {
            console.log("⏭️ Skipping frame generation - already in progress");
            return;
        }
        
        this.isGenerating = true;
        const frameStartTime = performance.now();
        
        try {
            // Prepare request for gRPC service
            const request = {
                model_name: this.modelName,
                frame_id: this.currentFrameIndex,
                audio_override: audioData ? this.audioDataToBase64(audioData) : null
            };
            
            console.log(`🎬 Generating frame ${this.currentFrameIndex} with ${audioData?.length || 0} audio samples`);
            
            // Call gRPC service via HTTP proxy
            const response = await this.callGrpcService('GenerateInference', request);
            
            if (response.success) {
                // Process the generated frame
                await this.processGeneratedFrame(response);
                
                const processingTime = performance.now() - frameStartTime;
                this.updateStats(processingTime);
                
                console.log(`✅ Frame ${this.currentFrameIndex} generated in ${processingTime.toFixed(2)}ms`);
                this.currentFrameIndex++;
                
            } else {
                console.error(`❌ Frame generation failed: ${response.error}`);
            }
            
        } catch (error) {
            console.error("❌ Error generating frame:", error);
        } finally {
            this.isGenerating = false;
        }
    }
    
    /**
     * Process generated frame and add to frame buffer
     */
    async processGeneratedFrame(response) {
        try {
            // Decode base64 JPEG data to create frame
            const imageUrl = `data:image/jpeg;base64,${response.prediction_data}`;
            
            // Create image element
            const img = new Image();
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
                img.src = imageUrl;
            });
            
            // Add to frame buffer
            const frameData = {
                frameId: response.frame_id,
                image: img,
                bounds: response.bounds,
                timestamp: performance.now(),
                processingTime: response.processing_time_ms,
                modelName: response.model_name,
                shape: response.prediction_shape
            };
            
            this.frameBuffer.push(frameData);
            
            // Keep buffer size manageable (last 100 frames)
            if (this.frameBuffer.length > 100) {
                this.frameBuffer.shift();
            }
            
            // Trigger frame display update
            this.triggerFrameUpdate(frameData);
            
        } catch (error) {
            console.error("❌ Error processing generated frame:", error);
        }
    }
    
    /**
     * Call gRPC service via HTTP proxy
     */
    async callGrpcService(method, request) {
        const response = await fetch(this.grpcEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                service: 'LipSyncService',
                method: method,
                request: request
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    /**
     * Convert audio data to base64 for gRPC transmission
     */
    audioDataToBase64(audioData) {
        if (!audioData || !audioData.length) return null;
        
        // Convert Int16Array to base64
        const bytes = new Uint8Array(audioData.buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    /**
     * Update performance statistics
     */
    updateStats(processingTime) {
        this.stats.totalFramesGenerated++;
        this.stats.totalProcessingTime += processingTime;
        this.stats.averageFrameTime = this.stats.totalProcessingTime / this.stats.totalFramesGenerated;
        
        // Calculate frame rate (frames per second)
        if (this.generationStartTime) {
            const elapsed = (performance.now() - this.generationStartTime) / 1000;
            this.stats.frameRate = this.stats.totalFramesGenerated / elapsed;
        } else {
            this.generationStartTime = performance.now();
        }
    }
    
    /**
     * Trigger frame update event for display
     */
    triggerFrameUpdate(frameData) {
        // Dispatch custom event for frame display
        const event = new CustomEvent('frameGenerated', {
            detail: {
                frame: frameData,
                stats: this.stats,
                bufferSize: this.frameBuffer.length
            }
        });
        
        window.dispatchEvent(event);
    }
    
    /**
     * Get current frame for display
     */
    getCurrentFrame() {
        return this.frameBuffer[this.frameBuffer.length - 1] || null;
    }
    
    /**
     * Get all frames in buffer
     */
    getAllFrames() {
        return [...this.frameBuffer];
    }
    
    /**
     * Get performance statistics
     */
    getStats() {
        return { ...this.stats };
    }
    
    /**
     * Clear frame buffer and reset stats
     */
    reset() {
        // Clean up image URLs
        this.frameBuffer.forEach(frame => {
            if (frame.image && frame.image.src) {
                URL.revokeObjectURL(frame.image.src);
            }
        });
        
        this.frameBuffer = [];
        this.currentFrameIndex = 0;
        this.isGenerating = false;
        this.generationStartTime = null;
        this.stats = {
            totalFramesGenerated: 0,
            totalProcessingTime: 0,
            averageFrameTime: 0,
            frameRate: 0
        };
        
        console.log("🔄 Frame generator reset");
    }
}

// Export for use in other modules
window.FrameGeneratorClient = FrameGeneratorClient;
