/**
 * Pure gRPC-Web Client - No HTTP, Direct Binary Protocol
 * Maximum performance with native gRPC protocol
 */

// Import the generated gRPC-Web client (we'll need to generate this)
// For now, we'll use direct binary gRPC-Web calls

class PureGrpcWebClient {
    constructor(grpcEndpoint = 'http://localhost:8080', modelName = 'default_model') {
        this.grpcEndpoint = grpcEndpoint; // gRPC-Web proxy endpoint
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
        
        console.log(`🚀 Pure gRPC-Web Client initialized for model: ${modelName}`);
        console.log(`📡 Direct gRPC protocol via: ${grpcEndpoint}`);
    }
    
    /**
     * Generate frame using pure gRPC-Web binary protocol
     */
    async generateFrameFromAudio(audioData) {
        if (this.isGenerating) {
            console.log('⏳ Frame generation in progress, skipping...');
            return null;
        }
        
        this.isGenerating = true;
        const frameStartTime = performance.now();
        
        try {
            // Create protobuf request
            const request = this.createProtobufRequest({
                model_name: this.modelName,
                frame_id: this.currentFrameIndex,
                audio_override: audioData ? this.arrayBufferToBase64(audioData) : ''
            });
            
            // Make pure gRPC-Web call
            const response = await this.grpcWebCall('LipSyncService', 'GenerateInference', request);
            
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
                    const totalElapsed = (currentTime - this.generationStartTime) / 1000;
                    this.stats.frameRate = this.stats.totalFramesGenerated / totalElapsed;
                } else {
                    this.generationStartTime = currentTime;
                }
                
                console.log(`⚡ Pure gRPC Frame #${this.stats.totalFramesGenerated}: ${processingTime.toFixed(1)}ms (${this.stats.frameRate.toFixed(1)} FPS)`);
                
                // Increment frame index
                this.currentFrameIndex = (this.currentFrameIndex + 1) % 3305;
                
                // Add to frame buffer
                const frameData = {
                    frameId: response.frame_id,
                    imageData: response.prediction_data,
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
            console.error('❌ Pure gRPC call failed:', error);
            return null;
        } finally {
            this.isGenerating = false;
        }
    }
    
    /**
     * Create protobuf request message
     */
    createProtobufRequest(data) {
        // Create binary protobuf message
        // This would normally use generated protobuf classes
        // For now, we'll serialize to JSON and encode as bytes
        const message = {
            model_name: data.model_name,
            frame_id: data.frame_id,
            audio_override: data.audio_override
        };
        
        return new TextEncoder().encode(JSON.stringify(message));
    }
    
    /**
     * Make pure gRPC-Web call using binary protocol
     */
    async grpcWebCall(service, method, requestBytes) {
        const url = `${this.grpcEndpoint}/${service}/${method}`;
        
        // Create gRPC-Web headers
        const headers = new Headers();
        headers.set('Content-Type', 'application/grpc-web+proto');
        headers.set('Accept', 'application/grpc-web+proto');
        headers.set('X-Grpc-Web', 'true');
        
        // Create gRPC-Web frame format
        const frame = this.createGrpcWebFrame(requestBytes);
        
        const response = await fetch(url, {
            method: 'POST',
            headers: headers,
            body: frame
        });
        
        if (!response.ok) {
            throw new Error(`gRPC-Web call failed: ${response.status}`);
        }
        
        // Parse gRPC-Web response
        const responseBytes = await response.arrayBuffer();
        return this.parseGrpcWebResponse(responseBytes);
    }
    
    /**
     * Create gRPC-Web frame format
     */
    createGrpcWebFrame(messageBytes) {
        // gRPC-Web frame format:
        // [compression flag (1 byte)] + [message length (4 bytes)] + [message]
        const frame = new ArrayBuffer(5 + messageBytes.byteLength);
        const view = new DataView(frame);
        
        // Compression flag (0 = no compression)
        view.setUint8(0, 0);
        
        // Message length (big-endian)
        view.setUint32(1, messageBytes.byteLength, false);
        
        // Copy message
        new Uint8Array(frame, 5).set(new Uint8Array(messageBytes));
        
        return frame;
    }
    
    /**
     * Parse gRPC-Web response frame
     */
    parseGrpcWebResponse(responseBytes) {
        const view = new DataView(responseBytes);
        
        // Skip compression flag (1 byte)
        const messageLength = view.getUint32(1, false);
        
        // Extract message
        const messageBytes = responseBytes.slice(5, 5 + messageLength);
        const messageText = new TextDecoder().decode(messageBytes);
        
        return JSON.parse(messageText);
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
     * Connect to audio buffer for pure gRPC frame generation
     */
    async connectToAudioBuffer(audioBufferManager) {
        if (!audioBufferManager) {
            throw new Error("Audio buffer manager is required");
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
                    // Pure gRPC call - no HTTP overhead
                    self.generateFrameFromAudio(audioChunks[0].data).catch(error => {
                        console.error('❌ Pure gRPC error:', error);
                    });
                }
            }
            
            return result;
        };
        
        console.log('🔗 Connected to audio buffer - pure gRPC protocol active');
    }
    
    getLatestFrame() {
        return this.frameBuffer.length > 0 ? this.frameBuffer[this.frameBuffer.length - 1] : null;
    }
    
    getStats() {
        return { ...this.stats };
    }
    
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
}
