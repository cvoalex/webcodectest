/**
 * Real-time Audio Buffer Manager for OpenAI + Lip Sync Integration
 * Handles audio buffering, synchronization, and controlled playback
 */

class AudioBufferManager {
    constructor() {
        // Audio configuration
        this.BUFFER_SIZE = 3000;           // 3000 elements
        this.CHUNK_DURATION_MS = 40;       // 40ms per chunk
        this.SAMPLE_RATE = 24000;          // OpenAI default sample rate
        this.SAMPLES_PER_CHUNK = (this.SAMPLE_RATE * this.CHUNK_DURATION_MS) / 1000; // 960 samples
        
        // Frame generation configuration
        this.FRAMES_FOR_INFERENCE = 16;    // 8 prev + 1 current + 7 future
        this.INFERENCE_BATCH_SIZE = 5;     // Process 5 frames at once
        this.LOOKAHEAD_FRAMES = 7;         // Need 7 future frames
        this.LOOKAHEAD_MS = this.LOOKAHEAD_FRAMES * this.CHUNK_DURATION_MS; // 280ms
        
        // Circular buffers
        this.audioBuffer = new Array(this.BUFFER_SIZE).fill(null);
        this.frameBuffer = new Array(this.BUFFER_SIZE).fill(null);
        
        // Buffer pointers
        this.audioWriteIndex = 0;
        this.audioReadIndex = 0;
        this.frameWriteIndex = 0;
        this.frameReadIndex = 0;
        
        // Playback control
        this.isBuffering = true;
        this.bufferFillThreshold = 20; // Need 20 chunks (800ms) before starting playback
        
        // Frame buffer management for jitter minimization
        this.targetFrameBuffer = 15;    // Target 15 frames ahead (600ms)
        this.minFrameBuffer = 8;        // Minimum 8 frames (320ms) 
        this.maxFrameBuffer = 25;       // Maximum 25 frames (1000ms)
        this.criticalFrameBuffer = 3;   // Critical low level (120ms)
        
        this.audioContext = null;
        this.audioSource = null;
        this.playbackStartTime = 0;
        
        // Audio accumulation for controlled playback
        this.accumulatedAudio = [];
        this.playbackQueue = [];
        
        this.initializeAudioContext();
    }
    
    /**
     * Initialize Web Audio API context for controlled playback
     */
    initializeAudioContext() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log("🎵 Audio context initialized for controlled playback");
    }
    
    /**
     * Disable WebRTC audio playback by not setting srcObject
     * and capturing audio data manually
     */
    setupWebRTCWithoutAutoPlay(peerConnection) {
        // Create audio element but don't set autoplay
        const audioElement = document.createElement("audio");
        audioElement.autoplay = false; // CRITICAL: Disable autoplay
        audioElement.muted = true;     // Mute to prevent any accidental playback
        
        // Capture remote streams but don't play them
        peerConnection.ontrack = (event) => {
            console.log("🎤 Received remote audio stream (not auto-playing)");
            // Don't set srcObject - we'll handle audio manually
            this.captureAudioFromStream(event.streams[0]);
        };
        
        return audioElement;
    }
    
    /**
     * Capture audio data from WebRTC stream using Web Audio API
     */
    captureAudioFromStream(stream) {
        try {
            // Use Web Audio API to process the stream directly
            const source = this.audioContext.createMediaStreamSource(stream);
            const processor = this.audioContext.createScriptProcessor(1024, 1, 1);
            
            processor.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                
                // Convert to PCM16 and process
                this.processAudioSamples(inputData);
            };
            
            source.connect(processor);
            processor.connect(this.audioContext.destination);
            
            console.log("🎵 Audio capture started with Web Audio API");
            
        } catch (error) {
            console.error("❌ Error setting up audio capture:", error);
        }
    }
    
    /**
     * Process audio samples from Web Audio API
     */
    processAudioSamples(samples) {
        try {
            // Check if we have any actual audio data
            let hasAudio = false;
            let maxValue = 0;
            for (let i = 0; i < samples.length; i++) {
                const absValue = Math.abs(samples[i]);
                if (absValue > 0.001) { // Threshold for silence
                    hasAudio = true;
                }
                maxValue = Math.max(maxValue, absValue);
            }
            
            // Only process if we have actual audio content
            if (hasAudio) {
                // Convert Float32 samples to PCM16
                const pcm16Data = new Int16Array(samples.length);
                for (let i = 0; i < samples.length; i++) {
                    const sample = Math.max(-1, Math.min(1, samples[i]));
                    pcm16Data[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                }
                
                // Add to circular buffer for 40ms chunks processing
                this.addAudioToBuffer(pcm16Data);
                
                console.log(`🎵 Processed ${samples.length} audio samples from WebRTC stream (max amplitude: ${maxValue.toFixed(4)})`);
            } else {
                // Log silence but less frequently
                if (this.silenceCounter % 100 === 0) {
                    console.log(`🔇 Receiving silence from WebRTC stream (${samples.length} samples)`);
                }
                this.silenceCounter = (this.silenceCounter || 0) + 1;
            }
            
        } catch (error) {
            console.error("❌ Error processing audio samples:", error);
        }
    }
    
    /**
     * Process audio delta from OpenAI data channel events
     */
    processOpenAIAudioDelta(base64Audio) {
        try {
            // Decode base64 audio
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Convert to PCM16 format
            const pcmData = new Int16Array(bytes.buffer);
            
            // Add to circular buffer
            this.addAudioToBuffer(pcmData);
            
        } catch (error) {
            console.error("❌ Error processing OpenAI audio delta:", error);
        }
    }
    
    /**
     * Add audio data to circular buffer
     */
    addAudioToBuffer(pcmData) {
        // Store in circular buffer
        this.audioBuffer[this.audioWriteIndex] = {
            data: pcmData,
            timestamp: performance.now(),
            index: this.audioWriteIndex
        };
        
        // Advance write pointer
        this.audioWriteIndex = (this.audioWriteIndex + 1) % this.BUFFER_SIZE;
        
        // Add to accumulation for controlled playback
        this.accumulatedAudio.push(pcmData);
        
        console.log(`🎵 Audio chunk added to buffer. Index: ${this.audioWriteIndex}, Buffer fill: ${this.getBufferFillLevel()}`);
        
        // Check if we can start frame generation
        this.checkFrameGeneration();
        
        // Check if we can start playback
        this.checkPlaybackStart();
    }
    
    /**
     * Check if we have enough audio to generate frames
     */
    checkFrameGeneration() {
        const fillLevel = this.getBufferFillLevel();
        const frameBufferLevel = this.getFrameBufferFillLevel();
        
        // Generate frames if we have enough audio AND need more frames
        const needMoreFrames = frameBufferLevel < this.targetFrameBuffer;
        const haveEnoughAudio = fillLevel >= this.FRAMES_FOR_INFERENCE + 5;
        
        if (needMoreFrames && haveEnoughAudio) {
            console.log(`🎬 Generating frames: Audio=${fillLevel}, Frames=${frameBufferLevel}, Target=${this.targetFrameBuffer}`);
            this.generateFrameBatch();
        }
        
        // Check for critical low frame buffer
        if (frameBufferLevel <= this.criticalFrameBuffer) {
            console.warn(`⚠️ Critical frame buffer low: ${frameBufferLevel} frames remaining`);
        }
    }
    
    /**
     * Generate a batch of lip sync frames
     */
    async generateFrameBatch() {
        try {
            // Get 16 consecutive audio chunks for inference
            const audioChunks = this.getConsecutiveAudioChunks(this.FRAMES_FOR_INFERENCE);
            
            if (audioChunks.length < this.FRAMES_FOR_INFERENCE) {
                return; // Not enough audio yet
            }
            
            // Combine audio chunks into 640ms audio for inference
            const combinedAudio = this.combineAudioChunks(audioChunks);
            
            // Send to gRPC service for frame generation
            const frames = await this.generateLipSyncFrames(combinedAudio);
            
            // Store frames in circular buffer
            for (const frame of frames) {
                this.frameBuffer[this.frameWriteIndex] = {
                    data: frame,
                    timestamp: performance.now(),
                    audioIndex: this.audioReadIndex
                };
                
                this.frameWriteIndex = (this.frameWriteIndex + 1) % this.BUFFER_SIZE;
            }
            
            console.log(`🎬 Generated ${frames.length} frames. Frame buffer fill: ${this.getFrameBufferFillLevel()}`);
            
        } catch (error) {
            console.error("❌ Error generating frame batch:", error);
        }
    }
    
    /**
     * Check if we should start audio playback
     */
    checkPlaybackStart() {
        if (this.isBuffering && this.getBufferFillLevel() >= this.bufferFillThreshold) {
            console.log(`🎵 Buffer threshold reached (${this.bufferFillThreshold} chunks). Starting controlled playback...`);
            this.startControlledPlayback();
            this.isBuffering = false;
        }
    }
    
    /**
     * Start controlled audio playback with delay
     */
    startControlledPlayback() {
        // Calculate delay to allow frame generation
        const delayMs = this.LOOKAHEAD_MS + 100; // 280ms + 100ms buffer
        
        console.log(`🎵 Starting audio playback with ${delayMs}ms delay for synchronization`);
        
        setTimeout(() => {
            this.playBufferedAudio();
        }, delayMs);
    }
    
    /**
     * Play buffered audio using Web Audio API
     */
    async playBufferedAudio() {
        try {
            // Combine accumulated audio chunks
            const totalSamples = this.accumulatedAudio.reduce((sum, chunk) => sum + chunk.length, 0);
            const combinedAudio = new Float32Array(totalSamples);
            
            let offset = 0;
            for (const chunk of this.accumulatedAudio) {
                // Convert Int16 to Float32 for Web Audio API
                for (let i = 0; i < chunk.length; i++) {
                    combinedAudio[offset + i] = chunk[i] / 32768.0; // Normalize to [-1, 1]
                }
                offset += chunk.length;
            }
            
            // Create audio buffer
            const audioBuffer = this.audioContext.createBuffer(
                1, // mono
                combinedAudio.length,
                this.SAMPLE_RATE
            );
            
            audioBuffer.copyToChannel(combinedAudio, 0);
            
            // Create and start audio source
            this.audioSource = this.audioContext.createBufferSource();
            this.audioSource.buffer = audioBuffer;
            this.audioSource.connect(this.audioContext.destination);
            
            this.playbackStartTime = this.audioContext.currentTime;
            this.audioSource.start();
            
            console.log("🎵 Controlled audio playback started!");
            
            // Clear accumulated audio
            this.accumulatedAudio = [];
            
        } catch (error) {
            console.error("❌ Error starting controlled playback:", error);
        }
    }
    
    /**
     * Get consecutive audio chunks for inference
     */
    getConsecutiveAudioChunks(count) {
        const chunks = [];
        let currentIndex = this.audioReadIndex;
        
        for (let i = 0; i < count; i++) {
            if (this.audioBuffer[currentIndex] !== null) {
                chunks.push(this.audioBuffer[currentIndex]);
                currentIndex = (currentIndex + 1) % this.BUFFER_SIZE;
            } else {
                break; // Not enough consecutive chunks
            }
        }
        
        return chunks;
    }
    
    /**
     * Combine audio chunks into single array for inference
     */
    combineAudioChunks(chunks) {
        const totalSamples = chunks.reduce((sum, chunk) => sum + chunk.data.length, 0);
        const combined = new Int16Array(totalSamples);
        
        let offset = 0;
        for (const chunk of chunks) {
            combined.set(chunk.data, offset);
            offset += chunk.data.length;
        }
        
        return combined;
    }
    
    /**
     * Send audio to gRPC service for lip sync frame generation
     */
    async generateLipSyncFrames(audioData) {
        // This would connect to your gRPC service
        // For now, return mock frames
        console.log(`🎬 Generating lip sync frames for ${audioData.length} audio samples`);
        
        // Mock frame generation - replace with actual gRPC call
        return new Array(this.INFERENCE_BATCH_SIZE).fill(null).map((_, i) => ({
            frameData: new Uint8Array(1024), // Mock frame data
            frameIndex: i
        }));
    }
    
    /**
     * Convert Web Audio API AudioBuffer to PCM16
     */
    convertToPCM16(audioBuffer) {
        const channelData = audioBuffer.getChannelData(0); // Get first channel
        const pcm16 = new Int16Array(channelData.length);
        
        for (let i = 0; i < channelData.length; i++) {
            // Convert float [-1, 1] to int16 [-32768, 32767]
            pcm16[i] = Math.max(-32768, Math.min(32767, channelData[i] * 32768));
        }
        
        return pcm16;
    }
    
    /**
     * Get current buffer fill level
     */
    getBufferFillLevel() {
        let count = 0;
        for (let i = 0; i < this.BUFFER_SIZE; i++) {
            if (this.audioBuffer[i] !== null) count++;
        }
        return count;
    }
    
    /**
     * Get current frame buffer fill level
     */
    getFrameBufferFillLevel() {
        let count = 0;
        for (let i = 0; i < this.BUFFER_SIZE; i++) {
            if (this.frameBuffer[i] !== null) count++;
        }
        return count;
    }
    
    /**
     * Get frame buffer status for jitter prevention
     */
    getFrameBufferStatus() {
        const fillLevel = this.getFrameBufferFillLevel();
        
        let status = 'optimal';
        if (fillLevel <= this.criticalFrameBuffer) {
            status = 'critical';
        } else if (fillLevel < this.minFrameBuffer) {
            status = 'low';
        } else if (fillLevel > this.maxFrameBuffer) {
            status = 'high';
        }
        
        return {
            fillLevel,
            status,
            target: this.targetFrameBuffer,
            percentage: (fillLevel / this.targetFrameBuffer) * 100
        };
    }
    
    /**
     * Get current playback position for frame synchronization
     */
    getCurrentPlaybackPosition() {
        if (!this.audioSource || !this.playbackStartTime) return 0;
        
        const elapsed = this.audioContext.currentTime - this.playbackStartTime;
        return elapsed * 1000; // Return in milliseconds
    }
    
    /**
     * Get synchronized frame for current playback position
     */
    getCurrentFrame() {
        const playbackPosition = this.getCurrentPlaybackPosition();
        const frameIndex = Math.floor(playbackPosition / this.CHUNK_DURATION_MS);
        
        const bufferIndex = (this.frameReadIndex + frameIndex) % this.BUFFER_SIZE;
        return this.frameBuffer[bufferIndex];
    }
    
    /**
     * Convert ArrayBuffer to base64 string
     */
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    /**
     * Send audio data to frame generator via WebSocket
     */
    sendAudioToFrameGenerator(base64Audio) {
        if (this.frameGeneratorSocket && this.frameGeneratorSocket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'audio_chunk',
                data: base64Audio,
                timestamp: Date.now(),
                sampleRate: 24000,
                format: 'pcm16'
            };
            
            this.frameGeneratorSocket.send(JSON.stringify(message));
        }
    }
}

// Make AudioBufferManager available globally
window.AudioBufferManager = AudioBufferManager;
