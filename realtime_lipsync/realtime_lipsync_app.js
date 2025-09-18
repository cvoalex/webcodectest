/**
 * Real-time Lip Sync Application
 * Main application controller that coordinates WebRTC, audio processing, and frame display
 */

import RealtimeLipSyncClient from './realtime_lipsync_client.js';
import AudioBufferManager from './audio_buffer_manager.js';

class RealtimeLipSyncApp {
    constructor() {
        // Core components
        this.lipSyncClient = new RealtimeLipSyncClient();
        this.audioBufferManager = this.lipSyncClient.audioBufferManager;
        
        // New components for full-body compositing
        this.modelVideoManager = new ModelVideoManager();
        this.faceCompositor = new FaceCompositor();
        
        // Current model state
        this.currentModel = 'test_optimized_package_fixed_1';
        this.modelPreparationProgress = new Map(); // Track model loading progress
        
        // UI elements
        this.startBtn = document.getElementById('start-session-btn');
        this.stopBtn = document.getElementById('stop-session-btn');
        this.textInput = document.getElementById('text-input');
        this.sendTextBtn = document.getElementById('send-text-btn');
        this.modelSelect = document.getElementById('model-select');
        this.videoCanvas = document.getElementById('video-canvas');
        this.ctx = this.videoCanvas.getContext('2d');
        
        // Status indicators
        this.statusText = document.getElementById('status-text');
        this.statusDot = document.getElementById('status-dot');
        this.currentModelSpan = document.getElementById('current-model');
        
        // Statistics elements
        this.audioBufferFillSpan = document.getElementById('audio-buffer-fill');
        this.frameBufferFillSpan = document.getElementById('frame-buffer-fill');
        this.audioBufferBar = document.getElementById('audio-buffer-bar');
        this.frameBufferBar = document.getElementById('frame-buffer-bar');
        this.fpsCounter = document.getElementById('fps-counter');
        this.latencyCounter = document.getElementById('latency-counter');
        this.generationFps = document.getElementById('generation-fps');
        this.gpuMemory = document.getElementById('gpu-memory');
        this.inferenceTime = document.getElementById('inference-time');
        
        // Transcript
        this.transcriptContent = document.getElementById('transcript-content');
        this.eventsContainer = document.getElementById('events-container');
        
        // State
        this.isSessionActive = false;
        this.frameGeneratorWs = null;
        this.animationFrameId = null;
        this.fpsStats = {
            lastTime: 0,
            frameCount: 0,
            fps: 0
        };
        
        // Performance tracking
        this.performanceStats = {
            frameGenerationTimes: [],
            audioLatencies: [],
            lastStatsUpdate: 0
        };
        
        this.initializeEventListeners();
        this.connectFrameGenerator();
        this.startRenderLoop();
        this.startStatsLoop();
        
        // Initialize default model
        this.initializeDefaultModel();
    }
    
    /**
     * Initialize UI event listeners
     */
    initializeEventListeners() {
        // Session controls
        this.startBtn.addEventListener('click', () => this.startSession());
        this.stopBtn.addEventListener('click', () => this.stopSession());
        
        // Text input
        this.textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && this.textInput.value.trim()) {
                this.sendTextMessage();
            }
        });
        
        this.sendTextBtn.addEventListener('click', () => this.sendTextMessage());
        
        // Model selection
        this.modelSelect.addEventListener('change', (e) => {
            this.changeModel(e.target.value);
        });
        
        // Canvas click for debugging
        this.videoCanvas.addEventListener('click', () => {
            this.logDebugInfo();
        });
        
        console.log("🎮 Event listeners initialized");
    }
    
    /**
     * Connect to the Go integrated frame generator WebSocket
     */
    async connectFrameGenerator() {
        try {
            this.frameGeneratorWs = new WebSocket('ws://localhost:3000/ws');
            
            this.frameGeneratorWs.onopen = () => {
                console.log("🔗 Connected to frame generator");
                this.logEvent('frame_generator.connected', 'Connected to Python frame generator', 'system');
                this.updateConnectionStatus(true);
            };
            
            this.frameGeneratorWs.onmessage = (event) => {
                this.handleFrameGeneratorMessage(JSON.parse(event.data));
            };
            
            this.frameGeneratorWs.onclose = () => {
                console.log("🔌 Frame generator disconnected");
                this.logEvent('frame_generator.disconnected', 'Frame generator connection lost', 'error');
                this.updateConnectionStatus(false);
                
                // Attempt reconnection after 3 seconds
                setTimeout(() => {
                    if (!this.frameGeneratorWs || this.frameGeneratorWs.readyState === WebSocket.CLOSED) {
                        this.connectFrameGenerator();
                    }
                }, 3000);
            };
            
            this.frameGeneratorWs.onerror = (error) => {
                console.error("❌ Frame generator WebSocket error:", error);
                this.logEvent('frame_generator.error', `WebSocket error: ${error.message}`, 'error');
            };
            
        } catch (error) {
            console.error("❌ Failed to connect to frame generator:", error);
            this.logEvent('frame_generator.error', `Connection failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Handle messages from the Python frame generator
     */
    handleFrameGeneratorMessage(message) {
        switch (message.type) {
            case 'frame_data':
                this.displayFrame(message);
                break;
                
            case 'frames_generated':
                this.updateFrameStats(message);
                break;
                
            case 'stats':
                this.updateSystemStats(message);
                break;
                
            default:
                console.log("📡 Unknown frame generator message:", message.type);
        }
    }
    
    /**
     * Start WebRTC session
     */
    async startSession() {
        try {
            this.startBtn.disabled = true;
            this.startBtn.innerHTML = '<div class="spinner"></div> Starting...';
            
            // Start the lip sync client session
            await this.lipSyncClient.startSession();
            
            // Update UI state
            this.isSessionActive = true;
            this.stopBtn.disabled = false;
            this.textInput.disabled = false;
            this.sendTextBtn.disabled = false;
            this.modelSelect.disabled = false;
            
            this.startBtn.style.display = 'none';
            this.stopBtn.style.display = 'inline-flex';
            
            this.statusText.textContent = 'Session Active';
            this.statusDot.classList.add('connected');
            
            this.logEvent('session.started', 'WebRTC session started successfully', 'system');
            
            // Set up custom event listeners for lip sync client
            this.setupLipSyncEventHandlers();
            
        } catch (error) {
            console.error("❌ Failed to start session:", error);
            this.logEvent('session.error', `Failed to start: ${error.message}`, 'error');
            
            // Reset UI
            this.startBtn.disabled = false;
            this.startBtn.innerHTML = '<span>🎤</span> Start Session';
        }
    }
    
    /**
     * Stop WebRTC session
     */
    async stopSession() {
        try {
            await this.lipSyncClient.stopSession();
            
            // Update UI state
            this.isSessionActive = false;
            this.stopBtn.disabled = true;
            this.textInput.disabled = true;
            this.sendTextBtn.disabled = true;
            this.textInput.value = '';
            
            this.startBtn.style.display = 'inline-flex';
            this.stopBtn.style.display = 'none';
            this.startBtn.disabled = false;
            
            this.statusText.textContent = 'Disconnected';
            this.statusDot.classList.remove('connected');
            
            this.logEvent('session.stopped', 'WebRTC session stopped', 'system');
            
            // Clear canvas
            this.ctx.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
            this.drawPlaceholder();
            
        } catch (error) {
            console.error("❌ Failed to stop session:", error);
            this.logEvent('session.error', `Failed to stop: ${error.message}`, 'error');
        }
    }
    
    /**
     * Send text message to the model
     */
    sendTextMessage() {
        const message = this.textInput.value.trim();
        if (!message || !this.isSessionActive) return;
        
        this.lipSyncClient.sendTextMessage(message);
        this.textInput.value = '';
        
        this.logEvent('message.sent', `Text: "${message}"`, 'client');
    }
    
    /**
     * Change the active model
     */
    async changeModel(modelName) {
        if (!this.frameGeneratorWs || this.frameGeneratorWs.readyState !== WebSocket.OPEN) {
            console.error("❌ Frame generator not connected");
            return;
        }

        // Update current model
        this.currentModel = modelName;
        
        // Send model change to frame generator
        const message = {
            type: 'set_model',
            model_name: modelName
        };
        
        this.frameGeneratorWs.send(JSON.stringify(message));
        this.currentModelSpan.textContent = modelName;
        
        // Prepare the new model video in the background
        if (!this.modelVideoManager.isModelReady(modelName)) {
            console.log(`🎬 Preparing new model video: ${modelName}`);
            this.prepareCurrentModel().catch(error => {
                console.error(`❌ Failed to prepare model ${modelName}:`, error);
            });
        }
        
        this.logEvent('model.changed', `Switched to ${modelName}`, 'system');
    }
    
    /**
     * Set up event handlers for lip sync client events
     */
    setupLipSyncEventHandlers() {
        // Override the lip sync client's event handlers to update our UI
        const originalHandleEvent = this.lipSyncClient.handleServerEvent.bind(this.lipSyncClient);
        
        this.lipSyncClient.handleServerEvent = (event) => {
            // Call original handler
            originalHandleEvent(event);
            
            // Update our UI
            this.handleLipSyncEvent(event);
        };
        
        // Override audio processing to send to frame generator
        const originalProcessAudio = this.audioBufferManager.processOpenAIAudioDelta.bind(this.audioBufferManager);
        
        this.audioBufferManager.processOpenAIAudioDelta = (base64Audio) => {
            // Call original processing
            originalProcessAudio(base64Audio);
            
            // Send to frame generator
            if (this.frameGeneratorWs && this.frameGeneratorWs.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'audio_chunk',
                    audio_data: base64Audio,
                    timestamp: performance.now()
                };
                
                this.frameGeneratorWs.send(JSON.stringify(message));
            }
        };
    }
    
    /**
     * Handle lip sync events and update UI
     */
    handleLipSyncEvent(event) {
        const eventType = event.type;
        const timestamp = new Date().toLocaleTimeString();
        
        switch (eventType) {
            case 'response.audio.delta':
                this.logEvent('audio.delta', `Audio chunk: ${event.delta?.length || 0} bytes`, 'server');
                break;
                
            case 'response.audio_transcript.delta':
                this.updateTranscript(event.delta);
                this.logEvent('transcript.delta', `"${event.delta}"`, 'server');
                break;
                
            case 'response.done':
                this.logEvent('response.done', 'Response completed', 'server');
                break;
                
            case 'session.created':
                this.logEvent('session.created', 'OpenAI session ready', 'server');
                break;
                
            case 'session.updated':
                this.logEvent('session.updated', 'Session configuration updated', 'server');
                break;
                
            default:
                this.logEvent(eventType, JSON.stringify(event, null, 2), 'server');
        }
    }
    
    /**
     * Display a generated frame on the canvas using full-body compositing
     */
    async displayFrame(frameMessage) {
        try {
            // Check if current model is ready
            if (!this.modelVideoManager.isModelReady(this.currentModel)) {
                console.log(`⏳ Model ${this.currentModel} not ready, preparing...`);
                await this.prepareCurrentModel();
                return;
            }

            // Get frame data and bounds
            const frameData = frameMessage.frame_data;
            const bounds = frameMessage.bounds;
            
            if (!frameData || !bounds || bounds.length < 4) {
                console.warn("⚠️ Invalid frame data or bounds, displaying mouth region only");
                this.displayMouthRegionOnly(frameData);
                return;
            }

            // Get the appropriate full-body frame
            // For now, use frame 0 as reference - in the future, this could cycle through frames
            // or be based on audio timing
            const frameIndex = this.getAppropriateFrameIndex();
            const fullBodyFrame = this.modelVideoManager.getModelFrame(this.currentModel, frameIndex);
            
            if (!fullBodyFrame) {
                console.warn("⚠️ No full-body frame available, displaying mouth region only");
                this.displayMouthRegionOnly(frameData);
                return;
            }

            // Composite the mouth region onto the full-body frame
            const compositeResult = await this.faceCompositor.compositeFrame(
                fullBodyFrame, 
                frameData, 
                bounds
            );

            // Display the composite result
            const img = new Image();
            img.onload = () => {
                // Clear canvas and draw composite frame
                this.ctx.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
                this.ctx.drawImage(img, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
                
                // Update FPS counter
                this.updateFpsCounter();
            };
            
            img.src = compositeResult;
            
        } catch (error) {
            console.error("❌ Error displaying composite frame:", error);
            // Fallback to mouth region only
            this.displayMouthRegionOnly(frameMessage.frame_data);
        }
    }

    /**
     * Fallback to display mouth region only (original behavior)
     */
    displayMouthRegionOnly(frameData) {
        try {
            if (!frameData) return;
            
            const blob = new Blob([Uint8Array.from(atob(frameData), c => c.charCodeAt(0))]);
            const img = new Image();
            img.onload = () => {
                this.ctx.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
                this.ctx.drawImage(img, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
                this.updateFpsCounter();
                URL.revokeObjectURL(img.src);
            };
            img.src = URL.createObjectURL(blob);
        } catch (error) {
            console.error("❌ Error displaying mouth region:", error);
        }
    }

    /**
     * Get appropriate frame index for current timing
     * In the future, this could be based on audio timing or other factors
     */
    getAppropriateFrameIndex() {
        // For now, just use frame 0 (neutral expression)
        // TODO: Implement proper frame selection based on audio timing or expression
        return 0;
    }

    /**
     * Prepare the current model for compositing
     */
    async prepareCurrentModel() {
        if (this.modelVideoManager.isModelReady(this.currentModel)) {
            return;
        }

        console.log(`🎬 Preparing model ${this.currentModel}...`);
        
        const success = await this.modelVideoManager.prepareModel(
            this.currentModel,
            (progress, status) => {
                this.updateModelPreparationProgress(this.currentModel, progress, status);
            }
        );

        if (success) {
            console.log(`✅ Model ${this.currentModel} ready for compositing`);
        } else {
            console.error(`❌ Failed to prepare model ${this.currentModel}`);
        }
    }

    /**
     * Update model preparation progress in UI
     */
    updateModelPreparationProgress(modelName, progress, status) {
        // Store progress for UI updates
        this.modelPreparationProgress.set(modelName, { progress, status });
        
        // Update status text if this is the current model
        if (modelName === this.currentModel) {
            this.statusText.textContent = `Preparing ${modelName}: ${progress.toFixed(1)}% (${status})`;
        }
        
        console.log(`📥 ${modelName}: ${progress.toFixed(1)}% (${status})`);
    }
    
    /**
     * Update transcript display
     */
    updateTranscript(delta) {
        if (this.transcriptContent.textContent === 'Waiting for speech...') {
            this.transcriptContent.textContent = '';
        }
        
        this.transcriptContent.textContent += delta;
        this.transcriptContent.scrollTop = this.transcriptContent.scrollHeight;
    }
    
    /**
     * Update system statistics display
     */
    updateSystemStats(stats) {
        // Update buffer fills
        this.audioBufferFillSpan.textContent = stats.audio_buffer_fill || 0;
        this.frameBufferFillSpan.textContent = stats.frame_buffer_fill || 0;
        
        // Update buffer bars
        const audioPercent = ((stats.audio_buffer_fill || 0) / 3000) * 100;
        const framePercent = ((stats.frame_buffer_fill || 0) / 3000) * 100;
        
        this.audioBufferBar.style.width = `${audioPercent}%`;
        this.frameBufferBar.style.width = `${framePercent}%`;
        
        // Update current model
        if (stats.current_model) {
            this.currentModelSpan.textContent = stats.current_model;
        }
    }
    
    /**
     * Update frame generation statistics
     */
    updateFrameStats(message) {
        this.logEvent('frames.generated', `Generated ${message.count} frames`, 'system');
        
        // Update generation FPS (simplified calculation)
        const now = performance.now();
        if (this.performanceStats.lastStatsUpdate > 0) {
            const timeDiff = (now - this.performanceStats.lastStatsUpdate) / 1000;
            const genFps = message.count / timeDiff;
            this.generationFps.textContent = genFps.toFixed(1);
        }
        this.performanceStats.lastStatsUpdate = now;
    }
    
    /**
     * Update FPS counter for display
     */
    updateFpsCounter() {
        const now = performance.now();
        this.fpsStats.frameCount++;
        
        if (now - this.fpsStats.lastTime >= 1000) {
            this.fpsStats.fps = this.fpsStats.frameCount;
            this.fpsCounter.textContent = this.fpsStats.fps;
            
            this.fpsStats.frameCount = 0;
            this.fpsStats.lastTime = now;
        }
    }
    
    /**
     * Update connection status indicator
     */
    updateConnectionStatus(connected) {
        if (connected) {
            this.statusDot.classList.add('connected');
            if (!this.isSessionActive) {
                this.statusText.textContent = 'Ready';
            }
        } else {
            this.statusDot.classList.remove('connected');
            if (!this.isSessionActive) {
                this.statusText.textContent = 'Disconnected';
            }
        }
    }
    
    /**
     * Log event to the events panel
     */
    logEvent(type, details, category = 'system') {
        const timestamp = new Date().toLocaleTimeString();
        
        const eventElement = document.createElement('div');
        eventElement.className = `event-item ${category}`;
        
        eventElement.innerHTML = `
            <div class="event-header">
                <span class="event-type">${type}</span>
                <span class="event-time">${timestamp}</span>
            </div>
            <div class="event-details">${details}</div>
        `;
        
        // Add to top of events container
        this.eventsContainer.insertBefore(eventElement, this.eventsContainer.firstChild);
        
        // Remove old events (keep last 50)
        while (this.eventsContainer.children.length > 50) {
            this.eventsContainer.removeChild(this.eventsContainer.lastChild);
        }
    }
    
    /**
     * Start the render loop for smooth video display
     */
    startRenderLoop() {
        const render = () => {
            // Request frame from frame generator if session is active
            if (this.isSessionActive && this.frameGeneratorWs && this.frameGeneratorWs.readyState === WebSocket.OPEN) {
                try {
                    this.frameGeneratorWs.send(JSON.stringify({ type: 'get_frame' }));
                } catch (error) {
                    // Ignore errors - WebSocket might be temporarily unavailable
                }
            }
            
            this.animationFrameId = requestAnimationFrame(render);
        };
        
        render();
    }
    
    /**
     * Start the statistics update loop
     */
    startStatsLoop() {
        const updateStats = () => {
            if (this.frameGeneratorWs && this.frameGeneratorWs.readyState === WebSocket.OPEN) {
                try {
                    this.frameGeneratorWs.send(JSON.stringify({ type: 'get_stats' }));
                } catch (error) {
                    // Ignore errors
                }
            }
            
            // Update every 500ms
            setTimeout(updateStats, 500);
        };
        
        updateStats();
    }
    
    /**
     * Draw placeholder when no video is active
     */
    drawPlaceholder() {
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
        
        this.ctx.fillStyle = '#666666';
        this.ctx.font = '24px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Start Session to Begin', this.videoCanvas.width / 2, this.videoCanvas.height / 2);
        
        this.ctx.font = '16px Arial';
        this.ctx.fillText('Real-time Lip Sync Video', this.videoCanvas.width / 2, this.videoCanvas.height / 2 + 40);
    }
    
    /**
     * Log debug information
     */
    logDebugInfo() {
        const stats = this.lipSyncClient.getPlaybackStats();
        console.log("🐛 Debug Info:", {
            sessionActive: this.isSessionActive,
            playbackStats: stats,
            frameGeneratorConnected: this.frameGeneratorWs?.readyState === WebSocket.OPEN,
            canvasSize: `${this.videoCanvas.width}x${this.videoCanvas.height}`
        });
    }
    
    /**
     * Cleanup when page unloads
     */
    cleanup() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        
        if (this.frameGeneratorWs) {
            this.frameGeneratorWs.close();
        }
        
        if (this.isSessionActive) {
            this.lipSyncClient.stopSession();
        }
    }

    /**
     * Initialize the default model
     */
    async initializeDefaultModel() {
        console.log(`🎬 Initializing default model: ${this.currentModel}`);
        
        try {
            // Prepare the default model in the background
            await this.prepareCurrentModel();
        } catch (error) {
            console.error(`❌ Failed to initialize default model:`, error);
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 Initializing Real-time Lip Sync Console");
    
    const app = new RealtimeLipSyncApp();
    
    // Draw initial placeholder
    app.drawPlaceholder();
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        app.cleanup();
    });
    
    // Expose app to global scope for debugging
    window.lipSyncApp = app;
    
    console.log("✅ Real-time Lip Sync Console ready");
});

export default RealtimeLipSyncApp;
