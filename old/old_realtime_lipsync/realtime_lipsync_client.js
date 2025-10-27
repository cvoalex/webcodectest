/**
 * Modified WebRTC setup for controlled audio playback with lip sync integration
 */

// AudioBufferManager is loaded globally

class RealtimeLipSyncClient {
    constructor() {
        this.audioBufferManager = new AudioBufferManager();
        this.peerConnection = null;
        this.dataChannel = null;
        this.isSessionActive = false;
    }
    
    /**
     * Start WebRTC session with audio playback and lip sync
     */
    async startSession() {
        try {
            // Get ephemeral token
            const tokenResponse = await fetch("/token");
            const data = await tokenResponse.json();
            const EPHEMERAL_KEY = data.value;
            
            // Create peer connection
            this.peerConnection = new RTCPeerConnection();
            
            // Setup audio capture WITH playback for user experience
            this.setupAudioCaptureWithPlayback();
            
            // Add local audio track (microphone)
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 24000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            this.peerConnection.addTrack(mediaStream.getTracks()[0]);
            
            // Set up data channel for events
            this.dataChannel = this.peerConnection.createDataChannel("oai-events");
            this.setupDataChannelEventHandlers();
            
            // Start session using SDP
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);
            
            const baseUrl = "https://api.openai.com/v1/realtime/calls";
            const model = "gpt-realtime";
            const sdpResponse = await fetch(`${baseUrl}?model=${model}`, {
                method: "POST",
                body: offer.sdp,
                headers: {
                    Authorization: `Bearer ${EPHEMERAL_KEY}`,
                    "Content-Type": "application/sdp",
                },
            });
            
            const answer = {
                type: "answer",
                sdp: await sdpResponse.text(),
            };
            await this.peerConnection.setRemoteDescription(answer);
            
            console.log("🎤 WebRTC session started with controlled audio playback");
            
        } catch (error) {
            console.error("❌ Error starting session:", error);
        }
    }
    
    /**
     * Setup audio capture WITH playback so user can hear responses
     */
    setupAudioCaptureWithPlayback() {
        this.peerConnection.ontrack = (event) => {
            console.log("🎵 Received remote audio track (with playback)");
            
            // Create audio element for playback so user can hear the AI
            const audioElement = document.createElement('audio');
            audioElement.autoplay = true;
            audioElement.srcObject = event.streams[0];
            audioElement.style.display = 'none'; // Hidden but functional
            document.body.appendChild(audioElement);
            
            // ALSO capture the stream for lip sync processing
            const stream = event.streams[0];
            this.audioBufferManager.captureAudioFromStream(stream);
            
            console.log("🎵 Audio stream routed to both playback and buffer manager");
        };
    }
    
    /**
     * Setup data channel event handlers for audio processing
     */
    setupDataChannelEventHandlers() {
        this.dataChannel.addEventListener("open", () => {
            console.log("📡 Data channel opened");
            this.isSessionActive = true;
            this.configureSession();
        });
        
        this.dataChannel.addEventListener("message", (event) => {
            const serverEvent = JSON.parse(event.data);
            this.handleServerEvent(serverEvent);
        });
        
        this.dataChannel.addEventListener("error", (error) => {
            console.error("❌ Data channel error:", error);
        });
    }
    
    /**
     * Configure session for optimal audio processing
     */
    configureSession() {
        const sessionConfig = {
            type: "session.update",
            session: {
                modalities: ["text", "audio"],
                instructions: "You are a helpful assistant. Speak naturally and expressively for optimal lip synchronization. Use clear pronunciation and natural pacing.",
                input_audio_format: "pcm16",
                output_audio_format: "pcm16",
                turn_detection: {
                    type: "server_vad",
                    threshold: 0.5,
                    prefix_padding_ms: 300,
                    silence_duration_ms: 200
                },
                tool_choice: "auto",
                temperature: 0.8
            }
        };
        
        this.dataChannel.send(JSON.stringify(sessionConfig));
        console.log("⚙️ Session configured for streaming audio output with PCM16 format");
    }
    
    /**
     * Handle incoming server events
     */
    handleServerEvent(event) {
        console.log("📡 Server event:", event.type);
        
        switch (event.type) {
            case "session.created":
                console.log("✅ Session created successfully");
                break;
                
            case "session.updated":
                console.log("✅ Session updated successfully");
                break;
                
            case "response.audio.delta":
            case "response.output_audio.delta":
                // CRITICAL: Process audio delta for lip sync (handle both event names)
                this.handleAudioDelta(event);
                break;
                
            case "response.audio.done":
            case "response.output_audio.done":
                console.log("🎵 Audio response complete");
                this.handleAudioComplete(event);
                break;
                
            case "response.audio_transcript.delta":
            case "response.output_audio_transcript.delta":
                // Optional: Display real-time transcript
                this.handleTranscriptDelta(event);
                break;
                
            case "response.done":
                console.log("✅ Response complete");
                break;
                
            case "output_audio_buffer.started":
                console.log("🎵 Audio buffer started");
                break;
                
            case "output_audio_buffer.stopped":
                console.log("🎵 Audio buffer stopped");
                break;
                
            default:
                console.log("📡 Unhandled event:", event.type);
        }
    }
    
    /**
     * Handle audio delta events for lip sync generation
     */
    handleAudioDelta(event) {
        console.log("🔍 Audio delta event received:", event);
        
        if (event.delta && event.delta.length > 0) {
            console.log(`🎵 Received audio delta: ${event.delta.length} bytes`);
            
            // Send base64 audio to buffer manager
            this.audioBufferManager.processOpenAIAudioDelta(event.delta);
        } else {
            console.warn("⚠️ Audio delta event has no data:", event);
        }
    }

    /**
     * Handle complete audio response - check if we can get audio data here
     */
    handleAudioComplete(event) {
        console.log("🔍 Audio complete event received:", event);
        console.log("🔍 Event keys:", Object.keys(event));
        
        // The OpenAI Realtime API might not send complete audio in done events
        // Instead, it sends streaming audio deltas during generation
        // Let's check all possible locations for audio data
        const possibleAudioPaths = [
            event.audio,
            event.data,
            event.content,
            event.response?.audio,
            event.response?.data,
            event.item?.content?.[0]?.audio,
            event.output?.audio
        ];
        
        let foundAudio = false;
        for (let i = 0; i < possibleAudioPaths.length; i++) {
            const audioData = possibleAudioPaths[i];
            if (audioData) {
                console.log(`🎵 Found audio data at path ${i}: ${typeof audioData === 'string' ? audioData.length + ' chars' : 'object'}`);
                this.audioBufferManager.processOpenAIAudioDelta(audioData);
                foundAudio = true;
                break;
            }
        }
        
        if (!foundAudio) {
            console.warn("⚠️ Audio complete event has no audio data in any expected location");
            console.log("🔍 Full event structure:", JSON.stringify(event, null, 2));
        }
    }
    
    /**
     * Handle transcript delta for real-time captions
     */
    handleTranscriptDelta(event) {
        if (event.delta) {
            console.log("📝 Transcript delta:", event.delta);
            // Update UI with real-time transcript
            this.updateTranscriptDisplay(event.delta);
        }
    }
    
    /**
     * Send text message to the model
     */
    sendTextMessage(message) {
        if (!this.isSessionActive || !this.dataChannel) {
            console.error("❌ Session not active");
            return;
        }
        
        const event = {
            type: "conversation.item.create",
            item: {
                type: "message",
                role: "user",
                content: [
                    {
                        type: "input_text",
                        text: message,
                    },
                ],
            },
        };
        
        this.dataChannel.send(JSON.stringify(event));
        
        // Create response - use simple format, modalities are set in session config
        const responseConfig = {
            type: "response.create"
        };
        
        this.dataChannel.send(JSON.stringify(responseConfig));
        
        console.log("📤 Text message sent:", message);
    }
    
    /**
     * Stop session and cleanup
     */
    stopSession() {
        if (this.dataChannel) {
            this.dataChannel.close();
        }
        
        if (this.peerConnection) {
            this.peerConnection.getSenders().forEach((sender) => {
                if (sender.track) {
                    sender.track.stop();
                }
            });
            this.peerConnection.close();
        }
        
        this.isSessionActive = false;
        console.log("🛑 Session stopped");
    }
    
    /**
     * Get current synchronized frame for display
     */
    getCurrentFrame() {
        return this.audioBufferManager.getCurrentFrame();
    }
    
    /**
     * Get playback statistics
     */
    getPlaybackStats() {
        return {
            audioBufferFill: this.audioBufferManager.getBufferFillLevel(),
            frameBufferFill: this.audioBufferManager.getFrameBufferFillLevel(),
            playbackPosition: this.audioBufferManager.getCurrentPlaybackPosition(),
            isBuffering: this.audioBufferManager.isBuffering
        };
    }
    
    /**
     * Update transcript display in UI
     */
    updateTranscriptDisplay(delta) {
        // Implementation for updating UI with transcript
        const transcriptElement = document.getElementById('transcript');
        if (transcriptElement) {
            transcriptElement.textContent += delta;
        }
    }
}

// Make RealtimeLipSyncClient available globally
window.RealtimeLipSyncClient = RealtimeLipSyncClient;