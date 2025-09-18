/**
 * Model Video Manager for Real-time Lip Sync
 * Downloads and manages full-body reference videos for each model
 */
class ModelVideoManager {
    constructor() {
        this.modelVideos = new Map(); // model_name -> video_data
        this.modelFrames = new Map(); // model_name -> extracted_frames_array (memory cache)
        this.modelFrameFiles = new Map(); // model_name -> frame_file_info
        this.downloadProgress = new Map(); // model_name -> progress_info
        this.isExtracting = new Map(); // model_name -> boolean
        
        // Configuration
        this.frameExtractionWorker = null;
        this.maxCachedModels = 3; // Cache fewer models in memory
        this.maxCachedFrames = 50; // Cache max 50 frames per model in memory
        this.useWebWorker = false; // Disable Web Worker, use optimized main thread
        this.useFilesystem = true; // Extract frames to filesystem/IndexedDB
        this.extractionOptions = {
            fps: 25,              // Required 25fps - no compromise on frame rate
            maxDimension: 256,    // Smaller resolution for speed (will scale up if needed)
            quality: 0.6,         // Lower quality for much faster processing
            format: 'jpeg'        // JPEG for smaller file sizes
        };
        
        // Initialize IndexedDB for frame storage
        this.db = null;
        this.initializeFrameStorage();
    }

    /**
     * Initialize IndexedDB for persistent frame storage
     */
    async initializeFrameStorage() {
        try {
            this.db = await this.openIndexedDB('ModelFramesDB', 1);
            console.log('🗄️ Frame storage initialized (IndexedDB)');
        } catch (error) {
            console.warn('⚠️ IndexedDB not available, using memory only:', error);
            this.useFilesystem = false;
        }
    }

    /**
     * Open IndexedDB connection
     */
    openIndexedDB(dbName, version) {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(dbName, version);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Create frames store
                if (!db.objectStoreNames.contains('frames')) {
                    const framesStore = db.createObjectStore('frames', { keyPath: 'id' });
                    framesStore.createIndex('modelName', 'modelName', { unique: false });
                    framesStore.createIndex('frameIndex', 'frameIndex', { unique: false });
                }
                
                // Create models store
                if (!db.objectStoreNames.contains('models')) {
                    const modelsStore = db.createObjectStore('models', { keyPath: 'modelName' });
                }
            };
        });
    }

    /**
     * Initialize Web Worker for frame extraction
     */
    initializeWorker() {
        if (!this.useWebWorker || this.frameExtractionWorker) return;
        
        try {
            this.frameExtractionWorker = new Worker('./frame_extraction_worker.js');
            console.log('🔧 Frame extraction Web Worker initialized');
        } catch (error) {
            console.warn('⚠️ Web Worker not available, using main thread extraction:', error);
            this.useWebWorker = false;
        }
    }

    /**
     * Download and prepare a model's reference video
     * @param {string} modelName - Name of the model
     * @param {function} progressCallback - Progress callback (progress, status)
     * @returns {Promise<boolean>} - Success status
     */
    async prepareModel(modelName, progressCallback = null) {
        try {
            // Check if already prepared in memory
            if (this.modelFrames.has(modelName)) {
                console.log(`✅ Model ${modelName} already prepared in memory`);
                if (progressCallback) progressCallback(100, 'ready');
                return true;
            }

            // Check if frames exist in storage
            if (this.useFilesystem && await this.checkFramesInStorage(modelName)) {
                console.log(`🗄️ Model ${modelName} frames found in storage, skipping extraction`);
                this.modelFrameFiles.set(modelName, { fromStorage: true });
                if (progressCallback) progressCallback(100, 'ready');
                return true;
            }

            console.log(`🎬 Preparing model: ${modelName}`);
            
            // Step 1: Download video
            if (progressCallback) progressCallback(0, 'downloading');
            const videoBlob = await this.downloadModelVideo(modelName, (progress) => {
                if (progressCallback) progressCallback(progress * 0.7, 'downloading'); // 70% for download
            });

            if (!videoBlob) {
                throw new Error('Failed to download model video');
            }

            // Step 2: Extract frames
            if (progressCallback) progressCallback(70, 'extracting');
            
            // Use fast main thread extraction with filesystem storage
            const frames = await this.extractFramesFromVideo(videoBlob, modelName, (progress) => {
                if (progressCallback) progressCallback(70 + (progress * 0.3), 'extracting');
            });

            // Step 3: Store frames (in memory for immediate use, filesystem for persistence)
            if (!this.useFilesystem) {
                this.modelFrames.set(modelName, frames);
            }
            this.modelVideos.set(modelName, videoBlob);

            // Mark model as extracted in storage
            if (this.useFilesystem) {
                await this.markModelExtracted(modelName, frames ? frames.length : 0);
                this.modelFrameFiles.set(modelName, { fromStorage: true });
            }

            // Clean up old models if cache is full
            this.cleanupCache();

            const frameCount = frames ? frames.length : 'filesystem';
            console.log(`✅ Model ${modelName} prepared with ${frameCount} frames`);
            if (progressCallback) progressCallback(100, 'ready');
            return true;

        } catch (error) {
            console.error(`❌ Error preparing model ${modelName}:`, error);
            if (progressCallback) progressCallback(0, 'error');
            return false;
        }
    }

    /**
     * Download model video from server
     * @param {string} modelName - Name of the model
     * @param {function} progressCallback - Progress callback
     * @returns {Promise<Blob>} - Video blob
     */
    async downloadModelVideo(modelName, progressCallback = null) {
        try {
            const response = await fetch(`/api/model-video/${encodeURIComponent(modelName)}`, {
                method: 'GET',
                headers: {
                    'Accept': 'video/mp4'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const contentLength = parseInt(response.headers.get('Content-Length') || '0');
            const reader = response.body.getReader();
            const chunks = [];
            let receivedLength = 0;

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                chunks.push(value);
                receivedLength += value.length;
                
                if (progressCallback && contentLength > 0) {
                    const progress = (receivedLength / contentLength) * 100;
                    progressCallback(progress);
                }
            }

            // Combine chunks into blob
            const videoBlob = new Blob(chunks, { type: 'video/mp4' });
            console.log(`📥 Downloaded ${modelName} video: ${(videoBlob.size / 1024 / 1024).toFixed(2)}MB`);
            
            return videoBlob;

        } catch (error) {
            console.error(`❌ Error downloading video for ${modelName}:`, error);
            return null;
        }
    }

    /**
     * Extract frames from video blob using fast batch processing
     * @param {Blob} videoBlob - Video blob data
     * @param {string} modelName - Model name for progress tracking
     * @param {function} progressCallback - Progress callback
     * @returns {Promise<Array>} - Array of frame ImageData objects
     */
    async extractFramesFromVideo(videoBlob, modelName, progressCallback = null) {
        return new Promise((resolve, reject) => {
            try {
                const video = document.createElement('video');
                video.muted = true;
                video.playsInline = true;
                
                // Use OffscreenCanvas for better performance (fallback to regular canvas)
                let canvas, ctx;
                try {
                    canvas = new OffscreenCanvas(1, 1);
                    ctx = canvas.getContext('2d', { willReadFrequently: true });
                } catch (e) {
                    canvas = document.createElement('canvas');
                    ctx = canvas.getContext('2d', { willReadFrequently: true });
                }

                const frames = [];

                video.onloadedmetadata = async () => {
                    try {
                        // Optimize canvas size - reduce resolution for faster processing
                        const scale = Math.min(1, 384 / Math.max(video.videoWidth, video.videoHeight));
                        canvas.width = Math.floor(video.videoWidth * scale);
                        canvas.height = Math.floor(video.videoHeight * scale);

                        const fps = this.extractionOptions.fps; // Use configured fps (25fps required)
                        const duration = video.duration;
                        const totalFrames = Math.floor(duration * fps);
                        const frameInterval = 1 / fps;
                        
                        console.log(`🎬 Video loaded: ${video.videoWidth}x${video.videoHeight}, duration: ${duration.toFixed(2)}s`);
                        console.log(`🎬 Extracting ${totalFrames} frames at ${fps}fps (scaled to ${canvas.width}x${canvas.height})`);

                        // Use ultra-fast batch extraction method
                        await this.extractFramesBatch(video, canvas, ctx, fps, totalFrames, frameInterval, progressCallback, resolve, reject);
                        
                    } catch (error) {
                        console.error('❌ Error in onloadedmetadata:', error);
                        reject(error);
                    }
                };

                video.onerror = (error) => {
                    reject(new Error(`Video load error: ${error}`));
                };

                // Load video
                const videoURL = URL.createObjectURL(videoBlob);
                video.src = videoURL;
                video.load();

            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Fast frame extraction using requestVideoFrameCallback (Chrome/Edge)
     */
    async extractFramesWithCallback(video, canvas, ctx, fps, totalFrames, progressCallback, resolve, reject) {
        const frames = [];
        let frameCount = 0;
        const frameInterval = 1 / fps;

        const extractFrame = () => {
            if (frameCount >= totalFrames) {
                resolve(frames);
                return;
            }

            video.requestVideoFrameCallback(() => {
                try {
                    // Draw and extract frame
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    
                    frames.push({
                        data: imageData,
                        timestamp: video.currentTime,
                        frameIndex: frameCount
                    });

                    frameCount++;
                    
                    if (progressCallback) {
                        progressCallback((frameCount / totalFrames) * 100);
                    }

                    // Seek to next frame
                    video.currentTime = frameCount * frameInterval;
                    
                    // Continue extraction
                    setTimeout(extractFrame, 5); // Minimal delay

                } catch (error) {
                    console.error('Frame extraction error:', error);
                    frameCount++;
                    setTimeout(extractFrame, 5);
                }
            });
        };

        video.currentTime = 0;
        extractFrame();
    }

    /**
     * Optimized seek-based frame extraction (fallback)
     */
    async extractFramesWithSeek(video, canvas, ctx, fps, totalFrames, frameInterval, progressCallback, resolve, reject) {
        const frames = [];
        let currentFrame = 0;
        let timeoutId = null;
        
        console.log(`🔧 Starting seek-based extraction: ${totalFrames} frames`);
        
        const extractNextFrame = () => {
            if (currentFrame >= totalFrames) {
                console.log(`✅ Extraction complete: ${frames.length} frames extracted`);
                resolve(frames);
                return;
            }

            const timeToSeek = currentFrame * frameInterval;
            
            // Clear any existing timeout
            if (timeoutId) clearTimeout(timeoutId);
            
            // Set a timeout in case seeking gets stuck
            timeoutId = setTimeout(() => {
                console.warn(`⚠️ Seek timeout for frame ${currentFrame}, skipping...`);
                currentFrame++;
                extractNextFrame();
            }, 1000); // 1 second timeout per frame
            
            video.currentTime = timeToSeek;
        };

        video.onseeked = () => {
            try {
                // Clear timeout
                if (timeoutId) {
                    clearTimeout(timeoutId);
                    timeoutId = null;
                }
                
                // Draw current frame
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Get image data
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                frames.push({
                    data: imageData,
                    timestamp: video.currentTime,
                    frameIndex: currentFrame
                });

                currentFrame++;
                
                if (progressCallback) {
                    const progress = (currentFrame / totalFrames) * 100;
                    progressCallback(progress);
                }

                // Process next frame
                setTimeout(extractNextFrame, 10); // Small delay to prevent blocking

            } catch (error) {
                console.error(`❌ Error extracting frame ${currentFrame}:`, error);
                currentFrame++;
                setTimeout(extractNextFrame, 10);
            }
        };

        video.onerror = (error) => {
            console.error('❌ Video error during frame extraction:', error);
            reject(new Error(`Video error during frame extraction: ${error}`));
        };

        // Start extraction
        extractNextFrame();
    }

    /**
     * Ultra-fast batch frame extraction - processes frames in batches for 25fps
     * @param {HTMLVideoElement} video - Video element
     * @param {Canvas} canvas - Canvas for drawing
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {number} fps - Frames per second (25)
     * @param {number} totalFrames - Total frames to extract
     * @param {number} frameInterval - Time between frames
     * @param {function} progressCallback - Progress callback
     * @param {function} resolve - Promise resolve
     * @param {function} reject - Promise reject
     */
    async extractFramesBatch(video, canvas, ctx, fps, totalFrames, frameInterval, progressCallback, resolve, reject) {
        const frames = [];
        let currentFrame = 0;
        let isExtracting = false;
        const batchSize = 5; // Process 5 frames per batch for speed
        
        // Pre-create ImageData object for reuse (memory optimization)
        let imageDataTemplate = null;
        
        const processBatch = async () => {
            if (currentFrame >= totalFrames) {
                const message = this.useFilesystem ? 
                    `✅ Batch extraction completed: ${totalFrames} frames saved to storage at 25fps` :
                    `✅ Batch extraction completed: ${frames.length} frames in memory at 25fps`;
                console.log(message);
                resolve(this.useFilesystem ? [] : frames); // Return empty array if using filesystem
                return;
            }

            if (isExtracting) return; // Avoid overlapping extractions
            isExtracting = true;

            const batchEnd = Math.min(currentFrame + batchSize, totalFrames);
            const framesToProcess = [];
            
            // Collect batch of frame times
            for (let i = currentFrame; i < batchEnd; i++) {
                framesToProcess.push(i * frameInterval);
            }

            // Process each frame in the batch rapidly
            for (const timeToSeek of framesToProcess) {
                try {
                    video.currentTime = timeToSeek;
                    
                    // Wait for seek to complete (using Promise to avoid callback hell)
                    await new Promise(async (seekResolve) => {
                        const onSeeked = async () => {
                            video.removeEventListener('seeked', onSeeked);
                            
                            try {
                                // Fast canvas draw with optimized settings
                                ctx.globalCompositeOperation = 'source-over';
                                ctx.imageSmoothingEnabled = false; // Disable smoothing for speed
                                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                                
                                // Reuse ImageData object if possible
                                if (!imageDataTemplate || imageDataTemplate.width !== canvas.width) {
                                    imageDataTemplate = ctx.createImageData(canvas.width, canvas.height);
                                }
                                
                                if (this.useFilesystem && this.db) {
                                    // Save frame to IndexedDB as compressed JPEG
                                    await this.saveFrameToStorage(canvas, currentFrame, video.currentTime, modelName);
                                } else {
                                    // Fallback: keep in memory
                                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                                    frames.push({
                                        data: imageData,
                                        timestamp: video.currentTime,
                                        frameIndex: currentFrame,
                                        width: canvas.width,
                                        height: canvas.height
                                    });
                                }

                                currentFrame++;
                                seekResolve();
                                
                            } catch (drawError) {
                                console.warn('⚠️ Frame draw error, skipping:', drawError);
                                currentFrame++;
                                seekResolve();
                            }
                        };
                        
                        video.addEventListener('seeked', onSeeked);
                        
                        // Timeout fallback
                        setTimeout(() => {
                            video.removeEventListener('seeked', onSeeked);
                            console.warn('⚠️ Seek timeout, skipping frame');
                            currentFrame++;
                            seekResolve();
                        }, 100);
                    });
                    
                } catch (error) {
                    console.warn('⚠️ Frame extraction error, skipping:', error);
                    currentFrame++;
                }
            }

            // Update progress
            if (progressCallback) {
                const progress = (currentFrame / totalFrames) * 100;
                progressCallback(progress);
            }

            isExtracting = false;
            
            // Continue with next batch immediately
            setTimeout(processBatch, 1); // Minimal delay
        };

        video.onerror = (error) => {
            console.error('❌ Video error during batch extraction:', error);
            reject(new Error(`Video error: ${error}`));
        };

        // Start batch processing
        processBatch();
    }

    /**
     * Save frame to IndexedDB storage as compressed image
     * @param {Canvas} canvas - Canvas with frame data
     * @param {number} frameIndex - Frame index
     * @param {number} timestamp - Video timestamp
     * @param {string} modelName - Model name
     */
    async saveFrameToStorage(canvas, frameIndex, timestamp, modelName) {
        try {
            // Convert canvas to compressed blob
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, `image/${this.extractionOptions.format}`, this.extractionOptions.quality);
            });

            // Store in IndexedDB
            const transaction = this.db.transaction(['frames'], 'readwrite');
            const store = transaction.objectStore('frames');
            
            const frameData = {
                id: `${modelName}_${frameIndex}`,
                modelName: modelName,
                frameIndex: frameIndex,
                timestamp: timestamp,
                blob: blob,
                width: canvas.width,
                height: canvas.height,
                format: this.extractionOptions.format,
                created: Date.now()
            };

            await new Promise((resolve, reject) => {
                const request = store.put(frameData);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

        } catch (error) {
            console.warn('⚠️ Failed to save frame to storage:', error);
        }
    }

    /**
     * Load frame from IndexedDB storage
     * @param {string} modelName - Model name
     * @param {number} frameIndex - Frame index
     * @returns {Promise<ImageData|null>} - Frame image data
     */
    async loadFrameFromStorage(modelName, frameIndex) {
        try {
            if (!this.db) return null;

            const transaction = this.db.transaction(['frames'], 'readonly');
            const store = transaction.objectStore('frames');
            
            const frameData = await new Promise((resolve, reject) => {
                const request = store.get(`${modelName}_${frameIndex}`);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });

            if (!frameData) return null;

            // Convert blob back to ImageData
            const img = new Image();
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            return new Promise((resolve) => {
                img.onload = () => {
                    canvas.width = frameData.width;
                    canvas.height = frameData.height;
                    ctx.drawImage(img, 0, 0);
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    
                    resolve({
                        data: imageData,
                        timestamp: frameData.timestamp,
                        frameIndex: frameData.frameIndex,
                        width: frameData.width,
                        height: frameData.height
                    });
                };
                
                img.onerror = () => resolve(null);
                img.src = URL.createObjectURL(frameData.blob);
            });

        } catch (error) {
            console.warn('⚠️ Failed to load frame from storage:', error);
            return null;
        }
    }

    /**
     * Check if model frames exist in storage
     * @param {string} modelName - Model name
     * @returns {Promise<boolean>} - Whether frames exist
     */
    async checkFramesInStorage(modelName) {
        try {
            if (!this.db) return false;

            const transaction = this.db.transaction(['models'], 'readonly');
            const store = transaction.objectStore('models');
            
            const modelData = await new Promise((resolve, reject) => {
                const request = store.get(modelName);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });

            return modelData && modelData.framesExtracted;
        } catch (error) {
            console.warn('⚠️ Failed to check frames in storage:', error);
            return false;
        }
    }

    /**
     * Mark model as having frames extracted
     * @param {string} modelName - Model name
     * @param {number} totalFrames - Total number of frames
     */
    async markModelExtracted(modelName, totalFrames) {
        try {
            if (!this.db) return;

            const transaction = this.db.transaction(['models'], 'readwrite');
            const store = transaction.objectStore('models');
            
            const modelData = {
                modelName: modelName,
                totalFrames: totalFrames,
                framesExtracted: true,
                extractedAt: Date.now(),
                fps: this.extractionOptions.fps
            };

            await new Promise((resolve, reject) => {
                const request = store.put(modelData);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

        } catch (error) {
            console.warn('⚠️ Failed to mark model as extracted:', error);
        }
    }

    /**
     * Extract frames using Web Worker (fastest method)
     * @param {Blob} videoBlob - Video blob data
     * @param {string} modelName - Model name for progress tracking
     * @param {function} progressCallback - Progress callback
     * @returns {Promise<Array>} - Array of frame data
     */
    async extractFramesWithWorker(videoBlob, modelName, progressCallback = null) {
        return new Promise((resolve, reject) => {
            // Set up worker message handling
            const handleWorkerMessage = (e) => {
                const { type, progress, frames, error } = e.data;
                
                switch (type) {
                    case 'progress':
                        if (progressCallback) {
                            progressCallback(progress);
                        }
                        break;
                        
                    case 'success':
                        this.frameExtractionWorker.removeEventListener('message', handleWorkerMessage);
                        console.log(`✅ Web Worker extracted ${frames.length} frames for ${modelName}`);
                        resolve(frames);
                        break;
                        
                    case 'error':
                        this.frameExtractionWorker.removeEventListener('message', handleWorkerMessage);
                        console.error(`❌ Web Worker extraction failed for ${modelName}:`, error);
                        reject(new Error(error));
                        break;
                }
            };

            this.frameExtractionWorker.addEventListener('message', handleWorkerMessage);
            
            // Send extraction job to worker
            this.frameExtractionWorker.postMessage({
                videoBlob: videoBlob,
                modelName: modelName,
                options: this.extractionOptions
            });
        });
    }

    /**
     * Get a specific frame for a model (async for filesystem support)
     * @param {string} modelName - Model name
     * @param {number} frameIndex - Frame index
     * @returns {Promise<ImageData|null>} - Frame image data or null if not available
     */
    async getModelFrame(modelName, frameIndex) {
        // Try memory cache first
        const frames = this.modelFrames.get(modelName);
        if (frames && frameIndex >= 0 && frameIndex < frames.length) {
            return frames[frameIndex].data;
        }

        // Try filesystem storage
        if (this.useFilesystem && this.modelFrameFiles.has(modelName)) {
            const frameData = await this.loadFrameFromStorage(modelName, frameIndex);
            return frameData ? frameData.data : null;
        }

        return null;
    }

    /**
     * Get a specific frame for a model (synchronous version for backwards compatibility)
     * @param {string} modelName - Model name
     * @param {number} frameIndex - Frame index
     * @returns {ImageData|null} - Frame image data or null if not available
     */
    getModelFrameSync(modelName, frameIndex) {
        const frames = this.modelFrames.get(modelName);
        if (!frames || frameIndex < 0 || frameIndex >= frames.length) {
            return null;
        }
        return frames[frameIndex].data;
    }

    /**
     * Get total frame count for a model
     * @param {string} modelName - Model name
     * @returns {number} - Total frame count
     */
    getFrameCount(modelName) {
        const frames = this.modelFrames.get(modelName);
        return frames ? frames.length : 0;
    }

    /**
     * Check if model is ready
     * @param {string} modelName - Model name
     * @returns {boolean} - True if model is ready
     */
    isModelReady(modelName) {
        return this.modelFrames.has(modelName) || this.modelFrameFiles.has(modelName);
    }

    /**
     * Clean up cache when it gets too large
     */
    cleanupCache() {
        if (this.modelFrames.size <= this.maxCachedModels) {
            return;
        }

        // Remove oldest models (simple FIFO for now)
        const modelNames = Array.from(this.modelFrames.keys());
        const modelsToRemove = modelNames.slice(0, modelNames.length - this.maxCachedModels);

        for (const modelName of modelsToRemove) {
            this.modelFrames.delete(modelName);
            this.modelVideos.delete(modelName);
            console.log(`🧹 Cleaned up cached model: ${modelName}`);
        }
    }

    /**
     * Get download progress for a model
     * @param {string} modelName - Model name
     * @returns {Object|null} - Progress info or null
     */
    getProgress(modelName) {
        return this.downloadProgress.get(modelName) || null;
    }

    /**
     * Preload commonly used models
     * @param {Array<string>} modelNames - Array of model names to preload
     */
    async preloadModels(modelNames) {
        console.log(`🚀 Preloading ${modelNames.length} models...`);
        
        for (const modelName of modelNames) {
            try {
                await this.prepareModel(modelName, (progress, status) => {
                    console.log(`📥 ${modelName}: ${progress.toFixed(1)}% (${status})`);
                });
            } catch (error) {
                console.error(`❌ Failed to preload ${modelName}:`, error);
            }
        }
        
        console.log(`✅ Preloading complete`);
    }

    /**
     * Get memory usage statistics
     * @returns {Object} - Memory usage info
     */
    getMemoryStats() {
        let totalFrames = 0;
        let totalVideoSize = 0;

        for (const [modelName, frames] of this.modelFrames) {
            totalFrames += frames.length;
            const videoBlob = this.modelVideos.get(modelName);
            if (videoBlob) {
                totalVideoSize += videoBlob.size;
            }
        }

        // Estimate frame memory (width * height * 4 bytes per pixel)
        const avgFrameSize = 512 * 512 * 4; // Assuming 512x512 frames
        const estimatedFrameMemory = totalFrames * avgFrameSize;

        return {
            cachedModels: this.modelFrames.size,
            totalFrames,
            videoSizeMB: (totalVideoSize / 1024 / 1024).toFixed(2),
            estimatedFrameMemoryMB: (estimatedFrameMemory / 1024 / 1024).toFixed(2)
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelVideoManager;
} else {
    window.ModelVideoManager = ModelVideoManager;
}
