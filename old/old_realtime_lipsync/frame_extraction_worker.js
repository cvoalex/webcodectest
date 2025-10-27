/**
 * Web Worker for fast frame extraction from video
 * Runs in background thread to avoid blocking UI
 */

self.onmessage = async function(e) {
    const { videoBlob, modelName, options = {} } = e.data;
    
    try {
        const frames = await extractFramesInWorker(videoBlob, modelName, options);
        self.postMessage({
            type: 'success',
            frames: frames,
            modelName: modelName
        });
    } catch (error) {
        self.postMessage({
            type: 'error',
            error: error.message,
            modelName: modelName
        });
    }
};

async function extractFramesInWorker(videoBlob, modelName, options) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.muted = true;
        video.playsInline = true;
        
        // Use OffscreenCanvas for worker thread
        const canvas = new OffscreenCanvas(1, 1);
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        
        const frames = [];
        const fps = options.fps || 15;
        const maxDimension = options.maxDimension || 512;

        video.onloadedmetadata = () => {
            // Optimize resolution
            const scale = Math.min(1, maxDimension / Math.max(video.videoWidth, video.videoHeight));
            canvas.width = Math.floor(video.videoWidth * scale);
            canvas.height = Math.floor(video.videoHeight * scale);

            const duration = video.duration;
            const totalFrames = Math.floor(duration * fps);
            const frameInterval = 1 / fps;
            let currentFrame = 0;

            const extractNextFrame = () => {
                if (currentFrame >= totalFrames) {
                    resolve(frames);
                    return;
                }

                video.currentTime = currentFrame * frameInterval;
            };

            video.onseeked = () => {
                try {
                    // Draw and extract frame
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Convert to transferable ImageBitmap for faster transfer
                    canvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 }).then(blob => {
                        frames.push({
                            blob: blob,
                            timestamp: video.currentTime,
                            frameIndex: currentFrame,
                            width: canvas.width,
                            height: canvas.height
                        });

                        currentFrame++;
                        
                        // Send progress update
                        self.postMessage({
                            type: 'progress',
                            progress: (currentFrame / totalFrames) * 100,
                            modelName: modelName
                        });

                        // Continue extraction
                        extractNextFrame();
                    });

                } catch (error) {
                    console.error('Frame extraction error:', error);
                    currentFrame++;
                    extractNextFrame();
                }
            };

            video.onerror = reject;
            extractNextFrame();
        };

        video.onerror = reject;
        
        // Load video
        const videoURL = URL.createObjectURL(videoBlob);
        video.src = videoURL;
        video.load();
    });
}
