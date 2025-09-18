/**
 * Face Compositor for Real-time Lip Sync
 * Handles compositing generated mouth regions onto full-body reference frames
 */
class FaceCompositor {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.tempCanvas = document.createElement('canvas');
        this.tempCtx = this.tempCanvas.getContext('2d');
        
        // Default canvas size - will be updated when frames are loaded
        this.canvas.width = 512;
        this.canvas.height = 512;
        this.tempCanvas.width = 320;
        this.tempCanvas.height = 320;
    }

    /**
     * Composite a generated mouth region onto a full-body reference frame
     * @param {ImageData} fullBodyFrame - Full-body reference frame from model video
     * @param {string} mouthRegionData - Base64 encoded mouth region (320x320)
     * @param {Array} bounds - [xmin, ymin, xmax, ymax] face bounds in the full-body frame
     * @returns {Promise<string>} - Base64 encoded composite image
     */
    async compositeFrame(fullBodyFrame, mouthRegionData, bounds) {
        if (!fullBodyFrame || !mouthRegionData || !bounds || bounds.length < 4) {
            throw new Error("Invalid input data for compositing");
        }

        return new Promise((resolve, reject) => {
            try {
                // Update canvas size to match full-body frame
                this.canvas.width = fullBodyFrame.width;
                this.canvas.height = fullBodyFrame.height;

                // Draw the full-body reference frame
                this.ctx.putImageData(fullBodyFrame, 0, 0);

                // Load the mouth region image
                const mouthImg = new Image();
                mouthImg.onload = () => {
                    try {
                        const [xmin, ymin, xmax, ymax] = bounds;
                        const faceWidth = xmax - xmin;
                        const faceHeight = ymax - ymin;

                        // Composite the mouth region onto the face area
                        this.ctx.drawImage(
                            mouthImg,                           // Source: generated mouth region (320x320)
                            0, 0, mouthImg.width, mouthImg.height,  // Source rectangle (full mouth image)
                            xmin, ymin, faceWidth, faceHeight   // Destination rectangle (face bounds in full frame)
                        );

                        // Convert to base64
                        const compositeData = this.canvas.toDataURL('image/jpeg', 0.92);
                        resolve(compositeData);

                    } catch (error) {
                        reject(error);
                    }
                };
                mouthImg.onerror = reject;

                // Handle base64 mouth region data
                if (mouthRegionData.startsWith('data:') || mouthRegionData.startsWith('blob:')) {
                    mouthImg.src = mouthRegionData;
                } else {
                    // Convert base64 bytes to blob URL
                    try {
                        const blob = new Blob([Uint8Array.from(atob(mouthRegionData), c => c.charCodeAt(0))], {type: 'image/jpeg'});
                        mouthImg.src = URL.createObjectURL(blob);
                    } catch (error) {
                        // Fallback: try as direct base64
                        mouthImg.src = `data:image/jpeg;base64,${mouthRegionData}`;
                    }
                }

            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Composite with blend modes for better integration
     * @param {ImageData} fullBodyFrame - Full-body reference frame
     * @param {string} mouthRegionData - Base64 encoded mouth region
     * @param {Array} bounds - Face bounds
     * @param {string} blendMode - Canvas blend mode (default: 'source-over')
     * @returns {Promise<string>} - Base64 encoded composite image
     */
    async compositeWithBlending(fullBodyFrame, mouthRegionData, bounds, blendMode = 'source-over') {
        if (!fullBodyFrame || !mouthRegionData || !bounds || bounds.length < 4) {
            throw new Error("Invalid input data for compositing");
        }

        return new Promise((resolve, reject) => {
            try {
                // Update canvas size to match full-body frame
                this.canvas.width = fullBodyFrame.width;
                this.canvas.height = fullBodyFrame.height;

                // Draw the full-body reference frame
                this.ctx.globalCompositeOperation = 'source-over';
                this.ctx.putImageData(fullBodyFrame, 0, 0);

                const mouthImg = new Image();
                mouthImg.onload = () => {
                    try {
                        const [xmin, ymin, xmax, ymax] = bounds;
                        const faceWidth = xmax - xmin;
                        const faceHeight = ymax - ymin;

                        // Set blend mode for mouth region
                        this.ctx.globalCompositeOperation = blendMode;

                        // Optional: Apply slight feathering for better blending
                        this.ctx.filter = 'blur(0.5px)';

                        // Composite the mouth region
                        this.ctx.drawImage(
                            mouthImg,
                            0, 0, mouthImg.width, mouthImg.height,
                            xmin, ymin, faceWidth, faceHeight
                        );

                        // Reset filters and blend mode
                        this.ctx.filter = 'none';
                        this.ctx.globalCompositeOperation = 'source-over';

                        // Convert to base64
                        const compositeData = this.canvas.toDataURL('image/jpeg', 0.92);
                        resolve(compositeData);

                    } catch (error) {
                        reject(error);
                    }
                };
                mouthImg.onerror = reject;

                // Handle base64 mouth region data
                if (mouthRegionData.startsWith('data:') || mouthRegionData.startsWith('blob:')) {
                    mouthImg.src = mouthRegionData;
                } else {
                    try {
                        const blob = new Blob([Uint8Array.from(atob(mouthRegionData), c => c.charCodeAt(0))], {type: 'image/jpeg'});
                        mouthImg.src = URL.createObjectURL(blob);
                    } catch (error) {
                        mouthImg.src = `data:image/jpeg;base64,${mouthRegionData}`;
                    }
                }

            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Get appropriate frame index based on audio timing
     * @param {number} audioTimestamp - Current audio timestamp
     * @param {number} totalFrames - Total frames in model video
     * @param {number} videoDuration - Total video duration in seconds
     * @returns {number} - Frame index to use
     */
    getFrameIndex(audioTimestamp, totalFrames, videoDuration) {
        // Simple approach: loop the video based on audio timing
        const normalizedTime = (audioTimestamp % videoDuration) / videoDuration;
        const frameIndex = Math.floor(normalizedTime * totalFrames);
        return Math.max(0, Math.min(frameIndex, totalFrames - 1));
    }

    /**
     * Create a test composite to verify the system is working
     * @param {ImageData} testFrame - Test full-body frame
     * @param {string} testMouth - Test mouth data
     * @param {Array} testBounds - Test bounds
     * @returns {Promise<string>} - Test composite result
     */
    async testComposite(testFrame, testMouth, testBounds) {
        try {
            console.log("🧪 Running compositor test...");
            console.log(`   Frame size: ${testFrame.width}x${testFrame.height}`);
            console.log(`   Bounds: [${testBounds.join(', ')}]`);
            
            const result = await this.compositeFrame(testFrame, testMouth, testBounds);
            console.log("✅ Compositor test successful");
            return result;
        } catch (error) {
            console.error("❌ Compositor test failed:", error);
            throw error;
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceCompositor;
} else {
    window.FaceCompositor = FaceCompositor;
}
