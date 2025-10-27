from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import base64
import cv2
import numpy as np
import time
import json
import tempfile
import os

from config import settings
from multi_model_engine import multi_model_engine
from multi_model_cache import multi_cache_manager
from dynamic_model_manager import dynamic_model_manager

# Initialize FastAPI app
app = FastAPI(
    title="SyncTalk2D Multi-Model Fast Inference Service",
    description="Real-time frame-by-frame lip-sync generation with multi-model support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class FrameRequest(BaseModel):
    model_name: str
    frame_id: int
    audio_override: Optional[str] = None  # Base64 encoded audio

class MultiAudioFrameRequest(BaseModel):
    model_name: str
    frame_id: int
    audio_override: Optional[str] = None

class MultiBatchFrameRequest(BaseModel):
    requests: List[MultiAudioFrameRequest]

class ModelLoadRequest(BaseModel):
    model_name: str
    package_path: str
    audio_override: Optional[str] = None

class PreloadRequest(BaseModel):
    model_name: str
    start_frame: int = 0
    end_frame: int = 100

class WebSocketFrameRequest(BaseModel):
    type: str = "frame_request"
    model_name: str
    frame_id: int
    audio_override: Optional[str] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Utility functions
def encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode frame to base64 for JSON response - ultra-fast version"""
    # For performance testing: use raw numpy array (no compression)
    # This eliminates JPEG/PNG compression overhead entirely
    frame_bytes = frame.tobytes()
    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
    # Include shape info for reconstruction: height,width,channels:data
    shape_info = f"{frame.shape[0]},{frame.shape[1]},{frame.shape[2]}:"
    return shape_info + frame_b64

def decode_audio_from_base64(audio_b64: str) -> str:
    """Decode base64 audio to temporary file"""
    audio_data = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(audio_data)
        return temp_file.name

# Model management endpoints
@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a new model package"""
    
    try:
        print(f"🎯 Service: Starting model load for '{request.model_name}' with package_path='{request.package_path}'")
        
        # Handle audio override
        temp_audio_path = None
        if request.audio_override:
            temp_audio_path = decode_audio_from_base64(request.audio_override)
        
        print(f"🎯 Service: About to call multi_model_engine.load_model...")
        result = await multi_model_engine.load_model(
            request.model_name, 
            request.package_path,
            audio_override=temp_audio_path
        )
        
        print(f"🎯 Service: Result from engine: {result}")
        print(f"🎯 Service: Models in engine after load: {len(multi_model_engine.models)}")
        print(f"🎯 Service: Model keys: {list(multi_model_engine.models.keys())}")
        
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        
        return {
            "success": True,
            "model_name": request.model_name,
            "message": f"Model {request.model_name} loaded successfully",
            "details": result
        }
        
    except Exception as e:
        print(f"🎯 Service: Exception during model load: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a specific model"""
    
    try:
        result = await multi_model_engine.unload_model(model_name)
        
        return {
            "success": True,
            "model_name": model_name,
            "message": f"Model {model_name} unloaded successfully",
            "details": result
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/models")
async def list_models():
    """List all loaded models and locally available models"""
    
    try:
        # Get loaded models
        loaded_models = multi_model_engine.list_models()
        
        # Get local models (extracted and zipped)
        local_models = dynamic_model_manager.list_local_models()
        
        # Debug: check what's actually in the engine
        debug_info = {
            "engine_models_keys": list(multi_model_engine.models.keys()),
            "engine_models_count": len(multi_model_engine.models)
        }
        
        return {
            "success": True,
            "loaded_models": loaded_models,
            "total_loaded": len(loaded_models),
            "local_models": local_models,
            "total_local": local_models["total"],
            "debug": debug_info
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/models/registry")
async def list_registry_models():
    """List models available in the central registry"""
    
    try:
        # Mock registry models - in real implementation this would query the API
        registry_models = [
            {
                "name": "default_model",
                "version": "1.0.0", 
                "description": "Default SyncTalk2D model",
                "size_mb": 86.2,
                "available": True
            },
            {
                "name": "enhanced_model",
                "version": "1.1.0",
                "description": "Enhanced quality model", 
                "size_mb": 120.5,
                "available": True
            },
            {
                "name": "fast_model",
                "version": "1.0.1",
                "description": "Optimized for speed",
                "size_mb": 65.8,
                "available": True
            }
        ]
        
        return {
            "success": True,
            "registry_models": registry_models,
            "total_available": len(registry_models),
            "registry_url": dynamic_model_manager.central_repo_url
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/models/download")
async def download_model(model_name: str):
    """Manually download and extract a model from the registry"""
    
    try:
        result = await dynamic_model_manager.ensure_model_available(model_name)
        
        return {
            "success": result["success"],
            "model_name": model_name,
            "actions_taken": result["actions_taken"],
            "model_path": result.get("model_path"),
            "error": result.get("error")
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/models/local/{model_name}")
async def cleanup_local_model(model_name: str, remove_zip: bool = False):
    """Clean up local model files"""
    
    try:
        # Unload from memory first if loaded
        if multi_model_engine.is_model_loaded(model_name):
            await multi_model_engine.unload_model(model_name)
        
        # Clean up local files
        result = await dynamic_model_manager.cleanup_model(model_name, remove_zip)
        
        return {
            "success": True,
            "model_name": model_name,
            "cleanup_result": result
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Frame generation endpoints
@app.post("/generate/frame")
async def generate_frame(request: FrameRequest):
    """Generate a single frame - with automatic model loading"""
    
    try:
        # Check cache first
        cached_frame = await multi_cache_manager.get_frame(request.model_name, request.frame_id)
        if cached_frame is not None:
            return {
                "success": True,
                "model_name": request.model_name,
                "frame_id": request.frame_id,
                "frame": encode_frame_to_base64(cached_frame),
                "from_cache": True,
                "processing_time_ms": 0,
                "auto_loaded": False
            }
        
        # Handle audio override
        temp_audio_path = None
        if request.audio_override:
            temp_audio_path = decode_audio_from_base64(request.audio_override)
        
        # Generate frame (this will auto-load the model if needed)
        start_time = time.time()
        frame, metadata = await multi_model_engine.generate_frame(
            request.model_name, 
            request.frame_id,
            audio_override=temp_audio_path
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Cache the result
        await multi_cache_manager.set_frame(request.model_name, request.frame_id, frame)
        
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        
        return {
            "success": True,
            "model_name": request.model_name,
            "frame_id": request.frame_id,
            "frame": encode_frame_to_base64(frame),
            "from_cache": False,
            "processing_time_ms": int(processing_time),
            "metadata": metadata,
            "auto_loaded": metadata.get("auto_loaded", False)
        }
        
    except Exception as e:
        # Clean up temp file on error
        if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
            
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "model_name": request.model_name,
                "frame_id": request.frame_id,
                "error": str(e)
            }
        )

@app.get("/generate/frame/fast/{model_name}/{frame_id}")
async def generate_frame_fast(model_name: str, frame_id: int):
    """Ultra-fast frame generation - returns raw PNG data without JSON overhead"""
    
    try:
        start_time = time.time()
        
        # Generate frame using the optimized engine
        frame, metadata = await multi_model_engine.generate_frame(model_name, frame_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Encode directly to PNG bytes (no base64)
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # Fast compression
        _, img_bytes = cv2.imencode('.png', frame, encode_param)
        
        # Return binary response with timing in headers
        return Response(
            content=img_bytes.tobytes(),
            media_type="image/png",
            headers={
                "X-Processing-Time-Ms": str(int(processing_time)),
                "X-Model-Name": model_name,
                "X-Frame-Id": str(frame_id),
                "X-Auto-Loaded": str(metadata.get("auto_loaded", False))
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/generate/frame/jpeg/{model_name}/{frame_id}")
async def generate_frame_jpeg(model_name: str, frame_id: int):
    """Ultra-fast frame generation with JPEG compression for smaller size"""
    
    try:
        start_time = time.time()
        
        # Generate frame using the optimized engine
        frame, metadata = await multi_model_engine.generate_frame(model_name, frame_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Encode to JPEG with high quality but much smaller size
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
        _, img_bytes = cv2.imencode('.jpg', frame, encode_param)
        
        # Return binary response with timing in headers
        return Response(
            content=img_bytes.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Processing-Time-Ms": str(int(processing_time)),
                "X-Model-Name": model_name,
                "X-Frame-Id": str(frame_id),
                "X-Auto-Loaded": str(metadata.get("auto_loaded", False)),
                "X-Size-Bytes": str(len(img_bytes))
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/generate/inference/{model_name}/{frame_id}")
async def generate_inference_only(model_name: str, frame_id: int):
    """Return only the inference result + bounds for client-side compositing"""
    
    try:
        start_time = time.time()
        
        # Generate only inference result (no full frame compositing)
        prediction, bounds, metadata = await multi_model_engine.generate_inference_only(model_name, frame_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Encode prediction to JPEG (much smaller than PNG)
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
        _, prediction_bytes = cv2.imencode('.jpg', prediction, encode_param)
        
        # Serialize bounds as compact binary data
        bounds_bytes = bounds.astype(np.float32).tobytes()
        
        # Create response with prediction image + bounds in headers
        return Response(
            content=prediction_bytes.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Processing-Time-Ms": str(int(processing_time)),
                "X-Model-Name": model_name,
                "X-Frame-Id": str(frame_id),
                "X-Auto-Loaded": str(metadata.get("auto_loaded", False)),
                "X-Prediction-Size": str(len(prediction_bytes)),
                "X-Bounds-Data": base64.b64encode(bounds_bytes).decode('ascii'),
                "X-Bounds-Shape": f"{bounds.shape[0]}",  # Length of bounds array
                "X-Prediction-Shape": f"{prediction.shape[0]}x{prediction.shape[1]}x{prediction.shape[2]}"
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/generate/batch")
async def generate_multi_audio_batch(request: MultiBatchFrameRequest):
    """Generate multiple frames with different audio for each frame"""
    
    try:
        start_time = time.time()
        results = {}
        processed_count = 0
        cached_count = 0
        auto_loaded_models = set()
        temp_audio_paths = []
        
        # Process each frame request
        for frame_req in request.requests:
            try:
                # Check cache first
                cached_frame = await multi_cache_manager.get_frame(frame_req.model_name, frame_req.frame_id)
                if cached_frame is not None:
                    results[f"{frame_req.model_name}_{frame_req.frame_id}"] = {
                        "success": True,
                        "model_name": frame_req.model_name,
                        "frame_id": frame_req.frame_id,
                        "frame": encode_frame_to_base64(cached_frame),
                        "from_cache": True,
                        "auto_loaded": False
                    }
                    cached_count += 1
                    continue
                
                # Handle audio override
                temp_audio_path = None
                if frame_req.audio_override:
                    temp_audio_path = decode_audio_from_base64(frame_req.audio_override)
                    temp_audio_paths.append(temp_audio_path)
                
                # Generate frame (auto-loads model if needed)
                frame, metadata = await multi_model_engine.generate_frame(
                    frame_req.model_name,
                    frame_req.frame_id,
                    audio_override=temp_audio_path
                )
                
                # Track auto-loaded models
                if metadata.get("auto_loaded"):
                    auto_loaded_models.add(frame_req.model_name)
                
                # Cache the result
                await multi_cache_manager.set_frame(frame_req.model_name, frame_req.frame_id, frame)
                
                results[f"{frame_req.model_name}_{frame_req.frame_id}"] = {
                    "success": True,
                    "model_name": frame_req.model_name,
                    "frame_id": frame_req.frame_id,
                    "frame": encode_frame_to_base64(frame),
                    "from_cache": False,
                    "auto_loaded": metadata.get("auto_loaded", False)
                }
                processed_count += 1
                
            except Exception as frame_error:
                # Handle individual frame errors without failing the whole batch
                results[f"{frame_req.model_name}_{frame_req.frame_id}"] = {
                    "success": False,
                    "model_name": frame_req.model_name,
                    "frame_id": frame_req.frame_id,
                    "error": str(frame_error)
                }
        
        processing_time = (time.time() - start_time) * 1000
        
        # Clean up temp audio files
        for temp_path in temp_audio_paths:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return {
            "success": True,
            "results": results,
            "total_requests": len(request.requests),
            "processed_count": processed_count,
            "cached_count": cached_count,
            "failed_count": len(request.requests) - processed_count - cached_count,
            "auto_loaded_models": list(auto_loaded_models),
            "processing_time_ms": int(processing_time)
        }
        
    except Exception as e:
        # Clean up temp audio files on error
        if 'temp_audio_paths' in locals():
            for temp_path in temp_audio_paths:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

# Cache management endpoints
@app.post("/cache/preload")
async def preload_frames(request: PreloadRequest):
    """Preload frames into cache for faster access"""
    
    try:
        # Check model is loaded
        if not multi_model_engine.is_model_loaded(request.model_name):
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{request.model_name}' not loaded"
            )
        
        frame_range = range(request.start_frame, request.end_frame + 1)
        result = await multi_cache_manager.preload_frames(request.model_name, frame_range, multi_model_engine)
        
        return {
            "success": True,
            "preload_result": result
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/cache/{model_name}")
async def clear_model_cache(model_name: str):
    """Clear cache for specific model"""
    
    try:
        cleared_count = await multi_cache_manager.clear_model_cache(model_name)
        
        return {
            "success": True,
            "model_name": model_name,
            "cleared_items": cleared_count,
            "message": f"Cache cleared for model {model_name}"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.delete("/cache")
async def clear_all_cache():
    """Clear all cache"""
    
    try:
        result = await multi_cache_manager.clear_all_cache()
        
        return {
            "success": True,
            "clear_result": result,
            "message": "All cache cleared"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Status and monitoring endpoints
@app.get("/status")
async def get_status():
    """Get service status"""
    
    try:
        redis_connected = multi_cache_manager.ping()
        loaded_models = multi_model_engine.list_models()
        
        return {
            "service": "SyncTalk2D Multi-Model Fast Inference",
            "version": "2.0.0",
            "status": "running",
            "redis_connected": redis_connected,
            "loaded_models": loaded_models,
            "total_models": len(loaded_models),
            "device": str(multi_model_engine.device) if hasattr(multi_model_engine, 'device') else "unknown"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@app.get("/stats")
async def get_stats():
    """Get comprehensive statistics"""
    
    try:
        # Get engine stats
        engine_stats = multi_model_engine.get_stats()
        
        # Get cache stats
        cache_stats = multi_cache_manager.get_cache_stats()
        
        return {
            "success": True,
            "timestamp": time.time(),
            "engine_stats": engine_stats,
            "cache_stats": cache_stats
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# WebSocket endpoint for real-time streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time frame generation"""
    
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            try:
                # Parse request
                model_name = request_data["model_name"]
                frame_id = request_data["frame_id"]
                audio_override = request_data.get("audio_override")
                
                # Check model is loaded
                if not multi_model_engine.is_model_loaded(model_name):
                    await websocket.send_text(json.dumps({
                        "success": False,
                        "error": f"Model '{model_name}' not loaded"
                    }))
                    continue
                
                # Check cache first
                cached_frame = await multi_cache_manager.get_frame(model_name, frame_id)
                
                if cached_frame is not None:
                    # Send cached frame
                    response = {
                        "success": True,
                        "model_name": model_name,
                        "frame_id": frame_id,
                        "frame": encode_frame_to_base64(cached_frame),
                        "from_cache": True,
                        "processing_time_ms": 0
                    }
                    await websocket.send_text(json.dumps(response))
                else:
                    # Generate frame
                    start_time = time.time()
                    
                    # Handle audio override
                    temp_audio_path = None
                    if audio_override:
                        temp_audio_path = decode_audio_from_base64(audio_override)
                    
                    frame, metadata = await multi_model_engine.generate_frame(
                        model_name, 
                        frame_id,
                        audio_override=temp_audio_path
                    )
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Cache the result
                    await multi_cache_manager.set_frame(model_name, frame_id, frame)
                    
                    # Clean up temp file
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    
                    # Send response
                    response = {
                        "success": True,
                        "model_name": model_name,
                        "frame_id": frame_id,
                        "frame": encode_frame_to_base64(frame),
                        "from_cache": False,
                        "processing_time_ms": int(processing_time),
                        "metadata": metadata
                    }
                    await websocket.send_text(json.dumps(response))
                    
            except Exception as e:
                # Send error response
                error_response = {
                    "success": False,
                    "error": str(e)
                }
                await websocket.send_text(json.dumps(error_response))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SyncTalk2D Multi-Model Fast Inference Service",
        "version": "2.0.0",
        "description": "Real-time frame-by-frame lip-sync generation with multi-model support",
        "endpoints": {
            "models": "/models (GET, POST, DELETE)",
            "generate_frame": "/generate/frame (POST)",
            "generate_batch": "/generate/batch (POST)",
            "cache": "/cache (DELETE)",
            "status": "/status (GET)",
            "stats": "/stats (GET)",
            "websocket": "/ws",
            "health": "/health (GET)",
            "docs": "/docs"
        },
        "features": [
            "Multi-model support",
            "Audio override per request",
            "Redis caching with model isolation", 
            "WebSocket streaming",
            "Batch processing",
            "Background preloading",
            "Comprehensive statistics"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
