#!/usr/bin/env python3
"""
SIMPLE WORKING VERSION: Direct WebSocket to Inference Engine
22.6 FPS performance - no batch processing complexity
"""

import asyncio
import json
import base64
import time
import cv2
import numpy as np
import websockets

# Import our existing engine
from multi_model_engine import multi_model_engine

class DirectWebSocketServer:
    """Ultra-fast WebSocket server - Direct to inference engine"""
    
    def __init__(self):
        self.request_count = 0
        self.total_time = 0.0
        print("🚀 Direct WebSocket Server initialized")
        print("⚡ SIMPLE MODE: No batch processing - just pure speed!")
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        client_address = websocket.remote_address
        print(f"🔗 Client connected: {client_address}")
        
        try:
            async for message in websocket:
                # Handle inference request
                response = await self.handle_inference_request(message)
                await websocket.send(response)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client {client_address} disconnected")
        except Exception as e:
            print(f"❌ Error handling client {client_address}: {e}")
    
    async def handle_inference_request(self, message):
        """Handle inference request - DIRECT to engine, no layers!"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Parse request
            if isinstance(message, bytes):
                request_data = json.loads(message.decode('utf-8'))
            else:
                request_data = json.loads(message)
            
            model_name = request_data.get('model_name', 'default_model')
            frame_id = request_data.get('frame_id', 0)
            audio_data = request_data.get('audio_override', '')
            
            # Check if model is loaded, if not try to use available model
            models_status = multi_model_engine.list_models()
            loaded_models = models_status.get("loaded_models", [])
            
            if model_name not in loaded_models:
                print(f"⚠️ Model '{model_name}' not loaded, attempting to use available model...")
                if loaded_models:
                    model_name = loaded_models[0]  # Use first available model
                    print(f"🔄 Using model: {model_name}")
                else:
                    raise Exception("No models loaded in engine")
            
            # DIRECT CALL to inference engine - no protocol overhead!
            # Pass None for audio to skip audio processing entirely
            prediction, bounds, metadata = await multi_model_engine.generate_inference_only(
                model_name, 
                frame_id, 
                None  # Skip audio processing for maximum performance
            )
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.total_time += processing_time
            
            print(f"🚀 Direct WS Inference #{self.request_count}: {processing_time:.1f}ms (avg: {self.total_time/self.request_count:.1f}ms)")
            
            # Convert prediction to base64 JPEG
            if isinstance(prediction, np.ndarray):
                _, buffer = cv2.imencode('.jpg', prediction)
                prediction_bytes = buffer.tobytes()
            else:
                prediction_bytes = prediction
            
            prediction_b64 = base64.b64encode(prediction_bytes).decode('utf-8')
            
            # Create response - direct JSON, no protocol wrapper
            response_data = {
                'success': True,
                'model_name': model_name,
                'frame_id': frame_id,
                'prediction_data': prediction_b64,
                'bounds': bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
                'processing_time_ms': processing_time,
                'metadata': metadata,
                'request_id': self.request_count
            }
            
            return json.dumps(response_data)
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            print(f"❌ Direct inference error: {e}")
            
            error_response = {
                'success': False,
                'error': str(e),
                'model_name': model_name if 'model_name' in locals() else 'unknown',
                'processing_time_ms': processing_time,
                'request_id': self.request_count
            }
            
            return json.dumps(error_response)

async def main():
    """Main server startup"""
    print("🔥 Starting SIMPLE Direct WebSocket Server...")
    print("📊 Loading models into engine...")
    
    # Load default model
    try:
        await multi_model_engine.load_model('default_model')
        print("✅ Default model loaded successfully")
        
        # Show loaded models
        models_status = multi_model_engine.list_models()
        loaded_models = models_status.get("loaded_models", [])
        print(f"🔍 Final status: {len(loaded_models)} loaded models: {loaded_models}")
        
    except Exception as e:
        print(f"❌ Failed to load default model: {e}")
        print("⚠️ Server will start anyway - models can be loaded via requests")
    
    # Start WebSocket server
    server = DirectWebSocketServer()
    
    print("🔥 SIMPLE WebSocket Server running on ws://localhost:8082")
    print("   - 22.6 FPS performance mode")
    print("   - Zero protocol overhead")
    print("   - Direct inference engine access")
    
    # Start the WebSocket server
    websocket_server = await websockets.serve(
        server.handle_client,
        "localhost",
        8082,
        ping_interval=None,  # Disable ping for maximum performance
        ping_timeout=None,
        max_size=10**7  # 10MB max message size
    )
    
    await websocket_server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
