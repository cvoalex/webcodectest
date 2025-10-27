#!/usr/bin/env python3
"""
ULTIMATE PERFORMANCE: Direct WebSocket to Inference Engine
No gRPC, no HTTP, no protocol overhead - Maximum speed!
"""

import asyncio
import json
import base64
import time
import websockets
import cv2
import numpy as np

# Import our existing engine directly
from multi_model_engine import multi_model_engine

class DirectWebSocketServer:
    def __init__(self):
        self.request_count = 0
        self.total_time = 0.0
        self.connected_clients = set()
        
    async def handle_client(self, websocket):
        """Handle WebSocket client connections"""
        client_addr = websocket.remote_address
        self.connected_clients.add(websocket)
        print(f"🔗 Client connected: {client_addr} (Total: {len(self.connected_clients)})")
        
        try:
            async for message in websocket:
                # Handle incoming inference requests
                response = await self.handle_inference_request(message)
                await websocket.send(response)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client disconnected: {client_addr}")
        except Exception as e:
            print(f"❌ WebSocket error for {client_addr}: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_inference_request(self, message):
        """Handle inference request - DIRECT to engine, supports BATCH processing!"""
        start_time = time.time()
        
        try:
            # Parse request
            if isinstance(message, bytes):
                request_data = json.loads(message.decode('utf-8'))
            else:
                request_data = json.loads(message)
            
            # Check if this is a batch request
            if 'batch_frames' in request_data:
                return await self.handle_batch_request(request_data, start_time)
            else:
                return await self.handle_single_request(request_data, start_time)
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            print(f"❌ Direct inference error: {e}")
            
            error_response = {
                'success': False,
                'error': str(e),
                'processing_time_ms': processing_time,
                'request_id': self.request_count
            }
            
            return json.dumps(error_response)
    
    async def handle_single_request(self, request_data, start_time):
        """Handle single frame request (original logic)"""
        self.request_count += 1
        
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
    
    async def handle_batch_request(self, request_data, start_time):
        """Handle batch frame request - SMART BATCH PROCESSING!"""
        model_name = request_data.get('model_name', 'default_model')
        batch_frames = request_data.get('batch_frames', [])
        batch_size = len(batch_frames)
        
        print(f"🔥 BATCH REQUEST: {batch_size} frames")
        
        # Check if model is loaded
        models_status = multi_model_engine.list_models()
        loaded_models = models_status.get("loaded_models", [])
        
        if model_name not in loaded_models:
            if loaded_models:
                model_name = loaded_models[0]
                print(f"🔄 Using model: {model_name}")
            else:
                raise Exception("No models loaded in engine")
        
        # Process batch frames efficiently
        batch_results = []
        batch_start = time.time()
        
        for i, frame_request in enumerate(batch_frames):
            frame_id = frame_request.get('frame_id', i)
            
            # DIRECT CALL to inference engine
            prediction, bounds, metadata = await multi_model_engine.generate_inference_only(
                model_name, 
                frame_id, 
                None  # Skip audio processing for maximum performance
            )
            
            # Convert prediction to base64 JPEG
            if isinstance(prediction, np.ndarray):
                _, buffer = cv2.imencode('.jpg', prediction)
                prediction_bytes = buffer.tobytes()
            else:
                prediction_bytes = prediction
            
            prediction_b64 = base64.b64encode(prediction_bytes).decode('utf-8')
            
            frame_result = {
                'frame_id': frame_id,
                'prediction_data': prediction_b64,
                'bounds': bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
                'metadata': metadata
            }
            
            batch_results.append(frame_result)
            self.request_count += 1
        
        batch_time = (time.time() - batch_start) * 1000
        avg_frame_time = batch_time / batch_size
        self.total_time += batch_time
        
        print(f"🚀 BATCH COMPLETE: {batch_size} frames in {batch_time:.1f}ms ({avg_frame_time:.1f}ms/frame) - {batch_size/batch_time*1000:.1f} FPS!")
        
        # Create batch response
        response_data = {
            'success': True,
            'batch': True,
            'model_name': model_name,
            'batch_size': batch_size,
            'frames': batch_results,
            'total_processing_time_ms': batch_time,
            'average_frame_time_ms': avg_frame_time,
            'batch_fps': batch_size / batch_time * 1000,
            'request_id': self.request_count
        }
        
        return json.dumps(response_data)
    
    async def broadcast_stats(self):
        """Broadcast performance stats to all clients"""
        while True:
            await asyncio.sleep(5)  # Every 5 seconds
            
            if self.connected_clients and self.request_count > 0:
                stats = {
                    'type': 'stats',
                    'total_requests': self.request_count,
                    'average_time_ms': self.total_time / self.request_count,
                    'connected_clients': len(self.connected_clients)
                }
                
                # Broadcast to all connected clients
                disconnected = []
                for client in self.connected_clients:
                    try:
                        await client.send(json.dumps(stats))
                    except:
                        disconnected.append(client)
                
                # Remove disconnected clients
                for client in disconnected:
                    self.connected_clients.discard(client)

async def main():
    """Start the ultimate performance WebSocket server"""
    print("🚀 Starting DIRECT WebSocket Inference Server...")
    print("   - No gRPC layer")
    print("   - No HTTP overhead") 
    print("   - Direct engine calls")
    print("   - Maximum performance!")
    
    # Auto-load default model on startup (same as gRPC server)
    print("📦 Auto-loading default model...")
    try:
        result = await multi_model_engine.load_model(
            "default_model",
            "models/default_model",
            None
        )
        
        if result["status"] == "loaded" or result["status"] == "already_loaded":
            print("✅ Default model loaded successfully")
        else:
            print(f"⚠️ Default model load failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error loading default model: {e}")
    
    # Check final model status
    models_status = multi_model_engine.list_models()
    loaded_models = models_status.get("loaded_models", [])
    print(f"🔍 Final status: {len(loaded_models)} loaded models: {loaded_models}")
    
    # Create server instance
    server = DirectWebSocketServer()
    
    # Start WebSocket server
    websocket_server = await websockets.serve(
        server.handle_client,
        "localhost",
        8082,
        ping_interval=None,      # Disable ping for max performance
        ping_timeout=None,
        max_size=10**7,          # 10MB max message size
        compression=None         # No compression for speed
    )
    
    print("🔥 DIRECT WebSocket Server running on ws://localhost:8082")
    print("   - Ultimate performance mode")
    print("   - Zero protocol overhead")
    print("   - Direct inference engine access")
    
    # Start background stats broadcaster
    stats_task = asyncio.create_task(server.broadcast_stats())
    
    try:
        # Keep servers running
        await asyncio.gather(
            websocket_server.wait_closed(),
            stats_task
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down DIRECT WebSocket server...")
        websocket_server.close()
        await websocket_server.wait_closed()
        stats_task.cancel()

if __name__ == '__main__':
    asyncio.run(main())
