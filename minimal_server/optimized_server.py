#!/usr/bin/env python3
"""
🚀 ULTRA-OPTIMIZED Binary WebSocket Server
Uses pre-processed model packages with maximum performance:
- Pre-loaded videos in RAM
- Memory-mapped audio features
- Cached metadata
- Zero I/O overhead during inference
"""

import asyncio
import json
import base64
import time
import websockets
import cv2
import numpy as np
import struct

from optimized_inference_engine import optimized_engine


class OptimizedBinaryWebSocketServer:
    def __init__(self):
        self.request_count = 0
        self.total_time = 0.0
        self.connected_clients = set()
        self.binary_request_count = 0
        self.json_request_count = 0
        
        print("🚀 Optimized Binary WebSocket Server initialized")
        
    async def handle_client(self, websocket):
        """Handle WebSocket client connections"""
        client_addr = websocket.remote_address
        self.connected_clients.add(websocket)
        print(f"🔗 Client connected: {client_addr} (Total: {len(self.connected_clients)})")
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary message - high performance frame request
                    response = await self.handle_binary_request(message)
                    await websocket.send(response)
                else:
                    # Text/JSON message - control messages and fallback
                    response = await self.handle_json_request(message)
                    await websocket.send(response)
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client disconnected: {client_addr}")
        except Exception as e:
            print(f"❌ WebSocket error for {client_addr}: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def handle_binary_request(self, binary_data):
        """Handle binary inference request"""
        start_time = time.time()
        self.request_count += 1
        self.binary_request_count += 1
        
        try:
            # Parse binary protocol:
            # [4 bytes: model_name_length][model_name][4 bytes: frame_id][4 bytes: audio_length][audio_data]
            
            offset = 0
            if len(binary_data) < 12:
                raise ValueError("Binary message too short")
            
            # Read model name
            model_name_len = struct.unpack('<I', binary_data[offset:offset+4])[0]
            offset += 4
            
            if offset + model_name_len > len(binary_data):
                raise ValueError("Invalid model name length")
                
            model_name = binary_data[offset:offset+model_name_len].decode('utf-8')
            offset += model_name_len
            
            # Read frame ID
            if offset + 4 > len(binary_data):
                raise ValueError("Missing frame ID")
            frame_id = struct.unpack('<I', binary_data[offset:offset+4])[0]
            offset += 4
            
            # Read audio data length (we don't use it for pre-processed packages, but keep protocol)
            if offset + 4 > len(binary_data):
                raise ValueError("Missing audio length")
            audio_length = struct.unpack('<I', binary_data[offset:offset+4])[0]
            offset += 4
            
            print(f"🚀 Binary request: model={model_name}, frame={frame_id}")
            
            # Check if model is loaded
            models_status = optimized_engine.list_models()
            loaded_models = models_status.get("loaded_models", [])
            
            if model_name not in loaded_models:
                if loaded_models:
                    model_name = loaded_models[0]
                    print(f"🔄 Using available model: {model_name}")
                else:
                    raise Exception("No models loaded in engine")
            
            # 🚀 OPTIMIZED: All data pre-loaded, instant access!
            prediction, bounds, metadata = await optimized_engine.generate_inference_only(
                model_name, 
                frame_id
            )
            
            processing_time = (time.time() - start_time) * 1000
            self.total_time += processing_time
            
            print(f"⚡ OPTIMIZED inference #{self.binary_request_count}: {processing_time:.1f}ms")
            
            # Encode prediction to JPEG
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
            _, image_buffer = cv2.imencode('.jpg', prediction, encode_param)
            image_bytes = image_buffer.tobytes()
            
            # Create binary response
            response = bytearray()
            
            # Success flag (1 byte)
            response.extend(struct.pack('<B', 1))
            
            # Frame ID (4 bytes)
            response.extend(struct.pack('<I', frame_id))
            
            # Processing time (4 bytes)
            response.extend(struct.pack('<I', int(processing_time)))
            
            # Image data length and data
            response.extend(struct.pack('<I', len(image_bytes)))
            response.extend(image_bytes)
            
            # Bounds data (as float32 array)
            bounds_array = np.array(bounds, dtype=np.float32)
            bounds_bytes = bounds_array.tobytes()
            response.extend(struct.pack('<I', len(bounds_bytes)))
            response.extend(bounds_bytes)
            
            print(f"📤 Binary response: {len(response)} bytes (image: {len(image_bytes)} bytes)")
            
            return bytes(response)
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            print(f"❌ Binary request error: {e}")
            
            # Create binary error response
            error_msg = str(e).encode('utf-8')
            
            response = bytearray()
            response.extend(struct.pack('<B', 0))  # success = 0
            response.extend(struct.pack('<I', 0))  # frame_id = 0
            response.extend(struct.pack('<I', int(processing_time)))
            response.extend(struct.pack('<I', len(error_msg)))
            response.extend(error_msg)
            
            return bytes(response)
    
    async def handle_json_request(self, message):
        """Handle JSON request (control messages and fallback)"""
        start_time = time.time()
        self.request_count += 1
        self.json_request_count += 1
        
        try:
            request_data = json.loads(message)
            
            # Handle control messages
            if request_data.get('type') == 'get_stats':
                return await self.handle_stats_request()
            elif request_data.get('type') == 'get_models':
                models = optimized_engine.list_models()
                return json.dumps({
                    'type': 'models',
                    'models': models
                })
            elif request_data.get('type') == 'get_model_stats':
                model_name = request_data.get('model_name')
                if model_name:
                    stats = optimized_engine.get_model_stats(model_name)
                    return json.dumps({
                        'type': 'model_stats',
                        'stats': stats
                    })
            
            # Handle regular inference request (fallback)
            model_name = request_data.get('model_name', 'sanders')
            frame_id = request_data.get('frame_id', 0)
            
            # Check if model is loaded
            models_status = optimized_engine.list_models()
            loaded_models = models_status.get("loaded_models", [])
            
            if model_name not in loaded_models:
                if loaded_models:
                    model_name = loaded_models[0]
                else:
                    raise Exception("No models loaded in engine")
            
            # Generate frame
            prediction, bounds, metadata = await optimized_engine.generate_inference_only(
                model_name, 
                frame_id
            )
            
            processing_time = (time.time() - start_time) * 1000
            self.total_time += processing_time
            
            print(f"🐌 JSON inference #{self.json_request_count}: {processing_time:.1f}ms")
            
            # Convert prediction to base64 JPEG
            _, buffer = cv2.imencode('.jpg', prediction)
            prediction_bytes = buffer.tobytes()
            prediction_b64 = base64.b64encode(prediction_bytes).decode('utf-8')
            
            # Create JSON response
            response_data = {
                'success': True,
                'model_name': model_name,
                'frame_id': frame_id,
                'prediction_data': prediction_b64,
                'bounds': bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
                'processing_time_ms': processing_time,
                'metadata': metadata,
                'request_id': self.request_count,
                'protocol': 'json_optimized'
            }
            
            return json.dumps(response_data)
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            print(f"❌ JSON request error: {e}")
            
            error_response = {
                'success': False,
                'error': str(e),
                'processing_time_ms': processing_time,
                'request_id': self.request_count
            }
            
            return json.dumps(error_response)
    
    async def handle_stats_request(self):
        """Handle stats request"""
        stats = {
            'type': 'stats',
            'total_requests': self.request_count,
            'binary_requests': self.binary_request_count,
            'json_requests': self.json_request_count,
            'binary_percentage': (self.binary_request_count / max(1, self.request_count)) * 100,
            'average_time_ms': self.total_time / max(1, self.request_count),
            'connected_clients': len(self.connected_clients)
        }
        return json.dumps(stats)
    
    async def broadcast_stats(self):
        """Broadcast performance stats to all clients"""
        while True:
            await asyncio.sleep(5)
            
            if self.connected_clients and self.request_count > 0:
                stats = {
                    'type': 'stats',
                    'total_requests': self.request_count,
                    'binary_requests': self.binary_request_count,
                    'json_requests': self.json_request_count,
                    'binary_percentage': (self.binary_request_count / self.request_count) * 100,
                    'average_time_ms': self.total_time / self.request_count,
                    'connected_clients': len(self.connected_clients)
                }
                
                disconnected = []
                for client in self.connected_clients:
                    try:
                        await client.send(json.dumps(stats))
                    except:
                        disconnected.append(client)
                
                for client in disconnected:
                    self.connected_clients.discard(client)


async def main():
    """Start the optimized binary WebSocket server"""
    print("=" * 80)
    print("🚀 STARTING ULTRA-OPTIMIZED BINARY WEBSOCKET SERVER")
    print("=" * 80)
    print("   ⚡ Pre-loaded videos in RAM")
    print("   ⚡ Memory-mapped audio features")
    print("   ⚡ Cached metadata")
    print("   ⚡ Zero I/O overhead")
    print("=" * 80)
    
    # Auto-load optimized sanders model
    print("\n📦 Loading optimized sanders model...")
    try:
        result = await optimized_engine.load_package(
            "sanders",
            "models/sanders"
        )
        
        if result["status"] == "success":
            print("\n✅ Sanders model loaded successfully!")
            print(f"   Frame count: {result['frame_count']}")
            print(f"   Initialization time: {result['initialization_time_s']:.2f}s")
            print(f"   Device: {result['device']}")
            print(f"   Videos loaded: {', '.join(result['videos_loaded'])}")
            print(f"   Audio features shape: {result['audio_features_shape']}")
            print(f"   Memory-mapped audio: {result['memory_mapped_audio']}")
        else:
            print(f"⚠️ Model load failed: {result.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        print(f"❌ Error loading sanders model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check final model status
    models_status = optimized_engine.list_models()
    loaded_models = models_status.get("loaded_models", [])
    print(f"\n🔍 Loaded models: {loaded_models}")
    
    # Create server instance
    server = OptimizedBinaryWebSocketServer()
    
    # Start WebSocket server
    websocket_server = await websockets.serve(
        server.handle_client,
        "localhost",
        8085,  # Different port from original server
        ping_interval=None,
        ping_timeout=None,
        max_size=10**7,
        compression=None
    )
    
    print("\n" + "=" * 80)
    print("⚡ OPTIMIZED WebSocket Server running on ws://localhost:8085")
    print("=" * 80)
    print("   🚀 Binary protocol: Maximum speed")
    print("   🚀 Pre-loaded data: Zero I/O overhead")
    print("   🚀 Memory-mapped audio: Instant access")
    print("   🚀 Cached metadata: No file reads")
    print("=" * 80)
    print("\n💡 Press Ctrl+C to stop the server")
    
    # Start background stats broadcaster
    stats_task = asyncio.create_task(server.broadcast_stats())
    
    try:
        await asyncio.gather(
            websocket_server.wait_closed(),
            stats_task
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down OPTIMIZED WebSocket server...")
        websocket_server.close()
        await websocket_server.wait_closed()
        stats_task.cancel()
        print("✅ Server stopped gracefully")


if __name__ == '__main__':
    asyncio.run(main())
