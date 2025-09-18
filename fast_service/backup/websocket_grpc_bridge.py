#!/usr/bin/env python3
"""
WebSocket to gRPC Bridge - Maximum Performance
Binary WebSocket connection directly to gRPC service
"""

import asyncio
import json
import base64
import websockets
import grpc
from concurrent.futures import ThreadPoolExecutor

# Import gRPC code
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc

class WebSocketGrpcBridge:
    def __init__(self):
        self.grpc_channel = None
        self.grpc_stub = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def start_grpc_connection(self):
        """Initialize gRPC connection"""
        self.grpc_channel = grpc.insecure_channel('localhost:50051')
        self.grpc_stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(self.grpc_channel)
        print("✅ gRPC connection established")
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        print(f"🔗 WebSocket client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                # Handle binary or text messages
                if isinstance(message, bytes):
                    # Handle binary gRPC-like messages
                    response = await self.handle_binary_message(message)
                    await websocket.send(response)
                else:
                    # Handle JSON messages
                    response = await self.handle_json_message(message)
                    await websocket.send(response)
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 WebSocket client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
    
    async def handle_json_message(self, message):
        """Handle JSON WebSocket messages"""
        try:
            request_data = json.loads(message)
            
            # Create gRPC request
            grpc_request = lipsyncsrv_pb2.InferenceRequest(
                model_name=request_data.get('model_name', 'default_model'),
                frame_id=request_data.get('frame_id', 0),
                audio_override=request_data.get('audio_override', '')
            )
            
            # Make gRPC call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            grpc_response = await loop.run_in_executor(
                self.executor,
                self.grpc_stub.GenerateInference,
                grpc_request
            )
            
            # Create response
            response_data = {
                'success': grpc_response.success,
                'model_name': grpc_response.model_name,
                'frame_id': grpc_request.frame_id,
                'prediction_data': base64.b64encode(grpc_response.prediction_data).decode('utf-8'),
                'bounds': list(grpc_response.bounds),
                'processing_time_ms': grpc_response.processing_time_ms,
                'error': grpc_response.error if hasattr(grpc_response, 'error') else ''
            }
            
            return json.dumps(response_data)
            
        except Exception as e:
            error_response = {'success': False, 'error': str(e)}
            return json.dumps(error_response)
    
    async def handle_binary_message(self, message):
        """Handle binary WebSocket messages (future optimization)"""
        # For now, treat as JSON
        try:
            json_message = message.decode('utf-8')
            return await self.handle_json_message(json_message)
        except:
            return json.dumps({'success': False, 'error': 'Invalid message format'})
    
    def close_grpc_connection(self):
        """Close gRPC connection"""
        if self.grpc_channel:
            self.grpc_channel.close()

async def main():
    """Start WebSocket to gRPC bridge"""
    print("🚀 Starting WebSocket to gRPC Bridge...")
    
    bridge = WebSocketGrpcBridge()
    await bridge.start_grpc_connection()
    
    # Start WebSocket server
    server = await websockets.serve(
        bridge.handle_websocket,
        "localhost",
        8081,
        ping_interval=None,  # Disable ping for maximum performance
        ping_timeout=None,
        max_size=10**7,     # 10MB max message size for images
        compression=None    # No compression for speed
    )
    
    print("🔥 WebSocket Bridge running on ws://localhost:8081")
    print("   - Binary WebSocket connection")
    print("   - Direct gRPC calls")
    print("   - No HTTP overhead")
    print("   - Persistent connection")
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down WebSocket bridge...")
        bridge.close_grpc_connection()

if __name__ == '__main__':
    asyncio.run(main())
