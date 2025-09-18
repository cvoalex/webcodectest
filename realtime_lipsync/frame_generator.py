#!/usr/bin/env python3
"""
Real-time Lip Sync Frame Generator
Connects to gRPC service for generating lip sync frames from audio buffers.
"""

import asyncio
import grpc
import numpy as np
import time
from typing import List, Optional
import websockets
import json
import base64
import wave
import io

# Import your existing gRPC modules
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc

class RealtimeLipSyncFrameGenerator:
    def __init__(self, grpc_host='localhost', grpc_port=50051, websocket_port=8080):
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.websocket_port = websocket_port
        
        # Audio configuration
        self.BUFFER_SIZE = 3000
        self.CHUNK_DURATION_MS = 40
        self.SAMPLE_RATE = 24000
        self.SAMPLES_PER_CHUNK = int((self.SAMPLE_RATE * self.CHUNK_DURATION_MS) / 1000)  # 960 samples
        
        # Frame generation configuration
        self.FRAMES_FOR_INFERENCE = 16    # 8 prev + 1 current + 7 future
        self.INFERENCE_BATCH_SIZE = 5     # Process 5 frames at once
        self.LOOKAHEAD_FRAMES = 7         # Need 7 future frames
        
        # Frame buffer management for jitter minimization
        self.TARGET_FRAME_BUFFER = 15     # Target 15 frames ahead (600ms)
        self.MIN_FRAME_BUFFER = 8         # Minimum 8 frames (320ms)
        self.MAX_FRAME_BUFFER = 25        # Maximum 25 frames (1000ms) 
        self.CRITICAL_FRAME_BUFFER = 3    # Critical low level (120ms)
        
        # Circular buffers
        self.audio_buffer = [None] * self.BUFFER_SIZE
        self.frame_buffer = [None] * self.BUFFER_SIZE
        
        # Buffer pointers
        self.audio_write_index = 0
        self.frame_write_index = 0
        
        # gRPC connection
        self.grpc_channel = None
        self.grpc_stub = None
        
        # WebSocket server for browser communication
        self.websocket_server = None
        self.connected_clients = set()
        
        # Model selection
        self.current_model = "test_optimized_package_fixed_1"
        
        # Reference image for compositing (stored per model)
        self.reference_images = {}  # model_name -> reference_image_data
        self.reference_bounds = {}  # model_name -> bounds_data
        
    async def initialize(self):
        """Initialize gRPC connection and WebSocket server"""
        # Initialize gRPC connection
        await self.connect_grpc()
        
        # Start WebSocket server for browser communication
        await self.start_websocket_server()
        
        print("🚀 Real-time Lip Sync Frame Generator initialized")
    
    async def connect_grpc(self):
        """Connect to gRPC lip sync service"""
        try:
            self.grpc_channel = grpc.aio.insecure_channel(f'{self.grpc_host}:{self.grpc_port}')
            self.grpc_stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(self.grpc_channel)
            
            # Test connection by checking if channel is ready
            await self.grpc_channel.channel_ready()
            print(f"✅ Connected to gRPC service on {self.grpc_host}:{self.grpc_port}")
                
        except Exception as e:
            print(f"❌ gRPC connection error: {e}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for browser communication"""
        async def handle_client(websocket):
            print(f"🔗 Client connected: {websocket.remote_address}")
            self.connected_clients.add(websocket)
            
            try:
                async for message in websocket:
                    await self.handle_websocket_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                print(f"🔌 Client disconnected: {websocket.remote_address}")
            finally:
                self.connected_clients.discard(websocket)
        
        self.websocket_server = await websockets.serve(
            handle_client, 
            'localhost', 
            self.websocket_port
        )
        
        print(f"🌐 WebSocket server started on ws://localhost:{self.websocket_port}")
    
    async def handle_websocket_message(self, websocket, message):
        """Handle incoming WebSocket messages from browser"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'audio_chunk':
                await self.process_audio_chunk(data)
            elif message_type == 'get_frame':
                await self.send_current_frame(websocket)
            elif message_type == 'set_model':
                await self.change_model(data.get('model_name'))
            elif message_type == 'get_stats':
                await self.send_stats(websocket)
            elif message_type == 'get_reference_image':
                await self.send_reference_image(websocket, data.get('model_name'), data.get('request_id'))
            
        except Exception as e:
            print(f"❌ Error handling WebSocket message: {e}")
    
    async def process_audio_chunk(self, data):
        """Process incoming audio chunk from browser"""
        try:
            # Decode base64 audio
            base64_audio = data.get('audio_data')
            if not base64_audio:
                return
            
            # Decode and convert to numpy array
            audio_bytes = base64.b64decode(base64_audio)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Add to circular buffer
            self.add_audio_to_buffer(audio_data)
            
            # Check if we can generate frames
            if self.can_generate_frames():
                await self.generate_frame_batch()
            
        except Exception as e:
            print(f"❌ Error processing audio chunk: {e}")
    
    def add_audio_to_buffer(self, audio_data: np.ndarray):
        """Add audio data to circular buffer"""
        self.audio_buffer[self.audio_write_index] = {
            'data': audio_data,
            'timestamp': time.time(),
            'index': self.audio_write_index
        }
        
        self.audio_write_index = (self.audio_write_index + 1) % self.BUFFER_SIZE
        
        print(f"🎵 Audio chunk added. Buffer fill: {self.get_audio_buffer_fill()}/3000")
    
    def can_generate_frames(self) -> bool:
        """Check if we have enough audio to generate frames and need more frames"""
        audio_fill = self.get_audio_buffer_fill()
        frame_fill = self.get_frame_buffer_fill()
        
        # Need enough audio for inference AND frame buffer below target
        need_more_frames = frame_fill < self.TARGET_FRAME_BUFFER
        have_enough_audio = audio_fill >= self.FRAMES_FOR_INFERENCE + 5
        
        return need_more_frames and have_enough_audio
    
    async def generate_frame_batch(self):
        """Generate a batch of lip sync frames using gRPC service"""
        try:
            # Get 16 consecutive audio chunks for inference
            audio_chunks = self.get_consecutive_audio_chunks(self.FRAMES_FOR_INFERENCE)
            
            if len(audio_chunks) < self.FRAMES_FOR_INFERENCE:
                return  # Not enough audio
            
            # Combine audio chunks into 640ms audio segment
            combined_audio = self.combine_audio_chunks(audio_chunks)
            
            # Convert to WAV format for gRPC service
            wav_data = self.convert_to_wav(combined_audio)
            
            # Generate frames using gRPC service
            frames = await self.call_grpc_inference(wav_data)
            
            # Store frames in circular buffer with bounds data
            for i, frame_response in enumerate(frames):
                self.frame_buffer[self.frame_write_index] = {
                    'data': frame_response.prediction_data,
                    'bounds': frame_response.bounds,
                    'timestamp': time.time(),
                    'audio_index': audio_chunks[0]['index'] + i,  # Link to source audio
                    'model_name': frame_response.model_name,
                    'prediction_shape': frame_response.prediction_shape
                }
                
                self.frame_write_index = (self.frame_write_index + 1) % self.BUFFER_SIZE
            
            print(f"🎬 Generated {len(frames)} frames. Frame buffer fill: {self.get_frame_buffer_fill()}/3000")
            
            # Notify connected clients about new frames
            await self.notify_clients_new_frames(len(frames))
            
        except Exception as e:
            print(f"❌ Error generating frame batch: {e}")
    
    async def call_grpc_inference(self, wav_data: bytes) -> List:
        """Call gRPC service to generate lip sync frames"""
        try:
            # Create batch request for 5 frames
            frame_ids = list(range(self.INFERENCE_BATCH_SIZE))
            
            request = lipsyncsrv_pb2.BatchInferenceRequest(
                model_name=self.current_model,
                frame_ids=frame_ids,
                audio_data=wav_data  # Assuming your service accepts audio data
            )
            
            start_time = time.time()
            response = await self.grpc_stub.GenerateBatchInference(request)
            inference_time = (time.time() - start_time) * 1000
            
            frames = []
            for frame_response in response.responses:
                if frame_response.success:
                    # Return full response object instead of just frame_data
                    frames.append(frame_response)
                else:
                    print(f"❌ Frame generation failed: {frame_response.error}")
            
            print(f"⚡ Batch inference completed: {len(frames)} frames in {inference_time:.1f}ms")
            return frames
            
        except Exception as e:
            print(f"❌ gRPC inference error: {e}")
            return []
    
    def get_consecutive_audio_chunks(self, count: int) -> List[dict]:
        """Get consecutive audio chunks from buffer"""
        chunks = []
        
        # Start from oldest unprocessed audio
        start_index = (self.audio_write_index - self.get_audio_buffer_fill()) % self.BUFFER_SIZE
        
        for i in range(count):
            index = (start_index + i) % self.BUFFER_SIZE
            if self.audio_buffer[index] is not None:
                chunks.append(self.audio_buffer[index])
            else:
                break
        
        return chunks
    
    def combine_audio_chunks(self, chunks: List[dict]) -> np.ndarray:
        """Combine audio chunks into single array"""
        total_samples = sum(len(chunk['data']) for chunk in chunks)
        combined = np.zeros(total_samples, dtype=np.int16)
        
        offset = 0
        for chunk in chunks:
            data = chunk['data']
            combined[offset:offset + len(data)] = data
            offset += len(data)
        
        return combined
    
    def convert_to_wav(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV bytes"""
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())
        
        return buffer.getvalue()
    
    async def send_current_frame(self, websocket):
        """Send current frame to client"""
        try:
            # Get frame synchronized with current playback position
            frame = self.get_synchronized_frame()
            
            if frame:
                message = {
                    'type': 'frame_data',
                    'frame_data': base64.b64encode(frame['data']).decode('utf-8'),
                    'timestamp': frame['timestamp']
                }
                
                await websocket.send(json.dumps(message))
            
        except Exception as e:
            print(f"❌ Error sending frame: {e}")
    
    def get_synchronized_frame(self) -> Optional[dict]:
        """Get frame synchronized with current playback position"""
        # This would calculate the appropriate frame based on timing
        # For now, return the most recent frame
        
        for i in range(self.BUFFER_SIZE):
            index = (self.frame_write_index - 1 - i) % self.BUFFER_SIZE
            if self.frame_buffer[index] is not None:
                return self.frame_buffer[index]
        
        return None
    
    async def notify_clients_new_frames(self, frame_count: int):
        """Notify all connected clients about new frames"""
        if self.connected_clients:
            message = {
                'type': 'frames_generated',
                'count': frame_count,
                'buffer_fill': self.get_frame_buffer_fill()
            }
            
            # Send to all connected clients
            disconnected = set()
            for client in self.connected_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected
    
    async def send_stats(self, websocket):
        """Send buffer statistics to client"""
        stats = {
            'type': 'stats',
            'audio_buffer_fill': self.get_audio_buffer_fill(),
            'frame_buffer_fill': self.get_frame_buffer_fill(),
            'can_generate': self.can_generate_frames(),
            'current_model': self.current_model
        }
        
        await websocket.send(json.dumps(stats))
    
    def get_audio_buffer_fill(self) -> int:
        """Get current audio buffer fill level"""
        return sum(1 for chunk in self.audio_buffer if chunk is not None)
    
    def get_frame_buffer_fill(self) -> int:
        """Get current frame buffer fill level"""
        return sum(1 for frame in self.frame_buffer if frame is not None)
    
    async def change_model(self, model_name: str):
        """Change the active lip sync model"""
        try:
            # For now, just update the model name without validation
            # since GetModelInfo method doesn't exist in the service
            self.current_model = model_name
            print(f"✅ Switched to model: {model_name}")
                
        except Exception as e:
            print(f"❌ Error changing model: {e}")
    
    async def send_reference_image(self, websocket, model_name: str, request_id: str):
        """Send reference image for the specified model"""
        try:
            # Check if we have a cached reference image for this model
            if model_name in self.reference_images:
                response = {
                    'type': 'reference_image',
                    'request_id': request_id,
                    'model_name': model_name,
                    'image_data': self.reference_images[model_name],
                    'bounds': self.reference_bounds.get(model_name, [])
                }
                await websocket.send(json.dumps(response))
                return
            
            # If no cached reference, try to get one from the service or create from first frame
            # For now, we'll wait for the first frame and use it as reference
            response = {
                'type': 'reference_image',
                'request_id': request_id,
                'model_name': model_name,
                'image_data': None,  # Will be set when first frame is generated
                'bounds': [],
                'message': 'Reference will be available after first frame generation'
            }
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            print(f"❌ Error sending reference image: {e}")
    
    def set_reference_from_frame(self, frame_data: dict):
        """Set reference image from the first generated frame"""
        try:
            model_name = frame_data.get('model_name', self.current_model)
            
            # Only set reference if we don't have one for this model
            if model_name not in self.reference_images:
                # Convert frame data to base64 for storage
                frame_bytes = frame_data['data']
                if isinstance(frame_bytes, bytes):
                    reference_data = base64.b64encode(frame_bytes).decode('utf-8')
                else:
                    reference_data = frame_bytes
                
                self.reference_images[model_name] = reference_data
                self.reference_bounds[model_name] = frame_data.get('bounds', [])
                
                print(f"✅ Set reference image for model: {model_name}")
                
        except Exception as e:
            print(f"❌ Error setting reference image: {e}")
    
    async def send_current_frame(self, websocket):
        """Send current frame with bounds data for compositing"""
        try:
            frame_data = self.get_synchronized_frame()
            
            if frame_data:
                # Set reference from first frame if not already set
                self.set_reference_from_frame(frame_data)
                
                # Convert frame data to base64 if it's raw bytes
                frame_bytes = frame_data['data']
                if isinstance(frame_bytes, bytes):
                    frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
                else:
                    frame_b64 = frame_bytes
                
                message = {
                    'type': 'current_frame',
                    'frame_data': frame_b64,
                    'bounds': frame_data.get('bounds', []),
                    'timestamp': frame_data.get('timestamp'),
                    'model_name': frame_data.get('model_name', self.current_model),
                    'prediction_shape': frame_data.get('prediction_shape', '320x320x3')
                }
                
                await websocket.send(json.dumps(message))
            else:
                # No frame available
                message = {
                    'type': 'current_frame',
                    'frame_data': None,
                    'message': 'No frame available'
                }
                
                await websocket.send(json.dumps(message))
            
        except Exception as e:
            print(f"❌ Error sending frame: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.grpc_channel:
            await self.grpc_channel.close()
        
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        print("🧹 Cleanup completed")

async def main():
    """Main function to start the real-time lip sync frame generator"""
    generator = RealtimeLipSyncFrameGenerator()
    
    try:
        await generator.initialize()
        
        print("🚀 Real-time Lip Sync Frame Generator running...")
        print("📡 Connect your browser to ws://localhost:8080")
        print("🎬 gRPC service on localhost:50051")
        print("Press Ctrl+C to stop")
        
        # Keep server running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        await generator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
