#!/usr/bin/env python3
"""
Combined HTTP + gRPC Server for Direct Browser Access
Eliminates Go middleman performance bottleneck
"""

import asyncio
import time
import json
import base64
import threading
from concurrent import futures
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import grpc

# Import existing gRPC code
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc
from multi_model_engine import multi_model_engine

class CORSHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler with CORS support for direct browser access"""
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle frame generation requests"""
        if self.path == '/generate-frame':
            self.handle_frame_generation()
        else:
            self.send_error(404, "Not Found")
    
    def handle_frame_generation(self):
        """Generate frame directly without Go middleman"""
        # Set CORS headers
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        try:
            # Parse request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            model_name = request_data.get('model_name', 'default_model')
            frame_id = request_data.get('frame_id', 0)
            audio_data = request_data.get('audio_data', '')
            
            # Run inference using our optimized engine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_time = time.time()
            try:
                prediction, bounds, metadata = loop.run_until_complete(
                    multi_model_engine.generate_inference_only(
                        model_name, 
                        frame_id, 
                        audio_data if audio_data else None
                    )
                )
            finally:
                loop.close()
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Encode prediction as base64 JPEG
            if isinstance(prediction, bytes):
                prediction_data = base64.b64encode(prediction).decode('utf-8')
            else:
                # Convert numpy array to JPEG bytes
                import cv2
                _, buffer = cv2.imencode('.jpg', prediction)
                prediction_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            # Return response in same format as Go server
            response = {
                'success': True,
                'model_name': model_name,
                'frame_id': frame_id,
                'prediction_data': prediction_data,
                'bounds': bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
                'processing_time_ms': processing_time,
                'metadata': metadata
            }
            
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            error_response = {
                'success': False,
                'error': str(e),
                'model_name': model_name if 'model_name' in locals() else 'unknown'
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

class LipSyncServicer(lipsyncsrv_pb2_grpc.LipSyncServiceServicer):
    """Existing gRPC servicer"""
    
    def __init__(self):
        self.request_count = 0
        self.total_time = 0.0
    
    def GenerateInference(self, request, context):
        """Single frame inference with maximum speed"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Run inference using our optimized engine
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                prediction, bounds, metadata = loop.run_until_complete(
                    multi_model_engine.generate_inference_only(
                        request.model_name, 
                        request.frame_id, 
                        request.audio_override if request.audio_override else None
                    )
                )
            finally:
                loop.close()
            
            processing_time = (time.time() - start_time) * 1000
            self.total_time += processing_time
            
            print(f"⚡ gRPC Inference #{self.request_count}: {processing_time:.1f}ms (avg: {self.total_time/self.request_count:.1f}ms)")
            
            # Convert prediction to bytes if needed
            if isinstance(prediction, np.ndarray):
                import cv2
                _, buffer = cv2.imencode('.jpg', prediction)
                prediction_bytes = buffer.tobytes()
            else:
                prediction_bytes = prediction
            
            response = lipsyncsrv_pb2.InferenceResponse(
                success=True,
                model_name=request.model_name,
                prediction_data=prediction_bytes,
                bounds=bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
                prediction_shape=str(prediction.shape) if hasattr(prediction, 'shape') else "",
                processing_time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            print(f"❌ gRPC Error: {e}")
            return lipsyncsrv_pb2.InferenceResponse(
                success=False,
                error=str(e),
                model_name=request.model_name
            )

def run_http_server(port=8080):
    """Run HTTP server in separate thread"""
    httpd = HTTPServer(('localhost', port), CORSHTTPRequestHandler)
    print(f"🌐 HTTP server running on http://localhost:{port}")
    httpd.serve_forever()

def run_grpc_server(port=50051):
    """Run gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = LipSyncServicer()
    lipsyncsrv_pb2_grpc.add_LipSyncServiceServicer_to_server(servicer, server)
    listen_addr = f'[::]:{port}'
    server.add_insecure_port(listen_addr)
    
    print(f"🚀 gRPC server running on port {port}")
    server.start()
    
    return server

async def main():
    """Main entry point with dual server support"""
    print("🎬 Starting Combined HTTP + gRPC Lip Sync Server...")
    
    # Load default model
    print("📦 Loading default model...")
    await multi_model_engine.load_model("default_model", "/path/to/default/model")
    print("✅ Default model loaded successfully")
    
    # Start gRPC server
    grpc_server = run_grpc_server(50051)
    
    # Start HTTP server in separate thread
    http_thread = threading.Thread(target=run_http_server, args=(8080,), daemon=True)
    http_thread.start()
    
    print("🔥 Both servers running! Use:")
    print("   - HTTP: http://localhost:8080/generate-frame (for browsers)")
    print("   - gRPC: localhost:50051 (for direct gRPC clients)")
    
    try:
        # Keep the servers running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        grpc_server.stop(0)

if __name__ == '__main__':
    asyncio.run(main())
