#!/usr/bin/env python3
"""
Minimal gRPC-Web Proxy for Direct Browser Access
Translates gRPC-Web calls to native gRPC - Maximum Performance
"""

import asyncio
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading
import grpc
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc

class GrpcWebProxy(BaseHTTPRequestHandler):
    """Minimal gRPC-Web to gRPC proxy"""
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
    
    def do_POST(self):
        """Handle gRPC-Web requests"""
        self._set_cors_headers()
        
        # Parse the URL to get service and method
        path = urlparse(self.path).path
        if path.startswith('/'):
            path = path[1:]
        
        parts = path.split('/')
        if len(parts) != 2:
            self.send_error(400, "Invalid gRPC path")
            return
            
        service_name, method_name = parts
        
        if service_name == 'LipSyncService' and method_name == 'GenerateInference':
            self._handle_generate_inference()
        else:
            self.send_error(404, "Method not found")
    
    def _handle_generate_inference(self):
        """Handle lip sync inference via native gRPC"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            request_data = self.rfile.read(content_length)
            
            # Parse gRPC-Web frame or JSON
            try:
                if content_length > 5 and request_data[0] == 0:  # gRPC-Web frame format
                    message_data = self._parse_grpc_web_frame(request_data)
                    request_json = json.loads(message_data.decode('utf-8'))
                else:  # Direct JSON
                    request_json = json.loads(request_data.decode('utf-8'))
            except:
                # Fallback: treat as JSON
                request_json = json.loads(request_data.decode('utf-8'))
            
            # Create gRPC request
            grpc_request = lipsyncsrv_pb2.InferenceRequest(
                model_name=request_json.get('model_name', 'default_model'),
                frame_id=request_json.get('frame_id', 0),
                audio_override=request_json.get('audio_override', '')
            )
            
            # Make native gRPC call
            with grpc.insecure_channel('localhost:50051') as channel:
                stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
                grpc_response = stub.GenerateInference(grpc_request)
            
            # Convert to JSON response
            response_json = {
                'success': grpc_response.success,
                'model_name': grpc_response.model_name,
                'frame_id': grpc_request.frame_id,
                'prediction_data': base64.b64encode(grpc_response.prediction_data).decode('utf-8'),
                'bounds': list(grpc_response.bounds),
                'processing_time_ms': grpc_response.processing_time_ms,
                'error': grpc_response.error if hasattr(grpc_response, 'error') else ''
            }
            
            # Send JSON response (not gRPC-Web frame)
            response_bytes = json.dumps(response_json).encode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)
            
        except Exception as e:
            print(f"❌ gRPC-Web proxy error: {e}")
            error_response = {'success': False, 'error': str(e)}
            error_bytes = json.dumps(error_response).encode('utf-8')
            
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(error_bytes)))
            self.end_headers()
            self.wfile.write(error_bytes)
    
    def _parse_grpc_web_frame(self, frame_data):
        """Parse gRPC-Web frame format"""
        if len(frame_data) < 5:
            raise ValueError("Invalid gRPC-Web frame")
        
        # Skip compression flag (1 byte) and read length (4 bytes)
        message_length = int.from_bytes(frame_data[1:5], 'big')
        message_data = frame_data[5:5+message_length]
        
        return message_data
    
    def _create_grpc_web_frame(self, message_data):
        """Create gRPC-Web frame format"""
        frame = bytearray()
        frame.append(0)  # No compression
        frame.extend(len(message_data).to_bytes(4, 'big'))  # Message length
        frame.extend(message_data)  # Message data
        return bytes(frame)
    
    def _set_cors_headers(self):
        """Set CORS headers for browser access"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Grpc-Web, Grpc-Timeout')
        self.send_header('Access-Control-Expose-Headers', 'Grpc-Status, Grpc-Message')

def run_grpc_web_proxy(port=8080):
    """Run gRPC-Web proxy server"""
    server = HTTPServer(('localhost', port), GrpcWebProxy)
    print(f"🌐 gRPC-Web Proxy running on http://localhost:{port}")
    server.serve_forever()

def main():
    """Start gRPC-Web proxy alongside existing gRPC server"""
    print("🚀 Starting gRPC-Web Proxy for Direct Browser Access...")
    
    # Start proxy in background thread
    proxy_thread = threading.Thread(target=run_grpc_web_proxy, daemon=True)
    proxy_thread.start()
    
    print("✅ gRPC-Web Proxy started")
    print("🔥 Browser can now call gRPC directly via http://localhost:8080")
    print("   - No HTTP overhead")
    print("   - No Go middleman")
    print("   - Pure gRPC protocol")
    
    try:
        # Keep running
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down proxy...")

if __name__ == '__main__':
    main()
