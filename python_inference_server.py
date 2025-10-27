#!/usr/bin/env python3
"""
Python Inference Server using ONNX Runtime

This server handles:
1. Audio encoder inference (mel-spectrogram → audio features)
2. UNet model inference (frames + audio → mouth regions)

The Go inference server sends processed audio/frames here for GPU inference.
"""

import grpc
from concurrent import futures
import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import generated protobuf code
from go_inference_server.proto import inference_pb2
from go_inference_server.proto import inference_pb2_grpc


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.audio_encoder_session = None
        self.unet_sessions = {}  # model_id -> session
        self.load_audio_encoder()
    
    def load_audio_encoder(self):
        """Load the audio encoder ONNX model"""
        possible_paths = [
            "../audio_encoder.onnx",
            "../../audio_encoder.onnx",
            "audio_encoder.onnx",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading audio encoder from: {path}")
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.audio_encoder_session = ort.InferenceSession(path, providers=providers)
                print(f"✅ Audio encoder loaded on: {self.audio_encoder_session.get_providers()}")
                return
        
        raise FileNotFoundError(f"audio_encoder.onnx not found in: {possible_paths}")
    
    def encode_audio_features(self, mel_window):
        """
        Encode mel-spectrogram window to audio features
        
        Args:
            mel_window: numpy array of shape [80, 16]
        
        Returns:
            features: numpy array of shape [512]
        """
        # Reshape to [1, 1, 80, 16] for ONNX
        input_data = mel_window.reshape(1, 1, 80, 16).astype(np.float32)
        
        # Run inference
        input_name = self.audio_encoder_session.get_inputs()[0].name
        output_name = self.audio_encoder_session.get_outputs()[0].name
        
        features = self.audio_encoder_session.run([output_name], {input_name: input_data})[0]
        
        # Return flattened [512] array
        return features.reshape(512)
    
    def InferBatch(self, request, context):
        """
        Handle batch inference request
        
        The Go server sends:
        - Preprocessed visual frames (already decoded)
        - Raw audio OR pre-computed mel-spectrogram
        
        We return:
        - Raw mouth region outputs
        """
        try:
            print(f"\n🔥 Received InferBatch request:")
            print(f"   Model: {request.model_id}")
            print(f"   Batch size: {request.batch_size}")
            print(f"   Has raw_audio: {len(request.raw_audio) > 0}")
            print(f"   Has audio_features: {len(request.audio_features) > 0}")
            
            # For now, return a stub response
            # TODO: Implement actual UNet inference
            
            response = inference_pb2.InferBatchResponse()
            response.inference_time_ms = 10.5
            response.audio_processing_ms = 2.3
            
            # Create dummy outputs for each batch item
            for i in range(request.batch_size):
                mouth_region = inference_pb2.RawMouthRegion()
                mouth_region.width = 320
                mouth_region.height = 320
                mouth_region.channels = 3
                
                # TODO: Replace with actual inference output
                mouth_region.data = np.zeros((320, 320, 3), dtype=np.float32).tobytes()
                
                response.outputs.append(mouth_region)
            
            print(f"✅ Returning {len(response.outputs)} outputs")
            return response
            
        except Exception as e:
            print(f"❌ Error in InferBatch: {e}")
            import traceback
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.InferBatchResponse()
    
    def ListModels(self, request, context):
        """List available models"""
        response = inference_pb2.ListModelsResponse()
        
        # TODO: Scan for available model packages
        model_info = inference_pb2.ModelInfo()
        model_info.model_id = "sanders"
        model_info.loaded = False
        model_info.model_path = "model/sanders"
        
        response.models.append(model_info)
        
        return response
    
    def LoadModel(self, request, context):
        """Load a UNet model into memory"""
        print(f"📦 Loading model: {request.model_id}")
        
        response = inference_pb2.LoadModelResponse()
        response.success = False
        response.message = "Model loading not yet implemented"
        
        return response
    
    def UnloadModel(self, request, context):
        """Unload a model from memory"""
        response = inference_pb2.UnloadModelResponse()
        response.success = True
        return response
    
    def GetModelStats(self, request, context):
        """Get model statistics"""
        response = inference_pb2.GetModelStatsResponse()
        return response
    
    def Health(self, request, context):
        """Health check"""
        response = inference_pb2.HealthResponse()
        response.status = "healthy"
        response.gpu_available = True
        
        # Add GPU info
        gpu_info = inference_pb2.GpuInfo()
        gpu_info.device_id = 0
        gpu_info.name = "CUDA GPU"
        gpu_info.total_memory_bytes = 0  # TODO: Get actual GPU memory
        gpu_info.used_memory_bytes = 0
        gpu_info.loaded_models = len(self.unet_sessions)
        
        response.gpus.append(gpu_info)
        
        return response


def serve():
    """Start the Python inference server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    
    servicer = InferenceServicer()
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    
    port = "50052"  # Different from Go server (50051)
    server.add_insecure_port(f'[::]:{port}')
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  🐍 Python Inference Server (ONNX Runtime)               ║
╠══════════════════════════════════════════════════════════╣
║  Port: {port}                                            ║
║  Audio Encoder: Loaded                                    ║
║  UNet Models: 0 loaded                                    ║
╚══════════════════════════════════════════════════════════╝
""")
    
    server.start()
    print("✅ Server started. Press Ctrl+C to stop.")
    server.wait_for_termination()


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
