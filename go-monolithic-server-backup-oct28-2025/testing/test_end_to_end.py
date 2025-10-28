#!/usr/bin/env python3
"""
Quick End-to-End Test for Monolithic Lip-Sync Server
Tests audio → mel → encoder → inference → compositing pipeline
"""

import grpc
import numpy as np
import sys
import time
import struct
import cv2
from pathlib import Path

# Add proto to path
sys.path.insert(0, str(Path(__file__).parent / "proto"))
import monolithic_pb2
import monolithic_pb2_grpc


def load_wav_as_float32(filepath):
    """Load WAV file and return float32 samples"""
    import wave
    
    with wave.open(filepath, 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        
        audio_bytes = wav.readframes(n_frames)
        
        # Convert to numpy array
        if sample_width == 2:  # 16-bit
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            # Normalize to [-1, 1]
            audio = audio.astype(np.float32) / 32768.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert to mono if stereo
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        return audio, framerate


def test_end_to_end():
    """Test complete audio → lip-sync video pipeline"""
    
    print("=" * 80)
    print("🧪 END-TO-END LIP-SYNC SERVER TEST")
    print("=" * 80)
    
    # Configuration
    SERVER_ADDR = "localhost:50053"
    MODEL_ID = "sanders"
    AUDIO_FILE = "../aud.wav"
    BATCH_SIZE = 24
    
    # Connect to server
    print(f"\n🔌 Connecting to server at {SERVER_ADDR}...")
    channel = grpc.insecure_channel(
        SERVER_ADDR,
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    stub = monolithic_pb2_grpc.MonolithicServiceStub(channel)
    
    # Check server health
    print("📊 Checking server health...")
    try:
        health_resp = stub.Health(monolithic_pb2.HealthRequest())
        if health_resp.healthy:
            print("✅ Server Status: Healthy")
            print(f"   Loaded Models: {health_resp.loaded_models}/{health_resp.max_models}")
            print(f"   GPUs: {health_resp.gpu_ids}")
        else:
            print("❌ Server Status: Unhealthy")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Load audio
    print(f"\n🎵 Loading audio: {AUDIO_FILE}")
    try:
        audio_samples, sample_rate = load_wav_as_float32(AUDIO_FILE)
        print(f"   Samples: {len(audio_samples)} ({len(audio_samples)/sample_rate:.2f}s)")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Range: [{audio_samples.min():.3f}, {audio_samples.max():.3f}]")
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return False
    
    # Load real visual frames from sanders model
    print(f"\n🖼️  Loading visual frames from sanders model (batch={BATCH_SIZE})...")
    
    # Load 6-channel face crops (3 channels original + 3 channels masked)
    crops_video = "../old/old_minimal_server/models/sanders/crops_328_video.mp4"
    rois_video = "../old/old_minimal_server/models/sanders/rois_320_video.mp4"
    
    import cv2
    visual_frames_list = []
    
    cap_crops = cv2.VideoCapture(crops_video)
    cap_rois = cv2.VideoCapture(rois_video)
    
    if not cap_crops.isOpened() or not cap_rois.isOpened():
        print(f"❌ Failed to open video files")
        return False
    
    for i in range(BATCH_SIZE):
        ret1, face_frame = cap_crops.read()
        ret2, masked_frame = cap_rois.read()
        
        if not ret1 or not ret2:
            print(f"❌ Failed to read frame {i}")
            cap_crops.release()
            cap_rois.release()
            return False
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 320x320 if needed
        if face_rgb.shape[:2] != (320, 320):
            face_rgb = cv2.resize(face_rgb, (320, 320))
        if masked_rgb.shape[:2] != (320, 320):
            masked_rgb = cv2.resize(masked_rgb, (320, 320))
        
        # Normalize to [0, 1]
        face_norm = face_rgb.astype(np.float32) / 255.0
        masked_norm = masked_rgb.astype(np.float32) / 255.0
        
        # Transpose to [C, H, W] and concatenate to 6 channels
        face_tensor = np.transpose(face_norm, (2, 0, 1))  # [3, 320, 320]
        masked_tensor = np.transpose(masked_norm, (2, 0, 1))  # [3, 320, 320]
        six_channel = np.concatenate([face_tensor, masked_tensor], axis=0)  # [6, 320, 320]
        
        visual_frames_list.append(six_channel)
    
    cap_crops.release()
    cap_rois.release()
    
    # Stack to [batch_size, 6, 320, 320]
    visual_frames = np.stack(visual_frames_list, axis=0).astype(np.float32)
    visual_bytes = visual_frames.tobytes()
    print(f"   Visual frames shape: {visual_frames.shape}")
    print(f"   Visual frames size: {len(visual_bytes):,} bytes")
    print(f"   Visual range: [{visual_frames.min():.3f}, {visual_frames.max():.3f}]")
    
    # Extract audio chunk for this batch
    # At 25fps, we need ~1s of audio per 25 frames
    audio_duration_needed = BATCH_SIZE / 25.0  # seconds
    audio_samples_needed = int(audio_duration_needed * sample_rate)
    audio_chunk = audio_samples[:audio_samples_needed]
    audio_bytes = audio_chunk.tobytes()
    print(f"   Audio chunk: {len(audio_chunk)} samples ({len(audio_chunk)/sample_rate:.2f}s)")
    print(f"   Audio chunk size: {len(audio_bytes):,} bytes")
    
    # Create frame indices
    frame_indices = list(range(BATCH_SIZE))
    
    # Send inference request
    print(f"\n🚀 Sending inference request...")
    print(f"   Model: {MODEL_ID}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    request = monolithic_pb2.CompositeBatchRequest(
        model_id=MODEL_ID,
        batch_size=BATCH_SIZE,
        start_frame_idx=0,  # Start from frame 0
        visual_frames=visual_bytes,
        raw_audio=audio_bytes,
        sample_rate=sample_rate
    )
    
    try:
        start_time = time.time()
        response = stub.InferBatchComposite(request)
        end_time = time.time()
        
        if not response.success:
            print(f"❌ Inference failed: {response.error}")
            return False
        
        # Success!
        total_time = (end_time - start_time) * 1000
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS! End-to-end pipeline works!")
        print("=" * 80)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Total time: {response.total_time_ms:.2f} ms")
        print(f"   Audio processing: {response.audio_processing_ms:.2f} ms")
        print(f"   Inference time: {response.inference_time_ms:.2f} ms")
        print(f"   Compositing time: {response.composite_time_ms:.2f} ms")
        
        print(f"\n📦 Output:")
        print(f"   Frames received: {len(response.composited_frames)}")
        
        if len(response.composited_frames) > 0:
            frame_sizes = [len(f) for f in response.composited_frames]
            print(f"   Frame sizes: {min(frame_sizes):,} - {max(frame_sizes):,} bytes (JPEG)")
            print(f"   Total output: {sum(frame_sizes):,} bytes")
            
            # Save first frame
            output_path = "test_output_frame_0.jpg"
            with open(output_path, 'wb') as f:
                f.write(response.composited_frames[0])
            print(f"   Saved first frame: {output_path}")
        
        print("\n🎉 All systems operational!")
        print("   ✓ Audio processing (mel-spectrogram)")
        print("   ✓ Audio encoder (ONNX)")
        print("   ✓ Lip-sync inference (GPU)")
        print("   ✓ Compositing")
        print("   ✓ JPEG encoding")
        
        return True
        
    except grpc.RpcError as e:
        print(f"❌ gRPC error: {e.code()} - {e.details()}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
