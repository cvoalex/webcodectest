"""
Test audio batch optimization with real WAV audio file.
Loads aud.wav from project root and processes it using the new batch audio API.
"""

import grpc
import wave
import struct
import numpy as np
from pathlib import Path
import time
import optimized_lipsyncsrv_pb2 as pb
import optimized_lipsyncsrv_pb2_grpc as pb_grpc

def load_wav_file(wav_path):
    """
    Load WAV file and return audio samples as int16 array.
    Validates that it's 16kHz mono audio.
    """
    print(f"Loading WAV file: {wav_path}")
    
    with wave.open(str(wav_path), 'rb') as wav_file:
        # Get audio parameters
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        
        print(f"WAV Info:")
        print(f"  Channels: {num_channels}")
        print(f"  Sample Width: {sample_width} bytes")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Total Frames: {num_frames}")
        print(f"  Duration: {num_frames / sample_rate:.2f} seconds")
        
        # Validate format
        if sample_rate != 16000:
            raise ValueError(f"Expected 16kHz sample rate, got {sample_rate} Hz")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit audio (2 bytes), got {sample_width} bytes")
        
        # Read all audio data
        audio_bytes = wav_file.readframes(num_frames)
        
        # Convert to int16 array
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert stereo to mono if needed
        if num_channels == 2:
            print(f"Converting stereo to mono by averaging channels...")
            # Reshape to (num_frames, 2) and average across channels
            samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
        elif num_channels != 1:
            raise ValueError(f"Unsupported channel count: {num_channels} (expected 1 or 2)")
        
        print(f"Loaded {len(samples)} samples")
        return samples

def audio_to_chunks(samples, chunk_size=640):
    """
    Convert audio samples to 40ms chunks (640 samples at 16kHz).
    """
    num_chunks = len(samples) // chunk_size
    chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = samples[start_idx:end_idx]
        
        # Convert to bytes (int16 little-endian)
        chunk_bytes = chunk.tobytes()
        chunks.append(chunk_bytes)
    
    print(f"Created {len(chunks)} chunks of {chunk_size} samples (40ms each)")
    return chunks

def calculate_frame_count(num_chunks):
    """
    Calculate how many frames can be generated from audio chunks.
    Each frame needs 16 chunks (8 previous + current + 7 future).
    So we can generate (total_chunks - 15) frames.
    """
    if num_chunks < 16:
        raise ValueError(f"Need at least 16 chunks (640ms) to generate frames, got {num_chunks}")
    
    frame_count = num_chunks - 15
    print(f"Can generate {frame_count} frames from {num_chunks} audio chunks")
    return frame_count

def test_audio_batch(model_name="sanders", start_frame=0):
    """
    Test the audio batch optimization with real audio file.
    """
    # Load audio file
    wav_path = Path("../aud.wav")
    if not wav_path.exists():
        wav_path = Path("aud.wav")
    if not wav_path.exists():
        raise FileNotFoundError("Could not find aud.wav in project root or current directory")
    
    samples = load_wav_file(wav_path)
    
    # Convert to chunks
    audio_chunks = audio_to_chunks(samples)
    
    # Calculate frame count
    frame_count = calculate_frame_count(len(audio_chunks))
    
    # Show bandwidth comparison
    print("\n" + "="*60)
    print("BANDWIDTH COMPARISON")
    print("="*60)
    
    # Old method: Each frame sends 16 chunks (8 previous + current + 7 future)
    old_method_chunks = frame_count * 16
    old_method_bytes = old_method_chunks * 640 * 2  # 640 samples * 2 bytes
    
    # New method: Send all chunks once (total_chunks = frame_count + 15)
    new_method_chunks = len(audio_chunks)
    new_method_bytes = new_method_chunks * 640 * 2
    
    savings = 1 - (new_method_bytes / old_method_bytes)
    
    print(f"Old Method: {old_method_chunks:,} chunks = {old_method_bytes:,} bytes")
    print(f"New Method: {new_method_chunks:,} chunks = {new_method_bytes:,} bytes")
    print(f"Savings: {savings*100:.1f}% reduction in audio data transfer")
    print("="*60 + "\n")
    
    # Connect to server
    print(f"Connecting to server at localhost:50051...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb_grpc.OptimizedLipSyncServiceStub(channel)
    
    # Create batch request with audio
    print(f"\nSending batch request:")
    print(f"  Model: {model_name}")
    print(f"  Start Frame: {start_frame}")
    print(f"  Frame Count: {frame_count}")
    print(f"  Audio Chunks: {len(audio_chunks)}")
    print(f"  Total Audio Data: {new_method_bytes:,} bytes")
    
    request = pb.BatchInferenceWithAudioRequest(
        model_name=model_name,
        start_frame_id=start_frame,
        frame_count=frame_count,
        audio_chunks=audio_chunks
    )
    
    # Send request and measure time
    print("\nProcessing frames...")
    start_time = time.time()
    
    try:
        response = stub.GenerateBatchWithAudio(request)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Process results
        print(f"\n{'='*60}")
        print(f"SUCCESS! Received {len(response.frames)} frames")
        print(f"{'='*60}")
        print(f"Processing time: {elapsed*1000:.1f}ms")
        print(f"Average per frame: {(elapsed*1000)/len(response.frames):.1f}ms")
        print(f"Throughput: {len(response.frames)/elapsed:.1f} FPS")
        
        # Save frames
        output_dir = Path("audio_batch_output")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nSaving frames to {output_dir}/...")
        for i, frame_data in enumerate(response.frames):
            frame_id = frame_data.frame_id
            image_bytes = frame_data.image_data
            
            output_path = output_dir / f"frame_{frame_id:04d}.jpg"
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            
            if i < 3 or i >= len(response.frames) - 3:
                print(f"  Frame {frame_id}: {len(image_bytes):,} bytes -> {output_path.name}")
            elif i == 3:
                print(f"  ... ({len(response.frames) - 6} more frames) ...")
        
        print(f"\n✓ All {len(response.frames)} frames saved successfully!")
        
        # Final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Audio Duration: {len(samples)/16000:.2f} seconds")
        print(f"Frames Generated: {len(response.frames)}")
        print(f"Audio Bandwidth Saved: {savings*100:.1f}%")
        print(f"Processing Speed: {len(response.frames)/elapsed:.1f} FPS")
        print(f"Average Latency: {(elapsed*1000)/len(response.frames):.1f}ms per frame")
        print(f"{'='*60}")
        
    except grpc.RpcError as e:
        print(f"\n✗ RPC Error: {e.code()}")
        print(f"  Details: {e.details()}")
        
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            print("\n⚠ The server does not have the GenerateBatchWithAudio method.")
            print("  Make sure you restart the server with the updated code.")
    
    channel.close()

if __name__ == "__main__":
    import sys
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "sanders"
    start_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    print("="*60)
    print("REAL AUDIO BATCH TEST")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Start Frame: {start_frame}")
    print("="*60 + "\n")
    
    test_audio_batch(model_name, start_frame)
