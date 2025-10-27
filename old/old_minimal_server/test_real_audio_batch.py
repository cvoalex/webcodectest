"""
Test audio batch inference with real audio file (aud.wav)
Demonstrates end-to-end audio processing with batch optimization
"""

import asyncio
import grpc
import time
import wave
import numpy as np
import struct

# Import generated protobuf
import optimized_lipsyncsrv_pb2 as pb2
import optimized_lipsyncsrv_pb2_grpc as pb2_grpc


def load_wav_file(wav_path):
    """Load WAV file and return audio data"""
    print(f"\n🎵 Loading audio file: {wav_path}")
    
    with wave.open(wav_path, 'rb') as wav:
        # Get WAV parameters
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        n_frames = wav.getnframes()
        
        duration = n_frames / frame_rate
        
        print(f"   Channels: {n_channels}")
        print(f"   Sample width: {sample_width} bytes")
        print(f"   Sample rate: {frame_rate} Hz")
        print(f"   Total frames: {n_frames}")
        print(f"   Duration: {duration:.2f} seconds")
        
        # Read audio data
        audio_data = wav.readframes(n_frames)
        
        # Convert to numpy array
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        audio_array = np.frombuffer(audio_data, dtype=dtype)
        
        # If stereo, convert to mono
        if n_channels == 2:
            print("   Converting stereo to mono...")
            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(dtype)
        
        return audio_array, frame_rate, duration


def create_audio_chunks(audio_array, sample_rate, chunk_duration_ms=40):
    """
    Split audio into chunks of specified duration
    
    Args:
        audio_array: Audio samples (numpy array)
        sample_rate: Sample rate in Hz
        chunk_duration_ms: Duration of each chunk in milliseconds
    
    Returns:
        List of audio chunks (each as bytes)
    """
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    total_samples = len(audio_array)
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    print(f"\n📊 Creating audio chunks:")
    print(f"   Chunk duration: {chunk_duration_ms}ms")
    print(f"   Samples per chunk: {chunk_samples}")
    print(f"   Total samples: {total_samples}")
    print(f"   Number of chunks: {num_chunks}")
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min((i + 1) * chunk_samples, total_samples)
        
        chunk_data = audio_array[start_idx:end_idx]
        
        # Pad last chunk if necessary
        if len(chunk_data) < chunk_samples:
            chunk_data = np.pad(chunk_data, (0, chunk_samples - len(chunk_data)), mode='constant')
        
        # Convert to bytes
        chunk_bytes = chunk_data.tobytes()
        chunks.append(chunk_bytes)
    
    print(f"   ✅ Created {len(chunks)} chunks ({len(chunks[0])} bytes each)")
    
    return chunks


def calculate_frame_range(audio_chunks, start_offset=8):
    """
    Calculate how many video frames we can generate from audio chunks
    
    Args:
        audio_chunks: List of audio chunks
        start_offset: Number of chunks before first frame (default: 8)
    
    Returns:
        (start_frame, max_frames)
    """
    total_chunks = len(audio_chunks)
    
    # We need: 8 before + N frames + 7 after
    # So: N = total_chunks - 15
    max_frames = max(1, total_chunks - 15)
    
    print(f"\n🎬 Frame calculation:")
    print(f"   Total audio chunks: {total_chunks}")
    print(f"   Required overhead: 15 chunks (8 before + 7 after)")
    print(f"   Maximum frames: {max_frames}")
    
    return 0, max_frames


async def test_real_audio_batch(wav_path, start_frame=0, frame_count=None, server_addr="localhost:50051"):
    """
    Test audio batch inference with real audio file
    """
    print("\n" + "="*70)
    print("🎵 REAL AUDIO BATCH INFERENCE TEST")
    print("="*70)
    
    # Load audio file
    audio_array, sample_rate, duration = load_wav_file(wav_path)
    
    # Create audio chunks (40ms each)
    audio_chunks = create_audio_chunks(audio_array, sample_rate, chunk_duration_ms=40)
    
    # Calculate frame range
    _, max_frames = calculate_frame_range(audio_chunks)
    
    # Determine how many frames to generate
    if frame_count is None or frame_count > max_frames:
        frame_count = min(max_frames, 20)  # Default to 20 frames max for testing
        print(f"\n   Using frame_count: {frame_count} (limited for testing)")
    
    # Extract audio chunks for the requested frames
    # Need: start_frame-8 through start_frame+frame_count+6
    audio_start_idx = max(0, start_frame - 8)
    audio_end_idx = start_frame + frame_count + 7
    
    if audio_end_idx > len(audio_chunks):
        print(f"\n❌ Not enough audio chunks!")
        print(f"   Need chunks up to index {audio_end_idx}, but only have {len(audio_chunks)}")
        return
    
    request_chunks = audio_chunks[audio_start_idx:audio_end_idx]
    required_chunks = frame_count + 15
    
    print(f"\n📦 Request configuration:")
    print(f"   Model: sanders")
    print(f"   Start frame: {start_frame}")
    print(f"   Frame count: {frame_count}")
    print(f"   Audio chunks: {len(request_chunks)} (expected: {required_chunks})")
    
    # Calculate bandwidth savings
    old_method_chunks = frame_count * 16
    savings_pct = (old_method_chunks - len(request_chunks)) / old_method_chunks * 100 if old_method_chunks > 0 else 0
    
    print(f"\n💾 Bandwidth comparison:")
    print(f"   Old method: {old_method_chunks} chunks")
    print(f"   New method: {len(request_chunks)} chunks")
    print(f"   Savings: {old_method_chunks - len(request_chunks)} chunks ({savings_pct:.1f}%)")
    
    # Connect to gRPC server
    print(f"\n🔌 Connecting to {server_addr}...")
    
    async with grpc.aio.insecure_channel(server_addr) as channel:
        stub = pb2_grpc.OptimizedLipSyncServiceStub(channel)
        
        print("✅ Connected!")
        
        # Create request
        request = pb2.BatchInferenceWithAudioRequest(
            model_name="sanders",
            start_frame_id=start_frame,
            frame_count=frame_count,
            audio_chunks=request_chunks
        )
        
        total_audio_size = sum(len(chunk) for chunk in request_chunks)
        print(f"\n🎯 Sending audio batch request...")
        print(f"   Total audio data: {total_audio_size / (1024*1024):.2f} MB")
        
        # Send request
        start_time = time.time()
        
        try:
            response = await stub.GenerateBatchWithAudio(request, timeout=60.0)
            
            elapsed = time.time() - start_time
            
            # Print results
            print(f"\n{'='*70}")
            print("📊 RESULTS")
            print(f"{'='*70}")
            
            print(f"\n✅ Received {len(response.responses)} responses\n")
            
            success_count = 0
            total_size = 0
            
            for i, r in enumerate(response.responses):
                frame_id = start_frame + i
                if r.success:
                    success_count += 1
                    size = len(r.prediction_data)
                    total_size += size
                    
                    print(f"  ✅ Frame {frame_id}: {r.processing_time_ms}ms "
                          f"({r.inference_time_ms:.2f}ms inference) - "
                          f"{size} bytes ({size/1024:.2f} KB)")
                    
                    # Save frame
                    import os
                    output_file = f"real_audio_frame_{frame_id}.jpg"
                    with open(output_file, 'wb') as f:
                        f.write(r.prediction_data)
                    print(f"       💾 Saved: {output_file}")
                else:
                    print(f"  ❌ Frame {frame_id}: ERROR - {r.error}")
            
            print(f"\n{'='*70}")
            print("📈 PERFORMANCE SUMMARY")
            print(f"{'='*70}")
            
            print(f"\n🎯 Results:")
            print(f"   Total Time: {elapsed*1000:.2f}ms")
            print(f"   Server Total: {response.total_processing_time_ms}ms")
            print(f"   Server Avg: {response.avg_frame_time_ms:.2f}ms per frame")
            print(f"   Success Rate: {success_count}/{frame_count} frames")
            
            if elapsed > 0:
                fps = frame_count / elapsed
                print(f"   Throughput: {fps:.2f} FPS")
            
            if total_size > 0:
                print(f"   Video Data: {total_size} bytes ({total_size/(1024*1024):.2f} MB)")
                data_rate = (total_size / (1024*1024)) / elapsed
                print(f"   Data Rate: {data_rate:.2f} MB/s")
            
            # Final bandwidth analysis
            print(f"\n{'='*70}")
            print("📊 BANDWIDTH ANALYSIS")
            print(f"{'='*70}")
            
            print(f"\n🎵 Audio Transfer:")
            print(f"   Chunks Sent: {len(request_chunks)}")
            print(f"   Old Method: {old_method_chunks} chunks")
            print(f"   Savings: {old_method_chunks - len(request_chunks)} chunks ({savings_pct:.1f}%)")
            print(f"   Audio Size: {total_audio_size/(1024*1024):.2f} MB")
            
            old_audio_size = old_method_chunks * len(request_chunks[0])
            print(f"   Old Method Size: {old_audio_size/(1024*1024):.2f} MB")
            print(f"   Bandwidth Saved: {(old_audio_size - total_audio_size)/(1024*1024):.2f} MB")
            
            print(f"\n{'='*70}\n")
            
        except grpc.RpcError as e:
            print(f"\n❌ RPC Error: {e.code()}")
            print(f"   Details: {e.details()}")
            return
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return


if __name__ == "__main__":
    import sys
    
    # Configuration
    WAV_FILE = "../aud.wav"  # Relative to minimal_server directory
    START_FRAME = 0
    FRAME_COUNT = 10  # Generate 10 frames (adjustable)
    SERVER_ADDR = "localhost:50051"
    
    print("\n🚀 REAL AUDIO BATCH TEST")
    print(f"   WAV File: {WAV_FILE}")
    print(f"   Frames: {START_FRAME} to {START_FRAME + FRAME_COUNT - 1}")
    
    # Run test
    asyncio.run(test_real_audio_batch(
        wav_path=WAV_FILE,
        start_frame=START_FRAME,
        frame_count=FRAME_COUNT,
        server_addr=SERVER_ADDR
    ))
