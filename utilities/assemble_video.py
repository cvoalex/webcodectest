#!/usr/bin/env python3
"""
Assemble generated frames with audio into MP4 video
Creates videos from both Python and Go outputs for quality comparison
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path
import subprocess
import json

def get_sorted_frames(frame_dir):
    """Get sorted list of frame files"""
    frame_files = []
    for f in os.listdir(frame_dir):
        if f.endswith('.png') and f.startswith('frame_'):
            frame_files.append(f)
    
    # Sort by frame number
    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return frame_files

def create_video_from_frames(frame_dir, output_video, audio_path, fps=25):
    """
    Create MP4 video from frames and audio
    
    Args:
        frame_dir: Directory containing frame_XXXX.png files
        output_video: Output video path
        audio_path: Path to audio file
        fps: Frames per second (default 25)
    """
    print(f"\n{'='*80}")
    print(f"🎬 Creating Video: {output_video}")
    print(f"{'='*80}")
    
    # Get frame files
    frame_files = get_sorted_frames(frame_dir)
    
    if not frame_files:
        print(f"❌ No frames found in {frame_dir}")
        return False
    
    print(f"📊 Found {len(frame_files)} frames")
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"❌ Could not read first frame: {first_frame_path}")
        return False
    
    height, width = first_frame.shape[:2]
    print(f"📐 Frame size: {width}x{height}")
    print(f"🎞️  FPS: {fps}")
    
    # Create temporary video without audio
    temp_video = output_video.replace('.mp4', '_temp.mp4')
    
    # Use H.264 codec for good quality and compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"❌ Could not open video writer")
        return False
    
    print(f"\n🔄 Writing frames to video...")
    
    # Write all frames
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"⚠️  Warning: Could not read {frame_file}, skipping")
            continue
        
        video_writer.write(frame)
        
        if (i + 1) % 10 == 0 or i == 0 or i == len(frame_files) - 1:
            print(f"   Written {i+1}/{len(frame_files)} frames ({(i+1)/len(frame_files)*100:.1f}%)")
    
    video_writer.release()
    print(f"✅ Temporary video created: {temp_video}")
    
    # Add audio using ffmpeg
    print(f"\n🎵 Adding audio track...")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        has_ffmpeg = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        has_ffmpeg = False
        print("⚠️  Warning: ffmpeg not found. Video will be created without audio.")
    
    if has_ffmpeg and os.path.exists(audio_path):
        # Use ffmpeg to combine video and audio
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', temp_video,  # Input video
            '-i', audio_path,  # Input audio
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'medium',  # Encoding speed/quality trade-off
            '-crf', '23',  # Quality (lower = better, 23 is default)
            '-c:a', 'aac',  # AAC audio codec
            '-b:a', '192k',  # Audio bitrate
            '-shortest',  # End video at shortest stream (audio or video)
            output_video
        ]
        
        print(f"🔧 Running ffmpeg...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Video with audio created: {output_video}")
            # Remove temporary video
            os.remove(temp_video)
            return True
        else:
            print(f"⚠️  ffmpeg failed, keeping video without audio")
            print(f"Error: {result.stderr[:500]}")
            os.rename(temp_video, output_video)
            return True
    else:
        # No ffmpeg or audio, just rename temp video
        if not os.path.exists(audio_path):
            print(f"⚠️  Audio file not found: {audio_path}")
        os.rename(temp_video, output_video)
        print(f"✅ Video created (without audio): {output_video}")
        return True

def compare_videos(python_video, go_video):
    """Print comparison information"""
    print(f"\n{'='*80}")
    print("📊 VIDEO COMPARISON")
    print(f"{'='*80}")
    
    for name, path in [("Python + ONNX", python_video), ("Go + ONNX", go_video)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"\n{name}:")
            print(f"   Path: {path}")
            print(f"   Size: {size_mb:.2f} MB")
            
            # Try to get video info with ffmpeg
            try:
                cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    import json
                    info = json.loads(result.stdout)
                    
                    # Get video stream
                    for stream in info.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            print(f"   Resolution: {stream.get('width')}x{stream.get('height')}")
                            print(f"   Codec: {stream.get('codec_name')}")
                            print(f"   FPS: {eval(stream.get('r_frame_rate', '0/1')):.2f}")
                            print(f"   Duration: {float(stream.get('duration', 0)):.2f}s")
                            break
            except:
                pass
        else:
            print(f"\n{name}:")
            print(f"   ❌ Not found: {path}")

def main():
    print("🎬 Video Assembly Tool")
    print("="*80)
    
    # Paths
    base_dir = Path(__file__).parent
    audio_path = base_dir / "aud.wav"
    
    python_frame_dir = base_dir / "fast_service" / "output_python_onnx"
    go_frame_dir = base_dir / "go-onnx-inference" / "cmd" / "test-real-audio" / "output_go_onnx"
    
    python_output = base_dir / "output_python_onnx.mp4"
    go_output = base_dir / "output_go_onnx.mp4"
    
    # Check audio file
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return 1
    
    print(f"✅ Audio file: {audio_path}")
    print(f"   Size: {audio_path.stat().st_size / 1024:.2f} KB")
    
    # Process Python output
    if python_frame_dir.exists():
        success = create_video_from_frames(
            str(python_frame_dir),
            str(python_output),
            str(audio_path),
            fps=25
        )
        if not success:
            print("⚠️  Failed to create Python video")
    else:
        print(f"\n⚠️  Python frames not found: {python_frame_dir}")
    
    # Process Go output
    if go_frame_dir.exists():
        success = create_video_from_frames(
            str(go_frame_dir),
            str(go_output),
            str(audio_path),
            fps=25
        )
        if not success:
            print("⚠️  Failed to create Go video")
    else:
        print(f"\n⚠️  Go frames not found: {go_frame_dir}")
    
    # Compare results
    compare_videos(str(python_output), str(go_output))
    
    print(f"\n{'='*80}")
    print("✅ VIDEO ASSEMBLY COMPLETE!")
    print(f"{'='*80}")
    print("\n📹 To view and compare:")
    print(f"   Python: {python_output}")
    print(f"   Go:     {go_output}")
    print("\n💡 Tips:")
    print("   - Open both videos side-by-side")
    print("   - Check lip-sync quality")
    print("   - Compare visual quality")
    print("   - Verify audio sync")
    print()

if __name__ == "__main__":
    sys.exit(main() or 0)
