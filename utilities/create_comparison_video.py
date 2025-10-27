#!/usr/bin/env python3
"""
Create side-by-side comparison video of Python vs Go outputs
"""

import subprocess
import sys
from pathlib import Path

def create_side_by_side(python_video, go_video, output_video, audio_path):
    """Create side-by-side comparison video using ffmpeg"""
    
    print("\n" + "="*80)
    print("🎬 Creating Side-by-Side Comparison Video")
    print("="*80)
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found. Please install ffmpeg to create comparison video.")
        return False
    
    # Create side-by-side video with labels
    # Filter explanation:
    # - Scale both videos to same height if needed
    # - Add text labels
    # - Stack horizontally
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-i', python_video,  # Left video (Python)
        '-i', go_video,      # Right video (Go)
        '-i', audio_path,    # Audio
        '-filter_complex',
        '[0:v]drawtext=text=\'Python + ONNX\':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10[left];'
        '[1:v]drawtext=text=\'Go + ONNX\':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=10[right];'
        '[left][right]hstack=inputs=2[v]',
        '-map', '[v]',
        '-map', '2:a',  # Use audio from third input
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        output_video
    ]
    
    print("🔧 Running ffmpeg to create comparison...")
    print(f"   Output: {output_video}")
    
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Side-by-side comparison created successfully!")
        
        # Get file size
        output_path = Path(output_video)
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")
        
        return True
    else:
        print(f"❌ Failed to create comparison video")
        print(f"Error: {result.stderr[:500]}")
        return False

def main():
    base_dir = Path(__file__).parent
    
    python_video = base_dir / "output_python_onnx.mp4"
    go_video = base_dir / "output_go_onnx.mp4"
    comparison_video = base_dir / "comparison_python_vs_go.mp4"
    audio_path = base_dir / "aud.wav"
    
    # Check if input videos exist
    if not python_video.exists():
        print(f"❌ Python video not found: {python_video}")
        print("   Run assemble_video.py first")
        return 1
    
    if not go_video.exists():
        print(f"❌ Go video not found: {go_video}")
        print("   Run assemble_video.py first")
        return 1
    
    if not audio_path.exists():
        print(f"❌ Audio not found: {audio_path}")
        return 1
    
    print(f"✅ Input videos found")
    print(f"   Python: {python_video}")
    print(f"   Go:     {go_video}")
    print(f"   Audio:  {audio_path}")
    
    success = create_side_by_side(
        str(python_video),
        str(go_video),
        str(comparison_video),
        str(audio_path)
    )
    
    if success:
        print("\n" + "="*80)
        print("✅ COMPARISON VIDEO READY!")
        print("="*80)
        print(f"\n📹 Open this file to compare: {comparison_video}")
        print("\n💡 What to look for:")
        print("   - Visual quality differences")
        print("   - Lip-sync accuracy")
        print("   - Artifacts or issues")
        print("   - Color/brightness differences")
        print()
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
