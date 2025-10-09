#!/usr/bin/env python3
"""
Batch Model Video Preparation Script

Prepares pre-rendered frame videos for multiple models in parallel.
Each model needs:
1. A video file: model_videos/{model_name}.mp4
2. Pre-rendered frames: data/{model_name}/*.jpg (frame_0000.jpg to frame_NNNN.jpg)

Usage:
    # Prepare all models in model_videos/ directory
    python batch_prepare_models.py
    
    # Prepare specific models
    python batch_prepare_models.py model1 model2 model3
    
    # Specify frame rate and quality
    python batch_prepare_models.py --fps 25 --quality 85
    
    # Parallel processing with N workers
    python batch_prepare_models.py --workers 4

Requirements:
    - FFmpeg installed and in PATH
    - Input videos in model_videos/ directory (MP4 format)
    - Sufficient disk space (~50MB per model)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional
import json
import time


def check_ffmpeg():
    """Check if FFmpeg is installed and available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ FFmpeg found")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("❌ FFmpeg not found. Please install FFmpeg:")
    print("   Windows: choco install ffmpeg  OR  download from https://ffmpeg.org/")
    print("   Linux: sudo apt install ffmpeg")
    print("   Mac: brew install ffmpeg")
    return False


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: int = 25,
    quality: int = 85
) -> tuple[bool, str, dict]:
    """
    Extract frames from video using FFmpeg.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frame rate to extract (frames per second)
        quality: JPEG quality (1-100, higher is better)
    
    Returns:
        (success, message, stats)
    """
    start_time = time.time()
    model_name = video_path.stem
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command to extract frames
    output_pattern = str(output_dir / "frame_%04d.jpg")
    
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'fps={fps}',  # Set frame rate
        '-q:v', str(int((100 - quality) / 10 + 2)),  # Quality (2=best, 31=worst)
        '-start_number', '0',  # Start at frame 0
        output_pattern,
        '-y'  # Overwrite existing files
    ]
    
    try:
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr[:200]}", {}
        
        # Count extracted frames
        frames = sorted(output_dir.glob("frame_*.jpg"))
        num_frames = len(frames)
        
        if num_frames == 0:
            return False, "No frames extracted", {}
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in frames)
        size_mb = total_size / (1024 * 1024)
        
        elapsed = time.time() - start_time
        
        stats = {
            'model_name': model_name,
            'num_frames': num_frames,
            'size_mb': round(size_mb, 2),
            'avg_frame_kb': round((total_size / num_frames) / 1024, 1),
            'fps': fps,
            'quality': quality,
            'elapsed_sec': round(elapsed, 1),
            'frames_per_sec': round(num_frames / elapsed, 1)
        }
        
        return True, f"Extracted {num_frames} frames ({size_mb:.1f} MB)", stats
        
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timeout (>5 minutes)", {}
    except Exception as e:
        return False, f"Error: {str(e)}", {}


def process_model(
    model_name: str,
    video_dir: Path,
    data_dir: Path,
    fps: int,
    quality: int
) -> tuple[str, bool, str, dict]:
    """
    Process a single model video.
    
    Returns:
        (model_name, success, message, stats)
    """
    video_path = video_dir / f"{model_name}.mp4"
    
    # Check if video exists
    if not video_path.exists():
        return model_name, False, f"Video not found: {video_path}", {}
    
    # Check video file size
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    if video_size_mb > 500:
        return model_name, False, f"Video too large: {video_size_mb:.1f} MB (limit: 500 MB)", {}
    
    output_dir = data_dir / model_name
    
    # Check if already processed
    existing_frames = sorted(output_dir.glob("frame_*.jpg")) if output_dir.exists() else []
    if existing_frames:
        num_frames = len(existing_frames)
        total_size = sum(f.stat().st_size for f in existing_frames)
        size_mb = total_size / (1024 * 1024)
        
        stats = {
            'model_name': model_name,
            'num_frames': num_frames,
            'size_mb': round(size_mb, 2),
            'avg_frame_kb': round((total_size / num_frames) / 1024, 1),
            'fps': fps,
            'quality': quality,
            'cached': True
        }
        
        return model_name, True, f"Already processed: {num_frames} frames ({size_mb:.1f} MB)", stats
    
    # Extract frames
    success, message, stats = extract_frames(video_path, output_dir, fps, quality)
    
    return model_name, success, message, stats


def main():
    parser = argparse.ArgumentParser(
        description='Batch prepare model videos for real-time lip-sync',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all models
  python batch_prepare_models.py
  
  # Prepare specific models
  python batch_prepare_models.py model1 model2 model3
  
  # Custom settings with 4 parallel workers
  python batch_prepare_models.py --fps 25 --quality 85 --workers 4
  
  # Force re-process even if frames exist
  python batch_prepare_models.py --force
        """
    )
    
    parser.add_argument(
        'models',
        nargs='*',
        help='Model names to process (default: all in model_videos/)'
    )
    parser.add_argument(
        '--video-dir',
        type=Path,
        default=Path('model_videos'),
        help='Directory containing input videos (default: model_videos/)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Directory to save extracted frames (default: data/)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='Frame rate to extract (default: 25)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=85,
        choices=range(1, 101),
        metavar='1-100',
        help='JPEG quality 1-100 (default: 85)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-process even if frames exist'
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        help='Save statistics to JSON file'
    )
    
    args = parser.parse_args()
    
    # Check FFmpeg
    if not check_ffmpeg():
        return 1
    
    # Get list of models to process
    if args.models:
        models = args.models
    else:
        # Auto-discover models from video directory
        if not args.video_dir.exists():
            print(f"❌ Video directory not found: {args.video_dir}")
            print(f"   Create it and add MP4 files: {args.video_dir.absolute()}")
            return 1
        
        video_files = list(args.video_dir.glob("*.mp4"))
        if not video_files:
            print(f"❌ No MP4 files found in {args.video_dir}")
            return 1
        
        models = [f.stem for f in video_files]
    
    print(f"\n{'='*70}")
    print(f"🎬 Batch Model Video Preparation")
    print(f"{'='*70}")
    print(f"Models to process: {len(models)}")
    print(f"Video directory: {args.video_dir.absolute()}")
    print(f"Data directory: {args.data_dir.absolute()}")
    print(f"Frame rate: {args.fps} fps")
    print(f"JPEG quality: {args.quality}")
    print(f"Parallel workers: {args.workers}")
    print(f"{'='*70}\n")
    
    # Delete existing frames if --force
    if args.force:
        for model in models:
            output_dir = args.data_dir / model
            if output_dir.exists():
                print(f"🗑️  Deleting existing frames: {model}")
                import shutil
                shutil.rmtree(output_dir)
    
    # Process models
    results = []
    failed = []
    
    if args.workers > 1:
        # Parallel processing
        print(f"⚡ Processing {len(models)} models with {args.workers} workers...\n")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_model,
                    model,
                    args.video_dir,
                    args.data_dir,
                    args.fps,
                    args.quality
                ): model
                for model in models
            }
            
            for future in as_completed(futures):
                model_name, success, message, stats = future.result()
                
                if success:
                    cached = stats.get('cached', False)
                    icon = "📦" if cached else "✅"
                    print(f"{icon} {model_name}: {message}")
                    results.append(stats)
                else:
                    print(f"❌ {model_name}: {message}")
                    failed.append(model_name)
    else:
        # Sequential processing
        print(f"⏳ Processing {len(models)} models sequentially...\n")
        
        for model in models:
            model_name, success, message, stats = process_model(
                model,
                args.video_dir,
                args.data_dir,
                args.fps,
                args.quality
            )
            
            if success:
                cached = stats.get('cached', False)
                icon = "📦" if cached else "✅"
                print(f"{icon} {model_name}: {message}")
                results.append(stats)
            else:
                print(f"❌ {model_name}: {message}")
                failed.append(model_name)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 Summary")
    print(f"{'='*70}")
    print(f"Total models: {len(models)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failed)}")
    
    if results:
        total_frames = sum(r['num_frames'] for r in results)
        total_size = sum(r['size_mb'] for r in results)
        avg_frames = total_frames / len(results)
        
        print(f"\nTotal frames: {total_frames:,}")
        print(f"Total size: {total_size:.1f} MB")
        print(f"Average frames per model: {avg_frames:.0f}")
        print(f"Average size per model: {total_size/len(results):.1f} MB")
        
        # Calculate estimated VRAM usage (each frame ~100KB in memory)
        estimated_vram_mb = (total_frames * 100) / 1024
        print(f"\nEstimated VRAM usage (all models): {estimated_vram_mb:.0f} MB")
    
    if failed:
        print(f"\n❌ Failed models:")
        for model in failed:
            print(f"   - {model}")
    
    # Save statistics to JSON
    if args.output_json:
        output_data = {
            'total_models': len(models),
            'successful': len(results),
            'failed': len(failed),
            'failed_models': failed,
            'models': results,
            'settings': {
                'fps': args.fps,
                'quality': args.quality,
                'workers': args.workers
            }
        }
        
        args.output_json.write_text(json.dumps(output_data, indent=2))
        print(f"\n💾 Statistics saved to: {args.output_json}")
    
    print(f"{'='*70}\n")
    
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
