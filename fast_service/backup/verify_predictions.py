#!/usr/bin/env python3
"""
Visual verification of inference predictions.
Creates comparison images and verifies prediction quality.
"""

import cv2
import numpy as np
import os
import time
import grpc
import lipsyncsrv_pb2
import lipsyncsrv_pb2_grpc

def create_comparison_grid():
    """Create a comparison grid of all prediction frames"""
    
    print("🖼️  Creating visual comparison of predictions...")
    
    # Load all prediction images
    http_frames = []
    grpc_frames = []
    
    for i in range(5):
        # Load HTTP REST predictions
        http_path = f"prediction_frame_{i}.jpg"
        if os.path.exists(http_path):
            http_img = cv2.imread(http_path)
            if http_img is not None:
                http_frames.append(http_img)
        
        # Load gRPC predictions  
        grpc_path = f"grpc_prediction_frame_{i}.jpg"
        if os.path.exists(grpc_path):
            grpc_img = cv2.imread(grpc_path)
            if grpc_img is not None:
                grpc_frames.append(grpc_img)
    
    if len(http_frames) == 0 and len(grpc_frames) == 0:
        print("❌ No prediction images found!")
        return
    
    # Use whichever set we have
    frames = grpc_frames if len(grpc_frames) > 0 else http_frames
    source = "gRPC" if len(grpc_frames) > 0 else "HTTP"
    
    if len(frames) < 5:
        print(f"⚠️ Only found {len(frames)} prediction images")
    
    # Create grid layout (1 row, 5 columns)
    grid_height = frames[0].shape[0]
    grid_width = frames[0].shape[1] * len(frames)
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place frames in grid
    for i, frame in enumerate(frames):
        x_start = i * frame.shape[1]
        x_end = x_start + frame.shape[1]
        grid[:, x_start:x_end] = frame
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    
    for i in range(len(frames)):
        x_pos = i * frames[0].shape[1] + 10
        y_pos = 30
        cv2.putText(grid, f"Frame {i}", (x_pos, y_pos), font, font_scale, color, thickness)
    
    # Save grid
    output_path = f"predictions_grid_{source.lower()}.jpg"
    cv2.imwrite(output_path, grid)
    print(f"✅ Saved prediction grid: {output_path}")
    
    # Also create individual resized images for easier viewing
    for i, frame in enumerate(frames):
        # Resize for easier viewing (2x scale)
        resized = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        
        # Add frame info text
        info_text = f"Frame {i} - {source} Prediction (320x320 -> 640x640)"
        cv2.putText(resized, info_text, (10, 30), font, 0.8, (255, 255, 255), 2)
        
        output_path = f"prediction_{source.lower()}_frame_{i}_large.jpg"
        cv2.imwrite(output_path, resized)
    
    print(f"✅ Saved {len(frames)} large prediction images")
    
    return frames

def test_prediction_quality():
    """Test and save high-quality predictions with analysis"""
    
    print("\n🔬 Testing Prediction Quality...")
    print("=" * 50)
    
    # Connect to gRPC server
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = lipsyncsrv_pb2_grpc.LipSyncServiceStub(channel)
        model_name = "test_optimized_package_fixed_3"
        
        predictions = []
        
        for frame_id in range(5):
            print(f"📷 Generating Frame {frame_id}...")
            
            start_time = time.time()
            
            # Create request
            request = lipsyncsrv_pb2.InferenceRequest(
                model_name=model_name,
                frame_id=frame_id
            )
            
            # Get prediction
            response = stub.GenerateInference(request)
            
            if response.success:
                # Save raw prediction
                prediction_path = f"quality_test_frame_{frame_id}.jpg"
                with open(prediction_path, "wb") as f:
                    f.write(response.prediction_data)
                
                # Load and analyze the image
                img = cv2.imread(prediction_path)
                predictions.append(img)
                
                # Calculate image statistics
                mean_brightness = np.mean(img)
                std_brightness = np.std(img)
                
                total_time = (time.time() - start_time) * 1000
                
                print(f"   ✅ Success: {len(response.prediction_data):,} bytes")
                print(f"   📊 Processing: {response.processing_time_ms}ms, Total: {total_time:.1f}ms")
                print(f"   🖼️  Image stats: mean={mean_brightness:.1f}, std={std_brightness:.1f}")
                print(f"   📐 Shape: {response.prediction_shape}")
                print(f"   📏 Bounds: {len(response.bounds)} values: {response.bounds}")
                
            else:
                print(f"   ❌ Failed: {response.error}")
        
        # Create enhanced comparison
        if len(predictions) >= 2:
            print(f"\n🎨 Creating enhanced visualizations...")
            
            # Create side-by-side comparison
            if len(predictions) >= 2:
                comparison = np.hstack([predictions[0], predictions[1]])
                cv2.putText(comparison, "Frame 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison, "Frame 1", (predictions[0].shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite("predictions_comparison.jpg", comparison)
                print("   ✅ Saved: predictions_comparison.jpg")
            
            # Create animation frames with enhanced details
            for i, pred in enumerate(predictions):
                # Create enhanced version with info overlay
                enhanced = pred.copy()
                
                # Add semi-transparent info panel
                overlay = enhanced.copy()
                cv2.rectangle(overlay, (0, 0), (320, 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, enhanced, 0.3, 0, enhanced)
                
                # Add detailed info
                cv2.putText(enhanced, f"Frame {i} - Lip Sync Prediction", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(enhanced, f"320x320 mouth region", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(enhanced, f"gRPC inference result", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Save enhanced version
                cv2.imwrite(f"enhanced_prediction_frame_{i}.jpg", enhanced)
            
            print(f"   ✅ Saved {len(predictions)} enhanced prediction frames")
        
    except Exception as e:
        print(f"💥 Quality test failed: {e}")

def analyze_prediction_images():
    """Analyze all saved prediction images"""
    
    print("\n📊 Analyzing Prediction Images...")
    print("=" * 50)
    
    # Find all prediction images
    prediction_files = []
    for filename in os.listdir('.'):
        if 'prediction' in filename and filename.endswith('.jpg'):
            prediction_files.append(filename)
    
    if not prediction_files:
        print("❌ No prediction images found!")
        return
    
    print(f"📁 Found {len(prediction_files)} prediction images:")
    
    for filename in sorted(prediction_files):
        try:
            img = cv2.imread(filename)
            if img is not None:
                height, width, channels = img.shape
                file_size = os.path.getsize(filename)
                mean_val = np.mean(img)
                
                print(f"   📷 {filename}")
                print(f"      📐 Size: {width}x{height}x{channels}")
                print(f"      💾 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                print(f"      🎨 Mean brightness: {mean_val:.1f}")
                
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")

if __name__ == "__main__":
    print("🖼️  Visual Verification of Lip Sync Predictions")
    print("=" * 60)
    
    # Create comparison grid from existing images
    create_comparison_grid()
    
    # Test new high-quality predictions
    test_prediction_quality()
    
    # Analyze all prediction images
    analyze_prediction_images()
    
    print("\n✅ Visual verification complete!")
    print("📁 Check the generated images:")
    print("   - predictions_grid_*.jpg (overview)")
    print("   - enhanced_prediction_frame_*.jpg (detailed)")
    print("   - predictions_comparison.jpg (side-by-side)")
    print("   - quality_test_frame_*.jpg (latest predictions)")
