#!/usr/bin/env python3
"""
Python dependencies installer and service starter for Real-time Lip Sync Console
"""

import subprocess
import sys
import os
import time
import asyncio
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle output"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_dependencies():
    """Check and install Python dependencies"""
    print("🐍 Checking Python dependencies...")
    
    required_packages = [
        "grpcio",
        "grpcio-tools", 
        "numpy",
        "websockets",
        "asyncio",
        "psutil"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            if not run_command(f"pip install {package}", f"Installing {package}"):
                return False
    
    return True

def check_grpc_service():
    """Check if gRPC service is available"""
    print("🎬 Checking gRPC lip sync service...")
    
    # Check if protobuf files exist
    fast_service_dir = Path("../fast_service")
    pb2_files = list(fast_service_dir.glob("*_pb2.py"))
    
    if not pb2_files:
        print("❌ gRPC protobuf files not found in ../fast_service")
        print("💡 Make sure your gRPC service is properly set up")
        return False
    
    print("✅ gRPC protobuf files found")
    return True

def start_services():
    """Start all required services"""
    print("\n🚀 Starting Real-time Lip Sync Console services...")
    
    # Start frame generator (this script)
    print("📡 Starting Python frame generator...")
    return True

async def main():
    """Main setup and startup function"""
    print("🎯 Real-time Lip Sync Console Setup")
    print("=" * 50)
    
    # Check Python dependencies
    if not check_python_dependencies():
        print("❌ Python dependency check failed")
        sys.exit(1)
    
    # Check gRPC service
    if not check_grpc_service():
        print("❌ gRPC service check failed")
        sys.exit(1)
    
    print("\n✅ All dependencies checked successfully!")
    print("\n🚀 Ready to start services")
    print("\nTo start the complete system:")
    print("1. Start gRPC service: cd ../fast_service && python grpc_server.py")
    print("2. Start frame generator: python frame_generator.py") 
    print("3. Start web server: npm run dev")
    print("4. Open browser: http://localhost:3000")
    
    # Import and start frame generator
    try:
        from frame_generator import main as frame_main
        print("\n📡 Starting frame generator...")
        await frame_main()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting frame generator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
