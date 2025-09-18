#!/usr/bin/env python3
"""
Service runner for SyncTalk2D FastAPI service
Handles startup, dependencies, and basic configuration
"""

import subprocess
import sys
import time
import requests
import os

def check_redis():
    """Check if Redis is running"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis is running")
        return True
    except Exception as e:
        print(f"❌ Redis not available: {e}")
        print("💡 Start Redis with: docker run -d -p 6379:6379 redis:alpine")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    required_packages = ['fastapi', 'uvicorn', 'redis', 'torch', 'cv2']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install fastapi uvicorn redis torch opencv-python")
        return False
    else:
        print("✅ All dependencies available")
        return True

def start_service():
    """Start the FastAPI service"""
    print("🚀 Starting SyncTalk2D FastAPI Service...")
    
    # Check if service.py exists
    if not os.path.exists('service.py'):
        print("❌ service.py not found in current directory")
        return False
    
    try:
        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 'service:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ])
        
        print("⏳ Waiting for service to start...")
        time.sleep(5)
        
        # Test if service is responding
        for attempt in range(10):
            try:
                response = requests.get('http://localhost:8000/health', timeout=2)
                if response.status_code == 200:
                    print("✅ Service started successfully!")
                    print("🌐 Service available at: http://localhost:8000")
                    print("📖 API docs at: http://localhost:8000/docs")
                    return True
            except:
                time.sleep(2)
                print(f"   Attempt {attempt + 1}/10...")
        
        print("❌ Service failed to start properly")
        process.terminate()
        return False
        
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        return False

def run_baseline_test():
    """Run baseline performance test"""
    print("\n🧪 Running baseline performance test...")
    
    if os.path.exists('test_baseline.py'):
        try:
            subprocess.run([sys.executable, 'test_baseline.py'], check=True)
        except subprocess.CalledProcessError:
            print("⚠️  Baseline test completed with issues")
    else:
        print("⚠️  test_baseline.py not found, skipping baseline test")

def main():
    """Main runner function"""
    print("🎬 SyncTalk2D Service Runner")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check Redis
    if not check_redis():
        return 1
    
    # Start service
    if start_service():
        print("\n🎯 Service is running! Ready for testing.")
        print("\nNext steps:")
        print("1. Run baseline test: python test_baseline.py")
        print("2. Test batch processing: python test_multi_batch.py") 
        print("3. Run performance benchmark: python benchmark.py")
        print("4. Start optimizations based on results")
        
        # Optionally run baseline test
        user_input = input("\nRun baseline test now? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            run_baseline_test()
        
        print(f"\n✨ Setup complete! Service ready for optimization work.")
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit(main())
