#!/bin/bash

# SyncTalk2D Fast Inference Service Startup Script

echo "🚀 Starting SyncTalk2D Fast Inference Service..."

# Check if Redis is running
echo "📦 Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis not running. Starting Redis with Docker..."
    docker run -d --name redis-fast-service -p 6379:6379 redis:alpine
    sleep 3
    
    if ! redis-cli ping > /dev/null 2>&1; then
        echo "❌ Failed to start Redis. Please start Redis manually:"
        echo "   docker run -d -p 6379:6379 redis:alpine"
        exit 1
    fi
fi

echo "✅ Redis is running"

# Check Python dependencies
echo "📦 Checking Python dependencies..."
if ! python -c "import fastapi, redis, torch" > /dev/null 2>&1; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies ready"

# Set environment variables
export DEBUG=true
export GPU_MEMORY_LIMIT=2048
export CACHE_SIZE=1000
export ENABLE_METRICS=true

echo "🔧 Configuration:"
echo "   - Debug: $DEBUG"
echo "   - GPU Memory Limit: ${GPU_MEMORY_LIMIT}MB"
echo "   - Cache Size: $CACHE_SIZE frames"
echo "   - Metrics: $ENABLE_METRICS"

# Start the service
echo "🌟 Starting FastAPI service..."
python service.py
