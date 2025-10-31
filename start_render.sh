#!/bin/bash
# Render startup script - ensures port binding happens quickly

echo "🚀 Starting SafeBite API on Render..."
echo "📦 Python version: $(python --version)"
echo "🌐 Binding to port: $PORT"
echo "🔧 Starting uvicorn..."

# Start uvicorn with single worker to minimize startup time
# The app uses lazy loading for ML models, so it starts fast
exec uvicorn src.app:app \
    --host 0.0.0.0 \
    --port ${PORT:-10000} \
    --workers 1 \
    --timeout-keep-alive 120 \
    --log-level info
