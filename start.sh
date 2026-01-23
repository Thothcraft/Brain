#!/bin/bash
# Fast startup script for Railway deployment

echo "Starting Thoth Backend..."

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Create logs directory
mkdir -p /app/logs

# Set default port if not provided
if [ -z "$PORT" ]; then
    PORT=8000
fi

echo "Using PORT: $PORT"

# Start with optimized settings
exec uvicorn server.main:app \
    --host=0.0.0.0 \
    --port=$PORT \
    --workers=1 \
    --timeout-keep-alive=65 \
    --timeout-graceful-shutdown=30 \
    --access-log \
    --log-level=info
