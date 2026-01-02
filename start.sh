#!/bin/bash
# Fast startup script for Railway deployment

echo "Starting Thoth Backend..."

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Create logs directory
mkdir -p /app/logs

# Start with optimized settings
exec uvicorn server.main:app \
    --host=0.0.0.0 \
    --port=${PORT:-8000} \
    --workers=1 \
    --worker-class=uvicorn.workers.UvicornWorker \
    --timeout-keep-alive=65 \
    --timeout-graceful-shutdown=30 \
    --access-log \
    --log-level=info
