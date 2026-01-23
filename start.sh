#!/bin/bash
# Fast startup script for Railway deployment
# NOTE: Prefer using 'python run.py' instead of this script

echo "Starting Thoth Backend..."

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Create logs directory
mkdir -p /app/logs

# Use Python to start - handles PORT env var properly
exec python run.py
