#!/usr/bin/env python3
"""Railway-compatible startup script for Thoth Backend."""

import os
import sys

# Ensure PYTHONPATH includes the app directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Get port from environment (Railway sets this)
port = int(os.environ.get("PORT", 8000))

print(f"Starting Thoth Backend on port {port}...")

# Start uvicorn
import uvicorn

uvicorn.run(
    "server.main:app",
    host="0.0.0.0",
    port=port,
    workers=1,
    timeout_keep_alive=65,
    access_log=True,
    log_level="info",
)
