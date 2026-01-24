#!/usr/bin/env python3
"""Railway-compatible startup script for Thoth Backend."""

import os
import sys
import traceback

# Ensure PYTHONPATH includes the app directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Get port from environment (Railway sets this)
port = int(os.environ.get("PORT", 8000))

print(f"Starting Thoth Backend on port {port}...", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)
print(f"PYTHONPATH: {sys.path}", flush=True)

try:
    print("Importing uvicorn...", flush=True)
    import uvicorn
    print("Uvicorn imported successfully", flush=True)
    
    print("Testing server.main import...", flush=True)
    from server.main import app
    print("server.main imported successfully", flush=True)
    
    print(f"Starting uvicorn on port {port}...", flush=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=65,
        access_log=True,
        log_level="info",
    )
except Exception as e:
    print(f"FATAL ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
