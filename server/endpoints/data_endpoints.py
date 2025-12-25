"""Data management endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/data", tags=["data"])

# Data endpoints have been consolidated into file_endpoints.py
# This module exists for backwards compatibility with routes.py
