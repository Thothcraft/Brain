from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., example="ok")
    timestamp: str = Field(..., example=datetime.utcnow().isoformat() + "Z")
    version: str = Field(..., example="1.0.0")
