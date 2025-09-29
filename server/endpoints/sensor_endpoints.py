"""Sensor Management Endpoints for Thoth Device (Raspberry Pi with Sense HAT).

This module handles all sensor-related operations including:
- Real-time sensor data collection
- Sensor control and configuration
- Historical data retrieval
- WebSocket streaming for live data
"""

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio
import json
import random
from collections import deque

# Import shared models
from .models import StandardResponse

router = APIRouter(prefix="/sensors", tags=["sensors"])

# ============================================================================
# MODELS
# ============================================================================

class SensorData(BaseModel):
    """Current sensor readings from Sense HAT."""
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    pressure: float = Field(..., description="Pressure in millibars")
    orientation: Dict[str, float] = Field(..., description="Pitch, roll, yaw")
    acceleration: Dict[str, float] = Field(..., description="X, Y, Z acceleration")
    compass: float = Field(..., description="Compass heading in degrees")
    timestamp: datetime = Field(default_factory=datetime.now)
    device_id: str = Field(..., description="Thoth device identifier")

class SensorControl(BaseModel):
    """Control which sensors are active."""
    temperature: bool = True
    humidity: bool = True
    pressure: bool = True
    motion: bool = True
    compass: bool = True

class SensorHistoryQuery(BaseModel):
    """Query parameters for sensor history."""
    device_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    sensor_types: Optional[List[str]] = None

# ============================================================================
# IN-MEMORY STORAGE (Replace with database in production)
# ============================================================================

# Store sensor configurations per device
sensor_configs: Dict[str, SensorControl] = {}

# Store historical sensor data (limited to last 10000 readings)
sensor_history: deque = deque(maxlen=10000)

# Active WebSocket connections
active_connections: List[WebSocket] = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_mock_sensor_data(device_id: str = "thoth-001") -> SensorData:
    """Generate mock sensor data for testing.
    
    In production, this would interface with actual Sense HAT hardware.
    """
    return SensorData(
        temperature=20.0 + random.uniform(-5, 5),
        humidity=40.0 + random.uniform(-10, 10),
        pressure=1013.25 + random.uniform(-20, 20),
        orientation={
            "pitch": random.uniform(-180, 180),
            "roll": random.uniform(-180, 180),
            "yaw": random.uniform(0, 360)
        },
        acceleration={
            "x": random.uniform(-2, 2),
            "y": random.uniform(-2, 2),
            "z": random.uniform(-2, 2)
        },
        compass=random.uniform(0, 360),
        device_id=device_id
    )

async def broadcast_sensor_data(data: SensorData):
    """Broadcast sensor data to all connected WebSocket clients."""
    message = data.model_dump_json()
    disconnected = []
    
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/sensors/current", response_model=SensorData)
async def get_current_sensor_data(device_id: str = Query("thoth-001", description="Device ID")):
    """Get the latest sensor readings from a Thoth device.
    
    Returns real-time data from Sense HAT including:
    - Temperature, humidity, pressure
    - Motion (orientation, acceleration)
    - Compass heading
    """
    try:
        # In production, this would query actual hardware
        # For now, generate mock data
        sensor_data = generate_mock_sensor_data(device_id)
        
        # Store in history
        sensor_history.append(sensor_data.model_dump())
        
        # Broadcast to WebSocket clients
        await broadcast_sensor_data(sensor_data)
        
        return sensor_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read sensors: {str(e)}")

@router.post("/sensors/control", response_model=StandardResponse)
async def control_sensors(
    device_id: str,
    control: SensorControl
):
    """Toggle individual sensors on/off for a device.
    
    Allows selective enabling/disabling of:
    - Temperature sensor
    - Humidity sensor
    - Pressure sensor
    - Motion sensors (IMU)
    - Compass
    """
    try:
        # Store configuration
        sensor_configs[device_id] = control
        
        # In production, this would send commands to actual hardware
        # via MQTT, WebSocket, or direct GPIO control
        
        return StandardResponse(
            success=True,
            message=f"Sensor configuration updated for device {device_id}",
            data={
                "device_id": device_id,
                "config": control.model_dump()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to control sensors: {str(e)}")

@router.websocket("/sensors/stream")
async def sensor_stream(websocket: WebSocket, device_id: str = "thoth-001"):
    """WebSocket endpoint for live sensor data streaming.
    
    Provides real-time sensor updates at configurable intervals.
    Clients receive JSON messages with current sensor readings.
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial configuration
        config = sensor_configs.get(device_id, SensorControl())
        await websocket.send_json({
            "type": "config",
            "data": config.model_dump()
        })
        
        # Stream sensor data
        while True:
            # Generate and send sensor data
            sensor_data = generate_mock_sensor_data(device_id)
            
            # Store in history
            sensor_history.append(sensor_data.model_dump())
            
            # Send to this client
            await websocket.send_json({
                "type": "sensor_data",
                "data": sensor_data.model_dump()
            })
            
            # Wait before next reading (configurable interval)
            await asyncio.sleep(1)  # 1 second interval
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"WebSocket error: {e}")

@router.get("/sensors/history", response_model=Dict[str, Any])
async def get_sensor_history(
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    sensor_types: Optional[str] = Query(None, description="Comma-separated sensor types"),
    start_time: Optional[datetime] = Query(None, description="Start time for filtering"),
    end_time: Optional[datetime] = Query(None, description="End time for filtering")
):
    """Retrieve paginated historical sensor data.
    
    Supports filtering by:
    - Device ID
    - Time range
    - Sensor types
    - Pagination (limit/offset)
    """
    try:
        # Filter history based on parameters
        filtered_history = list(sensor_history)
        
        # Apply device filter
        if device_id:
            filtered_history = [h for h in filtered_history if h.get("device_id") == device_id]
        
        # Apply time range filter
        if start_time:
            filtered_history = [h for h in filtered_history 
                              if datetime.fromisoformat(h["timestamp"]) >= start_time]
        if end_time:
            filtered_history = [h for h in filtered_history 
                              if datetime.fromisoformat(h["timestamp"]) <= end_time]
        
        # Apply sensor type filter
        if sensor_types:
            types = sensor_types.split(",")
            # Filter logic would go here based on sensor types
        
        # Apply pagination
        total = len(filtered_history)
        paginated = filtered_history[offset:offset + limit]
        
        return {
            "success": True,
            "data": paginated,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@router.get("/sensors/stats", response_model=Dict[str, Any])
async def get_sensor_statistics(
    device_id: str = Query("thoth-001", description="Device ID"),
    period: str = Query("1h", description="Time period (1h, 24h, 7d, 30d)")
):
    """Get aggregated sensor statistics for a device.
    
    Returns min, max, average, and trends for each sensor type
    over the specified time period.
    """
    try:
        # Parse period
        period_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        if period not in period_map:
            raise ValueError(f"Invalid period: {period}")
        
        cutoff_time = datetime.now() - period_map[period]
        
        # Filter data for device and time period
        device_data = [
            h for h in sensor_history 
            if h.get("device_id") == device_id 
            and datetime.fromisoformat(h["timestamp"]) >= cutoff_time
        ]
        
        if not device_data:
            return {
                "success": True,
                "message": "No data available for the specified period",
                "stats": {}
            }
        
        # Calculate statistics
        stats = {
            "temperature": {
                "min": min(d["temperature"] for d in device_data),
                "max": max(d["temperature"] for d in device_data),
                "avg": sum(d["temperature"] for d in device_data) / len(device_data),
                "current": device_data[-1]["temperature"] if device_data else None
            },
            "humidity": {
                "min": min(d["humidity"] for d in device_data),
                "max": max(d["humidity"] for d in device_data),
                "avg": sum(d["humidity"] for d in device_data) / len(device_data),
                "current": device_data[-1]["humidity"] if device_data else None
            },
            "pressure": {
                "min": min(d["pressure"] for d in device_data),
                "max": max(d["pressure"] for d in device_data),
                "avg": sum(d["pressure"] for d in device_data) / len(device_data),
                "current": device_data[-1]["pressure"] if device_data else None
            }
        }
        
        return {
            "success": True,
            "device_id": device_id,
            "period": period,
            "data_points": len(device_data),
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate statistics: {str(e)}")
