"""Sensor Data Collection Module.

This module provides advanced sensor device recognition and data collection
with configurable collection options for each sensor type.

Supported Sensors:
- Camera: Image/video capture with configurable intervals and duration
- CSI (WiFi): Channel State Information from ESP32 devices
- IMU: Inertial Measurement Unit (accelerometer, gyroscope)
- Radar (mmWave): Millimeter-wave radar for presence detection
- Microphone: Audio capture

Usage:
    from server.sensors import SensorRegistry, create_data_collector
    
    # List connected sensors
    sensors = SensorRegistry.detect_sensors()
    
    # Create collector with options
    collector = create_data_collector("camera", {
        "interval_seconds": 5,
        "total_captures": 30,
        "labels": ["empty", "office"],
    })
    
    # Start collection
    collector.start()
"""

from .base import (
    BaseSensor,
    SensorRegistry,
    SensorType,
    SensorCapabilities,
    CollectionConfig,
    CollectionResult,
)
from .data_collector import DataCollector, create_data_collector

# Import all sensors to register them
from . import sensor_camera
from . import sensor_csi
from . import sensor_imu
from . import sensor_radar
from . import sensor_microphone

__all__ = [
    "BaseSensor",
    "SensorRegistry",
    "SensorType",
    "SensorCapabilities",
    "CollectionConfig",
    "CollectionResult",
    "DataCollector",
    "create_data_collector",
]
