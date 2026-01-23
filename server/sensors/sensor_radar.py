"""Radar (mmWave) Sensor Implementation.

Supports millimeter-wave radar data collection for presence detection.
"""

import os
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import (
    BaseSensor,
    SensorType,
    SensorDevice,
    SensorCapabilities,
    CollectionConfig,
    CollectionResult,
    CollectionMode,
    register_sensor,
)

logger = logging.getLogger(__name__)


@register_sensor
class RadarSensor(BaseSensor):
    """mmWave radar sensor for presence and motion detection."""
    
    sensor_type = SensorType.RADAR
    sensor_name = "mmWave Radar"
    sensor_description = "Millimeter-wave radar for presence detection"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._serial_port = None
        self._baud_rate = 115200
    
    @classmethod
    def get_capabilities(cls) -> SensorCapabilities:
        return SensorCapabilities(
            sensor_type=cls.sensor_type,
            name=cls.sensor_name,
            description=cls.sensor_description,
            supported_modes=[CollectionMode.CONTINUOUS, CollectionMode.INTERVAL],
            supported_formats=["csv", "json"],
            max_sample_rate=100.0,
            min_sample_rate=1.0,
            supports_labels=True,
            supports_streaming=True,
            config_options=cls.get_config_options(),
        )
    
    @classmethod
    def get_config_options(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "duration_seconds", "type": "float", "default": 60.0, "description": "Collection duration"},
            {"name": "sample_rate_hz", "type": "float", "default": 20.0, "description": "Sample rate"},
            {"name": "include_point_cloud", "type": "bool", "default": True, "description": "Include point cloud data"},
            {"name": "include_vitals", "type": "bool", "default": False, "description": "Include vital signs"},
            {"name": "output_format", "type": "str", "default": "json", "options": ["csv", "json"], "description": "Output format"},
        ]
    
    async def detect(self) -> List[SensorDevice]:
        """Detect available radar devices."""
        devices = []
        try:
            import serial.tools.list_ports
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                if any(x in port.description.upper() for x in ["RADAR", "IWR", "AWR", "MMWAVE"]):
                    devices.append(SensorDevice(
                        device_id=f"radar_{port.device}",
                        sensor_type=SensorType.RADAR,
                        name=f"mmWave Radar ({port.device})",
                        description=port.description,
                        is_connected=True,
                        capabilities=self.get_capabilities(),
                    ))
        except Exception as e:
            logger.error(f"Error detecting radar: {e}")
        return devices
    
    async def connect(self, device_id: str) -> bool:
        try:
            import serial
            port = device_id.replace("radar_", "")
            self._serial_port = serial.Serial(port, self._baud_rate, timeout=1)
            self.is_connected = self._serial_port.is_open
            if self.is_connected:
                self.device = SensorDevice(device_id=device_id, sensor_type=SensorType.RADAR, name=f"Radar ({port})", is_connected=True)
            return self.is_connected
        except Exception as e:
            logger.error(f"Error connecting to radar: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self._serial_port:
            self._serial_port.close()
        self._serial_port = None
        self.is_connected = False
        return True
    
    async def capture(self, config: CollectionConfig) -> CollectionResult:
        self.is_collecting = True
        self._stop_collection = False
        started_at = datetime.now()
        files_created = []
        errors = []
        samples = []
        
        try:
            os.makedirs(config.output_directory, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label = config.labels[0] if config.labels else "unlabeled"
            filename = f"radar_{label}_{timestamp}.{config.output_format}"
            filepath = os.path.join(config.output_directory, filename)
            
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < config.duration_seconds:
                if self._stop_collection:
                    break
                if self._serial_port and self._serial_port.in_waiting:
                    line = self._serial_port.readline().decode('utf-8', errors='ignore').strip()
                    parsed = self._parse_radar_line(line)
                    if parsed:
                        parsed['timestamp'] = datetime.now().isoformat()
                        parsed['label'] = label
                        samples.append(parsed)
                await asyncio.sleep(1.0 / config.sample_rate_hz)
            
            if samples:
                with open(filepath, 'w') as f:
                    json.dump(samples, f, indent=2)
                files_created.append(filepath)
        except Exception as e:
            errors.append(str(e))
        finally:
            self.is_collecting = False
        
        return CollectionResult(
            success=len(errors) == 0,
            sensor_type=SensorType.RADAR,
            files_created=files_created,
            total_samples=len(samples),
            duration_seconds=(datetime.now() - started_at).total_seconds(),
            labels_used=config.labels,
            started_at=started_at,
            completed_at=datetime.now(),
            errors=errors,
        )
    
    def _parse_radar_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(line)
        except:
            return None
