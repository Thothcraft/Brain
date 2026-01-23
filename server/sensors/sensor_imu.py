"""IMU (Inertial Measurement Unit) Sensor Implementation.

Supports accelerometer and gyroscope data collection.
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
class IMUSensor(BaseSensor):
    """IMU sensor for accelerometer and gyroscope data."""
    
    sensor_type = SensorType.IMU
    sensor_name = "IMU"
    sensor_description = "Inertial Measurement Unit (accelerometer, gyroscope)"
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
            max_sample_rate=1000.0,
            min_sample_rate=1.0,
            supports_labels=True,
            supports_streaming=True,
            config_options=cls.get_config_options(),
        )
    
    @classmethod
    def get_config_options(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "duration_seconds", "type": "float", "default": 60.0, "min": 1, "max": 3600, "description": "Collection duration"},
            {"name": "sample_rate_hz", "type": "float", "default": 100.0, "min": 1, "max": 1000, "description": "Sample rate"},
            {"name": "include_accelerometer", "type": "bool", "default": True, "description": "Include accelerometer data"},
            {"name": "include_gyroscope", "type": "bool", "default": True, "description": "Include gyroscope data"},
            {"name": "include_magnetometer", "type": "bool", "default": False, "description": "Include magnetometer data"},
            {"name": "output_format", "type": "str", "default": "csv", "options": ["csv", "json"], "description": "Output format"},
        ]
    
    async def detect(self) -> List[SensorDevice]:
        """Detect available IMU devices."""
        devices = []
        
        try:
            import serial.tools.list_ports
            
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                if any(x in port.description.upper() for x in ["IMU", "MPU", "BNO", "LSM", "ARDUINO"]):
                    devices.append(SensorDevice(
                        device_id=f"imu_{port.device}",
                        sensor_type=SensorType.IMU,
                        name=f"IMU ({port.device})",
                        description=port.description,
                        is_connected=True,
                        capabilities=self.get_capabilities(),
                        metadata={"port": port.device},
                    ))
        except ImportError:
            logger.warning("pyserial not installed")
        except Exception as e:
            logger.error(f"Error detecting IMU devices: {e}")
        
        return devices
    
    async def connect(self, device_id: str) -> bool:
        """Connect to an IMU device."""
        try:
            import serial
            port = device_id.replace("imu_", "")
            self._serial_port = serial.Serial(port, self._baud_rate, timeout=1)
            
            if self._serial_port.is_open:
                self.is_connected = True
                self.device = SensorDevice(
                    device_id=device_id,
                    sensor_type=SensorType.IMU,
                    name=f"IMU ({port})",
                    is_connected=True,
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error connecting to IMU: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self._serial_port and self._serial_port.is_open:
            self._serial_port.close()
        self._serial_port = None
        self.is_connected = False
        return True
    
    async def capture(self, config: CollectionConfig) -> CollectionResult:
        """Capture IMU data."""
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
            filename = f"imu_{label}_{timestamp}.{config.output_format}"
            filepath = os.path.join(config.output_directory, filename)
            
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < config.duration_seconds:
                if self._stop_collection:
                    break
                
                if self._serial_port and self._serial_port.in_waiting:
                    line = self._serial_port.readline().decode('utf-8', errors='ignore').strip()
                    parsed = self._parse_imu_line(line)
                    if parsed:
                        parsed['timestamp'] = datetime.now().isoformat()
                        parsed['label'] = label
                        samples.append(parsed)
                
                await asyncio.sleep(1.0 / config.sample_rate_hz)
            
            if samples:
                if config.output_format == "csv":
                    self._save_csv(filepath, samples)
                else:
                    self._save_json(filepath, samples)
                files_created.append(filepath)
            
        except Exception as e:
            errors.append(str(e))
        finally:
            self.is_collecting = False
        
        return CollectionResult(
            success=len(errors) == 0,
            sensor_type=SensorType.IMU,
            files_created=files_created,
            total_samples=len(samples),
            duration_seconds=(datetime.now() - started_at).total_seconds(),
            labels_used=config.labels,
            started_at=started_at,
            completed_at=datetime.now(),
            errors=errors,
        )
    
    def _parse_imu_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            if "," in line:
                parts = [float(x) for x in line.split(",") if x.strip()]
                if len(parts) >= 6:
                    return {
                        "acc_x": parts[0], "acc_y": parts[1], "acc_z": parts[2],
                        "gyro_x": parts[3], "gyro_y": parts[4], "gyro_z": parts[5],
                    }
            return json.loads(line)
        except:
            return None
    
    def _save_csv(self, filepath: str, samples: List[Dict]):
        with open(filepath, 'w') as f:
            f.write("timestamp,label,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
            for s in samples:
                f.write(f"{s.get('timestamp','')},{s.get('label','')},{s.get('acc_x',0)},{s.get('acc_y',0)},{s.get('acc_z',0)},{s.get('gyro_x',0)},{s.get('gyro_y',0)},{s.get('gyro_z',0)}\n")
    
    def _save_json(self, filepath: str, samples: List[Dict]):
        with open(filepath, 'w') as f:
            json.dump(samples, f, indent=2)
