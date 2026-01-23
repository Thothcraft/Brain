"""CSI (Channel State Information) Sensor Implementation.

Supports WiFi CSI data collection from ESP32 devices.
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
class CSISensor(BaseSensor):
    """CSI sensor for WiFi Channel State Information collection."""
    
    sensor_type = SensorType.CSI
    sensor_name = "WiFi CSI"
    sensor_description = "Channel State Information from ESP32 devices"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._serial_port = None
        self._baud_rate = 921600
        self._buffer = []
    
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
            {"name": "duration_seconds", "type": "float", "default": 60.0, "min": 1, "max": 3600, "description": "Collection duration in seconds"},
            {"name": "sample_rate_hz", "type": "float", "default": 100.0, "min": 1, "max": 1000, "description": "Target sample rate"},
            {"name": "baud_rate", "type": "int", "default": 921600, "options": [115200, 460800, 921600], "description": "Serial baud rate"},
            {"name": "output_format", "type": "str", "default": "csv", "options": ["csv", "json"], "description": "Output format"},
            {"name": "include_rssi", "type": "bool", "default": True, "description": "Include RSSI values"},
            {"name": "include_timestamp", "type": "bool", "default": True, "description": "Include timestamps"},
        ]
    
    async def detect(self) -> List[SensorDevice]:
        """Detect available ESP32 CSI devices on serial ports."""
        devices = []
        
        try:
            import serial.tools.list_ports
            
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                # Check for ESP32 devices
                if "CP210" in port.description or "CH340" in port.description or "ESP" in port.description.upper():
                    devices.append(SensorDevice(
                        device_id=f"csi_{port.device}",
                        sensor_type=SensorType.CSI,
                        name=f"ESP32 CSI ({port.device})",
                        description=port.description,
                        is_connected=True,
                        capabilities=self.get_capabilities(),
                        metadata={
                            "port": port.device,
                            "description": port.description,
                            "hwid": port.hwid,
                        },
                    ))
        except ImportError:
            logger.warning("pyserial not installed, CSI device detection unavailable")
        except Exception as e:
            logger.error(f"Error detecting CSI devices: {e}")
        
        return devices
    
    async def connect(self, device_id: str) -> bool:
        """Connect to an ESP32 CSI device."""
        try:
            import serial
            
            # Extract port from device_id
            port = device_id.replace("csi_", "")
            self._serial_port = serial.Serial(port, self._baud_rate, timeout=1)
            
            if self._serial_port.is_open:
                self.is_connected = True
                self.device = SensorDevice(
                    device_id=device_id,
                    sensor_type=SensorType.CSI,
                    name=f"ESP32 CSI ({port})",
                    is_connected=True,
                )
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error connecting to CSI device: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the CSI device."""
        if self._serial_port and self._serial_port.is_open:
            self._serial_port.close()
        self._serial_port = None
        self.is_connected = False
        self.device = None
        return True
    
    async def capture(self, config: CollectionConfig) -> CollectionResult:
        """Capture CSI data according to configuration."""
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
            filename = f"csi_{label}_{timestamp}.{config.output_format}"
            filepath = os.path.join(config.output_directory, filename)
            
            start_time = datetime.now()
            sample_count = 0
            
            while (datetime.now() - start_time).total_seconds() < config.duration_seconds:
                if self._stop_collection:
                    break
                
                if self._serial_port and self._serial_port.in_waiting:
                    line = self._serial_port.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line and self._is_valid_csi_line(line):
                        parsed = self._parse_csi_line(line)
                        if parsed:
                            parsed['timestamp'] = datetime.now().isoformat()
                            parsed['label'] = label
                            samples.append(parsed)
                            sample_count += 1
                
                await asyncio.sleep(1.0 / config.sample_rate_hz)
            
            # Save collected data
            if samples:
                if config.output_format == "csv":
                    self._save_csv(filepath, samples)
                else:
                    self._save_json(filepath, samples)
                files_created.append(filepath)
            
            logger.info(f"Collected {sample_count} CSI samples: {filename}")
            
        except Exception as e:
            errors.append(str(e))
            logger.error(f"CSI capture error: {e}")
        finally:
            self.is_collecting = False
        
        return CollectionResult(
            success=len(errors) == 0,
            sensor_type=SensorType.CSI,
            files_created=files_created,
            total_samples=len(samples),
            duration_seconds=(datetime.now() - started_at).total_seconds(),
            labels_used=config.labels,
            started_at=started_at,
            completed_at=datetime.now(),
            errors=errors,
            metadata={"sample_count": len(samples)},
        )
    
    def _is_valid_csi_line(self, line: str) -> bool:
        """Check if line contains valid CSI data."""
        return "CSI_DATA" in line or (line.startswith("[") and "," in line)
    
    def _parse_csi_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a CSI data line."""
        try:
            # Handle different CSI formats
            if "CSI_DATA" in line:
                # Format: CSI_DATA,rssi,csi_array
                parts = line.split(",")
                if len(parts) >= 3:
                    rssi = int(parts[1])
                    csi_raw = [int(x) for x in parts[2:] if x.strip()]
                    return {"rssi": rssi, "csi": csi_raw}
            else:
                # Try JSON format
                data = json.loads(line)
                return data
        except Exception:
            return None
    
    def _save_csv(self, filepath: str, samples: List[Dict]):
        """Save samples to CSV file."""
        with open(filepath, 'w') as f:
            if samples:
                # Write header
                f.write("timestamp,label,rssi," + ",".join([f"csi_{i}" for i in range(128)]) + "\n")
                
                for sample in samples:
                    csi = sample.get('csi', [0] * 128)
                    csi_str = ",".join(str(x) for x in csi[:128])
                    f.write(f"{sample.get('timestamp', '')},{sample.get('label', '')},{sample.get('rssi', 0)},{csi_str}\n")
    
    def _save_json(self, filepath: str, samples: List[Dict]):
        """Save samples to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(samples, f, indent=2)
