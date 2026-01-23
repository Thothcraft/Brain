"""Base classes and registry for sensor devices.

This module defines the base class and registry pattern for all sensor types.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class SensorType(str, Enum):
    """Types of supported sensors."""
    CAMERA = "camera"
    CSI = "csi"
    IMU = "imu"
    RADAR = "radar"
    MICROPHONE = "microphone"
    GENERIC = "generic"


class CollectionMode(str, Enum):
    """Data collection modes."""
    SINGLE = "single"           # Single capture
    INTERVAL = "interval"       # Capture at intervals
    CONTINUOUS = "continuous"   # Continuous stream
    TRIGGERED = "triggered"     # Event-triggered capture


@dataclass
class SensorCapabilities:
    """Capabilities of a sensor device."""
    sensor_type: SensorType
    name: str
    description: str
    supported_modes: List[CollectionMode] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    max_sample_rate: float = 0.0  # Hz
    min_sample_rate: float = 0.0
    supports_labels: bool = True
    supports_streaming: bool = False
    config_options: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CollectionConfig:
    """Configuration for data collection."""
    # Basic settings
    mode: CollectionMode = CollectionMode.INTERVAL
    labels: List[str] = field(default_factory=list)
    output_format: str = "auto"
    
    # Interval mode settings
    interval_seconds: float = 5.0
    total_captures: int = 30
    
    # Continuous mode settings
    duration_seconds: float = 60.0
    sample_rate_hz: float = 100.0
    
    # Output settings
    output_directory: str = ""
    filename_prefix: str = "capture"
    auto_label: bool = False
    
    # Camera-specific
    resolution: tuple = (640, 480)
    fps: int = 30
    
    # Audio-specific
    audio_channels: int = 1
    audio_sample_rate: int = 44100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "labels": self.labels,
            "output_format": self.output_format,
            "interval_seconds": self.interval_seconds,
            "total_captures": self.total_captures,
            "duration_seconds": self.duration_seconds,
            "sample_rate_hz": self.sample_rate_hz,
        }


@dataclass
class CollectionResult:
    """Result from a data collection session."""
    success: bool
    sensor_type: SensorType
    files_created: List[str] = field(default_factory=list)
    total_samples: int = 0
    duration_seconds: float = 0.0
    labels_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class SensorDevice:
    """Represents a detected sensor device."""
    device_id: str
    sensor_type: SensorType
    name: str
    description: str = ""
    is_connected: bool = True
    capabilities: Optional[SensorCapabilities] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSensor(ABC):
    """Base class for all sensor implementations.
    
    All sensors must inherit from this class and implement:
    - detect(): Detect available devices
    - connect(): Connect to a device
    - capture(): Capture data
    - get_capabilities(): Return sensor capabilities
    """
    
    sensor_type: SensorType = SensorType.GENERIC
    sensor_name: str = "Base Sensor"
    sensor_description: str = "Base sensor implementation"
    version: str = "1.0.0"
    
    def __init__(self):
        self.device: Optional[SensorDevice] = None
        self.is_connected: bool = False
        self.is_collecting: bool = False
        self._stop_collection: bool = False
    
    @abstractmethod
    async def detect(self) -> List[SensorDevice]:
        """Detect available sensor devices.
        
        Returns:
            List of detected SensorDevice instances
        """
        pass
    
    @abstractmethod
    async def connect(self, device_id: str) -> bool:
        """Connect to a specific device.
        
        Args:
            device_id: ID of the device to connect to
            
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the current device."""
        pass
    
    @abstractmethod
    async def capture(self, config: CollectionConfig) -> CollectionResult:
        """Capture data according to configuration.
        
        Args:
            config: Collection configuration
            
        Returns:
            CollectionResult with captured data info
        """
        pass
    
    @classmethod
    def get_capabilities(cls) -> SensorCapabilities:
        """Get sensor capabilities."""
        return SensorCapabilities(
            sensor_type=cls.sensor_type,
            name=cls.sensor_name,
            description=cls.sensor_description,
        )
    
    @classmethod
    def get_config_options(cls) -> List[Dict[str, Any]]:
        """Get configuration options for this sensor."""
        return []
    
    def stop_collection(self):
        """Signal to stop ongoing collection."""
        self._stop_collection = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sensor status."""
        return {
            "sensor_type": self.sensor_type.value,
            "is_connected": self.is_connected,
            "is_collecting": self.is_collecting,
            "device": self.device.device_id if self.device else None,
        }


class SensorRegistry:
    """Registry for sensor implementations."""
    
    _sensors: Dict[SensorType, Type[BaseSensor]] = {}
    _instances: Dict[SensorType, BaseSensor] = {}
    
    @classmethod
    def register(cls, sensor_class: Type[BaseSensor]):
        """Register a sensor implementation."""
        sensor_type = sensor_class.sensor_type
        cls._sensors[sensor_type] = sensor_class
        logger.debug(f"Registered sensor: {sensor_type.value}")
    
    @classmethod
    def get(cls, sensor_type: SensorType) -> Optional[Type[BaseSensor]]:
        """Get a sensor class by type."""
        return cls._sensors.get(sensor_type)
    
    @classmethod
    def get_instance(cls, sensor_type: SensorType) -> Optional[BaseSensor]:
        """Get or create a sensor instance."""
        if sensor_type not in cls._instances:
            sensor_class = cls.get(sensor_type)
            if sensor_class:
                cls._instances[sensor_type] = sensor_class()
        return cls._instances.get(sensor_type)
    
    @classmethod
    async def detect_all_sensors(cls) -> List[SensorDevice]:
        """Detect all available sensors across all types."""
        all_devices = []
        
        for sensor_type, sensor_class in cls._sensors.items():
            try:
                sensor = cls.get_instance(sensor_type)
                if sensor:
                    devices = await sensor.detect()
                    all_devices.extend(devices)
            except Exception as e:
                logger.error(f"Error detecting {sensor_type.value} sensors: {e}")
        
        return all_devices
    
    @classmethod
    def list_sensors(cls) -> List[Dict[str, Any]]:
        """List all registered sensor types with capabilities."""
        return [
            {
                "type": sensor_type.value,
                "name": sensor_class.sensor_name,
                "description": sensor_class.sensor_description,
                "capabilities": sensor_class.get_capabilities().__dict__,
                "config_options": sensor_class.get_config_options(),
            }
            for sensor_type, sensor_class in cls._sensors.items()
        ]


def register_sensor(cls: Type[BaseSensor]) -> Type[BaseSensor]:
    """Decorator to register a sensor."""
    SensorRegistry.register(cls)
    return cls
