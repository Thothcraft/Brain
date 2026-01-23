"""Camera Sensor Implementation.

Supports image and video capture from connected cameras.
"""

import os
import asyncio
import logging
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
class CameraSensor(BaseSensor):
    """Camera sensor for image and video capture."""
    
    sensor_type = SensorType.CAMERA
    sensor_name = "Camera"
    sensor_description = "Image and video capture from connected cameras"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._camera = None
        self._camera_index = 0
    
    @classmethod
    def get_capabilities(cls) -> SensorCapabilities:
        return SensorCapabilities(
            sensor_type=cls.sensor_type,
            name=cls.sensor_name,
            description=cls.sensor_description,
            supported_modes=[CollectionMode.SINGLE, CollectionMode.INTERVAL, CollectionMode.CONTINUOUS],
            supported_formats=["jpg", "png", "mp4", "avi"],
            max_sample_rate=60.0,
            min_sample_rate=0.1,
            supports_labels=True,
            supports_streaming=True,
            config_options=cls.get_config_options(),
        )
    
    @classmethod
    def get_config_options(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "resolution", "type": "tuple", "default": [640, 480], "options": [[640, 480], [1280, 720], [1920, 1080]], "description": "Image resolution"},
            {"name": "fps", "type": "int", "default": 30, "min": 1, "max": 60, "description": "Frames per second (video)"},
            {"name": "interval_seconds", "type": "float", "default": 5.0, "min": 0.1, "max": 3600, "description": "Interval between captures"},
            {"name": "total_captures", "type": "int", "default": 30, "min": 1, "max": 10000, "description": "Total number of images to capture"},
            {"name": "duration_seconds", "type": "float", "default": 60.0, "min": 1, "max": 3600, "description": "Video duration in seconds"},
            {"name": "output_format", "type": "str", "default": "jpg", "options": ["jpg", "png", "mp4"], "description": "Output format"},
        ]
    
    async def detect(self) -> List[SensorDevice]:
        """Detect available cameras."""
        devices = []
        
        try:
            import cv2
            
            # Check up to 5 camera indices
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Get camera info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    devices.append(SensorDevice(
                        device_id=f"camera_{i}",
                        sensor_type=SensorType.CAMERA,
                        name=f"Camera {i}",
                        description=f"USB/Built-in Camera ({width}x{height} @ {fps}fps)",
                        is_connected=True,
                        capabilities=self.get_capabilities(),
                        metadata={
                            "index": i,
                            "width": width,
                            "height": height,
                            "fps": fps,
                        },
                    ))
                    cap.release()
                else:
                    cap.release()
                    break
        except ImportError:
            logger.warning("OpenCV not installed, camera detection unavailable")
        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")
        
        return devices
    
    async def connect(self, device_id: str) -> bool:
        """Connect to a camera."""
        try:
            import cv2
            
            # Extract camera index from device_id
            index = int(device_id.split("_")[1])
            self._camera = cv2.VideoCapture(index)
            
            if self._camera.isOpened():
                self._camera_index = index
                self.is_connected = True
                self.device = SensorDevice(
                    device_id=device_id,
                    sensor_type=SensorType.CAMERA,
                    name=f"Camera {index}",
                    is_connected=True,
                )
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error connecting to camera: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the camera."""
        if self._camera:
            self._camera.release()
            self._camera = None
        self.is_connected = False
        self.device = None
        return True
    
    async def capture(self, config: CollectionConfig) -> CollectionResult:
        """Capture images or video according to configuration."""
        import cv2
        
        self.is_collecting = True
        self._stop_collection = False
        started_at = datetime.now()
        files_created = []
        errors = []
        
        try:
            if not self._camera or not self._camera.isOpened():
                # Try to connect
                self._camera = cv2.VideoCapture(self._camera_index)
            
            # Set resolution
            if config.resolution:
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            
            os.makedirs(config.output_directory, exist_ok=True)
            
            if config.mode == CollectionMode.CONTINUOUS and config.output_format in ["mp4", "avi"]:
                # Video capture
                files_created, errors = await self._capture_video(config)
            else:
                # Image capture (single or interval)
                files_created, errors = await self._capture_images(config)
            
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Capture error: {e}")
        finally:
            self.is_collecting = False
        
        return CollectionResult(
            success=len(errors) == 0,
            sensor_type=SensorType.CAMERA,
            files_created=files_created,
            total_samples=len(files_created),
            duration_seconds=(datetime.now() - started_at).total_seconds(),
            labels_used=config.labels,
            started_at=started_at,
            completed_at=datetime.now(),
            errors=errors,
        )
    
    async def _capture_images(self, config: CollectionConfig) -> tuple:
        """Capture images at intervals."""
        import cv2
        
        files_created = []
        errors = []
        
        for i in range(config.total_captures):
            if self._stop_collection:
                break
            
            ret, frame = self._camera.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                label = config.labels[i % len(config.labels)] if config.labels else "unlabeled"
                filename = f"{config.filename_prefix}_{label}_{timestamp}.{config.output_format}"
                filepath = os.path.join(config.output_directory, filename)
                
                cv2.imwrite(filepath, frame)
                files_created.append(filepath)
                
                logger.info(f"Captured image {i+1}/{config.total_captures}: {filename}")
            else:
                errors.append(f"Failed to capture frame {i+1}")
            
            if i < config.total_captures - 1:
                await asyncio.sleep(config.interval_seconds)
        
        return files_created, errors
    
    async def _capture_video(self, config: CollectionConfig) -> tuple:
        """Capture video for specified duration."""
        import cv2
        
        files_created = []
        errors = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = config.labels[0] if config.labels else "unlabeled"
        filename = f"{config.filename_prefix}_{label}_{timestamp}.{config.output_format}"
        filepath = os.path.join(config.output_directory, filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if config.output_format == "mp4" else cv2.VideoWriter_fourcc(*'XVID')
        width = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(filepath, fourcc, config.fps, (width, height))
        
        start_time = datetime.now()
        frame_count = 0
        
        while (datetime.now() - start_time).total_seconds() < config.duration_seconds:
            if self._stop_collection:
                break
            
            ret, frame = self._camera.read()
            if ret:
                out.write(frame)
                frame_count += 1
            
            await asyncio.sleep(1.0 / config.fps)
        
        out.release()
        files_created.append(filepath)
        
        logger.info(f"Captured video: {filename} ({frame_count} frames)")
        
        return files_created, errors
