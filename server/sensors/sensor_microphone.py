"""Microphone Sensor Implementation.

Supports audio capture from connected microphones.
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
class MicrophoneSensor(BaseSensor):
    """Microphone sensor for audio capture."""
    
    sensor_type = SensorType.MICROPHONE
    sensor_name = "Microphone"
    sensor_description = "Audio capture from connected microphones"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._device_index = None
    
    @classmethod
    def get_capabilities(cls) -> SensorCapabilities:
        return SensorCapabilities(
            sensor_type=cls.sensor_type,
            name=cls.sensor_name,
            description=cls.sensor_description,
            supported_modes=[CollectionMode.CONTINUOUS, CollectionMode.INTERVAL],
            supported_formats=["wav", "mp3"],
            max_sample_rate=48000.0,
            min_sample_rate=8000.0,
            supports_labels=True,
            supports_streaming=True,
            config_options=cls.get_config_options(),
        )
    
    @classmethod
    def get_config_options(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "duration_seconds", "type": "float", "default": 60.0, "description": "Recording duration"},
            {"name": "audio_sample_rate", "type": "int", "default": 44100, "options": [8000, 16000, 22050, 44100, 48000], "description": "Audio sample rate"},
            {"name": "audio_channels", "type": "int", "default": 1, "options": [1, 2], "description": "Audio channels (mono/stereo)"},
            {"name": "output_format", "type": "str", "default": "wav", "options": ["wav", "mp3"], "description": "Output format"},
        ]
    
    async def detect(self) -> List[SensorDevice]:
        """Detect available microphones."""
        devices = []
        try:
            import sounddevice as sd
            device_list = sd.query_devices()
            
            for i, dev in enumerate(device_list):
                if dev['max_input_channels'] > 0:
                    devices.append(SensorDevice(
                        device_id=f"mic_{i}",
                        sensor_type=SensorType.MICROPHONE,
                        name=dev['name'],
                        description=f"{dev['max_input_channels']} channels, {dev['default_samplerate']}Hz",
                        is_connected=True,
                        capabilities=self.get_capabilities(),
                        metadata={
                            "index": i,
                            "channels": dev['max_input_channels'],
                            "sample_rate": dev['default_samplerate'],
                        },
                    ))
        except ImportError:
            logger.warning("sounddevice not installed")
        except Exception as e:
            logger.error(f"Error detecting microphones: {e}")
        return devices
    
    async def connect(self, device_id: str) -> bool:
        try:
            self._device_index = int(device_id.split("_")[1])
            self.is_connected = True
            self.device = SensorDevice(
                device_id=device_id,
                sensor_type=SensorType.MICROPHONE,
                name=f"Microphone {self._device_index}",
                is_connected=True,
            )
            return True
        except Exception as e:
            logger.error(f"Error connecting to microphone: {e}")
            return False
    
    async def disconnect(self) -> bool:
        self._device_index = None
        self.is_connected = False
        return True
    
    async def capture(self, config: CollectionConfig) -> CollectionResult:
        self.is_collecting = True
        self._stop_collection = False
        started_at = datetime.now()
        files_created = []
        errors = []
        
        try:
            import sounddevice as sd
            import numpy as np
            from scipy.io import wavfile
            
            os.makedirs(config.output_directory, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label = config.labels[0] if config.labels else "unlabeled"
            filename = f"audio_{label}_{timestamp}.wav"
            filepath = os.path.join(config.output_directory, filename)
            
            # Record audio
            duration = config.duration_seconds
            sample_rate = config.audio_sample_rate
            channels = config.audio_channels
            
            logger.info(f"Recording {duration}s of audio at {sample_rate}Hz...")
            
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                device=self._device_index,
                dtype=np.int16,
            )
            
            # Wait for recording with stop check
            elapsed = 0
            while elapsed < duration and not self._stop_collection:
                await asyncio.sleep(0.1)
                elapsed += 0.1
            
            if self._stop_collection:
                sd.stop()
            else:
                sd.wait()
            
            # Save audio
            wavfile.write(filepath, sample_rate, recording)
            files_created.append(filepath)
            
            logger.info(f"Saved audio: {filename}")
            
        except ImportError:
            errors.append("sounddevice or scipy not installed")
        except Exception as e:
            errors.append(str(e))
        finally:
            self.is_collecting = False
        
        return CollectionResult(
            success=len(errors) == 0,
            sensor_type=SensorType.MICROPHONE,
            files_created=files_created,
            total_samples=1 if files_created else 0,
            duration_seconds=(datetime.now() - started_at).total_seconds(),
            labels_used=config.labels,
            started_at=started_at,
            completed_at=datetime.now(),
            errors=errors,
        )
