"""Data Collector - Unified interface for sensor data collection.

Provides a high-level interface for collecting data from multiple sensors
with advanced configuration options.
"""

import os
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .base import (
    SensorRegistry,
    SensorType,
    SensorDevice,
    CollectionConfig,
    CollectionResult,
    CollectionMode,
)

logger = logging.getLogger(__name__)


@dataclass
class CollectionJob:
    """Represents a data collection job."""
    job_id: str
    sensor_type: SensorType
    device_id: str
    config: CollectionConfig
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Optional[CollectionResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class DataCollector:
    """Unified data collector for all sensor types.
    
    Provides:
    - Device detection and management
    - Configurable collection jobs
    - Multi-sensor simultaneous collection
    - Progress tracking and callbacks
    """
    
    def __init__(self):
        self.jobs: Dict[str, CollectionJob] = {}
        self._job_counter = 0
    
    async def detect_sensors(self) -> List[SensorDevice]:
        """Detect all available sensors."""
        return await SensorRegistry.detect_all_sensors()
    
    def list_sensor_types(self) -> List[Dict[str, Any]]:
        """List all supported sensor types with capabilities."""
        return SensorRegistry.list_sensors()
    
    def get_config_options(self, sensor_type: str) -> List[Dict[str, Any]]:
        """Get configuration options for a sensor type."""
        try:
            st = SensorType(sensor_type)
            sensor_class = SensorRegistry.get(st)
            if sensor_class:
                return sensor_class.get_config_options()
        except ValueError:
            pass
        return []
    
    def create_collection_job(
        self,
        sensor_type: str,
        device_id: str,
        config: Dict[str, Any],
    ) -> CollectionJob:
        """Create a new collection job.
        
        Args:
            sensor_type: Type of sensor (camera, csi, imu, etc.)
            device_id: ID of the device to use
            config: Collection configuration
            
        Returns:
            CollectionJob instance
        """
        self._job_counter += 1
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._job_counter}"
        
        # Parse sensor type
        try:
            st = SensorType(sensor_type)
        except ValueError:
            st = SensorType.GENERIC
        
        # Create config
        collection_config = CollectionConfig(
            mode=CollectionMode(config.get("mode", "interval")),
            labels=config.get("labels", []),
            output_format=config.get("output_format", "auto"),
            interval_seconds=config.get("interval_seconds", 5.0),
            total_captures=config.get("total_captures", 30),
            duration_seconds=config.get("duration_seconds", 60.0),
            sample_rate_hz=config.get("sample_rate_hz", 100.0),
            output_directory=config.get("output_directory", f"./data/{sensor_type}"),
            filename_prefix=config.get("filename_prefix", "capture"),
            resolution=tuple(config.get("resolution", [640, 480])),
            fps=config.get("fps", 30),
            audio_channels=config.get("audio_channels", 1),
            audio_sample_rate=config.get("audio_sample_rate", 44100),
        )
        
        job = CollectionJob(
            job_id=job_id,
            sensor_type=st,
            device_id=device_id,
            config=collection_config,
        )
        
        self.jobs[job_id] = job
        return job
    
    async def start_collection(self, job_id: str) -> CollectionResult:
        """Start a collection job.
        
        Args:
            job_id: ID of the job to start
            
        Returns:
            CollectionResult when complete
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.status == "running":
            raise ValueError(f"Job already running: {job_id}")
        
        job.status = "running"
        job.started_at = datetime.now()
        
        try:
            # Get sensor instance
            sensor = SensorRegistry.get_instance(job.sensor_type)
            if not sensor:
                raise ValueError(f"Sensor not available: {job.sensor_type.value}")
            
            # Connect to device
            connected = await sensor.connect(job.device_id)
            if not connected:
                raise ValueError(f"Failed to connect to device: {job.device_id}")
            
            # Run collection
            result = await sensor.capture(job.config)
            
            # Disconnect
            await sensor.disconnect()
            
            job.result = result
            job.status = "completed" if result.success else "failed"
            job.completed_at = datetime.now()
            
            return result
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Collection job {job_id} failed: {e}")
            
            return CollectionResult(
                success=False,
                sensor_type=job.sensor_type,
                errors=[str(e)],
                started_at=job.started_at,
                completed_at=job.completed_at,
            )
    
    async def stop_collection(self, job_id: str) -> bool:
        """Stop a running collection job."""
        job = self.jobs.get(job_id)
        if not job or job.status != "running":
            return False
        
        sensor = SensorRegistry.get_instance(job.sensor_type)
        if sensor:
            sensor.stop_collection()
        
        job.status = "cancelled"
        job.completed_at = datetime.now()
        return True
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a collection job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "sensor_type": job.sensor_type.value,
            "device_id": job.device_id,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error,
            "result": {
                "success": job.result.success,
                "files_created": job.result.files_created,
                "total_samples": job.result.total_samples,
            } if job.result else None,
        }
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all collection jobs."""
        jobs = []
        for job in self.jobs.values():
            if status is None or job.status == status:
                jobs.append(self.get_job_status(job.job_id))
        return jobs


# Global collector instance
_collector: Optional[DataCollector] = None


def get_data_collector() -> DataCollector:
    """Get the global data collector instance."""
    global _collector
    if _collector is None:
        _collector = DataCollector()
    return _collector


def create_data_collector(
    sensor_type: str,
    config: Dict[str, Any],
    device_id: Optional[str] = None,
) -> CollectionJob:
    """Convenience function to create a data collection job.
    
    Args:
        sensor_type: Type of sensor
        config: Collection configuration
        device_id: Optional device ID (auto-detect if not provided)
        
    Returns:
        CollectionJob instance
    """
    collector = get_data_collector()
    
    # Auto-detect device if not provided
    if device_id is None:
        device_id = f"{sensor_type}_auto"
    
    return collector.create_collection_job(sensor_type, device_id, config)
