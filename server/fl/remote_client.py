"""Remote FL Client Support for Thoth Devices.

This module enables Thoth devices to participate as Flower clients in federated learning.
It implements the SuperNode architecture from Flower for distributed FL.

Flower Documentation References:
- Docker deployment: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
- SuperNode architecture: SuperNodes connect to SuperLink and execute ClientApps
- Multi-node simulations: https://flower.ai/docs/framework/how-to-run-simulations.html#multi-node-flower-simulations

Architecture:
1. Brain server acts as the FL coordinator (SuperLink equivalent)
2. Thoth devices register as remote FL clients (SuperNode equivalent)
3. When an FL session starts, registered devices can be selected as participants
4. Each device runs a ClientApp that connects back to the Brain server
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceStatus(str, Enum):
    """Status of a remote FL device."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"  # Currently participating in FL
    ERROR = "error"


@dataclass
class RemoteFLDevice:
    """Represents a Thoth device available for FL participation.
    
    Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
    In Flower's architecture, this is equivalent to a SuperNode that can execute ClientApps.
    """
    device_id: str
    device_name: str
    ip_address: str
    port: int = 9094  # Default Flower SuperNode ClientAppIO port
    
    # Device capabilities
    compute_capability: float = 1.0
    available_memory_mb: int = 0
    cpu_cores: int = 1
    has_gpu: bool = False
    gpu_memory_mb: int = 0
    
    # Status
    status: DeviceStatus = DeviceStatus.OFFLINE
    last_heartbeat: Optional[datetime] = None
    
    # Data info
    available_datasets: List[str] = field(default_factory=list)
    data_samples_available: int = 0
    
    # FL participation history
    sessions_participated: int = 0
    total_rounds_completed: int = 0
    avg_contribution_score: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # User association
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "ip_address": self.ip_address,
            "port": self.port,
            "compute_capability": self.compute_capability,
            "available_memory_mb": self.available_memory_mb,
            "cpu_cores": self.cpu_cores,
            "has_gpu": self.has_gpu,
            "gpu_memory_mb": self.gpu_memory_mb,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "available_datasets": self.available_datasets,
            "data_samples_available": self.data_samples_available,
            "sessions_participated": self.sessions_participated,
            "total_rounds_completed": self.total_rounds_completed,
            "avg_contribution_score": self.avg_contribution_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def is_available(self) -> bool:
        """Check if device is available for FL participation."""
        if self.status != DeviceStatus.ONLINE:
            return False
        if self.last_heartbeat is None:
            return False
        # Device must have sent heartbeat in last 5 minutes
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < 300  # 5 minutes


class RemoteDeviceManager:
    """Manages remote Thoth devices for FL participation.
    
    This class handles:
    - Device registration and discovery
    - Heartbeat monitoring
    - Device selection for FL sessions
    - Communication with remote devices
    
    Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
    """
    
    def __init__(self):
        self.devices: Dict[str, RemoteFLDevice] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    def register_device(
        self,
        device_id: str,
        device_name: str,
        ip_address: str,
        port: int = 9094,
        compute_capability: float = 1.0,
        available_memory_mb: int = 0,
        cpu_cores: int = 1,
        has_gpu: bool = False,
        gpu_memory_mb: int = 0,
        available_datasets: List[str] = None,
        data_samples_available: int = 0,
        user_id: Optional[str] = None,
    ) -> RemoteFLDevice:
        """Register a new remote device for FL participation.
        
        Args:
            device_id: Unique device identifier
            device_name: Human-readable device name
            ip_address: IP address of the device
            port: Port for Flower ClientAppIO (default 9094)
            compute_capability: Relative compute power (1.0 = baseline)
            available_memory_mb: Available RAM in MB
            cpu_cores: Number of CPU cores
            has_gpu: Whether device has GPU
            gpu_memory_mb: GPU memory in MB
            available_datasets: List of dataset names available on device
            data_samples_available: Number of data samples on device
            user_id: Optional user ID for ownership
        
        Returns:
            Registered RemoteFLDevice instance
        """
        device = RemoteFLDevice(
            device_id=device_id,
            device_name=device_name,
            ip_address=ip_address,
            port=port,
            compute_capability=compute_capability,
            available_memory_mb=available_memory_mb,
            cpu_cores=cpu_cores,
            has_gpu=has_gpu,
            gpu_memory_mb=gpu_memory_mb,
            available_datasets=available_datasets or [],
            data_samples_available=data_samples_available,
            user_id=user_id,
            status=DeviceStatus.ONLINE,
            last_heartbeat=datetime.now(),
        )
        
        self.devices[device_id] = device
        logger.info(f"[FL Remote] Registered device: {device_name} ({device_id}) at {ip_address}:{port}")
        
        return device
    
    def update_heartbeat(self, device_id: str) -> bool:
        """Update device heartbeat timestamp.
        
        Args:
            device_id: Device identifier
        
        Returns:
            True if device found and updated, False otherwise
        """
        device = self.devices.get(device_id)
        if not device:
            return False
        
        device.last_heartbeat = datetime.now()
        device.updated_at = datetime.now()
        if device.status == DeviceStatus.OFFLINE:
            device.status = DeviceStatus.ONLINE
            logger.info(f"[FL Remote] Device {device_id} came online")
        
        return True
    
    def update_device_status(self, device_id: str, status: DeviceStatus) -> bool:
        """Update device status.
        
        Args:
            device_id: Device identifier
            status: New status
        
        Returns:
            True if device found and updated, False otherwise
        """
        device = self.devices.get(device_id)
        if not device:
            return False
        
        old_status = device.status
        device.status = status
        device.updated_at = datetime.now()
        
        if old_status != status:
            logger.info(f"[FL Remote] Device {device_id} status: {old_status.value} -> {status.value}")
        
        return True
    
    def get_device(self, device_id: str) -> Optional[RemoteFLDevice]:
        """Get device by ID."""
        return self.devices.get(device_id)
    
    def list_devices(
        self,
        status: Optional[DeviceStatus] = None,
        min_capability: float = 0.0,
        has_dataset: Optional[str] = None,
    ) -> List[RemoteFLDevice]:
        """List devices with optional filtering.
        
        Args:
            status: Filter by status
            min_capability: Minimum compute capability
            has_dataset: Filter by dataset availability
        
        Returns:
            List of matching devices
        """
        devices = list(self.devices.values())
        
        if status is not None:
            devices = [d for d in devices if d.status == status]
        
        if min_capability > 0:
            devices = [d for d in devices if d.compute_capability >= min_capability]
        
        if has_dataset:
            devices = [d for d in devices if has_dataset in d.available_datasets]
        
        return devices
    
    def get_available_devices(
        self,
        min_capability: float = 0.0,
        required_dataset: Optional[str] = None,
        min_samples: int = 0,
    ) -> List[RemoteFLDevice]:
        """Get devices available for FL participation.
        
        Args:
            min_capability: Minimum compute capability required
            required_dataset: Dataset that must be available
            min_samples: Minimum data samples required
        
        Returns:
            List of available devices sorted by capability
        """
        available = []
        
        for device in self.devices.values():
            if not device.is_available():
                continue
            if device.compute_capability < min_capability:
                continue
            if required_dataset and required_dataset not in device.available_datasets:
                continue
            if device.data_samples_available < min_samples:
                continue
            available.append(device)
        
        # Sort by capability (highest first)
        available.sort(key=lambda d: d.compute_capability, reverse=True)
        
        return available
    
    def select_devices_for_session(
        self,
        num_devices: int,
        required_dataset: Optional[str] = None,
        min_capability: float = 0.0,
        selection_strategy: str = "capability",  # capability, random, round_robin
    ) -> List[RemoteFLDevice]:
        """Select devices for an FL session.
        
        Args:
            num_devices: Number of devices to select
            required_dataset: Dataset that must be available
            min_capability: Minimum compute capability
            selection_strategy: How to select devices
        
        Returns:
            List of selected devices
        """
        available = self.get_available_devices(
            min_capability=min_capability,
            required_dataset=required_dataset,
        )
        
        if len(available) < num_devices:
            logger.warning(f"[FL Remote] Only {len(available)} devices available, requested {num_devices}")
        
        if selection_strategy == "random":
            import random
            random.shuffle(available)
        elif selection_strategy == "round_robin":
            # Sort by sessions participated (least first)
            available.sort(key=lambda d: d.sessions_participated)
        # Default "capability" is already sorted by capability
        
        selected = available[:num_devices]
        
        # Mark selected devices as busy
        for device in selected:
            device.status = DeviceStatus.BUSY
        
        logger.info(f"[FL Remote] Selected {len(selected)} devices for FL session")
        return selected
    
    def release_devices(self, device_ids: List[str]) -> None:
        """Release devices after FL session completion.
        
        Args:
            device_ids: List of device IDs to release
        """
        for device_id in device_ids:
            device = self.devices.get(device_id)
            if device and device.status == DeviceStatus.BUSY:
                device.status = DeviceStatus.ONLINE
                logger.info(f"[FL Remote] Released device {device_id}")
    
    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device.
        
        Args:
            device_id: Device identifier
        
        Returns:
            True if device was removed, False if not found
        """
        if device_id in self.devices:
            del self.devices[device_id]
            logger.info(f"[FL Remote] Unregistered device {device_id}")
            return True
        return False
    
    async def check_stale_devices(self, timeout_seconds: int = 300) -> List[str]:
        """Check for and mark stale devices as offline.
        
        Args:
            timeout_seconds: Seconds since last heartbeat to consider stale
        
        Returns:
            List of device IDs marked as offline
        """
        stale_devices = []
        now = datetime.now()
        
        for device_id, device in self.devices.items():
            if device.status in [DeviceStatus.ONLINE, DeviceStatus.BUSY]:
                if device.last_heartbeat:
                    time_since = (now - device.last_heartbeat).total_seconds()
                    if time_since > timeout_seconds:
                        device.status = DeviceStatus.OFFLINE
                        stale_devices.append(device_id)
                        logger.warning(f"[FL Remote] Device {device_id} marked offline (no heartbeat for {time_since:.0f}s)")
        
        return stale_devices
    
    def get_device_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics about registered devices."""
        total = len(self.devices)
        online = sum(1 for d in self.devices.values() if d.status == DeviceStatus.ONLINE)
        busy = sum(1 for d in self.devices.values() if d.status == DeviceStatus.BUSY)
        offline = sum(1 for d in self.devices.values() if d.status == DeviceStatus.OFFLINE)
        
        total_samples = sum(d.data_samples_available for d in self.devices.values())
        total_capability = sum(d.compute_capability for d in self.devices.values())
        gpu_devices = sum(1 for d in self.devices.values() if d.has_gpu)
        
        return {
            "total_devices": total,
            "online": online,
            "busy": busy,
            "offline": offline,
            "total_data_samples": total_samples,
            "total_compute_capability": total_capability,
            "gpu_devices": gpu_devices,
            "available_for_fl": sum(1 for d in self.devices.values() if d.is_available()),
        }


# Global instance
remote_device_manager = RemoteDeviceManager()


def generate_client_script(
    device_id: str,
    server_address: str,
    dataset: str,
    partition_id: int,
    num_partitions: int,
) -> str:
    """Generate a Python script for a Thoth device to run as an FL client.
    
    This script can be executed on a Thoth device to participate in FL.
    
    Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
    
    Args:
        device_id: Unique device identifier
        server_address: Address of the FL server (Brain)
        dataset: Dataset to use for training
        partition_id: Data partition ID for this client
        num_partitions: Total number of partitions
    
    Returns:
        Python script as a string
    """
    script = f'''#!/usr/bin/env python3
"""Thoth FL Client Script - Auto-generated
Device ID: {device_id}
Server: {server_address}
Dataset: {dataset}
Partition: {partition_id}/{num_partitions}

Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
"""

import flwr as fl
from flwr.client import NumPyClient
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

# Import your model and data loading functions
# These should be available on the Thoth device
from thoth_fl_utils import load_local_data, get_model, train_model, evaluate_model

DEVICE_ID = "{device_id}"
SERVER_ADDRESS = "{server_address}"
DATASET = "{dataset}"
PARTITION_ID = {partition_id}
NUM_PARTITIONS = {num_partitions}


class ThothFlowerClient(NumPyClient):
    """Flower client for Thoth device."""
    
    def __init__(self, model, trainloader, valloader, device):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({{k: torch.tensor(v) for k, v in params_dict}})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = config.get("local_epochs", 5)
        lr = config.get("learning_rate", 0.01)
        
        loss, num_samples = train_model(
            self.model, self.trainloader, epochs, lr, self.device
        )
        
        return self.get_parameters(config), num_samples, {{"train_loss": loss}}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, num_samples = evaluate_model(
            self.model, self.valloader, self.device
        )
        return loss, num_samples, {{"accuracy": accuracy}}


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Thoth FL] Device: {{DEVICE_ID}}, using {{device}}")
    
    # Load local data
    trainloader, valloader = load_local_data(
        dataset=DATASET,
        partition_id=PARTITION_ID,
        num_partitions=NUM_PARTITIONS,
    )
    
    # Create model
    model = get_model(DATASET)
    
    # Create client
    client = ThothFlowerClient(model, trainloader, valloader, device)
    
    # Start Flower client
    # Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
    print(f"[Thoth FL] Connecting to {{SERVER_ADDRESS}}...")
    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
'''
    return script
