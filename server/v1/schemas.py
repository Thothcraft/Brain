"""Pydantic schemas for the Brain v1 contract (Architecture v3.0 Part III).

These mirror ``whispy.contracts`` - the same shapes validated on the
edge, in Brain, and in every client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Devices & sensors
# ---------------------------------------------------------------------------

class SensorV1(BaseModel):
    id: str
    type: str
    driver: str = ""
    driver_version: str = ""
    sample_rate: Optional[float] = None
    units: Dict[str, str] = Field(default_factory=dict)
    online: bool = True
    capabilities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeviceV1(BaseModel):
    id: str
    stable_uuid: str
    name: str
    owner: Optional[str] = None
    platform: str = ""
    architecture: str = ""
    software_version: str = ""
    whispy_version: str = ""
    online: bool = False
    last_seen: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    sensors: List[SensorV1] = Field(default_factory=list)
    health: Dict[str, Any] = Field(default_factory=dict)


class DeviceListV1(BaseModel):
    devices: List[DeviceV1]
    count: int


class SensorListV1(BaseModel):
    device_id: str
    sensors: List[SensorV1]


# ---------------------------------------------------------------------------
# Samples & streams
# ---------------------------------------------------------------------------

class SensorSampleV1(BaseModel):
    device_id: str
    sensor_id: str
    sensor_type: str
    timestamp: float
    sequence: int
    payload_type: str = "auto"
    payload: Any = None
    sample_rate: Optional[float] = None
    units: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamPageV1(BaseModel):
    """Cursor-paged sample stream response (remote streaming section 9.4)."""

    device_id: str
    sensor_id: str
    samples: List[SensorSampleV1] = Field(default_factory=list)
    cursor: Optional[str] = None
    state: str = "ok"                        # ok | disconnected | error


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class PredictionV1(BaseModel):
    id: str = ""
    device_id: str = ""
    runtime_model_id: str = ""
    timestamp: float = 0.0
    label: str
    confidence: float = 0.0
    scores: Dict[str, float] = Field(default_factory=dict)
    source_window: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PredictionListV1(BaseModel):
    device_id: str
    predictions: List[PredictionV1]


# ---------------------------------------------------------------------------
# Captures
# ---------------------------------------------------------------------------

class CaptureV1(BaseModel):
    id: str
    device_id: str
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    state: str = "active"
    sensors: List[str] = Field(default_factory=list)
    sample_counts: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CaptureListV1(BaseModel):
    captures: List[CaptureV1]


class CaptureStartRequestV1(BaseModel):
    sensors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Models & deployments
# ---------------------------------------------------------------------------

class ModelInputV1(BaseModel):
    sensor: str
    window_seconds: float = 1.0
    required_sample_rate: Optional[float] = None


class ModelManifestV1(BaseModel):
    """``whispy-model/v1`` artifact manifest (section 18).

    Legacy ``thoth-model/v1`` is still accepted by the registration
    endpoint during the rename transition.
    """

    format: str = "whispy-model/v1"
    name: str
    processor: str
    inputs: List[ModelInputV1] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    whispy_version: str = ""
    artifact_sha256: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelV1(BaseModel):
    id: str
    name: str
    processor: str = "torchscript"
    sensor: Optional[str] = None
    task: Optional[str] = None
    visibility: str = "private"
    manifest: Optional[ModelManifestV1] = None
    created_at: Optional[str] = None


class ModelListV1(BaseModel):
    models: List[ModelV1]


class DeploymentV1(BaseModel):
    deployment_id: str
    device_id: str
    model_id: str
    state: str = "queued"                    # queued|received|validated|installed|acknowledged|active|failed
    runtime_model_id: str = ""
    failure: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DeploymentListV1(BaseModel):
    deployments: List[DeploymentV1]


class DeploymentCreateV1(BaseModel):
    model_id: str
    device_id: str


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class AccountV1(BaseModel):
    user_id: int
    username: str
    email: Optional[str] = None
    plan: str = "free"
    entitlements: Dict[str, Any] = Field(default_factory=dict)
