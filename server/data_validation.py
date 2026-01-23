"""Data Validation and Loading System.

This module provides strict data validation for uploaded files:
- Enforces recognized file types (CSI, GeneralCSV, Image, Video)
- Requires metadata files with correct required fields
- Type-specific loaders with statistics reporting
- Automatic metadata extraction and validation

Recognized File Types:
- CSI (csv): 128 numbers in CSI array + RSSI per line
- GeneralCSV (csv): Comma-separated features with optional header
- Image (png, jpeg, jpg, gif, bmp, webp, tiff)
- Video (mp4, avi, mov, mkv, webm)
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# FILE TYPE DEFINITIONS
# ============================================================================

class FileType(str, Enum):
    """Recognized file types for the platform."""
    CSI = "csi"
    GENERAL_CSV = "general_csv"
    IMAGE = "image"
    VIDEO = "video"
    IMU = "imu"  # JSON-based IMU data
    UNKNOWN = "unknown"


# File extension mappings
FILE_TYPE_EXTENSIONS = {
    FileType.CSI: [".csv"],  # CSI files are CSV with specific format
    FileType.GENERAL_CSV: [".csv"],
    FileType.IMAGE: [".png", ".jpeg", ".jpg", ".gif", ".bmp", ".webp", ".tiff", ".heic"],
    FileType.VIDEO: [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"],
    FileType.IMU: [".json", ".jsonl"],
}

# MIME type mappings
MIME_TYPE_MAPPING = {
    "text/csv": [FileType.CSI, FileType.GENERAL_CSV],
    "application/json": [FileType.IMU],
    "image/png": [FileType.IMAGE],
    "image/jpeg": [FileType.IMAGE],
    "image/gif": [FileType.IMAGE],
    "image/webp": [FileType.IMAGE],
    "image/tiff": [FileType.IMAGE],
    "video/mp4": [FileType.VIDEO],
    "video/avi": [FileType.VIDEO],
    "video/quicktime": [FileType.VIDEO],
    "video/x-matroska": [FileType.VIDEO],
    "video/webm": [FileType.VIDEO],
}


# ============================================================================
# METADATA SCHEMA DEFINITIONS
# ============================================================================

@dataclass
class MetadataSchema:
    """Base metadata schema with required fields."""
    # Required fields for all file types
    file_type: str  # Must match FileType enum
    label: str  # Classification label
    description: Optional[str] = None
    
    # Auto-populated fields
    filename: Optional[str] = None
    file_size: Optional[int] = None
    uploaded_at: Optional[str] = None
    file_hash: Optional[str] = None
    
    # Validation status
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class CSIMetadata(MetadataSchema):
    """Metadata schema for CSI files."""
    # CSI-specific required fields
    num_subcarriers: int = 64  # Expected number of subcarriers
    sampling_rate_hz: Optional[float] = None
    antenna_config: Optional[str] = None  # e.g., "1x1", "2x2"
    bandwidth_mhz: Optional[int] = None  # 20, 40, 80, 160
    
    # Auto-extracted statistics
    num_lines: Optional[int] = None
    valid_lines: Optional[int] = None
    bad_lines: Optional[int] = None
    rssi_min: Optional[float] = None
    rssi_max: Optional[float] = None
    rssi_mean: Optional[float] = None
    duration_seconds: Optional[float] = None


@dataclass
class GeneralCSVMetadata(MetadataSchema):
    """Metadata schema for general CSV files."""
    # CSV-specific fields
    has_header: bool = True
    delimiter: str = ","
    num_columns: Optional[int] = None
    column_names: Optional[List[str]] = None
    
    # Auto-extracted statistics
    num_rows: Optional[int] = None
    feature_types: Optional[Dict[str, str]] = None  # column -> type mapping


@dataclass
class ImageMetadata(MetadataSchema):
    """Metadata schema for image files."""
    # Image-specific fields
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    color_mode: Optional[str] = None  # RGB, RGBA, L (grayscale)


@dataclass
class VideoMetadata(MetadataSchema):
    """Metadata schema for video files."""
    # Video-specific fields
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    num_frames: Optional[int] = None
    codec: Optional[str] = None


@dataclass
class IMUMetadata(MetadataSchema):
    """Metadata schema for IMU data files."""
    # IMU-specific fields
    sampling_rate_hz: Optional[float] = None
    axes: List[str] = field(default_factory=lambda: ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
    
    # Auto-extracted statistics
    num_samples: Optional[int] = None
    duration_seconds: Optional[float] = None


# ============================================================================
# VALIDATION RESULT
# ============================================================================

@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    file_type: FileType
    metadata: Optional[MetadataSchema] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "file_type": self.file_type.value,
            "metadata": asdict(self.metadata) if self.metadata else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics,
        }


# ============================================================================
# DATA LOADERS
# ============================================================================

class BaseDataLoader:
    """Base class for type-specific data loaders."""
    
    file_type: FileType = FileType.UNKNOWN
    
    def __init__(self):
        self.statistics: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, content: bytes, filename: str, metadata: Optional[Dict] = None) -> ValidationResult:
        """Validate file content and metadata."""
        raise NotImplementedError
    
    def load(self, content: bytes, config: Optional[Dict] = None) -> Tuple[Any, Dict[str, Any]]:
        """Load and parse file content. Returns (data, statistics)."""
        raise NotImplementedError
    
    def extract_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract metadata from file content."""
        raise NotImplementedError


class CSIDataLoader(BaseDataLoader):
    """Loader for CSI (Channel State Information) data files.
    
    Expected format: CSV with rows containing CSI arrays
    Each row: type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,[csi_data]
    CSI data: 128 complex numbers as [imag1,real1,imag2,real2,...]
    """
    
    file_type = FileType.CSI
    EXPECTED_SUBCARRIERS = 64  # 128 values = 64 complex numbers
    
    def validate(self, content: bytes, filename: str, metadata: Optional[Dict] = None) -> ValidationResult:
        """Validate CSI file format and metadata."""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            text = content.decode('utf-8', errors='ignore').strip()
            lines = text.split('\n')
            
            if len(lines) < 2:
                errors.append("CSI file must have at least a header and one data row")
                return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
            
            # Check header
            header = lines[0].strip()
            expected_header_start = "type,seq,mac,rssi"
            if not header.lower().startswith(expected_header_start.lower()):
                warnings.append(f"Header doesn't match expected CSI format. Expected to start with: {expected_header_start}")
            
            # Parse data rows
            valid_rows = 0
            bad_rows = 0
            rssi_values = []
            csi_lengths = []
            
            for i, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Check for CSI data array
                    if '[' not in line or ']' not in line:
                        bad_rows += 1
                        continue
                    
                    # Extract RSSI (4th column)
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            rssi = float(parts[3])
                            rssi_values.append(rssi)
                        except ValueError:
                            pass
                    
                    # Extract CSI array
                    csi_start = line.index('[')
                    csi_end = line.index(']')
                    csi_str = line[csi_start+1:csi_end]
                    csi_values = [float(x.strip()) for x in csi_str.split(',') if x.strip()]
                    
                    if len(csi_values) < 2:
                        bad_rows += 1
                        continue
                    
                    csi_lengths.append(len(csi_values))
                    valid_rows += 1
                    
                except Exception as e:
                    bad_rows += 1
                    continue
            
            # Compile statistics
            statistics = {
                "total_lines": len(lines) - 1,
                "valid_lines": valid_rows,
                "bad_lines": bad_rows,
                "valid_percentage": (valid_rows / max(1, len(lines) - 1)) * 100,
            }
            
            if rssi_values:
                statistics["rssi_min"] = min(rssi_values)
                statistics["rssi_max"] = max(rssi_values)
                statistics["rssi_mean"] = sum(rssi_values) / len(rssi_values)
            
            if csi_lengths:
                from collections import Counter
                most_common_len = Counter(csi_lengths).most_common(1)[0][0]
                statistics["csi_array_length"] = most_common_len
                statistics["num_subcarriers"] = most_common_len // 2
            
            # Validation checks
            if valid_rows == 0:
                errors.append("No valid CSI data rows found")
            elif valid_rows < 10:
                warnings.append(f"Only {valid_rows} valid rows found, may not be enough for training")
            
            if bad_rows > valid_rows:
                warnings.append(f"More bad rows ({bad_rows}) than valid rows ({valid_rows})")
            
            # Validate metadata if provided
            if metadata:
                required_fields = ["file_type", "label"]
                for field in required_fields:
                    if field not in metadata:
                        errors.append(f"Missing required metadata field: {field}")
                
                if metadata.get("file_type") != "csi":
                    errors.append(f"Metadata file_type must be 'csi', got: {metadata.get('file_type')}")
            
            # Create metadata object
            meta = CSIMetadata(
                file_type="csi",
                label=metadata.get("label", "") if metadata else "",
                description=metadata.get("description") if metadata else None,
                filename=filename,
                file_size=len(content),
                uploaded_at=datetime.utcnow().isoformat(),
                num_lines=statistics.get("total_lines"),
                valid_lines=statistics.get("valid_lines"),
                bad_lines=statistics.get("bad_lines"),
                rssi_min=statistics.get("rssi_min"),
                rssi_max=statistics.get("rssi_max"),
                rssi_mean=statistics.get("rssi_mean"),
                num_subcarriers=statistics.get("num_subcarriers"),
                is_valid=len(errors) == 0,
                validation_errors=errors,
            )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                file_type=self.file_type,
                metadata=meta,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
            )
            
        except Exception as e:
            logger.error(f"CSI validation error: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
    
    def load(self, content: bytes, config: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load CSI data and return numpy array with statistics."""
        config = config or {}
        window_size = config.get("window_size", 1000)
        include_phase = config.get("include_phase", True)
        filter_subcarriers = config.get("filter_subcarriers", True)
        subcarrier_start = config.get("subcarrier_start", 5)
        subcarrier_end = config.get("subcarrier_end", 32)
        
        # Use existing parse_csi_file function
        from server.ml_training import parse_csi_file
        windows, metadata = parse_csi_file(
            content,
            window_size=window_size,
            include_phase=include_phase,
            filter_subcarriers=filter_subcarriers,
            subcarrier_start=subcarrier_start,
            subcarrier_end=subcarrier_end,
        )
        
        if windows:
            data = np.stack(windows, axis=0)
        else:
            data = np.array([])
        
        return data, metadata
    
    def extract_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract metadata from CSI file."""
        result = self.validate(content, filename)
        return result.statistics


class GeneralCSVDataLoader(BaseDataLoader):
    """Loader for general CSV files with comma-separated features."""
    
    file_type = FileType.GENERAL_CSV
    
    def validate(self, content: bytes, filename: str, metadata: Optional[Dict] = None) -> ValidationResult:
        """Validate general CSV file format."""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            text = content.decode('utf-8', errors='ignore').strip()
            lines = text.split('\n')
            
            if len(lines) < 1:
                errors.append("CSV file is empty")
                return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
            
            # Detect delimiter
            import csv
            try:
                dialect = csv.Sniffer().sniff(lines[0])
                delimiter = dialect.delimiter
            except:
                delimiter = ','
            
            # Check for header
            has_header = metadata.get("has_header", True) if metadata else True
            
            # Parse rows
            reader = csv.reader(lines, delimiter=delimiter)
            rows = list(reader)
            
            if len(rows) < 2:
                errors.append("CSV must have at least header and one data row")
                return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
            
            header = rows[0] if has_header else [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows[1:] if has_header else rows
            
            # Validate column consistency
            num_columns = len(header)
            inconsistent_rows = sum(1 for row in data_rows if len(row) != num_columns)
            
            if inconsistent_rows > 0:
                warnings.append(f"{inconsistent_rows} rows have inconsistent column count")
            
            # Detect column types
            column_types = {}
            for i, col in enumerate(header):
                values = [row[i] for row in data_rows if len(row) > i]
                col_type = self._detect_column_type(values)
                column_types[col] = col_type
            
            statistics = {
                "num_rows": len(data_rows),
                "num_columns": num_columns,
                "column_names": header,
                "column_types": column_types,
                "delimiter": delimiter,
                "has_header": has_header,
                "inconsistent_rows": inconsistent_rows,
            }
            
            # Validate metadata
            if metadata:
                required_fields = ["file_type", "label"]
                for field in required_fields:
                    if field not in metadata:
                        errors.append(f"Missing required metadata field: {field}")
                
                if metadata.get("file_type") not in ["general_csv", "csv"]:
                    errors.append(f"Metadata file_type must be 'general_csv' or 'csv'")
            
            meta = GeneralCSVMetadata(
                file_type="general_csv",
                label=metadata.get("label", "") if metadata else "",
                description=metadata.get("description") if metadata else None,
                filename=filename,
                file_size=len(content),
                uploaded_at=datetime.utcnow().isoformat(),
                has_header=has_header,
                delimiter=delimiter,
                num_columns=num_columns,
                column_names=header,
                num_rows=len(data_rows),
                feature_types=column_types,
                is_valid=len(errors) == 0,
                validation_errors=errors,
            )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                file_type=self.file_type,
                metadata=meta,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
            )
            
        except Exception as e:
            logger.error(f"CSV validation error: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
    
    def _detect_column_type(self, values: List[str]) -> str:
        """Detect the type of a column based on sample values."""
        if not values:
            return "unknown"
        
        # Sample up to 100 values
        sample = values[:100]
        
        # Try numeric
        numeric_count = 0
        for v in sample:
            try:
                float(v)
                numeric_count += 1
            except:
                pass
        
        if numeric_count / len(sample) > 0.9:
            # Check if integer
            int_count = sum(1 for v in sample if v.isdigit() or (v.startswith('-') and v[1:].isdigit()))
            if int_count / len(sample) > 0.9:
                return "integer"
            return "float"
        
        # Check for categorical (few unique values)
        unique_ratio = len(set(sample)) / len(sample)
        if unique_ratio < 0.1:
            return "categorical"
        
        return "string"
    
    def load(self, content: bytes, config: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load CSV data as numpy array."""
        import csv
        import io
        
        config = config or {}
        has_header = config.get("has_header", True)
        
        text = content.decode('utf-8', errors='ignore').strip()
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        
        if has_header and rows:
            header = rows[0]
            data_rows = rows[1:]
        else:
            header = [f"col_{i}" for i in range(len(rows[0]))] if rows else []
            data_rows = rows
        
        # Convert to numeric array
        numeric_data = []
        for row in data_rows:
            try:
                numeric_row = [float(v) for v in row]
                numeric_data.append(numeric_row)
            except ValueError:
                continue
        
        data = np.array(numeric_data, dtype=np.float32) if numeric_data else np.array([])
        
        metadata = {
            "num_rows": len(data),
            "num_columns": data.shape[1] if len(data.shape) > 1 else 0,
            "column_names": header,
        }
        
        return data, metadata
    
    def extract_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        result = self.validate(content, filename)
        return result.statistics


class ImageDataLoader(BaseDataLoader):
    """Loader for image files."""
    
    file_type = FileType.IMAGE
    SUPPORTED_EXTENSIONS = [".png", ".jpeg", ".jpg", ".gif", ".bmp", ".webp", ".tiff"]
    
    def validate(self, content: bytes, filename: str, metadata: Optional[Dict] = None) -> ValidationResult:
        """Validate image file."""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            # Check file extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                errors.append(f"Unsupported image format: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")
            
            # Try to read image dimensions
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(content))
                width, height = img.size
                mode = img.mode
                format_name = img.format
                
                statistics = {
                    "width": width,
                    "height": height,
                    "channels": len(mode),
                    "color_mode": mode,
                    "format": format_name,
                    "file_size": len(content),
                }
            except ImportError:
                warnings.append("PIL not available, skipping image dimension extraction")
            except Exception as e:
                errors.append(f"Failed to read image: {str(e)}")
            
            # Validate metadata
            if metadata:
                required_fields = ["file_type", "label"]
                for field in required_fields:
                    if field not in metadata:
                        errors.append(f"Missing required metadata field: {field}")
                
                if metadata.get("file_type") != "image":
                    errors.append(f"Metadata file_type must be 'image'")
            
            meta = ImageMetadata(
                file_type="image",
                label=metadata.get("label", "") if metadata else "",
                description=metadata.get("description") if metadata else None,
                filename=filename,
                file_size=len(content),
                uploaded_at=datetime.utcnow().isoformat(),
                width=statistics.get("width"),
                height=statistics.get("height"),
                channels=statistics.get("channels"),
                format=statistics.get("format"),
                color_mode=statistics.get("color_mode"),
                is_valid=len(errors) == 0,
                validation_errors=errors,
            )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                file_type=self.file_type,
                metadata=meta,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
            )
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
    
    def load(self, content: bytes, config: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load image as numpy array."""
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(content))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            data = np.array(img)
            metadata = {
                "shape": data.shape,
                "dtype": str(data.dtype),
            }
            return data, metadata
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return np.array([]), {"error": str(e)}
    
    def extract_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        result = self.validate(content, filename)
        return result.statistics


class VideoDataLoader(BaseDataLoader):
    """Loader for video files."""
    
    file_type = FileType.VIDEO
    SUPPORTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]
    
    def validate(self, content: bytes, filename: str, metadata: Optional[Dict] = None) -> ValidationResult:
        """Validate video file."""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            # Check file extension
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                errors.append(f"Unsupported video format: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")
            
            # Try to extract video metadata
            try:
                import cv2
                import tempfile
                
                # Write to temp file for cv2
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                    f.write(content)
                    temp_path = f.name
                
                cap = cv2.VideoCapture(temp_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    statistics = {
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "num_frames": frame_count,
                        "duration_seconds": duration,
                        "file_size": len(content),
                    }
                    cap.release()
                else:
                    errors.append("Failed to open video file")
                
                os.unlink(temp_path)
                
            except ImportError:
                warnings.append("OpenCV not available, skipping video metadata extraction")
            except Exception as e:
                warnings.append(f"Could not extract video metadata: {str(e)}")
            
            # Validate metadata
            if metadata:
                required_fields = ["file_type", "label"]
                for field in required_fields:
                    if field not in metadata:
                        errors.append(f"Missing required metadata field: {field}")
                
                if metadata.get("file_type") != "video":
                    errors.append(f"Metadata file_type must be 'video'")
            
            meta = VideoMetadata(
                file_type="video",
                label=metadata.get("label", "") if metadata else "",
                description=metadata.get("description") if metadata else None,
                filename=filename,
                file_size=len(content),
                uploaded_at=datetime.utcnow().isoformat(),
                width=statistics.get("width"),
                height=statistics.get("height"),
                fps=statistics.get("fps"),
                duration_seconds=statistics.get("duration_seconds"),
                num_frames=statistics.get("num_frames"),
                is_valid=len(errors) == 0,
                validation_errors=errors,
            )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                file_type=self.file_type,
                metadata=meta,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
            )
            
        except Exception as e:
            logger.error(f"Video validation error: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
    
    def load(self, content: bytes, config: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load video frames as numpy array."""
        # Video loading is memory-intensive, return metadata only by default
        result = self.validate(content, "video.mp4")
        return np.array([]), result.statistics
    
    def extract_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        result = self.validate(content, filename)
        return result.statistics


class IMUDataLoader(BaseDataLoader):
    """Loader for IMU (Inertial Measurement Unit) data files."""
    
    file_type = FileType.IMU
    
    def validate(self, content: bytes, filename: str, metadata: Optional[Dict] = None) -> ValidationResult:
        """Validate IMU data file."""
        errors = []
        warnings = []
        statistics = {}
        
        try:
            text = content.decode('utf-8', errors='ignore').strip()
            
            # Try to parse as JSON
            samples = []
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    samples = data.get('samples', data.get('data', []))
                elif isinstance(data, list):
                    samples = data
            except json.JSONDecodeError:
                # Try JSONL format
                for line in text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            samples.append(obj)
                    except:
                        continue
            
            if not samples:
                errors.append("No valid IMU samples found")
                return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
            
            # Check for required IMU fields
            required_fields = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            alt_fields = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
            
            sample = samples[0]
            has_required = all(f in sample for f in required_fields)
            has_alt = all(f in sample for f in alt_fields)
            
            if not has_required and not has_alt:
                # Check for nested format
                if 'imu' in sample or 'accel' in sample:
                    pass  # Nested format is acceptable
                else:
                    warnings.append("IMU data may not have standard field names")
            
            statistics = {
                "num_samples": len(samples),
                "sample_fields": list(samples[0].keys()) if samples else [],
            }
            
            # Validate metadata
            if metadata:
                required_meta = ["file_type", "label"]
                for field in required_meta:
                    if field not in metadata:
                        errors.append(f"Missing required metadata field: {field}")
                
                if metadata.get("file_type") != "imu":
                    errors.append(f"Metadata file_type must be 'imu'")
            
            meta = IMUMetadata(
                file_type="imu",
                label=metadata.get("label", "") if metadata else "",
                description=metadata.get("description") if metadata else None,
                filename=filename,
                file_size=len(content),
                uploaded_at=datetime.utcnow().isoformat(),
                num_samples=len(samples),
                is_valid=len(errors) == 0,
                validation_errors=errors,
            )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                file_type=self.file_type,
                metadata=meta,
                errors=errors,
                warnings=warnings,
                statistics=statistics,
            )
            
        except Exception as e:
            logger.error(f"IMU validation error: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(False, self.file_type, None, errors, warnings, statistics)
    
    def load(self, content: bytes, config: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load IMU data as numpy array."""
        from server.ml_training import parse_imu_file
        
        config = config or {}
        window_size = config.get("window_size", 128)
        
        windows = parse_imu_file(content, window_size)
        
        if windows:
            data = np.stack(windows, axis=0)
        else:
            data = np.array([])
        
        metadata = {
            "num_windows": len(windows),
            "window_size": window_size,
        }
        
        return data, metadata
    
    def extract_metadata(self, content: bytes, filename: str) -> Dict[str, Any]:
        result = self.validate(content, filename)
        return result.statistics


# ============================================================================
# DATA LOADER REGISTRY
# ============================================================================

class DataLoaderRegistry:
    """Registry for type-specific data loaders."""
    
    _loaders: Dict[FileType, BaseDataLoader] = {}
    
    @classmethod
    def register(cls, file_type: FileType, loader: BaseDataLoader):
        """Register a loader for a file type."""
        cls._loaders[file_type] = loader
    
    @classmethod
    def get_loader(cls, file_type: FileType) -> Optional[BaseDataLoader]:
        """Get loader for a file type."""
        return cls._loaders.get(file_type)
    
    @classmethod
    def detect_file_type(cls, content: bytes, filename: str) -> FileType:
        """Detect file type from content and filename."""
        ext = os.path.splitext(filename)[1].lower()
        
        # Check by extension first
        if ext in [".png", ".jpeg", ".jpg", ".gif", ".bmp", ".webp", ".tiff", ".heic"]:
            return FileType.IMAGE
        elif ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]:
            return FileType.VIDEO
        elif ext in [".json", ".jsonl"]:
            return FileType.IMU
        elif ext == ".csv":
            # Distinguish between CSI and general CSV
            try:
                text = content.decode('utf-8', errors='ignore')[:2000]
                if 'CSI_DATA' in text or ('[' in text and ']' in text):
                    # Check for CSI array pattern
                    lines = text.split('\n')
                    for line in lines[1:5]:
                        if '[' in line and ']' in line:
                            return FileType.CSI
                return FileType.GENERAL_CSV
            except:
                return FileType.GENERAL_CSV
        
        return FileType.UNKNOWN
    
    @classmethod
    def validate_file(cls, content: bytes, filename: str, metadata: Optional[Dict] = None) -> ValidationResult:
        """Validate a file using the appropriate loader."""
        file_type = cls.detect_file_type(content, filename)
        
        if file_type == FileType.UNKNOWN:
            return ValidationResult(
                is_valid=False,
                file_type=file_type,
                errors=[f"Unrecognized file type for: {filename}"],
            )
        
        loader = cls.get_loader(file_type)
        if not loader:
            return ValidationResult(
                is_valid=False,
                file_type=file_type,
                errors=[f"No loader registered for file type: {file_type.value}"],
            )
        
        return loader.validate(content, filename, metadata)
    
    @classmethod
    def load_file(cls, content: bytes, filename: str, config: Optional[Dict] = None) -> Tuple[Any, Dict[str, Any]]:
        """Load a file using the appropriate loader."""
        file_type = cls.detect_file_type(content, filename)
        loader = cls.get_loader(file_type)
        
        if not loader:
            return None, {"error": f"No loader for file type: {file_type.value}"}
        
        return loader.load(content, config)


# Register default loaders
DataLoaderRegistry.register(FileType.CSI, CSIDataLoader())
DataLoaderRegistry.register(FileType.GENERAL_CSV, GeneralCSVDataLoader())
DataLoaderRegistry.register(FileType.IMAGE, ImageDataLoader())
DataLoaderRegistry.register(FileType.VIDEO, VideoDataLoader())
DataLoaderRegistry.register(FileType.IMU, IMUDataLoader())


# ============================================================================
# METADATA FILE VALIDATION
# ============================================================================

def validate_metadata_file(metadata_content: bytes, data_file_type: FileType) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Validate a metadata JSON file for a data file.
    
    Required fields for all types:
    - file_type: Must match the data file type
    - label: Classification label for the data
    
    Optional fields:
    - description: Human-readable description
    - Additional type-specific fields
    
    Returns:
        Tuple of (is_valid, parsed_metadata, errors)
    """
    errors = []
    
    try:
        metadata = json.loads(metadata_content.decode('utf-8'))
    except json.JSONDecodeError as e:
        return False, {}, [f"Invalid JSON in metadata file: {str(e)}"]
    
    # Check required fields
    required_fields = ["file_type", "label"]
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
    
    # Validate file_type matches
    if "file_type" in metadata:
        expected_types = {
            FileType.CSI: ["csi"],
            FileType.GENERAL_CSV: ["general_csv", "csv"],
            FileType.IMAGE: ["image"],
            FileType.VIDEO: ["video"],
            FileType.IMU: ["imu"],
        }
        
        valid_types = expected_types.get(data_file_type, [])
        if metadata["file_type"] not in valid_types:
            errors.append(f"file_type '{metadata['file_type']}' does not match expected types: {valid_types}")
    
    # Validate label is not empty
    if "label" in metadata and not metadata["label"].strip():
        errors.append("label cannot be empty")
    
    return len(errors) == 0, metadata, errors


def create_metadata_template(file_type: FileType) -> Dict[str, Any]:
    """Create a metadata template for a file type."""
    templates = {
        FileType.CSI: {
            "file_type": "csi",
            "label": "",
            "description": "",
            "num_subcarriers": 64,
            "sampling_rate_hz": None,
            "antenna_config": "1x1",
            "bandwidth_mhz": 20,
        },
        FileType.GENERAL_CSV: {
            "file_type": "general_csv",
            "label": "",
            "description": "",
            "has_header": True,
            "delimiter": ",",
        },
        FileType.IMAGE: {
            "file_type": "image",
            "label": "",
            "description": "",
        },
        FileType.VIDEO: {
            "file_type": "video",
            "label": "",
            "description": "",
        },
        FileType.IMU: {
            "file_type": "imu",
            "label": "",
            "description": "",
            "sampling_rate_hz": None,
        },
    }
    
    return templates.get(file_type, {"file_type": "unknown", "label": "", "description": ""})
