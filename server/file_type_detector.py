"""File Type Detection Module.

This module provides content-based file type detection using:
1. File extension
2. First-line/header analysis
3. Content pattern matching

Supports detection of:
- CSI (WiFi Channel State Information) CSV files
- General CSV files with time-dependent features
- IMU (JSON/JSONL) files
- Image files
- Video files

CSI files are identified by their specific header:
type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data

The 'data' column contains 128 numbers representing CSI values.
"""

import os
import json
import hashlib
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# CSI header columns (exact match for CSI detection)
CSI_HEADER_COLUMNS = [
    "type", "seq", "mac", "rssi", "rate", "noise_floor", "fft_gain",
    "agc_gain", "channel", "local_timestamp", "sig_len", "rx_state",
    "len", "first_word", "data"
]
CSI_HEADER_START = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"
CSI_DATA_ARRAY_LENGTH = 128  # Expected number of values in CSI data array
MINUTE_DIR_RE = re.compile(r"^\d{8}_\d{4}$")


class DetectedFileType(str, Enum):
    """Detected file types."""
    CSI = "csi"
    GENERAL_CSV = "general_csv"
    IMU = "imu"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    NUMPY = "numpy"
    UNKNOWN = "unknown"


@dataclass
class FileTypeDetectionResult:
    """Result of file type detection."""
    detected_type: DetectedFileType
    confidence: float  # 0.0 to 1.0
    detection_method: str  # "extension", "header", "content", "combined"
    
    # For CSV files
    header_columns: Optional[List[str]] = None
    delimiter: str = ","
    has_header: bool = True
    
    # For CSI files specifically
    is_csi: bool = False
    csi_data_column: Optional[str] = None
    csi_array_length: Optional[int] = None
    
    # Statistics from content analysis
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Errors/warnings during detection
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_type": self.detected_type.value,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "header_columns": self.header_columns,
            "delimiter": self.delimiter,
            "has_header": self.has_header,
            "is_csi": self.is_csi,
            "csi_data_column": self.csi_data_column,
            "csi_array_length": self.csi_array_length,
            "statistics": self.statistics,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# Extension mappings
EXTENSION_TO_TYPE = {
    # CSV types (need content analysis to distinguish)
    ".csv": [DetectedFileType.CSI, DetectedFileType.GENERAL_CSV],
    
    # JSON types
    ".json": [DetectedFileType.IMU],
    ".jsonl": [DetectedFileType.IMU],
    
    # Image types
    ".png": [DetectedFileType.IMAGE],
    ".jpg": [DetectedFileType.IMAGE],
    ".jpeg": [DetectedFileType.IMAGE],
    ".gif": [DetectedFileType.IMAGE],
    ".bmp": [DetectedFileType.IMAGE],
    ".webp": [DetectedFileType.IMAGE],
    ".tiff": [DetectedFileType.IMAGE],
    ".heic": [DetectedFileType.IMAGE],
    
    # Video types
    ".mp4": [DetectedFileType.VIDEO],
    ".avi": [DetectedFileType.VIDEO],
    ".mov": [DetectedFileType.VIDEO],
    ".mkv": [DetectedFileType.VIDEO],
    ".webm": [DetectedFileType.VIDEO],
    ".m4v": [DetectedFileType.VIDEO],
    
    # Audio types
    ".wav": [DetectedFileType.AUDIO],
    ".mp3": [DetectedFileType.AUDIO],
    ".ogg": [DetectedFileType.AUDIO],
    ".flac": [DetectedFileType.AUDIO],
    
    # NumPy types
    ".npy": [DetectedFileType.NUMPY],
    ".npz": [DetectedFileType.NUMPY],
}


def detect_file_type(
    content: bytes,
    filename: str,
    read_bytes: int = 8192
) -> FileTypeDetectionResult:
    """Detect file type based on extension and content analysis.
    
    This function does NOT use filename conventions (like csi_*, imu_*).
    Instead, it uses:
    1. File extension to narrow down possibilities
    2. First-line/header analysis for CSV files
    3. Content pattern matching for JSON files
    
    Args:
        content: File content bytes
        filename: Filename (used for extension only)
        read_bytes: Number of bytes to read for analysis
        
    Returns:
        FileTypeDetectionResult with detected type and metadata
    """
    ext = os.path.splitext(filename)[1].lower()
    
    # Get possible types from extension
    possible_types = EXTENSION_TO_TYPE.get(ext, [DetectedFileType.UNKNOWN])
    
    # If extension uniquely identifies the type (non-CSV, non-JSON)
    if len(possible_types) == 1 and possible_types[0] not in [
        DetectedFileType.CSI, DetectedFileType.GENERAL_CSV, DetectedFileType.IMU
    ]:
        return FileTypeDetectionResult(
            detected_type=possible_types[0],
            confidence=0.9,
            detection_method="extension",
        )
    
    # For CSV files, analyze content
    if ext == ".csv":
        return _detect_csv_type(content, filename, read_bytes)
    
    # For JSON files, analyze content
    if ext in [".json", ".jsonl"]:
        return _detect_json_type(content, filename, read_bytes)
    
    # Unknown extension
    return FileTypeDetectionResult(
        detected_type=DetectedFileType.UNKNOWN,
        confidence=0.0,
        detection_method="extension",
        errors=[f"Unknown file extension: {ext}"],
    )


def _detect_csv_type(
    content: bytes,
    filename: str,
    read_bytes: int = 8192
) -> FileTypeDetectionResult:
    """Detect CSV file type by analyzing the first line (header).
    
    CSI files have a specific header:
    type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data
    
    General CSV files have arbitrary headers where each column is a feature.
    """
    errors = []
    warnings = []
    statistics = {}
    
    try:
        # Decode content
        text = content[:read_bytes].decode('utf-8', errors='ignore')
        text = text.lstrip('\ufeff').strip()  # Remove BOM if present
        
        if not text:
            return FileTypeDetectionResult(
                detected_type=DetectedFileType.UNKNOWN,
                confidence=0.0,
                detection_method="content",
                errors=["Empty file content"],
            )
        
        lines = text.split('\n')
        if not lines:
            return FileTypeDetectionResult(
                detected_type=DetectedFileType.UNKNOWN,
                confidence=0.0,
                detection_method="content",
                errors=["No lines found in file"],
            )
        
        # Get first line (header)
        first_line = lines[0].strip()
        
        # Detect delimiter
        import csv
        try:
            dialect = csv.Sniffer().sniff(first_line)
            delimiter = dialect.delimiter
        except:
            delimiter = ','
        
        # Parse header columns
        header_columns = [col.strip().lower() for col in first_line.split(delimiter)]
        
        # Check if this is a CSI file by matching header
        is_csi = _is_csi_header(header_columns)
        
        if is_csi:
            # Validate CSI data by checking a few data rows
            csi_stats = _analyze_csi_content(content, lines, delimiter)
            statistics.update(csi_stats)
            
            return FileTypeDetectionResult(
                detected_type=DetectedFileType.CSI,
                confidence=0.95 if csi_stats.get("valid_csi_rows", 0) > 0 else 0.7,
                detection_method="header",
                header_columns=header_columns,
                delimiter=delimiter,
                has_header=True,
                is_csi=True,
                csi_data_column="data",
                csi_array_length=csi_stats.get("csi_array_length"),
                statistics=statistics,
                errors=errors,
                warnings=warnings,
            )
        else:
            # General CSV - analyze columns as features
            csv_stats = _analyze_general_csv(content, lines, header_columns, delimiter)
            statistics.update(csv_stats)
            
            return FileTypeDetectionResult(
                detected_type=DetectedFileType.GENERAL_CSV,
                confidence=0.85,
                detection_method="header",
                header_columns=header_columns,
                delimiter=delimiter,
                has_header=True,
                is_csi=False,
                statistics=statistics,
                errors=errors,
                warnings=warnings,
            )
            
    except Exception as e:
        logger.error(f"Error detecting CSV type: {e}")
        return FileTypeDetectionResult(
            detected_type=DetectedFileType.UNKNOWN,
            confidence=0.0,
            detection_method="content",
            errors=[f"Detection failed: {str(e)}"],
        )


def _is_csi_header(header_columns: List[str]) -> bool:
    """Check if header columns match CSI format.
    
    CSI header must start with: type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data
    """
    expected_columns = [col.lower() for col in CSI_HEADER_COLUMNS]
    
    # Check if all expected columns are present
    if len(header_columns) < len(expected_columns):
        return False
    
    # Check first N columns match exactly
    for i, expected in enumerate(expected_columns):
        if i >= len(header_columns):
            return False
        if header_columns[i] != expected:
            return False
    
    return True


def _analyze_csi_content(
    content: bytes,
    lines: List[str],
    delimiter: str
) -> Dict[str, Any]:
    """Analyze CSI file content to extract statistics.
    
    CSI data format:
    - Each row has metadata columns + a 'data' column
    - The 'data' column contains an array like [imag1,real1,imag2,real2,...] with 128 numbers
    """
    stats = {
        "total_lines": len(lines) - 1,  # Exclude header
        "valid_csi_rows": 0,
        "invalid_csi_rows": 0,
        "csi_array_length": None,
        "rssi_values": [],
    }
    
    csi_lengths = []
    
    # Analyze up to 100 data rows for statistics
    for line in lines[1:101]:
        line = line.strip()
        if not line:
            continue
        
        try:
            # Check for CSI data array (enclosed in brackets)
            if '[' not in line or ']' not in line:
                stats["invalid_csi_rows"] += 1
                continue
            
            # Extract RSSI (4th column, index 3)
            parts = line.split(delimiter)
            if len(parts) >= 4:
                try:
                    rssi = float(parts[3])
                    stats["rssi_values"].append(rssi)
                except ValueError:
                    pass
            
            # Extract and validate CSI array
            csi_start = line.index('[')
            csi_end = line.index(']')
            csi_str = line[csi_start+1:csi_end]
            csi_values = [float(x.strip()) for x in csi_str.split(',') if x.strip()]
            
            if len(csi_values) >= 2:
                csi_lengths.append(len(csi_values))
                stats["valid_csi_rows"] += 1
            else:
                stats["invalid_csi_rows"] += 1
                
        except Exception:
            stats["invalid_csi_rows"] += 1
    
    # Determine most common CSI array length
    if csi_lengths:
        from collections import Counter
        most_common = Counter(csi_lengths).most_common(1)[0][0]
        stats["csi_array_length"] = most_common
        stats["num_subcarriers"] = most_common // 2
    
    # Calculate RSSI statistics
    if stats["rssi_values"]:
        stats["rssi_min"] = min(stats["rssi_values"])
        stats["rssi_max"] = max(stats["rssi_values"])
        stats["rssi_mean"] = sum(stats["rssi_values"]) / len(stats["rssi_values"])
    
    # Remove raw values from stats (keep only aggregates)
    del stats["rssi_values"]
    
    return stats


def _analyze_general_csv(
    content: bytes,
    lines: List[str],
    header_columns: List[str],
    delimiter: str
) -> Dict[str, Any]:
    """Analyze general CSV file to extract column types and statistics.
    
    Each column is treated as a feature. Column types are auto-detected:
    - numeric (float/int)
    - categorical
    - timestamp
    - string
    """
    import csv
    import io
    
    stats = {
        "num_columns": len(header_columns),
        "column_names": header_columns,
        "column_types": {},
        "num_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
    }
    
    # Parse data rows
    try:
        text = content.decode('utf-8', errors='ignore')
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)
        
        if len(rows) > 1:
            data_rows = rows[1:]
            stats["num_rows"] = len(data_rows)
            
            # Analyze column types
            for i, col_name in enumerate(header_columns):
                values = [row[i] for row in data_rows if len(row) > i][:100]
                col_type = _detect_column_type(values)
                stats["column_types"][col_name] = col_type
            
            # Count valid/invalid rows
            expected_cols = len(header_columns)
            for row in data_rows:
                if len(row) == expected_cols:
                    stats["valid_rows"] += 1
                else:
                    stats["invalid_rows"] += 1
                    
    except Exception as e:
        logger.warning(f"Error analyzing CSV: {e}")
    
    return stats


def _detect_column_type(values: List[str]) -> str:
    """Detect the type of a column based on sample values."""
    if not values:
        return "unknown"
    
    sample = [v.strip() for v in values[:100] if v.strip()]
    if not sample:
        return "unknown"
    
    # Check for timestamp patterns
    timestamp_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # ISO date
        r'\d{2}/\d{2}/\d{4}',  # US date
        r'\d{10,13}',  # Unix timestamp
    ]
    import re
    timestamp_count = 0
    for v in sample:
        for pattern in timestamp_patterns:
            if re.match(pattern, v):
                timestamp_count += 1
                break
    if timestamp_count / len(sample) > 0.8:
        return "timestamp"
    
    # Check for numeric
    numeric_count = 0
    for v in sample:
        try:
            float(v)
            numeric_count += 1
        except:
            pass
    
    if numeric_count / len(sample) > 0.9:
        # Check if integer
        int_count = sum(1 for v in sample if v.lstrip('-').isdigit())
        if int_count / len(sample) > 0.9:
            return "integer"
        return "float"
    
    # Check for categorical (few unique values)
    unique_ratio = len(set(sample)) / len(sample)
    if unique_ratio < 0.1:
        return "categorical"
    
    return "string"


def _detect_json_type(
    content: bytes,
    filename: str,
    read_bytes: int = 8192
) -> FileTypeDetectionResult:
    """Detect JSON file type by analyzing content structure.
    
    IMU files typically have:
    - Array of samples with accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    - Or nested structure with imu.accel.x, imu.gyro.x, etc.
    """
    errors = []
    warnings = []
    statistics = {}
    
    try:
        text = content[:read_bytes].decode('utf-8', errors='ignore').strip()
        
        if not text:
            return FileTypeDetectionResult(
                detected_type=DetectedFileType.UNKNOWN,
                confidence=0.0,
                detection_method="content",
                errors=["Empty file content"],
            )
        
        # Try to parse as JSON
        samples = []
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                samples = data.get('samples', data.get('data', []))
                if not samples and any(k in data for k in ['accel_x', 'ax', 'imu', 'accel']):
                    samples = [data]
            elif isinstance(data, list):
                samples = data
        except json.JSONDecodeError:
            # Try JSONL format
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.endswith(','):
                    line = line[:-1]
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        samples.append(obj)
                except:
                    continue
        
        if not samples:
            return FileTypeDetectionResult(
                detected_type=DetectedFileType.UNKNOWN,
                confidence=0.3,
                detection_method="content",
                warnings=["No valid JSON samples found"],
            )
        
        # Check for IMU fields
        imu_fields = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        alt_fields = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        
        sample = samples[0] if samples else {}
        has_imu = (
            all(f in sample for f in imu_fields) or
            all(f in sample for f in alt_fields) or
            'imu' in sample or
            'accel' in sample
        )
        
        if has_imu:
            statistics["num_samples"] = len(samples)
            statistics["sample_fields"] = list(sample.keys())
            
            return FileTypeDetectionResult(
                detected_type=DetectedFileType.IMU,
                confidence=0.9,
                detection_method="content",
                statistics=statistics,
                errors=errors,
                warnings=warnings,
            )
        else:
            # Generic JSON data
            statistics["num_samples"] = len(samples)
            statistics["sample_fields"] = list(sample.keys()) if sample else []
            
            return FileTypeDetectionResult(
                detected_type=DetectedFileType.UNKNOWN,
                confidence=0.5,
                detection_method="content",
                statistics=statistics,
                warnings=["JSON structure not recognized as IMU data"],
            )
            
    except Exception as e:
        logger.error(f"Error detecting JSON type: {e}")
        return FileTypeDetectionResult(
            detected_type=DetectedFileType.UNKNOWN,
            confidence=0.0,
            detection_method="content",
            errors=[f"Detection failed: {str(e)}"],
        )


@dataclass
class BrainFileMetadata:
    """Metadata schema for files in Brain.
    
    This metadata is created by Brain when files are uploaded or scanned.
    It includes auto-detected information plus user-fillable fields.
    """
    # Auto-detected fields
    file_id: str = ""
    filename: str = ""
    detected_type: str = "unknown"
    detection_confidence: float = 0.0
    file_extension: str = ""
    file_size: int = 0
    file_hash: str = ""
    
    # Timestamps
    created_at: str = ""
    modified_at: str = ""
    uploaded_at: str = ""
    
    # Content analysis results
    header_columns: Optional[List[str]] = None
    delimiter: str = ","
    has_header: bool = True
    
    # For CSI files
    is_csi: bool = False
    csi_array_length: Optional[int] = None
    num_subcarriers: Optional[int] = None
    
    # Validation statistics
    total_lines: int = 0
    valid_lines: int = 0
    invalid_lines: int = 0
    valid_percentage: float = 0.0
    
    # User-fillable fields
    labels: List[str] = field(default_factory=list)  # User-assigned labels
    primary_label: str = ""  # Main classification label
    description: str = ""
    subject_id: str = ""
    environment: str = ""
    activity: str = ""
    
    # Dataset assignment
    assigned_datasets: List[str] = field(default_factory=list)
    
    # Thoth metadata reference (if from thoth device)
    thoth_metadata_file: Optional[str] = None
    has_thoth_metadata: bool = False
    
    # Schema version
    schema_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrainFileMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Path) -> bool:
        """Save metadata to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False
    
    @classmethod
    def load(cls, path: Path) -> Optional["BrainFileMetadata"]:
        """Load metadata from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None


def create_brain_metadata(
    content: bytes,
    filename: str,
    file_path: Optional[str] = None,
    thoth_metadata: Optional[Dict[str, Any]] = None,
) -> BrainFileMetadata:
    """Create Brain metadata for a file by analyzing its content.
    
    Args:
        content: File content bytes
        filename: Filename
        file_path: Optional full file path for timestamp extraction
        thoth_metadata: Optional metadata from thoth device (.meta.json)
        
    Returns:
        BrainFileMetadata with auto-detected and user-fillable fields
    """
    import uuid
    
    # Detect file type
    detection = detect_file_type(content, filename)
    
    # Create metadata
    metadata = BrainFileMetadata(
        file_id=str(uuid.uuid4()),
        filename=filename,
        detected_type=detection.detected_type.value,
        detection_confidence=detection.confidence,
        file_extension=os.path.splitext(filename)[1].lower(),
        file_size=len(content),
        file_hash=hashlib.md5(content).hexdigest(),
        uploaded_at=datetime.utcnow().isoformat() + "Z",
        header_columns=detection.header_columns,
        delimiter=detection.delimiter,
        has_header=detection.has_header,
        is_csi=detection.is_csi,
        csi_array_length=detection.csi_array_length,
    )
    
    # Extract timestamps from file path if available
    if file_path:
        try:
            stat = os.stat(file_path)
            metadata.created_at = datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z"
            metadata.modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z"
        except:
            pass
    
    # Copy statistics
    stats = detection.statistics
    if stats:
        metadata.total_lines = stats.get("total_lines", 0)
        metadata.valid_lines = stats.get("valid_csi_rows", stats.get("valid_rows", 0))
        metadata.invalid_lines = stats.get("invalid_csi_rows", stats.get("invalid_rows", 0))
        if metadata.total_lines > 0:
            metadata.valid_percentage = (metadata.valid_lines / metadata.total_lines) * 100
        if "num_subcarriers" in stats:
            metadata.num_subcarriers = stats["num_subcarriers"]
    
    # Merge thoth metadata if provided
    if thoth_metadata:
        metadata.has_thoth_metadata = True
        
        # Copy labels from thoth metadata
        labels_info = thoth_metadata.get("labels", {})
        if labels_info:
            if labels_info.get("activity"):
                metadata.activity = labels_info["activity"]
                metadata.labels.append(labels_info["activity"])
            if labels_info.get("class_name"):
                metadata.primary_label = labels_info["class_name"]
            if labels_info.get("subject_id"):
                metadata.subject_id = labels_info["subject_id"]
            if labels_info.get("environment"):
                metadata.environment = labels_info["environment"]
    
    return metadata


def get_metadata_filename(data_filename: str) -> str:
    """Get the Brain metadata filename for a data file.
    
    Args:
        data_filename: Data file name
        
    Returns:
        Metadata filename (e.g., "myfile.brain.json")
    """
    base = Path(data_filename).stem
    return f"{base}.brain.json"


def get_thoth_metadata_filename(data_filename: str) -> str:
    """Get the thoth metadata filename for a data file.
    
    Args:
        data_filename: Data file name
        
    Returns:
        Thoth metadata filename (e.g., "myfile.meta.json")
    """
    base = Path(data_filename).stem
    return f"{base}.meta.json"


def validate_thoth_metadata(metadata_path: Path) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """Validate a thoth metadata file.
    
    Thoth metadata files (.meta.json) are required for files in thoth/data.
    Files without metadata are considered invalid.
    
    Args:
        metadata_path: Path to .meta.json file
        
    Returns:
        Tuple of (is_valid, metadata_dict, errors)
    """
    errors = []
    
    if not metadata_path.exists():
        return False, None, ["Metadata file does not exist"]
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        return False, None, [f"Invalid JSON: {str(e)}"]
    except Exception as e:
        return False, None, [f"Failed to read metadata: {str(e)}"]
    
    # Check required fields
    required_fields = ["file_id", "filename", "sensor_type", "device_id"]
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
    
    # Check collection info
    collection = metadata.get("collection", {})
    if collection.get("num_samples", 0) <= 0:
        errors.append("collection.num_samples must be positive")
    
    # Check quality info
    quality = metadata.get("quality", {})
    if quality.get("validated", False) and quality.get("validation_errors"):
        errors.append("File marked as validated but has validation errors")
    
    return len(errors) == 0, metadata, errors


def scan_thoth_data_directory(
    data_path: str,
    require_metadata: bool = True
) -> List[Dict[str, Any]]:
    """Scan thoth/data directory for files with their metadata.
    
    Args:
        data_path: Path to thoth/data directory
        require_metadata: If True, only return files with valid .meta.json
        
    Returns:
        List of file info dictionaries
    """
    files = []
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_path}")
        return files
    
    minute_dirs = []
    if MINUTE_DIR_RE.match(data_dir.name):
        minute_dirs = [data_dir]
    else:
        minute_dirs = sorted(
            [item for item in data_dir.iterdir() if item.is_dir() and MINUTE_DIR_RE.match(item.name)],
            key=lambda item: item.name,
        )

    for minute_dir in minute_dirs:
        for file_path in minute_dir.iterdir():
            if not file_path.is_file():
                continue

            raw_name = file_path.name
            filename = f"{minute_dir.name}_{raw_name}"

            if raw_name.endswith('.meta.json') or raw_name.endswith('.brain.json'):
                continue
            if raw_name in ['device_id.txt']:
                continue
            if raw_name.startswith('.'):
                continue

            meta_path = minute_dir / get_thoth_metadata_filename(raw_name)
            has_metadata = meta_path.exists()

            if require_metadata and not has_metadata:
                logger.debug(f"Skipping {filename}: no metadata file")
                continue

            try:
                with open(file_path, 'rb') as f:
                    content = f.read(8192)

                detection = detect_file_type(content, raw_name)

                thoth_meta = None
                if has_metadata:
                    is_valid, thoth_meta, meta_errors = validate_thoth_metadata(meta_path)
                    if not is_valid:
                        logger.warning(f"Invalid metadata for {filename}: {meta_errors}")

                file_info = {
                    'name': filename,
                    'relative_name': raw_name,
                    'minute': minute_dir.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'created': datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'detected_type': detection.detected_type.value,
                    'detection_confidence': detection.confidence,
                    'has_metadata': has_metadata,
                    'metadata_valid': has_metadata and thoth_meta is not None,
                    'thoth_metadata': thoth_meta,
                    'statistics': detection.statistics,
                }

                if detection.is_csi:
                    file_info['is_csi'] = True
                    file_info['csi_array_length'] = detection.csi_array_length

                files.append(file_info)

            except Exception as e:
                logger.error(f"Error scanning file {filename}: {e}")
                continue
    
    logger.info(f"Scanned {len(files)} files in {data_path}")
    return files
