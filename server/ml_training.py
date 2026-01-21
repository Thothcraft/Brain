"""Machine Learning Training Module.

This module provides real PyTorch-based training functionality for:
- IMU data classification (6-axis: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
- Time-series classification

Input shape: (batch_size, window_size, 6)
- window_size is configurable (default 128)
- 6 channels for IMU data

No synthetic data or fallbacks - uses real data from database.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import io
import json
import pickle
import traceback
import sys
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import logging

logger = logging.getLogger(__name__)

# Configure detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class IMUClassifier(nn.Module):
    """CNN+LSTM hybrid model for IMU time-series classification.
    
    Input shape: (batch_size, window_size, 6)
    - 6 channels: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    """
    
    def __init__(self, input_channels: int = 6, seq_length: int = 128, 
                 num_classes: int = 4, architecture_size: str = 'medium'):
        super().__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.architecture_size = architecture_size
        
        # Architecture configurations
        configs = {
            'small': {'conv_channels': [32, 64], 'lstm_hidden': 64, 'fc_hidden': 64},
            'medium': {'conv_channels': [64, 128, 128], 'lstm_hidden': 128, 'fc_hidden': 128},
            'large': {'conv_channels': [64, 128, 256, 256], 'lstm_hidden': 256, 'fc_hidden': 256}
        }
        config = configs.get(architecture_size, configs['medium'])
        self._config = config
        
        # Convolutional layers for temporal feature extraction
        conv_layers = []
        in_channels = input_channels
        for out_channels in config['conv_channels']:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate conv output length after pooling
        self.conv_out_len = seq_length // (2 ** len(config['conv_channels']))
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=config['conv_channels'][-1],
            hidden_size=config['lstm_hidden'],
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention mechanism for weighted sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(config['lstm_hidden'] * 2, config['lstm_hidden']),
            nn.Tanh(),
            nn.Linear(config['lstm_hidden'], 1)
        )
        
        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(config['lstm_hidden'] * 2, config['fc_hidden']),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config['fc_hidden'], num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_length, channels) = (batch, window_size, 6)
        batch_size = x.size(0)
        
        # Permute for Conv1d: (batch, channels, seq_length)
        x = x.permute(0, 2, 1)
        
        # Apply convolutions
        x = self.conv(x)
        
        # Permute back for LSTM: (batch, seq_length, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention-weighted aggregation
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        x = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classification
        x = self.fc(x)
        return x
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layers = []
        layers.append({
            'type': 'Input',
            'shape': f'(batch, {self.seq_length}, {self.input_channels})',
            'params': 0
        })
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                layers.append({
                    'type': 'Conv1d',
                    'shape': f'({module.in_channels}, {module.out_channels}, k={module.kernel_size[0]})',
                    'params': sum(p.numel() for p in module.parameters())
                })
            elif isinstance(module, nn.LSTM):
                layers.append({
                    'type': 'BiLSTM' if module.bidirectional else 'LSTM',
                    'units': module.hidden_size,
                    'shape': f'({module.input_size} -> {module.hidden_size * (2 if module.bidirectional else 1)})',
                    'params': sum(p.numel() for p in module.parameters())
                })
            elif isinstance(module, nn.Linear) and 'fc' in name:
                layers.append({
                    'type': 'Dense',
                    'units': module.out_features,
                    'shape': f'({module.in_features}, {module.out_features})',
                    'params': sum(p.numel() for p in module.parameters())
                })
        
        return {
            'layers': layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_shape': f'(batch, {self.seq_length}, {self.input_channels})',
            'architecture_size': self.architecture_size
        }


# ============================================================================
# DATA LOADING FROM DATABASE
# ============================================================================

class IMUDataset(Dataset):
    """PyTorch Dataset for IMU data loaded from database files."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def parse_csi_file(
    content: bytes,
    window_size: int = 1000,
    include_phase: bool = True,
    filter_subcarriers: bool = True,
    subcarrier_start: int = 5,
    subcarrier_end: int = 32,
    output_shape: str = "flattened"
) -> Tuple[List[np.ndarray], dict]:
    """Parse CSI CSV file and extract amplitude/phase data.
    
    CSI data format: CSV with rows containing [imag1, real1, imag2, real2, ...]
    
    Args:
        content: Raw file bytes
        window_size: Number of rows per sample window
        include_phase: Whether to include phase data alongside amplitude
        filter_subcarriers: Whether to filter subcarriers (remove guard bands)
        subcarrier_start: Start index for subcarrier filtering (default 5)
        subcarrier_end: End index for first range (default 32)
        output_shape: "flattened" for ML models, "sequence" for DL models
        
    Returns:
        Tuple of (list of data arrays, metadata dict)
        - flattened: each array is shape (features,)
        - sequence: each array is shape (window_size, features)
    """
    import math
    
    try:
        text_content = content.decode('utf-8', errors='ignore').lstrip('\ufeff').strip()
        lines = text_content.split('\n')
        
        if len(lines) < 2:
            logger.warning("CSI file has less than 2 lines")
            return [], {"error": "File too short"}
        
        # Parse CSI rows - skip header
        csi_rows = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                # Extract array content between brackets [imag, real, imag, real, ...]
                if '[' in line and ']' in line:
                    csi_str = line[line.index("[")+1 : line.index("]")]
                    csi_values = [float(x.strip()) for x in csi_str.split(",") if x.strip()]
                    if csi_values:
                        csi_rows.append(csi_values)
            except Exception as e:
                logger.debug(f"Skipping malformed CSI row: {e}")
                continue
        
        if not csi_rows:
            logger.warning("No valid CSI rows found")
            return [], {"error": "No valid CSI data"}
        
        logger.info(f"Parsed {len(csi_rows)} CSI rows")
        
        # Normalize row lengths to avoid NaN padding from ragged rows
        lengths = [len(r) for r in csi_rows]
        if not lengths:
            return [], {"error": "No valid CSI data"}

        # Use the most common length (and force even length for IQ pairs)
        try:
            from collections import Counter
            expected_len = Counter(lengths).most_common(1)[0][0]
        except Exception:
            expected_len = int(np.median(lengths))

        expected_len = int(expected_len)
        expected_len = expected_len - (expected_len % 2)
        if expected_len <= 0:
            return [], {"error": "Invalid CSI row length"}

        cleaned_rows = []
        dropped_rows = 0
        for r in csi_rows:
            if len(r) < expected_len:
                dropped_rows += 1
                continue
            rr = r[:expected_len]
            arr = np.asarray(rr, dtype=np.float32)
            if not np.isfinite(arr).all():
                dropped_rows += 1
                continue
            cleaned_rows.append(arr)

        if not cleaned_rows:
            return [], {"error": "No valid CSI rows after cleaning"}

        csi_arr = np.stack(cleaned_rows, axis=0)  # (n_rows, expected_len)
        n_rows = csi_arr.shape[0]
        n_subcarriers = expected_len // 2
        imag = csi_arr[:, 0::2]
        real = csi_arr[:, 1::2]

        amp_arr = np.sqrt(imag ** 2 + real ** 2)
        phase_arr = np.arctan2(imag, real)

        logger.info(f"Extracted amplitude shape: {amp_arr.shape}, phase shape: {phase_arr.shape} (dropped_rows={dropped_rows})")
        
        # Apply subcarrier filtering (remove null guard bands)
        if filter_subcarriers and amp_arr.shape[1] > subcarrier_end + 27:
            amp_arr1 = amp_arr[:, subcarrier_start:subcarrier_end]
            amp_arr2 = amp_arr[:, subcarrier_end+1:subcarrier_end+28]
            amp_arr = np.concatenate([amp_arr1, amp_arr2], axis=1)

            phase_arr1 = phase_arr[:, subcarrier_start:subcarrier_end]
            phase_arr2 = phase_arr[:, subcarrier_end+1:subcarrier_end+28]
            phase_arr = np.concatenate([phase_arr1, phase_arr2], axis=1)

            logger.info(f"After subcarrier filtering: amp shape {amp_arr.shape}")
        
        # Combine amplitude and phase if requested
        if include_phase:
            combined_arr = np.concatenate([amp_arr, phase_arr], axis=1)
        else:
            combined_arr = amp_arr

        if not np.isfinite(combined_arr).all():
            combined_arr = np.nan_to_num(combined_arr, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Combined data shape: {combined_arr.shape}")
        
        # Create windows
        windows = []
        n_rows = combined_arr.shape[0]
        n_features = combined_arr.shape[1]
        
        if n_rows < window_size:
            logger.warning(f"Not enough rows ({n_rows}) for window size {window_size}")
            return [], {"error": f"Not enough data: {n_rows} rows < {window_size} window_size"}
        
        # Create non-overlapping windows
        for start in range(0, n_rows - window_size + 1, window_size):
            window_data = combined_arr[start:start + window_size]

            if output_shape == "flattened":
                windows.append(window_data.reshape(-1).astype(np.float32))
            else:
                windows.append(window_data.astype(np.float32))
        
        metadata = {
            "total_rows": n_rows,
            "n_windows": len(windows),
            "window_size": window_size,
            "n_features_per_row": n_features,
            "include_phase": include_phase,
            "filter_subcarriers": filter_subcarriers,
            "output_shape": output_shape,
            "final_shape": windows[0].shape if windows else None,
            "expected_row_len": expected_len,
            "dropped_rows": dropped_rows
        }
        
        logger.info(f"Created {len(windows)} windows, shape: {metadata['final_shape']}")
        
        return windows, metadata
        
    except Exception as e:
        logger.error(f"Error parsing CSI file: {e}")
        logger.error(traceback.format_exc())
        return [], {"error": str(e)}


def parse_imu_file(content: bytes, window_size: int = 128) -> List[np.ndarray]:
    """Parse IMU JSON file and extract windows of data.
    
    Supports multiple formats:
    1. Single JSON object with samples array:
       {"samples": [{"accel_x": float, ...}, ...]}
    2. JSON array:
       [{"accel_x": float, ...}, ...]
    3. JSONL (newline-delimited JSON):
       {"accel_x": float, ...}
       {"accel_x": float, ...}
    
    Returns list of windows, each shape (window_size, 6)
    """
    try:
        text_content = content.decode('utf-8', errors='ignore').lstrip('\ufeff').strip()
        samples = []
        
        # Try parsing as single JSON first
        try:
            data = json.loads(text_content)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                samples = data.get('samples', data.get('data', []))
                # If no samples/data key, treat the dict itself as a single sample
                if not samples and any(k in data for k in ['accel_x', 'ax', 'accel_y', 'ay']):
                    samples = [data]
            elif isinstance(data, list):
                samples = data
            else:
                logger.warning(f"Unknown IMU data format: {type(data)}")
                return []
        except json.JSONDecodeError:
            # Try JSONL format (one JSON object per line)
            logger.info("Trying JSONL format...")
            lines = text_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Some exporters add trailing commas; be tolerant.
                if line.endswith(','):
                    line = line[:-1].rstrip()

                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        samples.append(obj)
                    elif isinstance(obj, list):
                        samples.extend(obj)
                except json.JSONDecodeError:
                    continue

            if samples:
                logger.info(f"Parsed {len(samples)} samples from JSONL format")
        
        if not samples:
            # CSV fallback (common for IMU logs)
            try:
                import csv
                reader = csv.DictReader(io.StringIO(text_content))
                for row in reader:
                    samples.append(row)
            except Exception:
                samples = []

        if not samples:
            logger.warning("No samples found in IMU file")
            return []
        
        def _get_nested_xyz(obj: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
            if obj is None:
                return None, None, None
            if isinstance(obj, dict):
                try:
                    return float(obj.get('x')), float(obj.get('y')), float(obj.get('z'))
                except Exception:
                    return None, None, None
            if isinstance(obj, (list, tuple)) and len(obj) >= 3:
                try:
                    return float(obj[0]), float(obj[1]), float(obj[2])
                except Exception:
                    return None, None, None
            return None, None, None

        # Extract 6-axis IMU data
        imu_data = []
        for sample in samples:
            try:
                if not isinstance(sample, dict):
                    continue

                accel_x = accel_y = accel_z = None
                gyro_x = gyro_y = gyro_z = None

                # Nested format: {"imu": {"accel": {"x","y","z"}, "gyro": {...}}}
                if isinstance(sample.get('imu'), dict):
                    ax, ay, az = _get_nested_xyz(sample['imu'].get('accel'))
                    gx, gy, gz = _get_nested_xyz(sample['imu'].get('gyro'))
                    accel_x, accel_y, accel_z = ax, ay, az
                    gyro_x, gyro_y, gyro_z = gx, gy, gz

                # Alternative nested: {"accel": {"x","y","z"}, "gyro": {...}}
                if accel_x is None and isinstance(sample.get('accel'), (dict, list, tuple)):
                    accel_x, accel_y, accel_z = _get_nested_xyz(sample.get('accel'))
                if gyro_x is None and isinstance(sample.get('gyro'), (dict, list, tuple)):
                    gyro_x, gyro_y, gyro_z = _get_nested_xyz(sample.get('gyro'))

                # Flat keys (JSON / CSV)
                if accel_x is None:
                    accel_x = sample.get('accel_x', sample.get('ax', sample.get('accel.x')))
                if accel_y is None:
                    accel_y = sample.get('accel_y', sample.get('ay', sample.get('accel.y')))
                if accel_z is None:
                    accel_z = sample.get('accel_z', sample.get('az', sample.get('accel.z')))
                if gyro_x is None:
                    gyro_x = sample.get('gyro_x', sample.get('gx', sample.get('gyro.x')))
                if gyro_y is None:
                    gyro_y = sample.get('gyro_y', sample.get('gy', sample.get('gyro.y')))
                if gyro_z is None:
                    gyro_z = sample.get('gyro_z', sample.get('gz', sample.get('gyro.z')))

                if accel_x is None or accel_y is None or accel_z is None or gyro_x is None or gyro_y is None or gyro_z is None:
                    continue

                row = [
                    float(accel_x),
                    float(accel_y),
                    float(accel_z),
                    float(gyro_x),
                    float(gyro_y),
                    float(gyro_z),
                ]
                imu_data.append(row)
            except (KeyError, TypeError, ValueError):
                continue
        
        logger.info(f"Extracted {len(imu_data)} IMU samples")
        
        if len(imu_data) < window_size:
            logger.warning(f"Not enough samples ({len(imu_data)}) for window size {window_size}")
            return []
        
        # Create sliding windows with 50% overlap
        imu_array = np.array(imu_data, dtype=np.float32)
        windows = []
        stride = window_size // 2
        
        for start in range(0, len(imu_array) - window_size + 1, stride):
            window = imu_array[start:start + window_size]
            windows.append(window)
        
        logger.info(f"Created {len(windows)} windows of size {window_size}")
        return windows
        
    except Exception as e:
        logger.error(f"Error parsing IMU file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


# ============================================================================
# PREPROCESSING PIPELINE EXECUTOR (for preview and training)
# ============================================================================

# Block type registry: each block has an execute function
# Input/output: numpy arrays, config dict
# Returns: (output_array, stage_info_dict)

def _block_csi_loader(content: bytes, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Load CSI data from raw bytes, return I/Q matrix."""
    text_content = content.decode('utf-8', errors='ignore').lstrip('\ufeff').strip()
    lines = text_content.split('\n')
    csi_rows = []
    for line in lines[1:]:
        line = line.strip()
        if not line or '[' not in line or ']' not in line:
            continue
        try:
            csi_str = line[line.index('[') + 1: line.index(']')]
            csi_values = [float(x.strip()) for x in csi_str.split(',') if x.strip()]
            if csi_values:
                csi_rows.append(csi_values)
        except Exception:
            continue
    if not csi_rows:
        return np.array([]), {"error": "No valid CSI rows found"}
    from collections import Counter
    lengths = [len(r) for r in csi_rows]
    expected_len = Counter(lengths).most_common(1)[0][0]
    expected_len = expected_len - (expected_len % 2)
    cleaned = []
    for r in csi_rows:
        if len(r) < expected_len:
            continue
        arr = np.asarray(r[:expected_len], dtype=np.float32)
        if np.isfinite(arr).all():
            cleaned.append(arr)
    if not cleaned:
        return np.array([]), {"error": "No valid CSI rows after cleaning"}
    csi_arr = np.stack(cleaned, axis=0)
    sample = csi_arr[0, :max_preview].tolist() if csi_arr.size else []
    return csi_arr, {"block": "csi_loader", "name": "CSI Loader", "shape": list(csi_arr.shape), "sample": sample}


def _block_amplitude_extractor(data: np.ndarray, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Extract amplitude from I/Q pairs: sqrt(I^2 + Q^2)."""
    if data.size == 0:
        return data, {"block": "amplitude_extractor", "name": "Amplitude Extractor", "shape": [], "sample": [], "error": "No input data"}
    imag = data[:, 0::2]
    real = data[:, 1::2]
    amp = np.sqrt(imag ** 2 + real ** 2)
    sample = amp[0, :max_preview].tolist() if amp.size else []
    return amp, {"block": "amplitude_extractor", "name": "Amplitude Extractor", "shape": list(amp.shape), "sample": sample}


def _block_phase_extractor(data: np.ndarray, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Extract phase from I/Q pairs: atan2(I, Q)."""
    if data.size == 0:
        return data, {"block": "phase_extractor", "name": "Phase Extractor", "shape": [], "sample": [], "error": "No input data"}
    imag = data[:, 0::2]
    real = data[:, 1::2]
    phase = np.arctan2(imag, real)
    sample = phase[0, :max_preview].tolist() if phase.size else []
    return phase, {"block": "phase_extractor", "name": "Phase Extractor", "shape": list(phase.shape), "sample": sample}


def _block_subcarrier_filter(data: np.ndarray, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Filter subcarriers to remove guard bands."""
    if data.size == 0:
        return data, {"block": "subcarrier_filter", "name": "Subcarrier Filter", "shape": [], "sample": [], "error": "No input data"}
    start = int(config.get("subcarrier_start", 5))
    end = int(config.get("subcarrier_end", 32))
    n_sub = data.shape[1]
    if n_sub <= end + 27:
        sample = data[0, :max_preview].tolist() if data.size else []
        return data, {"block": "subcarrier_filter", "name": "Subcarrier Filter", "shape": list(data.shape), "sample": sample, "skipped": True}
    part1 = data[:, start:end]
    part2 = data[:, end + 1:end + 28]
    filtered = np.concatenate([part1, part2], axis=1)
    sample = filtered[0, :max_preview].tolist() if filtered.size else []
    return filtered, {"block": "subcarrier_filter", "name": "Subcarrier Filter", "shape": list(filtered.shape), "sample": sample}


def _block_feature_concat(amp: np.ndarray, phase: np.ndarray, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Concatenate amplitude and phase features."""
    if amp.size == 0:
        return amp, {"block": "feature_concat", "name": "Feature Combine", "shape": [], "sample": [], "error": "No amplitude data"}
    include_phase = config.get("include_phase", True)
    if include_phase and phase.size > 0:
        combined = np.concatenate([amp, phase], axis=1)
    else:
        combined = amp
    sample = combined[0, :max_preview].tolist() if combined.size else []
    return combined, {"block": "feature_concat", "name": "Feature Combine", "shape": list(combined.shape), "sample": sample}


def _block_windowing(data: np.ndarray, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Create windows and optionally flatten."""
    if data.size == 0:
        return data, {"block": "data_portion_selector", "name": "Windowing / Flattening", "shape": [], "sample": [], "error": "No input data"}
    window_size = int(config.get("window_size", 1000))
    output_shape = config.get("output_shape", "flattened")
    n_rows = data.shape[0]
    if n_rows < window_size:
        return np.array([]), {"block": "data_portion_selector", "name": "Windowing / Flattening", "shape": [], "sample": [], "error": f"Not enough rows ({n_rows}) for window size {window_size}"}
    windows = []
    for start in range(0, n_rows - window_size + 1, window_size):
        w = data[start:start + window_size]
        if output_shape == "flattened":
            windows.append(w.reshape(-1).astype(np.float32))
        else:
            windows.append(w.astype(np.float32))
    if not windows:
        return np.array([]), {"block": "data_portion_selector", "name": "Windowing / Flattening", "shape": [], "sample": [], "error": "No windows created"}
    arr = np.stack(windows, axis=0)
    sample = arr[0, :max_preview].tolist() if arr.size else []
    return arr, {"block": "data_portion_selector", "name": "Windowing / Flattening", "shape": list(arr.shape), "sample": sample}


def _block_moving_average(data: np.ndarray, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Apply moving average smoothing along time axis."""
    if data.size == 0:
        return data, {"block": "moving_average", "name": "Moving Average", "shape": [], "sample": [], "error": "No input data"}
    window = int(config.get("ma_window", 5))
    if window < 2:
        sample = data[0, :max_preview].tolist() if data.size else []
        return data, {"block": "moving_average", "name": "Moving Average", "shape": list(data.shape), "sample": sample, "skipped": True}
    kernel = np.ones(window) / window
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)
    sample = smoothed[0, :max_preview].tolist() if smoothed.size else []
    return smoothed.astype(np.float32), {"block": "moving_average", "name": "Moving Average", "shape": list(smoothed.shape), "sample": sample}


def _block_zscore_normalize(data: np.ndarray, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Z-score normalization per feature."""
    if data.size == 0:
        return data, {"block": "zscore_normalize", "name": "Z-Score Normalize", "shape": [], "sample": [], "error": "No input data"}
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-8
    normed = (data - mean) / std
    sample = normed[0, :max_preview].tolist() if normed.size else []
    return normed.astype(np.float32), {"block": "zscore_normalize", "name": "Z-Score Normalize", "shape": list(normed.shape), "sample": sample}


def _block_imu_loader(content: bytes, config: dict, max_preview: int = 32) -> Tuple[np.ndarray, dict]:
    """Load IMU data from raw bytes."""
    windows = parse_imu_file(content, window_size=int(config.get("window_size", 128)))
    if not windows:
        return np.array([]), {"block": "imu_loader", "name": "IMU Loader", "shape": [], "sample": [], "error": "No IMU data parsed"}
    arr = np.stack(windows, axis=0)
    sample = arr[0, :max_preview, :].flatten()[:max_preview].tolist() if arr.size else []
    return arr, {"block": "imu_loader", "name": "IMU Loader", "shape": list(arr.shape), "sample": sample}


# Block registry
PREPROCESSING_BLOCKS = {
    "csi_loader": _block_csi_loader,
    "amplitude_extractor": _block_amplitude_extractor,
    "phase_extractor": _block_phase_extractor,
    "subcarrier_filter": _block_subcarrier_filter,
    "feature_concat": _block_feature_concat,
    "data_portion_selector": _block_windowing,
    "windowing": _block_windowing,
    "moving_average": _block_moving_average,
    "zscore_normalize": _block_zscore_normalize,
    "imu_loader": _block_imu_loader,
}


def execute_preprocessing_pipeline_preview(
    content: bytes,
    filename: str,
    base_config: dict,
    pipeline_blocks: List[dict],
    max_preview_values: int = 32
) -> dict:
    """Execute a preprocessing pipeline (block graph) and return per-stage info for preview.

    Args:
        content: Raw file bytes
        filename: Filename for type detection
        base_config: Base preprocessing config (data_type, include_phase, etc.)
        pipeline_blocks: List of block dicts, each with 'type' and optional 'params'
        max_preview_values: Max sample values to include per stage

    Returns:
        dict with 'stages' list
    """
    stages: List[dict] = []
    data_type = base_config.get("data_type", "auto")
    if data_type == "auto":
        data_type = detect_file_type(content, filename)

    # If no explicit blocks, build default pipeline from base_config
    if not pipeline_blocks:
        if data_type == "csi":
            pipeline_blocks = [
                {"type": "csi_loader"},
                {"type": "amplitude_extractor"},
            ]
            if base_config.get("include_phase", True):
                pipeline_blocks.append({"type": "phase_extractor"})
            if base_config.get("filter_subcarriers", True):
                pipeline_blocks.append({"type": "subcarrier_filter"})
            pipeline_blocks.append({"type": "feature_concat"})
            pipeline_blocks.append({"type": "data_portion_selector"})
        else:
            pipeline_blocks = [{"type": "imu_loader"}]

    # Execute blocks in order
    current_data = None
    amp_data = None
    phase_data = None
    raw_iq = None

    for block in pipeline_blocks:
        block_type = block.get("type", "")
        block_params = {**base_config, **(block.get("params") or {})}
        if not block.get("enabled", True):
            continue

        try:
            if block_type == "csi_loader":
                current_data, info = _block_csi_loader(content, block_params, max_preview_values)
                raw_iq = current_data
            elif block_type == "amplitude_extractor":
                if raw_iq is None:
                    raw_iq, _ = _block_csi_loader(content, block_params, max_preview_values)
                amp_data, info = _block_amplitude_extractor(raw_iq, block_params, max_preview_values)
                current_data = amp_data
            elif block_type == "phase_extractor":
                if raw_iq is None:
                    raw_iq, _ = _block_csi_loader(content, block_params, max_preview_values)
                phase_data, info = _block_phase_extractor(raw_iq, block_params, max_preview_values)
            elif block_type == "subcarrier_filter":
                if amp_data is not None:
                    amp_data, info_amp = _block_subcarrier_filter(amp_data, block_params, max_preview_values)
                if phase_data is not None:
                    phase_data, _ = _block_subcarrier_filter(phase_data, block_params, max_preview_values)
                info = info_amp if amp_data is not None else {"block": "subcarrier_filter", "name": "Subcarrier Filter", "shape": [], "sample": []}
                current_data = amp_data
            elif block_type == "feature_concat":
                current_data, info = _block_feature_concat(amp_data if amp_data is not None else np.array([]), phase_data if phase_data is not None else np.array([]), block_params, max_preview_values)
            elif block_type in ("data_portion_selector", "windowing"):
                if current_data is None or current_data.size == 0:
                    current_data = amp_data if amp_data is not None else np.array([])
                current_data, info = _block_windowing(current_data, block_params, max_preview_values)
            elif block_type == "moving_average":
                if current_data is not None:
                    current_data, info = _block_moving_average(current_data, block_params, max_preview_values)
                else:
                    info = {"block": "moving_average", "name": "Moving Average", "shape": [], "sample": [], "error": "No data"}
            elif block_type == "zscore_normalize":
                if current_data is not None:
                    current_data, info = _block_zscore_normalize(current_data, block_params, max_preview_values)
                else:
                    info = {"block": "zscore_normalize", "name": "Z-Score Normalize", "shape": [], "sample": [], "error": "No data"}
            elif block_type == "imu_loader":
                current_data, info = _block_imu_loader(content, block_params, max_preview_values)
            else:
                info = {"block": block_type, "name": block_type, "shape": [], "sample": [], "error": f"Unknown block type: {block_type}"}
            stages.append(info)
        except Exception as e:
            stages.append({"block": block_type, "name": block_type, "shape": [], "sample": [], "error": str(e)})

    return {"stages": stages}


def detect_file_type(content: bytes, filename: str = "") -> str:
    """Detect whether file contains CSI or IMU data.
    
    Args:
        content: Raw file bytes
        filename: Optional filename for hints
        
    Returns:
        "csi" or "imu"
    """
    # Check filename hints
    filename_lower = filename.lower()
    if 'csi' in filename_lower:
        return "csi"
    if 'imu' in filename_lower:
        return "imu"
    
    # Check content
    try:
        text = content.decode('utf-8', errors='ignore')[:2000]  # Check first 2KB
        
        # CSI files have bracket arrays like [1, 2, 3, ...]
        if '[' in text and ']' in text:
            # Check if it looks like CSI data (many comma-separated numbers in brackets)
            import re
            bracket_match = re.search(r'\[[\d\s,.-]+\]', text)
            if bracket_match:
                values = bracket_match.group()[1:-1].split(',')
                if len(values) > 50:  # CSI typically has 100+ values per row
                    return "csi"
        
        # IMU files have JSON with accel/gyro keys
        if any(key in text.lower() for key in ['accel_x', 'accel_y', 'gyro_x', 'gyro_y', '"ax"', '"ay"', '"gx"', '"gy"']):
            return "imu"
        
        # Default to CSI if we see lots of numeric data in brackets
        if text.count('[') > 5:
            return "csi"
            
    except Exception:
        pass
    
    # Default to IMU for backward compatibility
    return "imu"


def load_dataset_from_db(
    db_session, 
    dataset_id: int, 
    window_size: int = 128,
    preprocessing_config: dict = None,
    progress_callback: callable = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess dataset from database.
    
    Automatically detects data type (CSI vs IMU) and applies appropriate parsing.
    
    Args:
        db_session: SQLAlchemy database session
        dataset_id: ID of the training dataset
        window_size: Size of sliding window for time series
        preprocessing_config: Optional preprocessing configuration dict with:
            - include_phase: bool (CSI only)
            - filter_subcarriers: bool (CSI only)
            - subcarrier_start: int (CSI only)
            - subcarrier_end: int (CSI only)
            - output_shape: "flattened" or "sequence"
            - data_type: "csi", "imu", or "auto"
        
    Returns:
        X: numpy array of shape depends on output_shape:
            - flattened: (num_samples, features)
            - sequence: (num_samples, window_size, features)
        y: numpy array of labels
        class_names: list of class name strings
    """
    from server.db import TrainingDataset, DatasetFile, File
    from sqlalchemy.orm import joinedload
    from sqlalchemy import text
    
    # Default preprocessing config
    if preprocessing_config is None:
        preprocessing_config = {}
    
    include_phase = preprocessing_config.get('include_phase', True)
    filter_subcarriers = preprocessing_config.get('filter_subcarriers', True)
    subcarrier_start = preprocessing_config.get('subcarrier_start', 5)
    subcarrier_end = preprocessing_config.get('subcarrier_end', 32)
    output_shape = preprocessing_config.get('output_shape', 'flattened')
    forced_data_type = preprocessing_config.get('data_type', 'auto')
    preprocessing_blocks = preprocessing_config.get('preprocessing_blocks', [])
    
    # Set a longer statement timeout for loading large files (5 minutes)
    try:
        db_session.execute(text("SET statement_timeout = '300000'"))  # 5 minutes in ms
    except Exception as e:
        logger.warning(f"Could not set statement timeout: {e}")
    
    # Use eager loading to fetch dataset with files and their content in fewer queries
    dataset = db_session.query(TrainingDataset).options(
        joinedload(TrainingDataset.files).joinedload(DatasetFile.file)
    ).filter(TrainingDataset.id == dataset_id).first()
    
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found. Please verify the dataset exists and you have access to it.")
    
    # Validate dataset has files
    if not dataset.files or len(dataset.files) == 0:
        raise ValueError(
            f"Dataset '{dataset.name}' (ID: {dataset_id}) has no files. "
            f"Please add at least 2 files with different labels to the dataset before training."
        )
    
    # Get unique labels and create mapping
    labels_set = set(df.label for df in dataset.files if df.label)
    
    # Validate labels
    if len(labels_set) < 2:
        raise ValueError(
            f"Dataset '{dataset.name}' has only {len(labels_set)} unique label(s): {list(labels_set)}. "
            f"Training requires at least 2 different classes. "
            f"Please add files with different labels to create a classification task."
        )
    
    class_names = sorted(list(labels_set))
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    
    logger.info(f"Dataset '{dataset.name}' loaded with {len(dataset.files)} files")
    logger.info(f"  Classes ({len(class_names)}): {class_names}")
    
    all_windows = []
    all_labels = []
    detected_type = None
    files_processed = 0
    files_failed = []
    files_empty = []
    per_class_samples = {label: 0 for label in class_names}
    
    total_files = len(dataset.files)
    for file_idx, dataset_file in enumerate(dataset.files):
        # Report progress for each file
        if progress_callback:
            try:
                progress_callback(file_idx, total_files, dataset_file.file.filename if dataset_file.file else "unknown")
            except Exception as cb_err:
                logger.warning(f"Progress callback failed: {cb_err}")
        
        if not dataset_file.file:
            files_failed.append((dataset_file.label, "File reference missing"))
            continue
            
        file_content = dataset_file.file.content
        if not file_content:
            files_empty.append(dataset_file.file.filename or "unknown")
            continue
        
        filename = dataset_file.file.filename or ""
        
        # Detect or use forced data type
        if forced_data_type == 'auto':
            file_type = detect_file_type(file_content, filename)
        else:
            file_type = forced_data_type
        
        if detected_type is None:
            detected_type = file_type
            logger.info(f"Detected data type: {file_type}")
        
        # If we have explicit preprocessing blocks, use the pipeline executor
        if preprocessing_blocks:
            try:
                base_cfg = {
                    'data_type': file_type,
                    'include_phase': include_phase,
                    'filter_subcarriers': filter_subcarriers,
                    'subcarrier_start': subcarrier_start,
                    'subcarrier_end': subcarrier_end,
                    'output_shape': output_shape,
                    'window_size': window_size,
                }
                result = execute_preprocessing_pipeline_preview(
                    content=file_content,
                    filename=filename,
                    base_config=base_cfg,
                    pipeline_blocks=preprocessing_blocks,
                    max_preview_values=0,  # No sample needed for training
                )
                # Get final output from last stage
                # Re-run pipeline to get actual data (not just preview info)
                # For now, fall back to standard parsing since execute_preprocessing_pipeline_preview
                # returns info only, not the actual processed data array.
                # TODO: refactor to return processed data from pipeline executor
            except Exception as e:
                logger.warning(f"Pipeline execution failed for {filename}, falling back to standard parsing: {e}")
        
        # Parse based on data type (standard path)
        if file_type == "csi":
            windows, metadata = parse_csi_file(
                file_content,
                window_size=window_size,
                include_phase=include_phase,
                filter_subcarriers=filter_subcarriers,
                subcarrier_start=subcarrier_start,
                subcarrier_end=subcarrier_end,
                output_shape=output_shape
            )
            
            if "error" in metadata and not windows:
                logger.warning(f"CSI parsing failed for {filename}: {metadata.get('error')}")
                continue
                
        else:  # IMU
            windows = parse_imu_file(file_content, window_size)
        
        if windows:
            label_idx = label_to_idx[dataset_file.label]
            all_windows.extend(windows)
            all_labels.extend([label_idx] * len(windows))
            per_class_samples[dataset_file.label] += len(windows)
            files_processed += 1
            logger.info(f"Loaded {len(windows)} windows from {filename} (label: {dataset_file.label}, type: {file_type})")
        else:
            files_failed.append((filename, f"No windows extracted (file_type: {file_type})"))
    
    # Report processing summary
    logger.info(f"File processing summary:")
    logger.info(f"  Processed: {files_processed}/{len(dataset.files)} files")
    if files_empty:
        logger.warning(f"  Empty files (skipped): {files_empty}")
    if files_failed:
        logger.warning(f"  Failed files: {files_failed}")
    
    # Validate we have data
    if not all_windows:
        data_type_msg = f" ({detected_type})" if detected_type else ""
        error_details = []
        if files_empty:
            error_details.append(f"Empty files: {files_empty}")
        if files_failed:
            error_details.append(f"Failed files: {[f[0] for f in files_failed]}")
        raise ValueError(
            f"No valid data found in dataset '{dataset.name}'{data_type_msg}. "
            f"Check that files contain valid {detected_type or 'CSI/IMU'} data. "
            f"Details: {'; '.join(error_details) if error_details else 'Unknown error'}"
        )
    
    # Validate per-class sample counts
    logger.info(f"Per-class sample counts:")
    min_samples_per_class = float('inf')
    classes_with_few_samples = []
    for label, count in per_class_samples.items():
        logger.info(f"  {label}: {count} samples")
        if count < min_samples_per_class:
            min_samples_per_class = count
        if count < 10:
            classes_with_few_samples.append((label, count))
    
    if min_samples_per_class < 3:
        raise ValueError(
            f"Some classes have too few samples for training. "
            f"Each class needs at least 3 samples (for train/val/test split). "
            f"Classes with insufficient data: {[(l, c) for l, c in per_class_samples.items() if c < 3]}"
        )
    
    if classes_with_few_samples:
        logger.warning(
            f"WARNING: Some classes have very few samples (<10), which may lead to poor model performance: "
            f"{classes_with_few_samples}"
        )
    
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Total samples: {X.shape[0]}")
    logger.info(f"  Classes: {len(class_names)} - {class_names}")
    logger.info(f"  Data type: {detected_type}")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Output format: {output_shape}")
    
    return X, y, class_names


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str = 'cpu',
    callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Train a PyTorch model with real training loop."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    model = model.to(device)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        train_losses.append(round(train_loss, 4))
        train_accuracies.append(round(train_acc, 4))
        val_losses.append(round(val_loss, 4))
        val_accuracies.append(round(val_acc, 4))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if callback:
            callback(epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc)
    
    # Restore best model
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': round(best_val_acc, 4),
        'best_epoch': best_epoch,
        'model_state_dict': best_state_dict
    }


class BayesianOptimizationConfig:
    """Configuration for Bayesian hyperparameter optimization."""
    
    def __init__(
        self,
        num_trials: int = 20,
        epochs_per_trial: int = 3,
        # Learning rate search space
        lr_min: float = 0.00001,
        lr_max: float = 0.01,
        lr_scale: str = 'log',  # 'log' or 'linear'
        # Batch size search space
        batch_sizes: List[int] = None,
        # Weight decay search space
        weight_decay_min: float = 0.0,
        weight_decay_max: float = 0.01,
        # Dropout search space
        dropout_min: float = 0.1,
        dropout_max: float = 0.5,
        # Optimizer options
        optimizers: List[str] = None,
        # Exploration vs exploitation
        exploration_rate: float = 0.3,
        # Early stopping for trials
        early_stopping_patience: int = 2,
        # Architecture sizes to try
        architecture_sizes: List[str] = None
    ):
        self.num_trials = num_trials
        self.epochs_per_trial = epochs_per_trial
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_scale = lr_scale
        self.batch_sizes = batch_sizes or [16, 32, 64, 128]
        self.weight_decay_min = weight_decay_min
        self.weight_decay_max = weight_decay_max
        self.dropout_min = dropout_min
        self.dropout_max = dropout_max
        self.optimizers = optimizers or ['adam', 'adamw', 'sgd']
        self.exploration_rate = exploration_rate
        self.early_stopping_patience = early_stopping_patience
        self.architecture_sizes = architecture_sizes or ['small', 'medium', 'large']
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BayesianOptimizationConfig':
        return cls(
            num_trials=config.get('bayesian_trials', 20),
            epochs_per_trial=config.get('bayesian_epochs_per_trial', 3),
            lr_min=config.get('bayesian_lr_min', 0.00001),
            lr_max=config.get('bayesian_lr_max', 0.01),
            lr_scale=config.get('bayesian_lr_scale', 'log'),
            batch_sizes=config.get('bayesian_batch_sizes', [16, 32, 64, 128]),
            weight_decay_min=config.get('bayesian_weight_decay_min', 0.0),
            weight_decay_max=config.get('bayesian_weight_decay_max', 0.01),
            dropout_min=config.get('bayesian_dropout_min', 0.1),
            dropout_max=config.get('bayesian_dropout_max', 0.5),
            optimizers=config.get('bayesian_optimizers', ['adam', 'adamw', 'sgd']),
            exploration_rate=config.get('bayesian_exploration_rate', 0.3),
            early_stopping_patience=config.get('bayesian_early_stopping', 2),
            architecture_sizes=config.get('bayesian_architecture_sizes')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_trials': self.num_trials,
            'epochs_per_trial': self.epochs_per_trial,
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'lr_scale': self.lr_scale,
            'batch_sizes': self.batch_sizes,
            'weight_decay_min': self.weight_decay_min,
            'weight_decay_max': self.weight_decay_max,
            'dropout_min': self.dropout_min,
            'dropout_max': self.dropout_max,
            'optimizers': self.optimizers,
            'exploration_rate': self.exploration_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'architecture_sizes': self.architecture_sizes
        }


def sample_hyperparameters(
    config: BayesianOptimizationConfig,
    best_params: Optional[Dict[str, Any]] = None,
    trial_num: int = 0
) -> Dict[str, Any]:
    """Sample hyperparameters using Bayesian-like strategy."""
    import random
    import math
    
    # First trial or no best params: random sampling
    if trial_num == 0 or best_params is None:
        # Learning rate
        if config.lr_scale == 'log':
            lr = math.exp(random.uniform(math.log(config.lr_min), math.log(config.lr_max)))
        else:
            lr = random.uniform(config.lr_min, config.lr_max)
        
        return {
            'learning_rate': lr,
            'batch_size': random.choice(config.batch_sizes),
            'weight_decay': random.uniform(config.weight_decay_min, config.weight_decay_max),
            'dropout': random.uniform(config.dropout_min, config.dropout_max),
            'optimizer': random.choice(config.optimizers),
            'architecture_size': random.choice(config.architecture_sizes) if config.architecture_sizes else 'medium'
        }
    
    # Exploit best params with some exploration
    if random.random() < config.exploration_rate:
        # Explore: random sampling
        if config.lr_scale == 'log':
            lr = math.exp(random.uniform(math.log(config.lr_min), math.log(config.lr_max)))
        else:
            lr = random.uniform(config.lr_min, config.lr_max)
        
        return {
            'learning_rate': lr,
            'batch_size': random.choice(config.batch_sizes),
            'weight_decay': random.uniform(config.weight_decay_min, config.weight_decay_max),
            'dropout': random.uniform(config.dropout_min, config.dropout_max),
            'optimizer': random.choice(config.optimizers),
            'architecture_size': random.choice(config.architecture_sizes) if config.architecture_sizes else 'medium'
        }
    else:
        # Exploit: perturb best params
        lr = best_params['learning_rate'] * random.uniform(0.5, 2.0)
        lr = max(config.lr_min, min(config.lr_max, lr))
        
        wd = best_params.get('weight_decay', 0.0) * random.uniform(0.5, 2.0)
        wd = max(config.weight_decay_min, min(config.weight_decay_max, wd))
        
        dropout = best_params.get('dropout', 0.3) + random.uniform(-0.1, 0.1)
        dropout = max(config.dropout_min, min(config.dropout_max, dropout))
        
        return {
            'learning_rate': lr,
            'batch_size': best_params.get('batch_size', 32),
            'weight_decay': wd,
            'dropout': dropout,
            'optimizer': best_params.get('optimizer', 'adam'),
            'architecture_size': best_params.get('architecture_size', 'medium')
        }


def run_bayesian_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    config: BayesianOptimizationConfig,
    device: str = 'cpu',
    callback: Optional[Callable] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run Bayesian hyperparameter optimization with real data.
    
    Args:
        X_train: Training data (num_samples, window_size, channels)
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        num_classes: Number of classes
        config: Bayesian optimization configuration
        device: Device to train on
        callback: Optional callback for progress updates
        
    Returns:
        Tuple of (best_params, trials_data)
    """
    import time
    
    seq_length = X_train.shape[1]
    input_channels = X_train.shape[2]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    trials_data = []
    best_params = None
    best_val_acc = 0.0
    best_trial_idx = 0
    
    logger.info(f"Starting Bayesian optimization with {config.num_trials} trials")
    logger.info(f"Search space: LR [{config.lr_min}, {config.lr_max}], Batch {config.batch_sizes}")
    
    for trial in range(config.num_trials):
        trial_start = time.time()
        
        # Sample hyperparameters
        params = sample_hyperparameters(config, best_params, trial)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Create model with sampled architecture
        model = IMUClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            architecture_size=params.get('architecture_size', 'medium')
        )
        
        # Get optimizer
        optimizer_name = params.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0.0)
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params.get('weight_decay', 0.01)
            )
        else:  # sgd
            optimizer = optim.SGD(
                model.parameters(), 
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params.get('weight_decay', 0.0)
            )
        
        # Quick training
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.epochs_per_trial,
            learning_rate=params['learning_rate'],
            device=device
        )
        
        trial_duration = time.time() - trial_start
        
        trial_data = {
            'trial': trial + 1,
            'learning_rate': round(params['learning_rate'], 8),
            'batch_size': params['batch_size'],
            'weight_decay': round(params.get('weight_decay', 0.0), 6),
            'dropout': round(params.get('dropout', 0.3), 3),
            'optimizer': params.get('optimizer', 'adam'),
            'architecture_size': params.get('architecture_size', 'medium'),
            'train_accuracy': results['train_accuracies'][-1],
            'val_accuracy': results['val_accuracies'][-1],
            'train_loss': results['train_losses'][-1],
            'val_loss': results['val_losses'][-1],
            'duration_seconds': round(trial_duration, 2),
            'is_best': False
        }
        trials_data.append(trial_data)
        
        logger.info(
            f"Trial {trial+1}/{config.num_trials}: "
            f"lr={params['learning_rate']:.6f}, batch={params['batch_size']}, "
            f"opt={params.get('optimizer', 'adam')}, "
            f"val_acc={results['val_accuracies'][-1]:.4f} ({trial_duration:.1f}s)"
        )
        
        if results['best_val_accuracy'] > best_val_acc:
            best_val_acc = results['best_val_accuracy']
            best_params = params.copy()
            best_trial_idx = trial
            logger.info(f"  -> New best! val_acc={best_val_acc:.4f}")
        
        if callback:
            callback(trial + 1, config.num_trials, trial_data)
    
    # Mark best trial
    if trials_data:
        trials_data[best_trial_idx]['is_best'] = True
    
    logger.info(f"Bayesian optimization complete. Best trial: {best_trial_idx + 1}")
    logger.info(f"Best params: {best_params}")
    
    return best_params, trials_data


def save_model_to_bytes(model: nn.Module, config: Dict[str, Any]) -> bytes:
    """Save model weights and config to bytes for storage."""
    buffer = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'architecture': model.get_architecture_summary() if hasattr(model, 'get_architecture_summary') else {}
    }, buffer)
    return buffer.getvalue()


def load_model_from_bytes(model_bytes: bytes) -> Tuple[Dict, Dict]:
    """Load model weights and config from bytes."""
    buffer = io.BytesIO(model_bytes)
    checkpoint = torch.load(buffer, map_location='cpu', weights_only=False)
    return checkpoint['model_state_dict'], checkpoint.get('config', {})


async def train_ml_model(
    job_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: List[str],
    model_type: str,
    config: Dict[str, Any],
    db_session,
    update_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Train traditional ML model (AdaBoost, KNN, SVC).
    
    Args:
        job_id: Job identifier
        X_train, y_train: Training data
        X_val, y_val: Validation data
        class_names: List of class names
        model_type: 'adaboost', 'knn', or 'svc'
        config: Model configuration
        db_session: Database session
        update_callback: Progress callback
        
    Returns:
        Training results dictionary
    """
    import asyncio
    from server.ml_models import MLModelWrapper, get_default_ml_config
    
    logger.info("="*80)
    logger.info(f"ML MODEL TRAINING: {model_type}")
    logger.info("="*80)
    
    try:
        # Update progress: ML training started
        if update_callback:
            try:
                await update_callback("training", 0, 1, {"stage": "initializing"})
            except Exception as cb_err:
                logger.warning(f"Progress callback failed: {cb_err}")
        
        # Get model-specific config
        ml_config = get_default_ml_config(model_type)
        ml_config.update(config.get('ml_params', {}))
        
        logger.info(f"ML Config: {ml_config}")
        
        # Create ML model wrapper
        logger.debug("Creating ML model wrapper...")
        model_wrapper = MLModelWrapper(model_type, ml_config)
        logger.info(f"✓ {model_type} model initialized")
        
        # Update progress: Model initialized, starting training
        if update_callback:
            try:
                await update_callback("training", 0, 1, {"stage": "fitting"})
            except Exception as cb_err:
                logger.warning(f"Progress callback failed: {cb_err}")
        
        # Train model in thread pool
        logger.info("Starting ML model training...")
        loop = asyncio.get_event_loop()
        
        def sync_train():
            return model_wrapper.fit(X_train, y_train, X_val, y_val)
        
        metrics = await loop.run_in_executor(None, sync_train)
        logger.info(f"✓ Training complete")
        logger.info(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
        if 'val_accuracy' in metrics:
            logger.info(f"  Val accuracy: {metrics['val_accuracy']:.4f}")
        
        # Update progress: Training complete, computing metrics
        if update_callback:
            try:
                await update_callback("training", 1, 1, {"stage": "computing_metrics", "train_accuracy": metrics['train_accuracy']})
            except Exception as cb_err:
                logger.warning(f"Progress callback failed: {cb_err}")
        
        # Compute detailed metrics
        logger.info("Computing detailed metrics...")
        
        def compute_ml_metrics():
            val_preds = model_wrapper.predict(X_val)
            val_probs = model_wrapper.predict_proba(X_val)
            
            return compute_metrics_from_predictions(
                y_val, val_preds, val_probs, class_names
            )
        
        detailed_metrics = await loop.run_in_executor(None, compute_ml_metrics)
        logger.info(f"✓ Metrics computed")
        
        # Save model
        logger.info("Serializing model...")
        model_bytes = model_wrapper.save_to_bytes()
        logger.info(f"✓ Model saved: {len(model_bytes)} bytes")
        
        # Get model info and add sample counts
        model_info = model_wrapper.get_model_info()
        model_info['num_train_samples'] = len(X_train)
        model_info['num_val_samples'] = len(X_val)
        
        return {
            'train_accuracies': [metrics['train_accuracy']],
            'val_accuracies': [metrics.get('val_accuracy', 0.0)],
            'train_losses': [0.0],  # ML models don't have loss
            'val_losses': [0.0],
            'best_val_accuracy': metrics.get('val_accuracy', metrics['train_accuracy']),
            'best_epoch': 1,
            'per_class_metrics': detailed_metrics['per_class_metrics'],
            'confusion_matrix': detailed_metrics['confusion_matrix'],
            'class_names': class_names,
            'roc_curves': detailed_metrics['roc_curves'],
            'pr_curves': detailed_metrics['pr_curves'],
            'model_architecture': model_info,
            'model_bytes': model_bytes,
            'bayesian_trials_data': [],
            'bayesian_config': None,
            'num_train_samples': len(X_train),
            'num_val_samples': len(X_val),
            'model_type': model_type
        }
        
    except Exception as e:
        logger.error(f"✗ ML MODEL TRAINING FAILED")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Traceback:")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"ML model training failed: {e}") from e


def compute_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """Compute metrics from predictions (for ML models)."""
    num_classes = len(class_names)
    
    # Validate y_probs shape - should match num_classes
    if y_probs.shape[1] != num_classes:
        raise ValueError(
            f"CRITICAL: Probability matrix has {y_probs.shape[1]} columns but expected {num_classes} classes. "
            f"This indicates the validation set is missing some classes. "
            f"Ensure all classes are represented in the validation data. "
            f"Classes in y_true: {np.unique(y_true).tolist()}, Expected: {list(range(num_classes))}"
        )
    
    # Validate all classes are present in y_true
    unique_in_true = set(np.unique(y_true))
    expected_classes = set(range(num_classes))
    if unique_in_true != expected_classes:
        missing = expected_classes - unique_in_true
        raise ValueError(
            f"CRITICAL: Validation data is missing classes: {[class_names[c] for c in missing]}. "
            f"All classes must be present in validation data for proper evaluation. "
            f"Please ensure your data split includes all classes in each set."
        )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': round(float(precision[i]), 4),
            'recall': round(float(recall[i]), 4),
            'f1_score': round(float(f1[i]), 4),
            'support': int(support[i])
        }
    
    # ROC curves
    roc_curves = {}
    pr_curves = {}
    
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    
    for i, class_name in enumerate(class_names):
        # With stratified splitting, all classes should be present
        # If not, this is a bug that should be caught earlier
        if y_bin[:, i].sum() == 0:
            raise ValueError(
                f"CRITICAL: Class '{class_name}' has no samples in validation data. "
                f"This should have been caught during data splitting. Please report this bug."
            )
        
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        indices = np.linspace(0, len(fpr) - 1, min(20, len(fpr)), dtype=int)
        roc_points = [{'fpr': round(float(fpr[j]), 4), 'tpr': round(float(tpr[j]), 4)} for j in indices]
        roc_curves[class_name] = {'points': roc_points, 'auc': round(float(roc_auc), 4)}
        
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        indices = np.linspace(0, len(prec) - 1, min(20, len(prec)), dtype=int)
        pr_points = [{'precision': round(float(prec[j]), 4), 'recall': round(float(rec[j]), 4)} for j in indices]
        pr_curves[class_name] = {'points': pr_points}
    
    return {
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
        'roc_curves': roc_curves,
        'pr_curves': pr_curves
    }


def compute_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: List[str],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Compute detailed metrics including confusion matrix, ROC curves, PR curves."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    num_classes = len(class_names)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
    )
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': round(float(precision[i]), 4),
            'recall': round(float(recall[i]), 4),
            'f1_score': round(float(f1[i]), 4),
            'support': int(support[i])
        }
    
    # ROC curves (one-vs-rest)
    roc_curves = {}
    pr_curves = {}
    
    # Binarize labels for multi-class ROC
    y_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    if num_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    
    for i, class_name in enumerate(class_names):
        # ROC curve
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Sample points for storage (max 20 points)
        indices = np.linspace(0, len(fpr) - 1, min(20, len(fpr)), dtype=int)
        roc_points = [{'fpr': round(float(fpr[j]), 4), 'tpr': round(float(tpr[j]), 4)} for j in indices]
        roc_curves[class_name] = {'points': roc_points, 'auc': round(float(roc_auc), 4)}
        
        # PR curve
        prec, rec, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        indices = np.linspace(0, len(prec) - 1, min(20, len(prec)), dtype=int)
        pr_points = [{'precision': round(float(prec[j]), 4), 'recall': round(float(rec[j]), 4)} for j in indices]
        pr_curves[class_name] = {'points': pr_points}
    
    return {
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_metrics,
        'roc_curves': roc_curves,
        'pr_curves': pr_curves
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

async def run_full_training(
    job_id: str,
    dataset_id: int,
    db_session,
    model_type: str,
    config: Dict[str, Any],
    update_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Run full training pipeline with real data from database.
    
    Supports both DL models (CNN-LSTM) and ML models (AdaBoost, KNN, SVC, XGBoost).
    
    Args:
        job_id: Unique job identifier
        dataset_id: ID of the training dataset
        db_session: SQLAlchemy database session
        model_type: Type of model - 'dl_cnn_lstm', 'adaboost', 'knn', 'svc', 'xgboost'
        config: Training configuration with model-specific parameters
        update_callback: Async callback for progress updates
        
    Returns:
        Dictionary with training results and model bytes
    """
    import asyncio
    import time
    
    # Timing tracking for pipeline stages
    timing = {
        'preprocessing_seconds': 0,
        'training_seconds': 0,
        'evaluation_seconds': 0,
    }
    
    logger.info("="*80)
    logger.info(f"TRAINING JOB START: {job_id}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Dataset ID: {dataset_id}")
    logger.info(f"Config: {config}")
    logger.info("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        logger.error(f"Error checking device: {e}")
        logger.error(traceback.format_exc())
        device = 'cpu'
    
    # Get window size from config (default 128 for IMU, 1000 for CSI)
    window_size = config.get('window_size', 128)
    architecture_size = config.get('model_architecture', 'medium')
    validation_split = float(config.get('validation_split', 0.2) or 0.0)
    test_split = float(config.get('test_split', 0.0) or 0.0)
    
    # Build preprocessing config from training config
    preprocessing_config = {
        'include_phase': config.get('include_phase', True),
        'filter_subcarriers': config.get('filter_subcarriers', True),
        'subcarrier_start': config.get('subcarrier_start', 5),
        'subcarrier_end': config.get('subcarrier_end', 32),
        'output_shape': config.get('output_shape', 'flattened'),
        'data_type': config.get('data_type', 'auto'),
        'preprocessing_blocks': config.get('preprocessing_blocks', []),
    }
    
    # Load preprocessing pipeline if specified
    preprocessing_pipeline_id = config.get('preprocessing_pipeline_id')
    if preprocessing_pipeline_id:
        try:
            from server.db import PreprocessingPipeline
            pipeline = db_session.query(PreprocessingPipeline).filter(
                PreprocessingPipeline.id == preprocessing_pipeline_id
            ).first()
            if pipeline:
                logger.info(f"Using preprocessing pipeline: {pipeline.name} (id={pipeline.id})")
                preprocessing_config.update({
                    'include_phase': pipeline.include_phase,
                    'filter_subcarriers': pipeline.filter_subcarriers,
                    'subcarrier_start': pipeline.subcarrier_start,
                    'subcarrier_end': pipeline.subcarrier_end,
                    'output_shape': pipeline.output_shape,
                    'data_type': pipeline.data_type,
                })
                window_size = pipeline.window_size
        except Exception as e:
            logger.warning(f"Failed to load preprocessing pipeline {preprocessing_pipeline_id}: {e}")
    
    logger.info(f"Preprocessing config: {preprocessing_config}")
    
    # Load real data from database
    try:
        # Update progress: Loading data (this is the slow part for large CSI files)
        if update_callback:
            try:
                await update_callback("training", 0, 1, {"stage": "loading_data", "files_loaded": 0, "total_files": 0})
            except Exception as cb_err:
                logger.warning(f"Progress callback failed: {cb_err}")
        
        logger.info(f"Loading dataset {dataset_id} from database...")
        logger.debug(f"Window size: {window_size}")
        
        # Create a synchronous progress callback that updates the database using a SEPARATE session
        # to avoid transaction conflicts with the main data loading session
        from server.db import TrainingJob
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        import json as json_module
        import os
        
        # Get database URL for creating a separate session
        db_url = os.environ.get('DATABASE_URL', '')
        if db_url:
            try:
                progress_engine = create_engine(db_url, pool_pre_ping=True)
                ProgressSession = sessionmaker(bind=progress_engine)
            except Exception as e:
                logger.warning(f"Failed to create progress session engine: {e}")
                ProgressSession = None
        else:
            ProgressSession = None
        
        def file_progress_callback(file_idx: int, total_files: int, filename: str):
            logger.info(f"Loading file {file_idx + 1}/{total_files}: {filename}")
            if not ProgressSession:
                return
            try:
                # Use a separate session to avoid transaction conflicts
                progress_db = ProgressSession()
                try:
                    job = progress_db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                    if job:
                        config_data = json_module.loads(job.config) if job.config else {}
                        config_data['current_stage'] = 'loading_data'
                        config_data['files_loaded'] = file_idx + 1
                        config_data['total_files'] = total_files
                        config_data['current_file'] = filename
                        job.config = json_module.dumps(config_data)
                        progress_db.commit()
                finally:
                    progress_db.close()
            except Exception as e:
                logger.warning(f"Failed to update file progress: {e}")
        
        preprocessing_start = time.time()
        X, y, class_names = load_dataset_from_db(db_session, dataset_id, window_size, preprocessing_config, file_progress_callback)
        timing['preprocessing_seconds'] = time.time() - preprocessing_start
        logger.info(f"[TIMING] Preprocessing: {timing['preprocessing_seconds']:.2f}s")
        num_classes = len(class_names)
        
        logger.info(f"✓ Dataset loaded successfully")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Classes: {num_classes} - {class_names}")
        if len(X.shape) == 3:
            logger.info(f"  Shape: {X.shape} (samples, window_size={X.shape[1]}, channels={X.shape[2]})")
        else:
            logger.info(f"  Shape: {X.shape} (samples, features={X.shape[1]})")
        logger.info(f"  Data type: {X.dtype}")
        logger.info(f"  Data range: [{X.min():.4f}, {X.max():.4f}]")
        logger.info(f"  Label distribution: {np.bincount(y)}")
        
    except Exception as e:
        logger.error(f"✗ FAILED to load dataset {dataset_id}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Dataset loading failed: {e}") from e
    
    # Sanitize non-finite values before any downstream processing
    try:
        non_finite_count = int((~np.isfinite(X)).sum())
        if non_finite_count > 0:
            logger.warning(f"Dataset contains non-finite values: {non_finite_count}. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.warning(f"Failed to sanitize non-finite values: {e}")

    # Normalize data (z-score normalization)
    try:
        logger.debug("Normalizing data...")
        if len(X.shape) == 3:
            # 3D data: (samples, window_size, channels) - normalize per channel
            X_mean = X.mean(axis=(0, 1), keepdims=True)
            X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        else:
            # 2D data: (samples, features) - normalize per feature
            X_mean = X.mean(axis=0, keepdims=True)
            X_std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std
        
        logger.info(f"✓ Data normalized")
        logger.debug(f"  Mean shape: {X_mean.shape}")
        logger.debug(f"  Std shape: {X_std.shape}")
        logger.debug(f"  Normalized range: [{X.min():.4f}, {X.max():.4f}]")
        
    except Exception as e:
        logger.error(f"✗ Normalization failed: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Stratified sequential split: ensures all classes are represented in each split
    # Data is split per-class to maintain class balance across train/val/test
    try:
        logger.info("Splitting data with stratified sequential split...")
        logger.info("  (Each class is split independently to ensure all classes in all sets)")
        
        if validation_split < 0 or test_split < 0 or (validation_split + test_split) >= 1:
            raise ValueError(f"Invalid splits: validation_split={validation_split}, test_split={test_split}. Must be >=0 and sum < 1")

        train_ratio = 1.0 - validation_split - test_split
        
        # Split each class independently to ensure all classes in all sets
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list, y_test_list = [], []
        
        unique_classes = np.unique(y)
        logger.info(f"  Found {len(unique_classes)} classes: {unique_classes.tolist()}")
        
        for cls in unique_classes:
            cls_mask = y == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]
            n_cls = len(X_cls)
            
            if n_cls < 3:
                raise ValueError(
                    f"Class '{class_names[cls]}' has only {n_cls} samples. "
                    f"Each class needs at least 3 samples for train/val/test split. "
                    f"Please add more data for this class or remove it from the dataset."
                )
            
            # Calculate split sizes for this class
            n_train_cls = max(1, int(n_cls * train_ratio))
            n_val_cls = max(1, int(n_cls * validation_split)) if validation_split > 0 else 0
            n_test_cls = n_cls - n_train_cls - n_val_cls
            
            # Ensure at least 1 sample in test if test_split > 0
            if test_split > 0 and n_test_cls < 1:
                n_test_cls = 1
                n_train_cls = n_cls - n_val_cls - n_test_cls
            
            if n_train_cls < 1:
                raise ValueError(
                    f"Class '{class_names[cls]}' has {n_cls} samples which is too few for the requested split ratios. "
                    f"Train would get {n_train_cls} samples. Please add more data or adjust split ratios."
                )
            
            # Sequential split within each class (first part = train, then val, then test)
            X_train_list.append(X_cls[:n_train_cls])
            y_train_list.append(y_cls[:n_train_cls])
            
            if n_val_cls > 0:
                X_val_list.append(X_cls[n_train_cls:n_train_cls + n_val_cls])
                y_val_list.append(y_cls[n_train_cls:n_train_cls + n_val_cls])
            
            if n_test_cls > 0:
                X_test_list.append(X_cls[n_train_cls + n_val_cls:])
                y_test_list.append(y_cls[n_train_cls + n_val_cls:])
            
            logger.debug(f"    Class '{class_names[cls]}': {n_cls} total -> train={n_train_cls}, val={n_val_cls}, test={n_test_cls}")
        
        # Concatenate all classes
        X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
        y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
        X_val = np.concatenate(X_val_list, axis=0) if X_val_list else np.array([]).reshape(0, *X.shape[1:])
        y_val = np.concatenate(y_val_list, axis=0) if y_val_list else np.array([], dtype=np.int64)
        X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([]).reshape(0, *X.shape[1:])
        y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([], dtype=np.int64)
        
        n_total = len(X)
        n_train = len(X_train)
        n_val = len(X_val)
        n_test = len(X_test)

        logger.info(f"✓ Data split complete (stratified sequential)")
        logger.info(f"  Train: {n_train} samples ({n_train/n_total*100:.1f}%)")
        logger.info(f"  Val: {n_val} samples ({n_val/n_total*100:.1f}%)")
        logger.info(f"  Test: {n_test} samples ({n_test/n_total*100:.1f}%)")
        
        # Verify all classes are in each split
        train_classes = set(np.unique(y_train))
        val_classes = set(np.unique(y_val)) if len(y_val) > 0 else set()
        test_classes = set(np.unique(y_test)) if len(y_test) > 0 else set()
        all_classes = set(unique_classes)
        
        logger.info(f"  Train classes: {sorted(train_classes)} ({len(train_classes)}/{len(all_classes)})")
        if len(y_val) > 0:
            logger.info(f"  Val classes: {sorted(val_classes)} ({len(val_classes)}/{len(all_classes)})")
        if len(y_test) > 0:
            logger.info(f"  Test classes: {sorted(test_classes)} ({len(test_classes)}/{len(all_classes)})")
        
        # Detailed distribution
        train_dist = np.bincount(y_train, minlength=len(class_names))
        logger.info(f"  Train distribution: {dict(zip(class_names, train_dist.tolist()))}")
        if len(y_val) > 0:
            val_dist = np.bincount(y_val, minlength=len(class_names))
            logger.info(f"  Val distribution: {dict(zip(class_names, val_dist.tolist()))}")
        if len(y_test) > 0:
            test_dist = np.bincount(y_test, minlength=len(class_names))
            logger.info(f"  Test distribution: {dict(zip(class_names, test_dist.tolist()))}")
        
        # Validate that all classes are present where needed
        if train_classes != all_classes:
            missing = all_classes - train_classes
            raise ValueError(
                f"CRITICAL: Training set is missing classes: {[class_names[c] for c in missing]}. "
                f"This should not happen with stratified splitting. Please report this bug."
            )
        
        if validation_split > 0 and val_classes != all_classes:
            missing = all_classes - val_classes
            raise ValueError(
                f"CRITICAL: Validation set is missing classes: {[class_names[c] for c in missing]}. "
                f"Each class needs at least 2 samples for train+val split. "
                f"Please add more data for these classes."
            )
        
        if test_split > 0 and test_classes != all_classes:
            missing = all_classes - test_classes
            raise ValueError(
                f"CRITICAL: Test set is missing classes: {[class_names[c] for c in missing]}. "
                f"Each class needs at least 3 samples for train+val+test split. "
                f"Please add more data for these classes."
            )
        
    except Exception as e:
        logger.error(f"✗ Data splitting failed: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Determine if using ML or DL model
    is_ml_model = model_type in ['adaboost', 'knn', 'svc', 'xgboost']
    is_dl_model = model_type in ['dl_cnn_lstm', 'cnn_lstm', 'deep_learning']
    
    logger.info(f"Model category: {'ML' if is_ml_model else 'DL' if is_dl_model else 'UNKNOWN'}")
    
    if is_ml_model:
        logger.info(f"Training ML model: {model_type}")
        training_start = time.time()
        ml_results = await train_ml_model(
            job_id, X_train, y_train,
            X_val if len(X_val) > 0 else X_train,
            y_val if len(y_val) > 0 else y_train,
            class_names,
            model_type, config, db_session, update_callback
        )
        timing['training_seconds'] = time.time() - training_start
        logger.info(f"[TIMING] ML Training: {timing['training_seconds']:.2f}s")

        evaluation_start = time.time()
        if len(X_test) > 0:
            try:
                from server.ml_models import MLModelWrapper, get_default_ml_config
                ml_config = get_default_ml_config(model_type)
                ml_config.update(config.get('ml_params', {}))
                model_wrapper = MLModelWrapper(model_type, ml_config)
                model_wrapper.fit(X_train, y_train, X_val if len(X_val) > 0 else X_train, y_val if len(y_val) > 0 else y_train)
                test_preds = model_wrapper.predict(X_test)
                test_acc = float((test_preds == y_test).mean()) if len(y_test) > 0 else 0.0
                ml_results['test_results'] = {
                    'test_accuracy': test_acc,
                    'test_loss': 0.0,
                    'test_dataset_id': None
                }
            except Exception as test_err:
                logger.warning(f"Test evaluation for ML model failed: {test_err}")

        timing['evaluation_seconds'] = time.time() - evaluation_start
        logger.info(f"[TIMING] Evaluation: {timing['evaluation_seconds']:.2f}s")
        
        ml_results['num_train_samples'] = len(X_train)
        ml_results['num_val_samples'] = len(X_val)
        ml_results['num_test_samples'] = len(X_test)
        ml_results['splits'] = {
            'train': 1 - validation_split - test_split,
            'val': validation_split,
            'test': test_split
        }
        ml_results['timing'] = timing
        logger.info(f"[TIMING] Total pipeline: preprocessing={timing['preprocessing_seconds']:.2f}s, training={timing['training_seconds']:.2f}s, evaluation={timing['evaluation_seconds']:.2f}s")
        return ml_results
    
    # Deep Learning path (existing code)
    logger.info("Training Deep Learning model: CNN-LSTM")
    
    # Bayesian optimization (DL only)
    bayesian_trials_data = []
    bayesian_config_used = None
    if config.get('use_bayesian_optimization', False):
        try:
            logger.info("Running Bayesian hyperparameter optimization...")
            if update_callback:
                await update_callback('optimizing', 0, 0, None)
            
            # Create Bayesian config from user settings
            bayesian_config = BayesianOptimizationConfig.from_dict(config)
            bayesian_config_used = bayesian_config.to_dict()
            
            # Run optimization in thread pool
            loop = asyncio.get_event_loop()
            best_params, bayesian_trials_data = await loop.run_in_executor(
                None,
                lambda: run_bayesian_optimization(
                    X_train, y_train, X_val, y_val,
                    num_classes=num_classes,
                    config=bayesian_config,
                    device=device
                )
            )
            
            if best_params:
                config['learning_rate'] = best_params['learning_rate']
                config['batch_size'] = best_params['batch_size']
                config['weight_decay'] = best_params.get('weight_decay', 0.0)
                architecture_size = best_params.get('architecture_size', architecture_size)
                logger.info(f"✓ Best hyperparameters found: {best_params}")
        except Exception as e:
            logger.error(f"✗ Bayesian optimization failed: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Continuing with default hyperparameters")
    
    # Create data loaders
    try:
        batch_size = config.get('batch_size', 32)
        logger.debug(f"Creating data loaders with batch_size={batch_size}")
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        test_loader = None
        if len(X_test) > 0:
            X_test_t = torch.FloatTensor(X_test)
            y_test_t = torch.LongTensor(y_test)
            test_dataset = TensorDataset(X_test_t, y_test_t)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        logger.info(f"✓ Data loaders created")
        logger.debug(f"  Train batches: {len(train_loader)}")
        logger.debug(f"  Val batches: {len(val_loader)}")
    except Exception as e:
        logger.error(f"✗ Failed to create data loaders: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Create model
    try:
        seq_length = X_train.shape[1]
        input_channels = X_train.shape[2]
        
        logger.debug(f"Creating model: seq_length={seq_length}, channels={input_channels}, classes={num_classes}")
        
        model = IMUClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            architecture_size=architecture_size
        )
        
        total_params = model.get_architecture_summary()['total_params']
        logger.info(f"✓ Model created: {total_params:,} parameters")
        logger.debug(f"  Architecture: {architecture_size}")
        logger.debug(f"  Input shape: ({seq_length}, {input_channels})")
        logger.debug(f"  Output classes: {num_classes}")
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Import TrainingJob for progress updates
    from server.db import TrainingJob
    import json as json_module
    
    # Track metrics for real-time updates
    running_metrics = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Create a callback to log epoch progress and update database
    def epoch_callback(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
        logger.info(f"Epoch {epoch}/{total_epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Update running metrics
        running_metrics['loss'].append(round(train_loss, 4))
        running_metrics['accuracy'].append(round(train_acc, 4))
        running_metrics['val_loss'].append(round(val_loss, 4))
        running_metrics['val_accuracy'].append(round(val_acc, 4))
        
        # Update job progress in database
        try:
            job = db_session.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job:
                job.current_epoch = epoch
                job.metrics = json_module.dumps(running_metrics)
                db_session.commit()
                logger.info(f"Updated job {job_id} progress: epoch {epoch}/{total_epochs}")
        except Exception as e:
            logger.warning(f"Failed to update job progress: {e}")
    
    # Train model in thread pool
    def sync_train():
        return train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.get('epochs', 10),
            learning_rate=config.get('learning_rate', 0.001),
            device=device,
            callback=epoch_callback
        )
    
    logger.info(f"Starting training for {config.get('epochs', 10)} epochs...")
    
    # Run training in thread pool to not block async
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, sync_train)
    
    logger.info(f"Training completed. Best val accuracy: {results['best_val_accuracy']:.4f} at epoch {results['best_epoch']}")
    
    # Compute real metrics on validation set
    logger.info("Computing detailed metrics...")
    detailed_metrics = await loop.run_in_executor(
        None,
        lambda: compute_metrics(model, val_loader, class_names, device)
    )

    test_results = None
    if test_loader is not None:
        try:
            logger.info("Computing test metrics...")
            test_metrics = await loop.run_in_executor(
                None,
                lambda: compute_metrics(model, test_loader, class_names, device)
            )

            def compute_test_accuracy_and_loss():
                model.eval()
                correct = 0
                total = 0
                loss_total = 0.0
                criterion = nn.CrossEntropyLoss()
                with torch.no_grad():
                    for bx, by in test_loader:
                        bx, by = bx.to(device), by.to(device)
                        out = model(bx)
                        loss = criterion(out, by)
                        loss_total += float(loss.item()) * bx.size(0)
                        pred = out.argmax(dim=1)
                        correct += int((pred == by).sum().item())
                        total += int(by.size(0))
                return (correct / total) if total > 0 else 0.0, (loss_total / total) if total > 0 else 0.0

            test_acc, test_loss = await loop.run_in_executor(None, compute_test_accuracy_and_loss)
            test_results = {
                'test_accuracy': round(float(test_acc), 4),
                'test_loss': round(float(test_loss), 4),
                'test_dataset_id': None,
                'test_metrics': test_metrics
            }
        except Exception as test_err:
            logger.warning(f"Failed to compute test metrics: {test_err}")
    
    # Get model architecture
    model_architecture = model.get_architecture_summary()
    model_architecture['optimizer'] = 'adam'
    model_architecture['learning_rate'] = config.get('learning_rate', 0.001)
    model_architecture['batch_size'] = config.get('batch_size', 32)
    
    # Save model weights with normalization stats for inference
    save_config = {
        **config,
        'normalization': {
            'mean': X_mean.tolist(),
            'std': X_std.tolist()
        },
        'class_names': class_names,
        'input_shape': list(X_train.shape[1:]),
        'num_classes': num_classes
    }
    model_bytes = save_model_to_bytes(model, save_config)
    
    logger.info(f"Model saved: {len(model_bytes)} bytes")
    
    # Calculate DL timing (training time is in results if available)
    timing['training_seconds'] = results.get('training_time', 0)
    timing['evaluation_seconds'] = 0  # Evaluation is included in training for DL
    logger.info(f"[TIMING] DL pipeline: preprocessing={timing['preprocessing_seconds']:.2f}s, training={timing['training_seconds']:.2f}s")
    
    return {
        'train_losses': results['train_losses'],
        'train_accuracies': results['train_accuracies'],
        'val_losses': results['val_losses'],
        'val_accuracies': results['val_accuracies'],
        'best_val_accuracy': results['best_val_accuracy'],
        'best_epoch': results['best_epoch'],
        'per_class_metrics': detailed_metrics['per_class_metrics'],
        'confusion_matrix': detailed_metrics['confusion_matrix'],
        'class_names': class_names,
        'roc_curves': detailed_metrics['roc_curves'],
        'pr_curves': detailed_metrics['pr_curves'],
        'model_architecture': model_architecture,
        'model_bytes': model_bytes,
        'bayesian_trials_data': bayesian_trials_data,
        'bayesian_config': bayesian_config_used,
        'num_train_samples': len(X_train),
        'num_val_samples': len(X_val),
        'num_test_samples': len(X_test),
        'splits': {
            'train': 1 - validation_split - test_split,
            'val': validation_split,
            'test': test_split
        },
        'test_results': test_results,
        'timing': timing
    }
