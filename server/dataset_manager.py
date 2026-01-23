"""Dataset Manager - Flexible dataset management with label-based operations.

Provides advanced dataset management including:
- Label-based file filtering and bulk adding
- Automatic file validation and metadata extraction
- Dataset versioning and statistics
- Train/val/test split management
"""

import os
import logging
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class DatasetStatus(str, Enum):
    """Dataset status."""
    DRAFT = "draft"
    READY = "ready"
    TRAINING = "training"
    ARCHIVED = "archived"


@dataclass
class FileEntry:
    """Represents a file in a dataset."""
    file_id: int
    filename: str
    file_path: str
    file_type: str
    label: str
    size_bytes: int
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    added_at: datetime = field(default_factory=datetime.now)


@dataclass
class DatasetSplit:
    """Train/val/test split configuration."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_by_label: bool = True
    random_seed: int = 42


@dataclass
class DatasetStatistics:
    """Statistics for a dataset."""
    total_files: int = 0
    total_size_bytes: int = 0
    labels: Dict[str, int] = field(default_factory=dict)
    file_types: Dict[str, int] = field(default_factory=dict)
    valid_files: int = 0
    invalid_files: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0


class DatasetManager:
    """Manages datasets with flexible file inclusion and label-based operations.
    
    Features:
    - Add files by label, file type, or bulk selection
    - Filter files by multiple criteria
    - Automatic validation and metadata extraction
    - Train/val/test split management
    - Dataset statistics and versioning
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self._datasets: Dict[int, Dict[str, Any]] = {}
        self._file_cache: Dict[int, List[FileEntry]] = {}
    
    def create_dataset(
        self,
        name: str,
        description: str = "",
        user_id: int = None,
        data_type: str = "auto",
        labels: List[str] = None,
    ) -> Dict[str, Any]:
        """Create a new dataset.
        
        Args:
            name: Dataset name
            description: Dataset description
            user_id: Owner user ID
            data_type: Expected data type (csi, image, video, etc.)
            labels: Expected labels for the dataset
            
        Returns:
            Dataset info dictionary
        """
        dataset_id = len(self._datasets) + 1
        
        dataset = {
            "id": dataset_id,
            "name": name,
            "description": description,
            "user_id": user_id,
            "data_type": data_type,
            "expected_labels": labels or [],
            "status": DatasetStatus.DRAFT.value,
            "files": [],
            "split_config": DatasetSplit().__dict__,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1,
        }
        
        self._datasets[dataset_id] = dataset
        self._file_cache[dataset_id] = []
        
        logger.info(f"Created dataset: {name} (ID: {dataset_id})")
        return dataset
    
    def add_files_by_label(
        self,
        dataset_id: int,
        label: str,
        file_ids: List[int] = None,
        file_paths: List[str] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Add files to dataset with a specific label.
        
        Args:
            dataset_id: Target dataset ID
            label: Label to assign to files
            file_ids: List of file IDs from database
            file_paths: List of file paths (alternative to file_ids)
            validate: Whether to validate files before adding
            
        Returns:
            Result with added/skipped counts
        """
        if dataset_id not in self._datasets:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        added = 0
        skipped = 0
        errors = []
        
        files_to_add = []
        
        # Process file IDs
        if file_ids:
            for fid in file_ids:
                entry = self._create_file_entry_from_id(fid, label)
                if entry:
                    files_to_add.append(entry)
                else:
                    skipped += 1
                    errors.append(f"File ID {fid} not found")
        
        # Process file paths
        if file_paths:
            for path in file_paths:
                entry = self._create_file_entry_from_path(path, label)
                if entry:
                    files_to_add.append(entry)
                else:
                    skipped += 1
                    errors.append(f"File not found: {path}")
        
        # Validate and add
        for entry in files_to_add:
            if validate and not entry.is_valid:
                skipped += 1
                errors.append(f"Invalid file: {entry.filename}")
                continue
            
            self._file_cache[dataset_id].append(entry)
            added += 1
        
        # Update dataset
        self._datasets[dataset_id]["updated_at"] = datetime.now().isoformat()
        
        return {
            "added": added,
            "skipped": skipped,
            "errors": errors,
            "total_files": len(self._file_cache[dataset_id]),
        }
    
    def add_files_bulk(
        self,
        dataset_id: int,
        files: List[Dict[str, Any]],
        default_label: str = "unlabeled",
    ) -> Dict[str, Any]:
        """Bulk add files with individual labels.
        
        Args:
            dataset_id: Target dataset ID
            files: List of dicts with file_id/file_path and optional label
            default_label: Default label if not specified
            
        Returns:
            Result with added/skipped counts
        """
        added = 0
        skipped = 0
        errors = []
        
        for file_info in files:
            label = file_info.get("label", default_label)
            file_id = file_info.get("file_id")
            file_path = file_info.get("file_path")
            
            try:
                if file_id:
                    result = self.add_files_by_label(dataset_id, label, file_ids=[file_id])
                elif file_path:
                    result = self.add_files_by_label(dataset_id, label, file_paths=[file_path])
                else:
                    skipped += 1
                    continue
                
                added += result["added"]
                skipped += result["skipped"]
                errors.extend(result["errors"])
            except Exception as e:
                skipped += 1
                errors.append(str(e))
        
        return {"added": added, "skipped": skipped, "errors": errors}
    
    def filter_files(
        self,
        dataset_id: int,
        labels: List[str] = None,
        file_types: List[str] = None,
        min_size: int = None,
        max_size: int = None,
        valid_only: bool = True,
    ) -> List[FileEntry]:
        """Filter files in a dataset by criteria.
        
        Args:
            dataset_id: Dataset ID
            labels: Filter by labels
            file_types: Filter by file types
            min_size: Minimum file size
            max_size: Maximum file size
            valid_only: Only return valid files
            
        Returns:
            List of matching FileEntry objects
        """
        if dataset_id not in self._file_cache:
            return []
        
        files = self._file_cache[dataset_id]
        
        if labels:
            files = [f for f in files if f.label in labels]
        
        if file_types:
            files = [f for f in files if f.file_type in file_types]
        
        if min_size is not None:
            files = [f for f in files if f.size_bytes >= min_size]
        
        if max_size is not None:
            files = [f for f in files if f.size_bytes <= max_size]
        
        if valid_only:
            files = [f for f in files if f.is_valid]
        
        return files
    
    def remove_files(
        self,
        dataset_id: int,
        file_ids: List[int] = None,
        labels: List[str] = None,
    ) -> int:
        """Remove files from dataset.
        
        Args:
            dataset_id: Dataset ID
            file_ids: Specific file IDs to remove
            labels: Remove all files with these labels
            
        Returns:
            Number of files removed
        """
        if dataset_id not in self._file_cache:
            return 0
        
        original_count = len(self._file_cache[dataset_id])
        
        if file_ids:
            self._file_cache[dataset_id] = [
                f for f in self._file_cache[dataset_id]
                if f.file_id not in file_ids
            ]
        
        if labels:
            self._file_cache[dataset_id] = [
                f for f in self._file_cache[dataset_id]
                if f.label not in labels
            ]
        
        removed = original_count - len(self._file_cache[dataset_id])
        self._datasets[dataset_id]["updated_at"] = datetime.now().isoformat()
        
        return removed
    
    def get_statistics(self, dataset_id: int) -> DatasetStatistics:
        """Get dataset statistics.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            DatasetStatistics object
        """
        if dataset_id not in self._file_cache:
            return DatasetStatistics()
        
        files = self._file_cache[dataset_id]
        
        stats = DatasetStatistics(
            total_files=len(files),
            total_size_bytes=sum(f.size_bytes for f in files),
            valid_files=sum(1 for f in files if f.is_valid),
            invalid_files=sum(1 for f in files if not f.is_valid),
        )
        
        # Count by label
        for f in files:
            stats.labels[f.label] = stats.labels.get(f.label, 0) + 1
        
        # Count by file type
        for f in files:
            stats.file_types[f.file_type] = stats.file_types.get(f.file_type, 0) + 1
        
        return stats
    
    def get_labels(self, dataset_id: int) -> List[str]:
        """Get unique labels in dataset."""
        if dataset_id not in self._file_cache:
            return []
        return list(set(f.label for f in self._file_cache[dataset_id]))
    
    def configure_split(
        self,
        dataset_id: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Configure train/val/test split.
        
        Args:
            dataset_id: Dataset ID
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify: Whether to stratify by label
            seed: Random seed
            
        Returns:
            Split configuration
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        split_config = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "stratify_by_label": stratify,
            "random_seed": seed,
        }
        
        self._datasets[dataset_id]["split_config"] = split_config
        return split_config
    
    def generate_split(self, dataset_id: int) -> Dict[str, List[int]]:
        """Generate train/val/test split indices.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dictionary with train/val/test file indices
        """
        import random
        
        if dataset_id not in self._file_cache:
            return {"train": [], "val": [], "test": []}
        
        files = self._file_cache[dataset_id]
        config = self._datasets[dataset_id].get("split_config", {})
        
        random.seed(config.get("random_seed", 42))
        
        train_ratio = config.get("train_ratio", 0.7)
        val_ratio = config.get("val_ratio", 0.15)
        stratify = config.get("stratify_by_label", True)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        if stratify:
            # Group by label
            label_groups = {}
            for i, f in enumerate(files):
                if f.label not in label_groups:
                    label_groups[f.label] = []
                label_groups[f.label].append(i)
            
            # Split each group
            for label, indices in label_groups.items():
                random.shuffle(indices)
                n = len(indices)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])
        else:
            # Simple random split
            indices = list(range(len(files)))
            random.shuffle(indices)
            
            n = len(indices)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
        
        # Update statistics
        stats = self.get_statistics(dataset_id)
        stats.train_samples = len(train_indices)
        stats.val_samples = len(val_indices)
        stats.test_samples = len(test_indices)
        
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
    
    def export_dataset_info(self, dataset_id: int) -> Dict[str, Any]:
        """Export dataset information for training.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Complete dataset info for training
        """
        if dataset_id not in self._datasets:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        dataset = self._datasets[dataset_id]
        files = self._file_cache.get(dataset_id, [])
        stats = self.get_statistics(dataset_id)
        split = self.generate_split(dataset_id)
        
        return {
            "dataset": dataset,
            "files": [
                {
                    "file_id": f.file_id,
                    "filename": f.filename,
                    "file_path": f.file_path,
                    "file_type": f.file_type,
                    "label": f.label,
                    "size_bytes": f.size_bytes,
                    "metadata": f.metadata,
                }
                for f in files
            ],
            "statistics": {
                "total_files": stats.total_files,
                "total_size_bytes": stats.total_size_bytes,
                "labels": stats.labels,
                "file_types": stats.file_types,
            },
            "split": split,
            "label_mapping": {label: i for i, label in enumerate(sorted(stats.labels.keys()))},
        }
    
    def _create_file_entry_from_id(self, file_id: int, label: str) -> Optional[FileEntry]:
        """Create FileEntry from database file ID."""
        # In real implementation, query database
        return FileEntry(
            file_id=file_id,
            filename=f"file_{file_id}",
            file_path=f"/data/file_{file_id}",
            file_type="unknown",
            label=label,
            size_bytes=0,
        )
    
    def _create_file_entry_from_path(self, path: str, label: str) -> Optional[FileEntry]:
        """Create FileEntry from file path."""
        if not os.path.exists(path):
            return None
        
        filename = os.path.basename(path)
        size = os.path.getsize(path)
        
        # Detect file type from extension
        ext = os.path.splitext(filename)[1].lower()
        file_type_map = {
            ".csv": "csv",
            ".json": "json",
            ".jpg": "image",
            ".jpeg": "image",
            ".png": "image",
            ".mp4": "video",
            ".avi": "video",
            ".wav": "audio",
            ".mp3": "audio",
        }
        file_type = file_type_map.get(ext, "unknown")
        
        return FileEntry(
            file_id=hash(path) % 1000000,
            filename=filename,
            file_path=path,
            file_type=file_type,
            label=label,
            size_bytes=size,
        )


# Global instance
_dataset_manager: Optional[DatasetManager] = None


def get_dataset_manager() -> DatasetManager:
    """Get global dataset manager instance."""
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager
