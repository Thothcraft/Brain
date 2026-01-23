"""Datasets Module for Federated Learning.

This module provides dataset loading and partitioning using Flower Datasets (flwr-datasets).
Supports various partitioning strategies for simulating non-IID data distributions.

Supported Datasets:
- CIFAR-10, CIFAR-100
- MNIST, Fashion-MNIST, EMNIST
- SVHN
- Custom datasets

Partitioning Strategies:
- IID: Random uniform distribution
- Dirichlet: Label distribution skew via Dirichlet distribution
- Shard: Each client gets specific label shards
- Pathological: Extreme non-IID (few classes per client)
"""

from .loaders import (
    load_partition,
    load_centralized_testset,
    load_public_dataset,
    get_dataset_info,
    get_all_label_distributions,
    DATASET_MAPPING,
    DATASET_INFO,
)
from .partitioners import (
    get_partitioner,
    PartitionStrategy,
)
from .preprocessing import (
    FLPreprocessor,
    create_fl_transform,
    get_preprocessing_blocks,
    get_default_pipelines,
)
from ..core.config import FLDataset

__all__ = [
    "load_partition",
    "load_centralized_testset",
    "load_public_dataset",
    "get_dataset_info",
    "get_all_label_distributions",
    "get_partitioner",
    "DATASET_MAPPING",
    "DATASET_INFO",
    "FLDataset",
    "PartitionStrategy",
    # Preprocessing
    "FLPreprocessor",
    "create_fl_transform",
    "get_preprocessing_blocks",
    "get_default_pipelines",
]
