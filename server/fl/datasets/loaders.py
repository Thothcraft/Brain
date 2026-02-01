"""Dataset Loading for Federated Learning using Flower Datasets.

This module provides functions to load and partition datasets for FL experiments.
Uses flwr-datasets for efficient federated data handling.
"""

import logging
from typing import Dict, Any, Tuple, Optional, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from torchvision import datasets as tv_datasets

from flwr_datasets import FederatedDataset

from ..core.config import FLDataset
from .partitioners import get_partitioner, PartitionStrategy

logger = logging.getLogger(__name__)


# Dataset name mapping for Flower Datasets (HuggingFace format)
DATASET_MAPPING = {
    FLDataset.CIFAR10: "uoft-cs/cifar10",
    FLDataset.CIFAR100: "uoft-cs/cifar100",
    FLDataset.MNIST: "ylecun/mnist",
    FLDataset.FASHION_MNIST: "zalando-datasets/fashion_mnist",
    FLDataset.SVHN: "ufldl-stanford/svhn",
    FLDataset.EMNIST: "emnist",
}

# Image key mapping - different HuggingFace datasets use different keys
DATASET_IMAGE_KEY = {
    FLDataset.CIFAR10: "img",
    FLDataset.CIFAR100: "img",
    FLDataset.MNIST: "image",
    FLDataset.FASHION_MNIST: "image",
    FLDataset.SVHN: "image",
    FLDataset.EMNIST: "image",
}

# Dataset metadata
DATASET_INFO = {
    FLDataset.CIFAR10: {
        "name": "CIFAR-10",
        "description": "60,000 32x32 color images in 10 classes",
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "train_samples": 50000,
        "test_samples": 10000,
        "task": "image_classification",
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    FLDataset.CIFAR100: {
        "name": "CIFAR-100",
        "description": "60,000 32x32 color images in 100 classes",
        "num_classes": 100,
        "input_shape": (3, 32, 32),
        "train_samples": 50000,
        "test_samples": 10000,
        "task": "image_classification",
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    FLDataset.MNIST: {
        "name": "MNIST",
        "description": "70,000 28x28 grayscale handwritten digits",
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_samples": 60000,
        "test_samples": 10000,
        "task": "image_classification",
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    FLDataset.FASHION_MNIST: {
        "name": "Fashion-MNIST",
        "description": "70,000 28x28 grayscale fashion items",
        "num_classes": 10,
        "input_shape": (1, 28, 28),
        "train_samples": 60000,
        "test_samples": 10000,
        "task": "image_classification",
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    FLDataset.SVHN: {
        "name": "SVHN",
        "description": "Street View House Numbers",
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "train_samples": 73257,
        "test_samples": 26032,
        "task": "image_classification",
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },
    FLDataset.EMNIST: {
        "name": "EMNIST",
        "description": "Extended MNIST with letters and digits",
        "num_classes": 62,
        "input_shape": (1, 28, 28),
        "train_samples": 697932,
        "test_samples": 116323,
        "task": "image_classification",
        "mean": (0.1751,),
        "std": (0.3332,),
    },
}


def get_dataset_info(dataset: FLDataset) -> Dict[str, Any]:
    """Get information about a dataset.
    
    Args:
        dataset: Dataset enum value
    
    Returns:
        Dictionary with dataset metadata
    """
    if isinstance(dataset, str):
        dataset = FLDataset(dataset)
    
    return DATASET_INFO.get(dataset, {
        "name": "Custom",
        "num_classes": 10,
        "input_shape": (3, 32, 32),
    })


def get_transforms(
    dataset: FLDataset,
    train: bool = True,
    augmentation: bool = True
) -> Callable:
    """Get PyTorch transforms for a dataset.
    
    Args:
        dataset: Dataset to get transforms for
        train: Whether this is for training (enables augmentation)
        augmentation: Whether to apply data augmentation
    
    Returns:
        Transform function
    """
    info = get_dataset_info(dataset)
    mean = info.get("mean", (0.5,))
    std = info.get("std", (0.5,))
    
    # RGB datasets
    if dataset in [FLDataset.CIFAR10, FLDataset.CIFAR100, FLDataset.SVHN]:
        if train and augmentation:
            return Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean, std),
            ])
        else:
            return Compose([
                ToTensor(),
                Normalize(mean, std),
            ])
    
    # Grayscale datasets
    else:
        return Compose([
            ToTensor(),
            Normalize(mean, std),
        ])


def apply_transforms(batch: Dict[str, Any], dataset: FLDataset, train: bool = True) -> Dict[str, Any]:
    """Apply PyTorch transforms to a batch from FederatedDataset.
    
    Args:
        batch: Batch from HuggingFace dataset (can be single example or batch)
        dataset: Dataset type for selecting correct transforms
        train: Whether this is training data
    
    Returns:
        Transformed batch with 'img' and 'label' keys
    """
    image_key = DATASET_IMAGE_KEY.get(dataset, "image")
    transforms = get_transforms(dataset, train=train)
    
    images = batch[image_key]
    
    # Handle both single examples and batches
    # with_transform can pass either a single image or a list of images
    if isinstance(images, list):
        # Batch of images
        transformed_images = [transforms(img) for img in images]
        batch["img"] = torch.stack(transformed_images)
    else:
        # Single image (PIL Image or similar)
        batch["img"] = transforms(images)
    
    # Ensure labels are tensors
    if "label" in batch:
        labels = batch["label"]
        if isinstance(labels, list):
            batch["label"] = torch.tensor(labels)
        elif not isinstance(labels, torch.Tensor):
            batch["label"] = torch.tensor(labels)
    
    return batch


def load_partition(
    partition_id: int,
    num_partitions: int,
    dataset: FLDataset = FLDataset.CIFAR10,
    partition_strategy: PartitionStrategy = PartitionStrategy.IID,
    batch_size: int = 32,
    dirichlet_alpha: float = 0.5,
    num_shards_per_partition: int = 2,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Load a single partition for a client.
    
    Uses torchvision directly to avoid Python 3.14 pickle compatibility
    issues with flwr-datasets/HuggingFace datasets.
    
    Args:
        partition_id: The partition ID (0 to num_partitions-1)
        num_partitions: Total number of partitions
        dataset: The dataset to use
        partition_strategy: How to partition the data
        batch_size: Batch size for data loaders
        dirichlet_alpha: Alpha parameter for Dirichlet partitioning
        num_shards_per_partition: Shards per partition for shard strategy
        val_split: Fraction of data to use for validation
        seed: Random seed
        num_workers: Number of data loader workers
    
    Returns:
        Tuple of (trainloader, valloader)
    """
    if isinstance(dataset, str):
        dataset = FLDataset(dataset)
    if isinstance(partition_strategy, str):
        partition_strategy = PartitionStrategy(partition_strategy)
    
    # Get dataset info for normalization
    info = DATASET_INFO.get(dataset, DATASET_INFO[FLDataset.CIFAR10])
    
    # Build transforms
    if len(info["input_shape"]) == 3 and info["input_shape"][0] == 3:
        # Color images - add augmentation
        train_transform = Compose([
            RandomCrop(info["input_shape"][1], padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(info["mean"], info["std"]),
        ])
    else:
        # Grayscale images
        train_transform = Compose([
            ToTensor(),
            Normalize(info["mean"], info["std"]),
        ])
    
    val_transform = Compose([
        ToTensor(),
        Normalize(info["mean"], info["std"]),
    ])
    
    # Load full training dataset using torchvision
    data_dir = "./data"
    
    if dataset == FLDataset.CIFAR10:
        full_trainset = tv_datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        full_valset = tv_datasets.CIFAR10(root=data_dir, train=True, download=True, transform=val_transform)
    elif dataset == FLDataset.CIFAR100:
        full_trainset = tv_datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        full_valset = tv_datasets.CIFAR100(root=data_dir, train=True, download=True, transform=val_transform)
    elif dataset == FLDataset.MNIST:
        full_trainset = tv_datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
        full_valset = tv_datasets.MNIST(root=data_dir, train=True, download=True, transform=val_transform)
    elif dataset == FLDataset.FASHION_MNIST:
        full_trainset = tv_datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
        full_valset = tv_datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=val_transform)
    elif dataset == FLDataset.SVHN:
        full_trainset = tv_datasets.SVHN(root=data_dir, split="train", download=True, transform=train_transform)
        full_valset = tv_datasets.SVHN(root=data_dir, split="train", download=True, transform=val_transform)
    else:
        logger.warning(f"Unknown dataset {dataset}, falling back to CIFAR-10")
        full_trainset = tv_datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        full_valset = tv_datasets.CIFAR10(root=data_dir, train=True, download=True, transform=val_transform)
    
    # Get labels for partitioning
    if hasattr(full_trainset, 'targets'):
        labels = np.array(full_trainset.targets)
    elif hasattr(full_trainset, 'labels'):
        labels = np.array(full_trainset.labels)
    else:
        # Fallback - iterate through dataset
        labels = np.array([full_trainset[i][1] for i in range(len(full_trainset))])
    
    num_samples = len(full_trainset)
    
    # Partition the data
    np.random.seed(seed)
    
    if partition_strategy == PartitionStrategy.IID:
        # IID: random uniform split
        indices = np.random.permutation(num_samples)
        partition_size = num_samples // num_partitions
        start_idx = partition_id * partition_size
        end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else num_samples
        partition_indices = indices[start_idx:end_idx]
    
    elif partition_strategy == PartitionStrategy.DIRICHLET:
        # Non-IID Dirichlet distribution
        num_classes = len(np.unique(labels))
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([dirichlet_alpha] * num_partitions, num_classes)
        
        partition_indices = []
        for c in range(num_classes):
            class_idx = class_indices[c]
            np.random.shuffle(class_idx)
            
            # Split class indices according to proportions
            splits = (proportions[c] * len(class_idx)).astype(int)
            splits[-1] = len(class_idx) - splits[:-1].sum()  # Ensure all samples used
            
            start = sum(splits[:partition_id])
            end = start + splits[partition_id]
            partition_indices.extend(class_idx[start:end])
        
        partition_indices = np.array(partition_indices)
        np.random.shuffle(partition_indices)
    
    else:
        # Default to IID for other strategies
        indices = np.random.permutation(num_samples)
        partition_size = num_samples // num_partitions
        start_idx = partition_id * partition_size
        end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else num_samples
        partition_indices = indices[start_idx:end_idx]
    
    # Split partition into train/val
    np.random.shuffle(partition_indices)
    val_size = int(len(partition_indices) * val_split)
    val_indices = partition_indices[:val_size]
    train_indices = partition_indices[val_size:]
    
    # Create subset datasets
    trainset = torch.utils.data.Subset(full_trainset, train_indices.tolist())
    valset = torch.utils.data.Subset(full_valset, val_indices.tolist())
    
    # Create data loaders
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    logger.debug(f"Loaded partition {partition_id}/{num_partitions}: "
                f"{len(trainset)} train, {len(valset)} val")
    
    return trainloader, valloader


def load_centralized_testset(
    dataset: FLDataset = FLDataset.CIFAR10,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Load centralized test set for server-side evaluation.
    
    Uses torchvision directly to avoid Python 3.14 pickle compatibility
    issues with flwr-datasets/HuggingFace datasets.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size for data loader
        num_workers: Number of data loader workers
    
    Returns:
        Test data loader
    """
    if isinstance(dataset, str):
        dataset = FLDataset(dataset)
    
    # Get dataset info for normalization
    info = DATASET_INFO.get(dataset, DATASET_INFO[FLDataset.CIFAR10])
    
    # Build transform
    transform = Compose([
        ToTensor(),
        Normalize(info["mean"], info["std"]),
    ])
    
    # Map to torchvision dataset class
    data_dir = "./data"
    
    if dataset == FLDataset.CIFAR10:
        testset = tv_datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == FLDataset.CIFAR100:
        testset = tv_datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == FLDataset.MNIST:
        testset = tv_datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == FLDataset.FASHION_MNIST:
        testset = tv_datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == FLDataset.SVHN:
        testset = tv_datasets.SVHN(root=data_dir, split="test", download=True, transform=transform)
    else:
        # Fallback to CIFAR-10
        logger.warning(f"Unknown dataset {dataset}, falling back to CIFAR-10")
        testset = tv_datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def load_public_dataset(
    dataset: FLDataset = FLDataset.CIFAR10,
    size: int = 5000,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Load a public dataset for knowledge distillation FL.
    
    Uses torchvision directly to avoid Python 3.14 pickle compatibility
    issues with flwr-datasets/HuggingFace datasets.
    
    Args:
        dataset: Dataset to load
        size: Number of samples for public dataset
        batch_size: Batch size for data loader
        seed: Random seed for sampling
        num_workers: Number of data loader workers
    
    Returns:
        Public dataset data loader
    """
    if isinstance(dataset, str):
        dataset = FLDataset(dataset)
    
    # Get dataset info for normalization
    info = DATASET_INFO.get(dataset, DATASET_INFO[FLDataset.CIFAR10])
    
    # Build transform
    transform = Compose([
        ToTensor(),
        Normalize(info["mean"], info["std"]),
    ])
    
    # Map to torchvision dataset class
    data_dir = "./data"
    
    if dataset == FLDataset.CIFAR10:
        testset = tv_datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == FLDataset.CIFAR100:
        testset = tv_datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == FLDataset.MNIST:
        testset = tv_datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == FLDataset.FASHION_MNIST:
        testset = tv_datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == FLDataset.SVHN:
        testset = tv_datasets.SVHN(root=data_dir, split="test", download=True, transform=transform)
    else:
        logger.warning(f"Unknown dataset {dataset}, falling back to CIFAR-10")
        testset = tv_datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    # Sample subset
    torch.manual_seed(seed)
    if len(testset) > size:
        indices = torch.randperm(len(testset))[:size].tolist()
        testset = torch.utils.data.Subset(testset, indices)
    
    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def get_label_distribution(
    partition_id: int,
    num_partitions: int,
    dataset: FLDataset = FLDataset.CIFAR10,
    partition_strategy: PartitionStrategy = PartitionStrategy.IID,
    dirichlet_alpha: float = 0.5,
) -> Dict[int, int]:
    """Get the label distribution for a partition.
    
    Useful for visualizing non-IID data distributions.
    
    Args:
        partition_id: Partition to analyze
        num_partitions: Total number of partitions
        dataset: Dataset being used
        partition_strategy: Partitioning strategy
        dirichlet_alpha: Alpha for Dirichlet partitioning
    
    Returns:
        Dictionary mapping label -> count
    """
    if isinstance(dataset, str):
        dataset = FLDataset(dataset)
    
    dataset_name = DATASET_MAPPING.get(dataset, "uoft-cs/cifar10")
    
    partitioner = get_partitioner(
        strategy=partition_strategy,
        num_partitions=num_partitions,
        alpha=dirichlet_alpha,
    )
    
    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": partitioner}
    )
    
    partition = fds.load_partition(partition_id)
    
    # Count labels
    label_counts = {}
    for item in partition:
        label = item["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    return label_counts


def get_all_label_distributions(
    num_partitions: int,
    dataset: FLDataset = FLDataset.CIFAR10,
    partition_strategy: PartitionStrategy = PartitionStrategy.IID,
    dirichlet_alpha: float = 0.5,
) -> Dict[int, Dict[int, int]]:
    """Get label distributions for all partitions.
    
    Args:
        num_partitions: Number of partitions
        dataset: Dataset being used
        partition_strategy: Partitioning strategy
        dirichlet_alpha: Alpha for Dirichlet partitioning
    
    Returns:
        Dictionary mapping partition_id -> {label -> count}
    """
    distributions = {}
    for pid in range(num_partitions):
        distributions[pid] = get_label_distribution(
            partition_id=pid,
            num_partitions=num_partitions,
            dataset=dataset,
            partition_strategy=partition_strategy,
            dirichlet_alpha=dirichlet_alpha,
        )
    return distributions
