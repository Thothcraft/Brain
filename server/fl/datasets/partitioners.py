"""Data Partitioning Strategies for Federated Learning.

This module provides partitioners for simulating various data distributions
in federated learning scenarios using Flower Datasets.
"""

import logging
from typing import Optional
from enum import Enum

from flwr_datasets.partitioner import (
    Partitioner,
    IidPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
    PathologicalPartitioner,
)

logger = logging.getLogger(__name__)


class PartitionStrategy(str, Enum):
    """Data partitioning strategies for non-IID simulation."""
    IID = "iid"
    NON_IID_LABEL = "non_iid_label"
    NON_IID_DIRICHLET = "non_iid_dirichlet"
    PATHOLOGICAL = "pathological"
    SHARD = "shard"


def get_partitioner(
    strategy: PartitionStrategy,
    num_partitions: int,
    alpha: float = 0.5,
    num_shards_per_partition: int = 2,
    num_classes_per_partition: int = 2,
    partition_by: str = "label",
    min_partition_size: int = 10,
    seed: int = 42,
) -> Partitioner:
    """Create a Flower Datasets partitioner based on strategy.
    
    Args:
        strategy: Partitioning strategy to use
        num_partitions: Number of partitions (clients)
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        num_shards_per_partition: Number of label shards per client (for shard strategy)
        num_classes_per_partition: Number of classes per client (for pathological)
        partition_by: Column name to partition by (usually "label")
        min_partition_size: Minimum samples per partition
        seed: Random seed for reproducibility
    
    Returns:
        Flower Datasets Partitioner instance
    
    Strategy Details:
    - IID: Each client gets a random uniform sample of all classes
    - Dirichlet: Label distribution follows Dirichlet(alpha) - lower alpha = more skew
    - Shard: Each client gets specific label shards (extreme non-IID)
    - Pathological: Each client only has a few classes (most extreme)
    """
    strategy_str = strategy.value if isinstance(strategy, PartitionStrategy) else strategy
    
    if strategy_str == "iid":
        logger.info(f"Creating IID partitioner with {num_partitions} partitions")
        return IidPartitioner(num_partitions=num_partitions)
    
    elif strategy_str in ["non_iid_dirichlet", "dirichlet"]:
        logger.info(f"Creating Dirichlet partitioner: alpha={alpha}, partitions={num_partitions}")
        return DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=partition_by,
            alpha=alpha,
            min_partition_size=min_partition_size,
            self_balancing=True,
            seed=seed,
        )
    
    elif strategy_str in ["non_iid_label", "shard"]:
        logger.info(f"Creating Shard partitioner: {num_shards_per_partition} shards/client")
        return ShardPartitioner(
            num_partitions=num_partitions,
            partition_by=partition_by,
            num_shards_per_partition=num_shards_per_partition,
            seed=seed,
        )
    
    elif strategy_str == "pathological":
        logger.info(f"Creating Pathological partitioner: {num_classes_per_partition} classes/client")
        return PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by=partition_by,
            num_classes_per_partition=num_classes_per_partition,
            seed=seed,
        )
    
    else:
        logger.warning(f"Unknown strategy '{strategy_str}', defaulting to IID")
        return IidPartitioner(num_partitions=num_partitions)


def describe_partition_strategy(strategy: PartitionStrategy) -> dict:
    """Get description and recommended parameters for a partition strategy.
    
    Returns:
        Dictionary with strategy description and parameter recommendations
    """
    descriptions = {
        PartitionStrategy.IID: {
            "name": "IID (Independent and Identically Distributed)",
            "description": "Each client receives a random uniform sample of all classes. "
                          "This is the simplest case and serves as a baseline.",
            "non_iid_level": "None",
            "recommended_for": "Baseline experiments, homogeneous data scenarios",
            "parameters": {},
        },
        PartitionStrategy.NON_IID_DIRICHLET: {
            "name": "Dirichlet Non-IID",
            "description": "Label distribution follows Dirichlet(alpha). Lower alpha values "
                          "create more skewed distributions (more non-IID).",
            "non_iid_level": "Configurable (alpha)",
            "recommended_for": "Realistic non-IID scenarios, research experiments",
            "parameters": {
                "alpha": {
                    "description": "Concentration parameter",
                    "range": [0.1, 100],
                    "recommended": 0.5,
                    "effect": "Lower = more non-IID, Higher = more IID",
                }
            },
        },
        PartitionStrategy.SHARD: {
            "name": "Shard-based Non-IID",
            "description": "Each client receives specific label shards. Creates moderate "
                          "non-IID distribution where clients have subset of classes.",
            "non_iid_level": "Moderate to High",
            "recommended_for": "Controlled non-IID experiments",
            "parameters": {
                "num_shards_per_partition": {
                    "description": "Number of label shards per client",
                    "range": [1, 10],
                    "recommended": 2,
                    "effect": "Fewer shards = more non-IID",
                }
            },
        },
        PartitionStrategy.PATHOLOGICAL: {
            "name": "Pathological Non-IID",
            "description": "Each client only has samples from a few classes. This is the "
                          "most extreme non-IID setting.",
            "non_iid_level": "Extreme",
            "recommended_for": "Stress testing FL algorithms, worst-case scenarios",
            "parameters": {
                "num_classes_per_partition": {
                    "description": "Number of classes per client",
                    "range": [1, 5],
                    "recommended": 2,
                    "effect": "Fewer classes = more extreme non-IID",
                }
            },
        },
    }
    
    return descriptions.get(strategy, descriptions[PartitionStrategy.IID])
