"""Configuration dataclasses for Federated Learning.

All FL configuration is centralized here for easy modification and extension.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class FLAlgorithm(str, Enum):
    """Supported Federated Learning algorithms from Flower."""
    # Standard algorithms
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"
    FEDADAGRAD = "fedadagrad"
    FEDAVGM = "fedavgm"
    FEDOPT = "fedopt"
    # Byzantine-robust
    FEDMEDIAN = "fedmedian"
    FEDTRIMMEDAVG = "fedtrimmedavg"
    KRUM = "krum"
    BULYAN = "bulyan"
    # Fair FL
    QFEDAVG = "qfedavg"
    # Privacy-preserving
    DPFEDAVG_ADAPTIVE = "dpfedavg_adaptive"
    DPFEDAVG_FIXED = "dpfedavg_fixed"
    # Knowledge Distillation (heterogeneous models)
    FEDDF = "feddf"
    FEDMD = "fedmd"
    FEDGEN = "fedgen"


class FLDataset(str, Enum):
    """Built-in datasets for federated learning."""
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    EMNIST = "emnist"
    SVHN = "svhn"
    CELEBA = "celeba"
    SHAKESPEARE = "shakespeare"
    FEMNIST = "femnist"
    CUSTOM = "custom"


class PartitionStrategy(str, Enum):
    """Data partitioning strategies for non-IID simulation."""
    IID = "iid"
    NON_IID_LABEL = "non_iid_label"
    NON_IID_DIRICHLET = "non_iid_dirichlet"
    NON_IID_QUANTITY = "non_iid_quantity"
    PATHOLOGICAL = "pathological"
    SHARD = "shard"


class ModelArchitecture(str, Enum):
    """Supported model architectures for FL."""
    # Image models
    CNN = "cnn"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    MOBILENET = "mobilenet"
    MOBILENET_V2 = "mobilenet_v2"
    EFFICIENTNET = "efficientnet"
    EFFICIENTNET_B0 = "efficientnet_b0"
    VGG16 = "vgg16"
    # Sequence/time-series models
    LSTM = "lstm"
    GRU = "gru"
    CNN_LSTM = "cnn_lstm"
    TRANSFORMER = "transformer"
    TCN = "tcn"
    # Simple models
    MLP = "mlp"
    LOGISTIC = "logistic"
    # Custom
    CUSTOM = "custom"


class AggregationMethod(str, Enum):
    """Model aggregation methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    BULYAN = "bulyan"
    GEOMETRIC_MEDIAN = "geometric_median"


class ClientSelectionStrategy(str, Enum):
    """Client selection strategies."""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    POWER_OF_CHOICE = "power_of_choice"
    OORT = "oort"
    CLUSTERED = "clustered"
    ACTIVE = "active"


@dataclass
class PrivacyConfig:
    """Differential privacy and secure aggregation settings."""
    differential_privacy: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    secure_aggregation: bool = False
    encryption_bits: int = 256
    min_clients_for_aggregation: int = 3


@dataclass
class ClientConfig:
    """Per-client training configuration."""
    local_epochs: int = 5
    local_batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer: str = "sgd"  # sgd, adam, adamw
    lr_scheduler: Optional[str] = None  # step, cosine, exponential
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    gradient_clipping: Optional[float] = None
    data_augmentation: bool = True


@dataclass
class ServerConfig:
    """Server-side FL configuration."""
    num_rounds: int = 100
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 0.5
    accept_failures: bool = True
    
    def __post_init__(self):
        min_required = max(self.min_fit_clients, self.min_evaluate_clients)
        if self.min_available_clients < min_required:
            self.min_available_clients = min_required


@dataclass
class AlgorithmConfig:
    """Algorithm-specific hyperparameters."""
    # FedProx
    proximal_mu: float = 0.01
    
    # FedAdam/FedYogi/FedAdagrad (server-side optimizer)
    server_learning_rate: float = 1.0
    beta_1: float = 0.9
    beta_2: float = 0.99
    tau: float = 1e-3
    
    # FedAvgM
    server_momentum: float = 0.9
    
    # QFedAvg (fairness)
    q_param: float = 0.2
    
    # Byzantine-robust
    byzantine_fraction: float = 0.0
    trimmed_mean_beta: float = 0.1
    krum_num_closest: int = 2
    
    # Knowledge Distillation
    temperature: float = 3.0
    distillation_weight: float = 0.5
    public_dataset_size: int = 5000


@dataclass
class DataConfig:
    """Dataset and partitioning configuration."""
    dataset: FLDataset = FLDataset.CIFAR10
    num_partitions: int = 10
    partition_strategy: PartitionStrategy = PartitionStrategy.IID
    dirichlet_alpha: float = 0.5
    num_shards_per_partition: int = 2
    min_samples_per_client: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    custom_data_path: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Training monitoring and logging configuration."""
    log_interval: int = 1
    checkpoint_interval: int = 10
    early_stopping_patience: int = 20
    early_stopping_metric: str = "accuracy"
    early_stopping_mode: str = "max"
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    algorithm: FLAlgorithm = FLAlgorithm.FEDAVG
    model: ModelArchitecture = ModelArchitecture.CNN
    
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    algorithm_params: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    data: DataConfig = field(default_factory=DataConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Experiment settings
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps
    num_workers: int = 4
    pin_memory: bool = True
    
    # Multi-run settings
    num_runs: int = 1  # Number of times to repeat this experiment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "algorithm": self.algorithm.value,
            "model": self.model.value,
            "server": {
                "num_rounds": self.server.num_rounds,
                "min_fit_clients": self.server.min_fit_clients,
                "fraction_fit": self.server.fraction_fit,
            },
            "client": {
                "local_epochs": self.client.local_epochs,
                "learning_rate": self.client.learning_rate,
                "batch_size": self.client.local_batch_size,
            },
            "data": {
                "dataset": self.data.dataset.value,
                "num_partitions": self.data.num_partitions,
                "partition_strategy": self.data.partition_strategy.value,
            },
            "seed": self.seed,
            "num_runs": self.num_runs,
        }


# Alias for backward compatibility
FLConfig = ExperimentConfig
