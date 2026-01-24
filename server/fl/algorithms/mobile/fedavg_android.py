"""FedAvgAndroid - FedAvg Optimized for Android Mobile Devices.

Flower's implementation of FedAvg optimized for Android mobile devices.

Verification References:
- Flower Documentation: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvgAndroid.html
- Flower Android SDK: https://flower.ai/docs/framework/how-to-run-flower-on-android.html
- Parameters verified: Same as FedAvg (fraction_fit, fraction_evaluate, etc.)
- Note: Optimized for Android TFLite models; may not be available in all Flower versions
"""

import logging
from typing import Dict, Any

from flwr.server.strategy import FedAvg, Strategy

from ...core.config import ExperimentConfig, FLAlgorithm
from ..base import BaseStrategyWrapper, AlgorithmMetadata, register_algorithm

logger = logging.getLogger(__name__)

# Try to import FedAvgAndroid (may not be available in all Flower versions)
try:
    from flwr.server.strategy import FedAvgAndroid
    HAS_FEDAVG_ANDROID = True
except ImportError:
    FedAvgAndroid = None
    HAS_FEDAVG_ANDROID = False


@register_algorithm
class FedAvgAndroidWrapper(BaseStrategyWrapper):
    """FedAvgAndroid strategy wrapper.
    
    FedAvgAndroid is optimized for Android mobile devices with
    battery-efficient communication patterns.
    """
    
    algorithm_id = FLAlgorithm.FEDAVG_ANDROID
    flower_class = FedAvgAndroid if HAS_FEDAVG_ANDROID else None
    
    @classmethod
    def get_metadata(cls) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="FedAvg Android",
            description="FedAvg optimized for Android mobile devices",
            paper="Flower Team",
            category="mobile",
            params=[],
            pros=["Mobile-optimized", "Battery efficient"],
            cons=["Android-specific"],
        )
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if FedAvgAndroid is available."""
        return HAS_FEDAVG_ANDROID
    
    @classmethod
    def create_strategy(
        cls,
        config: ExperimentConfig,
        common_params: Dict[str, Any],
    ) -> Strategy:
        """Create FedAvgAndroid strategy (falls back to FedAvg if unavailable)."""
        if not HAS_FEDAVG_ANDROID:
            logger.warning("FedAvgAndroid not available, falling back to FedAvg")
            return FedAvg(**common_params)
        
        return FedAvgAndroid(**common_params)
