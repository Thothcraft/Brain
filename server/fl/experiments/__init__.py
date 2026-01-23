"""Experiments Module for Federated Learning.

This module provides:
- FLExperimentRunner: Run single or multiple FL experiments
- Default pipelines: Pre-configured experiment setups
- Reports: Per-model and comparative analysis
- Multi-run support with statistical analysis
"""

from .runner import (
    FLExperimentRunner,
    ExperimentResult,
    RunResult,
)
from .pipelines import (
    DEFAULT_PIPELINES,
    create_experiment,
    get_pipeline,
    list_pipelines,
    pipeline_to_experiments,
)
from .reports import (
    generate_experiment_report,
    generate_comparative_report,
    generate_statistical_summary,
    format_report_as_text,
)

__all__ = [
    "FLExperimentRunner",
    "ExperimentResult",
    "RunResult",
    "DEFAULT_PIPELINES",
    "create_experiment",
    "get_pipeline",
    "list_pipelines",
    "pipeline_to_experiments",
    "generate_experiment_report",
    "generate_comparative_report",
    "generate_statistical_summary",
    "format_report_as_text",
]
