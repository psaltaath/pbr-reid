"""Configuration system: dataclasses + YAML files + ``--section.key value`` CLI overrides."""

from .manager import ConfigManager, UnifiedConfig, create_default_config, load_config_from_yaml
from .training import TrainingConfig
from .model import ModelConfig
from .data import DataConfig
from .evaluation import EvaluationConfig
from .system import SystemConfig
from .loss import LossConfig

__all__ = [
    "ConfigManager",
    "UnifiedConfig",
    "create_default_config",
    "load_config_from_yaml",
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    "EvaluationConfig",
    "SystemConfig",
    "LossConfig",
]
