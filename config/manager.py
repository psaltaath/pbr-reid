"""Unified configuration: base.yaml + named config(s) + extra files + CLI overrides."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseConfig, ConfigError, load_yaml_config, merge_configs
from .data import DataConfig
from .evaluation import EvaluationConfig
from .loss import LossConfig
from .model import ModelConfig
from .system import SystemConfig
from .training import TrainingConfig

# <repo>/configs
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass
class UnifiedConfig(BaseConfig):
    """All sub-configurations."""

    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
    evaluation: EvaluationConfig
    system: SystemConfig
    loss: LossConfig

    def __post_init__(self):
        if isinstance(self.training, dict):
            self.training = TrainingConfig.from_dict(self.training)
        if isinstance(self.model, dict):
            self.model = ModelConfig.from_dict(self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig.from_dict(self.data)
        if isinstance(self.evaluation, dict):
            self.evaluation = EvaluationConfig.from_dict(self.evaluation)
        if isinstance(self.system, dict):
            self.system = SystemConfig.from_dict(self.system)
        if isinstance(self.loss, dict):
            self.loss = LossConfig.from_dict(self.loss)


def resolve_path(value: str, env_var: Optional[str] = None) -> str:
    """Expand ``~`` / ``$VARS`` in a configured path; ``env_var`` overrides it when set."""
    if env_var and os.environ.get(env_var):
        value = os.environ[env_var]
    return os.path.expanduser(os.path.expandvars(value))


class ConfigManager:
    """Load ``configs/base.yaml``, then named configs, then extra YAML files, then CLI overrides.

    Named configs are looked up recursively under the config dir, so
    ``market1501`` resolves to ``configs/datasets/market1501.yaml`` and
    ``vitl16`` to ``configs/backbones/vitl16.yaml``. A name that is an existing
    file path is loaded directly.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self.base_config_path = self.config_dir / "base.yaml"

    def resolve(self, name: str) -> Path:
        p = Path(name)
        if p.suffix in (".yaml", ".yml") and p.exists():
            return p
        matches = sorted(self.config_dir.rglob(f"{name}.yaml"))
        if not matches:
            raise ConfigError(
                f"Named configuration '{name}' not found under {self.config_dir} "
                f"(available: {', '.join(self.list_available_configs())})"
            )
        if len(matches) > 1:
            raise ConfigError(f"Ambiguous configuration name '{name}': {matches}")
        return matches[0]

    def load_config(
        self,
        config_name: Optional[Union[str, List[str]]] = None,
        config_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        cli_args: Optional[List[str]] = None,
    ) -> UnifiedConfig:
        config_dict: Dict[str, Any] = {}
        if self.base_config_path.exists():
            config_dict = load_yaml_config(self.base_config_path)

        names = [config_name] if isinstance(config_name, str) else (config_name or [])
        for name in names:
            config_dict = merge_configs(config_dict, load_yaml_config(self.resolve(name)))

        paths = [config_path] if isinstance(config_path, (str, Path)) else (config_path or [])
        for path in paths:
            config_dict = merge_configs(config_dict, load_yaml_config(path))

        if overrides:
            config_dict = merge_configs(config_dict, overrides)

        if cli_args:
            config_dict = merge_configs(config_dict, self._parse_cli_overrides(cli_args))

        return UnifiedConfig.from_dict(config_dict)

    def list_available_configs(self) -> List[str]:
        names = []
        for f in sorted(self.config_dir.rglob("*.yaml")):
            if f.name == "base.yaml":
                continue
            names.append(str(f.relative_to(self.config_dir).with_suffix("")))
        return names

    # ------------------------------------------------------------------ CLI --
    _RESERVED = {"config-file", "list-configs", "checkpoint", "output", "rerank"}

    def _parse_cli_overrides(self, cli_args: List[str]) -> Dict[str, Any]:
        """``--section.key value`` / ``--section.key=value`` / ``--section.flag`` -> nested dict."""
        overrides: Dict[str, Any] = {}
        i = 0
        while i < len(cli_args):
            arg = cli_args[i]
            if arg.startswith("--"):
                key = arg[2:]
                if "=" in key:
                    key, value = key.split("=", 1)
                    if "." in key:
                        overrides[key] = self._parse_value(value)
                elif i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
                    if "." in key:
                        overrides[key] = self._parse_value(cli_args[i + 1])
                    i += 1
                elif "." in key:
                    overrides[key] = True
            i += 1
        return self._expand_nested_keys(overrides)

    @staticmethod
    def _parse_value(value_str: str) -> Any:
        low = value_str.lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        if low in ("none", "null"):
            return None
        try:
            f = float(value_str)
            if f.is_integer() and "." not in value_str and "e" not in low:
                return int(f)
            return f
        except ValueError:
            pass
        if "," in value_str:
            return [ConfigManager._parse_value(s.strip()) for s in value_str.split(",")]
        return value_str

    @staticmethod
    def _expand_nested_keys(flat: Dict[str, Any]) -> Dict[str, Any]:
        nested: Dict[str, Any] = {}
        for key, value in flat.items():
            parts = key.split(".")
            cur = nested
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
        return nested


def load_config_from_yaml(file_path: Union[str, Path]) -> UnifiedConfig:
    return UnifiedConfig.from_dict(load_yaml_config(file_path))


def create_default_config() -> UnifiedConfig:
    return UnifiedConfig(
        training=TrainingConfig(),
        model=ModelConfig(),
        data=DataConfig(),
        evaluation=EvaluationConfig(),
        system=SystemConfig(),
        loss=LossConfig(),
    )
