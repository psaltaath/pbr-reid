"""
Base configuration classes and utilities for handling YAML configuration files.
"""

import yaml
from typing import Any, Dict, Type, TypeVar, Union
from pathlib import Path
from dataclasses import dataclass, fields, is_dataclass
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""

    pass


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        file_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        ConfigError: If the file cannot be loaded or parsed
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        logger.info(f"Loaded configuration from {file_path}")
        return config

    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML file {file_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration file {file_path}: {e}")


def save_yaml_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        file_path: Path where to save the configuration
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {file_path}")

    except Exception as e:
        raise ConfigError(f"Error saving configuration to {file_path}: {e}")


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary to merge (takes precedence)

    Returns:
        Merged configuration dictionary
    """
    if not isinstance(override_config, dict):
        return override_config

    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


@dataclass
class BaseConfig:
    """Base configuration class with YAML loading/saving capabilities."""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """
        Create configuration instance from dictionary.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            Configuration instance
        """
        if not is_dataclass(cls):
            raise ConfigError(f"{cls.__name__} is not a dataclass")

        field_names = {f.name for f in fields(cls)}
        valid_args = {k: v for k, v in config_dict.items() if k in field_names}

        # Log ignored keys
        ignored_keys = set(config_dict.keys()) - field_names
        if ignored_keys:
            logger.warning(
                f"Ignoring unknown configuration keys in {cls.__name__}: {ignored_keys}"
            )

        return cls(**valid_args)

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "BaseConfig":
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            Configuration instance
        """
        config_dict = load_yaml_config(file_path)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if not is_dataclass(self):
            raise ConfigError(f"{self.__class__.__name__} is not a dataclass")

        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if is_dataclass(value):
                result[field.name] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[field.name] = [
                    item.to_dict() if is_dataclass(item) else item for item in value
                ]
            else:
                result[field.name] = value

        return result

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            file_path: Path where to save the configuration
        """
        save_yaml_config(self.to_dict(), file_path)

    def merge_with(self: T, other: Union[T, Dict[str, Any]]) -> T:
        """
        Merge this configuration with another configuration or dictionary.

        Args:
            other: Configuration instance or dictionary to merge

        Returns:
            New configuration instance with merged values
        """
        base_dict = self.to_dict()

        if is_dataclass(other):
            override_dict = other.to_dict()
        elif isinstance(other, dict):
            override_dict = other
        else:
            raise ConfigError(f"Cannot merge with object of type {type(other)}")

        merged_dict = merge_configs(base_dict, override_dict)
        return self.__class__.from_dict(merged_dict)


def validate_config_field(value: Any, expected_type: Type, field_name: str) -> None:
    """
    Validate that a configuration field has the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type
        field_name: Name of the field being validated

    Raises:
        ConfigError: If validation fails
    """
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"Configuration field '{field_name}' expected {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
