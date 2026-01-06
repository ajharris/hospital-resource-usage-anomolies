"""
Configuration management utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
        self._load_defaults()
        self._load_env_overrides()
    
    def _load_defaults(self):
        """Load default configuration values."""
        defaults = {
            'data_dir': 'data',
            'cache_enabled': True,
            'log_level': 'INFO',
            'download_timeout': 30,
            'retry_attempts': 3,
        }
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mapping = {
            'PUBLICDATA_DATA_DIR': 'data_dir',
            'PUBLICDATA_CACHE_ENABLED': 'cache_enabled',
            'PUBLICDATA_LOG_LEVEL': 'log_level',
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string booleans
                if value.lower() in ('true', 'yes', '1'):
                    value = True
                elif value.lower() in ('false', 'no', '0'):
                    value = False
                self._config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration with a dictionary."""
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return self._config.copy()


# Global configuration instance
_global_config = Config()


def load_config(config_file: Path) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file: Path to YAML configuration file
    
    Returns:
        Config instance
    """
    config_file = Path(config_file)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    
    # Update global config
    _global_config.update(config_dict)
    
    return config


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a value from the global configuration.
    
    Args:
        key: Configuration key
        default: Default value if key not found
    
    Returns:
        Configuration value
    """
    return _global_config.get(key, default)


def set_config_value(key: str, value: Any):
    """
    Set a value in the global configuration.
    
    Args:
        key: Configuration key
        value: Configuration value
    """
    _global_config.set(key, value)
