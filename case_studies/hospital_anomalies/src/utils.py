"""
Compatibility utilities for logging, config, and path management.
These utilities provide the same API as the old publicdata_ca.utils module
but use standard Python libraries and the new publicdata_ca package.
"""

import logging
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================================
# Logging utilities
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# Configuration utilities
# ============================================================================

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
    
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
    
    return Config(config_dict)


# ============================================================================
# Path management utilities
# ============================================================================

def get_project_root() -> Path:
    """
    Get the project root directory.
    Assumes this file is in case_studies/hospital_anomalies/src/
    """
    return Path(__file__).parent.parent.parent.parent


def get_data_path(
    stage: str,
    source: str,
    filename: Optional[str] = None
) -> Path:
    """
    Get path for data files following the project convention.
    
    Args:
        stage: One of 'raw', 'interim', or 'processed'
        source: Data source name (e.g., 'cihi')
        filename: Optional filename to append
    
    Returns:
        Path to the data location
    """
    if stage not in ['raw', 'interim', 'processed']:
        raise ValueError(f"Invalid stage: {stage}. Must be 'raw', 'interim', or 'processed'")
    
    root = get_project_root()
    path = root / "data" / stage / source
    
    if filename:
        path = path / filename
    
    return path


def ensure_data_dirs(source: str = "cihi"):
    """
    Ensure data directories exist for a given source.
    
    Args:
        source: Data source name (e.g., 'cihi')
    """
    for stage in ['raw', 'interim', 'processed']:
        path = get_data_path(stage, source)
        path.mkdir(parents=True, exist_ok=True)
