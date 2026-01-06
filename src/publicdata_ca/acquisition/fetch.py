"""
Download helpers with HTTP retry logic and caching.
"""

import requests
from pathlib import Path
from typing import Optional
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session_with_retry(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (500, 502, 504)
) -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 8192,
    timeout: int = 30,
    force: bool = False
) -> Path:
    """
    Download a file from URL to destination with retry logic.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        chunk_size: Size of chunks for streaming download
        timeout: Request timeout in seconds
        force: If True, re-download even if file exists
    
    Returns:
        Path to downloaded file
    """
    destination = Path(destination)
    
    # Check if file already exists and caching is enabled
    if destination.exists() and not force:
        return destination
    
    # Ensure parent directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Download the file
    session = create_session_with_retry()
    try:
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        return destination
    except requests.RequestException as e:
        # Clean up partial download
        if destination.exists():
            destination.unlink()
        raise Exception(f"Failed to download {url}: {e}")


def fetch_dataset(
    dataset_id: str,
    destination: Optional[Path] = None,
    force_download: bool = False
) -> Path:
    """
    Fetch a dataset by ID from the registry.
    
    Args:
        dataset_id: ID of the dataset in the registry
        destination: Optional custom destination path
        force_download: If True, re-download even if cached
    
    Returns:
        Path to the downloaded dataset
    """
    from .registry import registry
    from .storage import get_data_path
    
    metadata = registry.get(dataset_id)
    if not metadata:
        raise ValueError(f"Dataset {dataset_id} not found in registry")
    
    if destination is None:
        destination = get_data_path("raw", "cihi", f"{dataset_id}.{metadata.format}")
    
    return download_file(metadata.url, destination, force=force_download)
