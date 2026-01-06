"""
Dataset registry with metadata for available public data sources.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DatasetMetadata:
    """Metadata for a registered dataset."""
    
    dataset_id: str
    name: str
    source: str
    url: str
    description: str
    format: str
    update_frequency: str
    last_updated: Optional[datetime] = None
    tags: Optional[List[str]] = None


class DatasetRegistry:
    """Registry of available public datasets."""
    
    def __init__(self):
        self._datasets: Dict[str, DatasetMetadata] = {}
        self._initialize_datasets()
    
    def _initialize_datasets(self):
        """Initialize the registry with known datasets."""
        # CIHI Hospital datasets
        self.register(DatasetMetadata(
            dataset_id="cihi_hospital_admissions",
            name="CIHI Hospital Admissions",
            source="Canadian Institute for Health Information",
            url="https://www.cihi.ca/en/data",
            description="Monthly inpatient admission data",
            format="csv",
            update_frequency="monthly",
            tags=["hospital", "admissions", "cihi"]
        ))
        
        self.register(DatasetMetadata(
            dataset_id="cihi_bed_occupancy",
            name="CIHI Bed Occupancy",
            source="Canadian Institute for Health Information",
            url="https://www.cihi.ca/en/data",
            description="Hospital bed occupancy rates",
            format="csv",
            update_frequency="monthly",
            tags=["hospital", "occupancy", "cihi"]
        ))
        
        self.register(DatasetMetadata(
            dataset_id="cihi_icu_utilization",
            name="CIHI ICU Utilization",
            source="Canadian Institute for Health Information",
            url="https://www.cihi.ca/en/data",
            description="ICU utilization statistics",
            format="csv",
            update_frequency="monthly",
            tags=["hospital", "icu", "cihi"]
        ))
    
    def register(self, metadata: DatasetMetadata):
        """Register a dataset in the registry."""
        self._datasets[metadata.dataset_id] = metadata
    
    def get(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """Get metadata for a specific dataset."""
        return self._datasets.get(dataset_id)
    
    def list_datasets(self, tag: Optional[str] = None) -> List[DatasetMetadata]:
        """List all datasets, optionally filtered by tag."""
        datasets = list(self._datasets.values())
        if tag:
            datasets = [d for d in datasets if d.tags and tag in d.tags]
        return datasets
    
    def search(self, query: str) -> List[DatasetMetadata]:
        """Search datasets by name or description."""
        query_lower = query.lower()
        return [
            d for d in self._datasets.values()
            if query_lower in d.name.lower() or query_lower in d.description.lower()
        ]


# Global registry instance
registry = DatasetRegistry()
