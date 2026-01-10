"""
Command-line interface for the hospital anomalies case study.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from case_studies.hospital_anomalies.src.pipeline import run_pipeline
from case_studies.hospital_anomalies.src.utils import setup_logging, get_logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Hospital Resource Usage Anomaly Detection Pipeline'
    )
    
    parser.add_argument(
        'command',
        choices=['run'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='case_studies/hospital_anomalies/config/default.yaml',
        help='Path to configuration file (default: case_studies/hospital_anomalies/config/default.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = get_logger(__name__)
    
    try:
        if args.command == 'run':
            config_path = Path(args.config)
            
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_path}")
                sys.exit(1)
            
            logger.info(f"Running pipeline with config: {config_path}")
            results = run_pipeline(config_path)
            logger.info("Pipeline completed successfully!")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
