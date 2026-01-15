"""
Command-line interface for the hospital anomalies case study.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from case_studies.hospital_anomalies.src.pipeline import run_pipeline
from case_studies.hospital_anomalies.src.ingest import ingest_cihi_data
from case_studies.hospital_anomalies.src.utils import setup_logging, get_logger, load_config


def _parse_iso_date(value: str):
    """Parse CLI date arguments as YYYY-MM-DD."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format YYYY-MM-DD."
        ) from exc


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Hospital Resource Usage Anomaly Detection Pipeline'
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'ingest'],
        help='Command to execute (run: full pipeline, ingest: data ingestion only)'
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

    parser.add_argument(
        '--start-date',
        type=_parse_iso_date,
        help='Override the configuration start date (YYYY-MM-DD, run command only)'
    )

    parser.add_argument(
        '--end-date',
        type=_parse_iso_date,
        help='Override the configuration end date (YYYY-MM-DD, run command only)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = get_logger(__name__)
    
    try:
        config_path = Path(args.config)
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        date_range_override = {}
        if args.start_date:
            date_range_override['start'] = args.start_date.isoformat()
        if args.end_date:
            date_range_override['end'] = args.end_date.isoformat()

        if args.start_date and args.end_date and args.start_date > args.end_date:
            logger.error("Start date must be on or before end date.")
            sys.exit(1)

        if args.command != 'run' and date_range_override:
            logger.warning("Date range overrides are ignored for the ingest command.")
            date_range_override = {}

        if args.command == 'run':
            logger.info(f"Running pipeline with config: {config_path}")
            if date_range_override:
                logger.info(
                    "Applying date range override: %s",
                    date_range_override
                )
            results = run_pipeline(
                config_path,
                date_range_override=date_range_override or None
            )
            logger.info("Pipeline completed successfully!")
        
        elif args.command == 'ingest':
            logger.info(f"Running data ingestion with config: {config_path}")
            config = load_config(config_path)
            dataset_ids = config.to_dict().get('datasets', [])
            
            if not dataset_ids:
                logger.error("No datasets specified in configuration file")
                sys.exit(1)
            
            datasets = ingest_cihi_data(dataset_ids, force_download=True)
            logger.info(f"Ingestion completed successfully! Fetched {len(datasets)} datasets.")
            
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
