"""
Date and time utilities.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Union
import pandas as pd


def parse_date(
    date_str: str,
    format: Optional[str] = None
) -> datetime:
    """
    Parse a date string to datetime object.
    
    Args:
        date_str: Date string to parse
        format: Optional format string (uses pandas flexible parsing if None)
    
    Returns:
        Parsed datetime object
    """
    if format:
        return datetime.strptime(date_str, format)
    return pd.to_datetime(date_str).to_pydatetime()


def format_date(
    date: datetime,
    format: str = "%Y-%m-%d"
) -> str:
    """
    Format a datetime object to string.
    
    Args:
        date: Datetime object to format
        format: Format string (default: YYYY-MM-DD)
    
    Returns:
        Formatted date string
    """
    return date.strftime(format)


def get_date_range(
    start: Union[str, datetime],
    end: Union[str, datetime],
    freq: str = 'D'
) -> List[datetime]:
    """
    Generate a list of dates between start and end.
    
    Args:
        start: Start date (string or datetime)
        end: End date (string or datetime)
        freq: Frequency string (D=daily, M=monthly, Y=yearly)
    
    Returns:
        List of datetime objects
    """
    if isinstance(start, str):
        start = parse_date(start)
    if isinstance(end, str):
        end = parse_date(end)
    
    date_range = pd.date_range(start=start, end=end, freq=freq)
    return [d.to_pydatetime() for d in date_range]


def get_month_range(year: int, month: int) -> tuple:
    """
    Get the start and end dates for a given month.
    
    Args:
        year: Year
        month: Month (1-12)
    
    Returns:
        Tuple of (start_date, end_date)
    """
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)
    return start, end


def get_quarter_dates(year: int, quarter: int) -> tuple:
    """
    Get start and end dates for a fiscal quarter.
    
    Args:
        year: Year
        quarter: Quarter (1-4)
    
    Returns:
        Tuple of (start_date, end_date)
    """
    if quarter not in [1, 2, 3, 4]:
        raise ValueError("Quarter must be 1, 2, 3, or 4")
    
    start_month = (quarter - 1) * 3 + 1
    start = datetime(year, start_month, 1)
    
    end_month = start_month + 2
    if end_month > 12:
        end = datetime(year + 1, end_month - 12 + 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, end_month + 1, 1) - timedelta(days=1)
    
    return start, end
