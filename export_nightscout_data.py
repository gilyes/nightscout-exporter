#!/usr/bin/env python3
"""
Nightscout Data Exporter
Fetches glucose monitoring data (CGM entries and treatments) from Nightscout API
and exports to CSV format.
"""

import argparse
import bisect
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv


def get_local_timezone():
    """Get the local timezone in a cross-platform way."""
    return datetime.now().astimezone().tzinfo


def load_environment() -> tuple[str, str]:
    """Load and validate environment variables from .env file."""
    # Load .env file from script directory
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    token = os.getenv("NIGHTSCOUT_TOKEN")
    base_url = os.getenv("NIGHTSCOUT_URL")

    if not token or not base_url:
        print("Error: Missing required environment variables.", file=sys.stderr)
        print("Please ensure .env file contains NIGHTSCOUT_TOKEN and NIGHTSCOUT_URL", file=sys.stderr)
        sys.exit(1)

    return base_url, token


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    # Default from_date is one month ago (calendar month, not 30 days)
    default_from_date = (datetime.now() - relativedelta(months=1)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="Export Nightscout data (CGM entries and treatments) to CSV format."
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=default_from_date,
        help="Start date in YYYY-MM-DD format (default: one month ago)"
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (optional, defaults to no upper limit)"
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=100000,
        help="Maximum number of records to fetch per API call (default: 100000)"
    )
    parser.add_argument(
        "--use-local-timezone",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Convert input dates and output timestamps to/from local timezone (default: True)"
    )
    parser.add_argument(
        "--convert-to-mmol",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Convert glucose values from mg/dL to mmol/L (default: True)"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./Output",
        help="Output folder path, relative or absolute (default: ./Output)"
    )

    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Please use YYYY-MM-DD format (e.g., 2025-01-01)",
              file=sys.stderr)
        sys.exit(1)


def convert_to_utc(dt: datetime) -> datetime:
    """Convert local datetime to UTC.

    Properly handles DST by treating the naive datetime as local time
    and converting to UTC timezone-aware.
    Cross-platform compatible (Windows, Linux, macOS).
    """
    # Treat naive datetime as local time
    local_tz = get_local_timezone()
    local_dt = dt.replace(tzinfo=local_tz)
    # Convert to UTC
    utc_dt = local_dt.astimezone(timezone.utc)

    return utc_dt


def convert_from_utc(dt: datetime) -> datetime:
    """Convert UTC datetime to local timezone.

    Properly handles DST by treating the naive datetime as UTC
    and converting to local timezone.
    Cross-platform compatible (Windows, Linux, macOS).
    """
    # Treat naive datetime as UTC
    utc_dt = dt.replace(tzinfo=timezone.utc)
    # Convert to local timezone (handles DST automatically for each datetime)
    local_tz = get_local_timezone()
    local_dt = utc_dt.astimezone(local_tz)

    return local_dt


def fetch_entries(base_url: str, token: str, max_count: int,
                  api_from_date: str, api_to_date: Optional[str],
                  use_local_timezone: bool, convert_to_mmol: bool) -> pd.DataFrame:
    """Fetch entries (CGM data) from Nightscout API."""
    print(f"Fetching entries from Nightscout API (max {max_count} records)...")

    # Build URL
    url = f"{base_url}/api/v1/entries"
    params = {
        "count": max_count,
        "token": token,
        "find[dateString][$gte]": api_from_date
    }
    if api_to_date:
        params["find[dateString][$lte]"] = api_to_date

    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        entries = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch entries: {e}", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("Warning: No entries returned from API")
        return pd.DataFrame()

    # Convert to DataFrame (exports all columns found across all entries)
    df = pd.DataFrame(entries)

    # Sort columns alphabetically for consistent ordering
    df = df.reindex(sorted(df.columns), axis=1)

    # Convert timestamps
    if 'dateString' in df.columns:
        df['dateString'] = pd.to_datetime(df['dateString'])
        if use_local_timezone:
            df['dateString'] = df['dateString'].apply(
                lambda x: convert_from_utc(x).strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            df['dateString'] = df['dateString'].dt.strftime("%Y-%m-%d %H:%M:%S")

    if 'sysTime' in df.columns:
        df['sysTime'] = pd.to_datetime(df['sysTime'])
        if use_local_timezone:
            df['sysTime'] = df['sysTime'].apply(
                lambda x: convert_from_utc(x).strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            df['sysTime'] = df['sysTime'].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Convert SGV from mg/dL to mmol/L if requested
    if convert_to_mmol and 'sgv' in df.columns:
        df['sgv'] = (df['sgv'] / 18.0182).round(1)

    print(f"Total entries: {len(df)}")
    return df


def fetch_treatments(base_url: str, token: str, max_count: int,
                     api_from_date: str, api_to_date: Optional[str],
                     use_local_timezone: bool) -> pd.DataFrame:
    """Fetch treatments from Nightscout API."""
    print(f"\nFetching treatments from Nightscout API (max {max_count} records)...")

    # Build URL
    url = f"{base_url}/api/v1/treatments"
    params = {
        "count": max_count,
        "token": token,
        "find[created_at][$gte]": api_from_date
    }
    if api_to_date:
        params["find[created_at][$lte]"] = api_to_date

    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        treatments = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch treatments: {e}", file=sys.stderr)
        sys.exit(1)

    if not treatments:
        print("Warning: No treatments returned from API")
        return pd.DataFrame()

    # Ensure insulin and amount properties exist in each treatment
    for treatment in treatments:
        if 'insulin' not in treatment:
            treatment['insulin'] = None
        if 'amount' not in treatment:
            treatment['amount'] = None

    # Convert to DataFrame (exports all columns found across all treatments)
    df = pd.DataFrame(treatments)

    # Sort columns alphabetically for consistent ordering
    df = df.reindex(sorted(df.columns), axis=1)

    # Convert timestamps
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        if use_local_timezone:
            df['created_at'] = df['created_at'].apply(
                lambda x: convert_from_utc(x).strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            df['created_at'] = df['created_at'].dt.strftime("%Y-%m-%d %H:%M:%S")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if use_local_timezone:
            df['timestamp'] = df['timestamp'].apply(
                lambda x: convert_from_utc(x).strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            df['timestamp'] = df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Total treatments: {len(df)}")
    return df


def fetch_devicestatus(base_url: str, token: str, max_count: int,
                       api_from_date: str, api_to_date: Optional[str],
                       use_local_timezone: bool) -> pd.DataFrame:
    """Fetch devicestatus from Nightscout API."""
    print(f"\nFetching devicestatus from Nightscout API (max {max_count} records)...")

    # Build URL
    url = f"{base_url}/api/v1/devicestatus"
    params = {
        "count": max_count,
        "token": token,
        "find[created_at][$gte]": api_from_date
    }
    if api_to_date:
        params["find[created_at][$lte]"] = api_to_date

    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        devicestatus = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch devicestatus: {e}", file=sys.stderr)
        sys.exit(1)

    if not devicestatus:
        print("Warning: No devicestatus returned from API")
        return pd.DataFrame()

    # Convert to DataFrame (exports all columns found across all devicestatus records)
    df = pd.DataFrame(devicestatus)

    # Sort columns alphabetically for consistent ordering
    df = df.reindex(sorted(df.columns), axis=1)

    # Convert timestamps
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'])
        if use_local_timezone:
            df['created_at'] = df['created_at'].apply(
                lambda x: convert_from_utc(x).strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            df['created_at'] = df['created_at'].dt.strftime("%Y-%m-%d %H:%M:%S")

    print(f"Total devicestatus records: {len(df)}")
    return df


def parse_devicestatus_cob(devicestatus_record) -> Optional[float]:
    """Parse COB value from devicestatus (Loop structure only).

    Args:
        devicestatus_record: A single devicestatus record (dict or Series)

    Returns:
        COB value as float, or None if not found
    """
    # Get loop data
    loop_data = None
    if isinstance(devicestatus_record, dict):
        loop_data = devicestatus_record.get('loop')
    elif hasattr(devicestatus_record, 'get'):
        loop_data = devicestatus_record.get('loop')

    # Try Loop structure (loop.cob.cob)
    if loop_data and isinstance(loop_data, dict):
        cob_obj = loop_data.get('cob')
        if isinstance(cob_obj, dict):
            cob_value = cob_obj.get('cob')
            if cob_value is not None:
                return float(cob_value)

    return None


def match_devicestatus_to_timestamps(combined_df: pd.DataFrame, devicestatus_df: pd.DataFrame,
                                     max_time_diff_minutes: int = 5) -> tuple:
    """Match devicestatus IOB/COB to combined data timestamps.

    Finds the nearest devicestatus entry within ±max_time_diff_minutes for each timestamp.
    When multiple devicestatus entries exist at the same time, uses the most recent one.

    Args:
        combined_df: Combined data DataFrame with DateTime column
        devicestatus_df: Devicestatus DataFrame with created_at column
        max_time_diff_minutes: Maximum time difference in minutes for matching

    Returns:
        tuple: (iob_values, cob_values) - lists matching combined_df rows
    """
    iob_values = []
    cob_values = []

    if devicestatus_df.empty or 'created_at' not in devicestatus_df.columns:
        # No devicestatus data, return None for all rows
        return [None] * len(combined_df), [None] * len(combined_df)

    # Parse devicestatus timestamps and build lookup list
    devicestatus_times = []
    for _, record in devicestatus_df.iterrows():
        timestamp = datetime.strptime(record['created_at'], "%Y-%m-%d %H:%M:%S")
        devicestatus_times.append((timestamp, record))

    # Sort by timestamp (and keep most recent if duplicates by using stable sort)
    devicestatus_times.sort(key=lambda x: x[0])

    # Build timestamp list for binary search
    timestamps = [t[0] for t in devicestatus_times]

    max_diff = timedelta(minutes=max_time_diff_minutes)

    # Match each combined row to nearest devicestatus
    for _, row in combined_df.iterrows():
        current_time = row['DateTime']

        # Find insertion point
        idx = bisect.bisect_left(timestamps, current_time)

        # Check candidates: idx-1 and idx
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, timestamps[idx - 1]))
        if idx < len(timestamps):
            candidates.append((idx, timestamps[idx]))

        # Find closest within time window
        best_match = None
        best_diff = max_diff

        for cand_idx, cand_time in candidates:
            time_diff = abs(current_time - cand_time)
            if time_diff <= max_diff and time_diff <= best_diff:
                best_diff = time_diff
                best_match = cand_idx

        if best_match is not None:
            _, record = devicestatus_times[best_match]

            # Extract IOB (Loop structure only)
            iob = None

            # Try Loop structure (loop.iob.iob)
            if 'loop' in record:
                loop_data = record.get('loop')
                if isinstance(loop_data, dict):
                    iob_obj = loop_data.get('iob')
                    if isinstance(iob_obj, dict):
                        iob_value = iob_obj.get('iob')
                        if iob_value is not None:
                            iob = float(iob_value)

            # Extract COB using parser (Loop structure only)
            cob = parse_devicestatus_cob(record)

            iob_values.append(iob)
            cob_values.append(cob)
        else:
            iob_values.append(None)
            cob_values.append(None)

    return iob_values, cob_values


def process_combined_data(entries_df: pd.DataFrame, treatments_df: pd.DataFrame,
                          from_datetime: datetime, to_datetime: Optional[datetime],
                          devicestatus_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Process and combine entries and treatments into unified format.

    Optionally adds IOB/COB from devicestatus (loop system values)."""
    print("\nProcessing and combining datasets...")

    processed_entries = []
    processed_treatments = []

    # Process entries
    if not entries_df.empty and 'sgv' in entries_df.columns:
        for _, entry in entries_df.iterrows():
            if pd.notna(entry.get('sgv')):
                entry_date = datetime.strptime(entry['dateString'], "%Y-%m-%d %H:%M:%S")

                # Apply date filtering
                if entry_date >= from_datetime and (to_datetime is None or entry_date <= to_datetime):
                    processed_entries.append({
                        'DateTime': entry['dateString'],
                        'eventType': 'SGV',
                        'sgv': entry['sgv'],
                        'insulin': None,
                        'amount': None,
                        'carbs': None
                    })

    # Process treatments
    if not treatments_df.empty and 'created_at' in treatments_df.columns:
        for _, treatment in treatments_df.iterrows():
            treatment_date = datetime.strptime(treatment['created_at'], "%Y-%m-%d %H:%M:%S")

            # Apply date filtering
            if treatment_date >= from_datetime and (to_datetime is None or treatment_date <= to_datetime):
                processed_treatments.append({
                    'DateTime': treatment['created_at'],
                    'eventType': treatment.get('eventType'),
                    'sgv': None,
                    'insulin': treatment.get('insulin'),
                    'amount': treatment.get('amount'),
                    'carbs': treatment.get('carbs')
                })

    # Combine datasets
    combined_data = processed_entries + processed_treatments

    if not combined_data:
        print("Warning: No data to combine")
        return pd.DataFrame()

    # Create DataFrame and sort by DateTime
    df = pd.DataFrame(combined_data)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')

    # Match devicestatus IOB/COB (from loop system)
    if devicestatus_df is not None and not devicestatus_df.empty:
        print("Matching devicestatus IOB/COB to timestamps...")
        iob_from_devicestatus, cob_from_devicestatus = match_devicestatus_to_timestamps(df, devicestatus_df)
        df['iob'] = iob_from_devicestatus
        df['cob'] = cob_from_devicestatus

        # Calculate match rate
        iob_match_count = sum(1 for v in iob_from_devicestatus if v is not None)
        cob_match_count = sum(1 for v in cob_from_devicestatus if v is not None)
        print(f"  IOB matched: {iob_match_count}/{len(df)} ({iob_match_count*100//len(df) if len(df) > 0 else 0}%)")
        print(f"  COB matched: {cob_match_count}/{len(df)} ({cob_match_count*100//len(df) if len(df) > 0 else 0}%)")
    else:
        df['iob'] = None
        df['cob'] = None

    # Convert DateTime back to string
    df['DateTime'] = df['DateTime'].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Load environment
    base_url, token = load_environment()

    # Parse and validate dates
    from_datetime = parse_date(args.from_date)
    to_datetime = parse_date(args.to_date) if args.to_date else None

    # Resolve output folder path
    output_folder = Path(args.output_folder)
    if not output_folder.is_absolute():
        output_folder = Path(__file__).parent / output_folder

    # Create output folder if needed
    output_folder.mkdir(parents=True, exist_ok=True)
    if not output_folder.exists():
        print(f"Created output folder: {output_folder}")

    # Create date range suffix for file names
    from_date_formatted = args.from_date.replace('-', '')
    if args.to_date:
        to_date_formatted = args.to_date.replace('-', '')
    else:
        to_date_formatted = datetime.now().strftime("%Y%m%d")
    date_range_suffix = f"{from_date_formatted}-{to_date_formatted}"

    # Print configuration
    print(f"Date range: From {args.from_date}", end="")
    if args.to_date:
        print(f" To {args.to_date}")
    else:
        print(" (no upper limit)")
    print(f"Use local timezone: {args.use_local_timezone}")
    print(f"Convert to mmol/L: {args.convert_to_mmol}")

    # Prepare API query dates
    if args.use_local_timezone:
        # Convert local dates to UTC for API query
        utc_from_datetime = convert_to_utc(from_datetime)
        api_from_date = utc_from_datetime.strftime("%Y-%m-%dT%H:%M:%S")

        if to_datetime:
            # End of day in local time
            utc_to_datetime = convert_to_utc(to_datetime + timedelta(days=1, seconds=-1))
            api_to_date = utc_to_datetime.strftime("%Y-%m-%dT%H:%M:%S")
        else:
            api_to_date = None

        print(f"API query (UTC): From {api_from_date}", end="")
        if api_to_date:
            print(f" To {api_to_date}")
        else:
            print()
    else:
        # Dates are already in UTC
        api_from_date = args.from_date
        api_to_date = args.to_date

    # Fetch entries
    entries_df = fetch_entries(base_url, token, args.max_count, api_from_date, api_to_date,
                               args.use_local_timezone, args.convert_to_mmol)

    # Save entries to CSV
    if not entries_df.empty:
        entries_file = output_folder / f"{date_range_suffix}-entries.csv"
        entries_df_sorted = entries_df.sort_values('dateString') if 'dateString' in entries_df.columns else entries_df
        entries_df_sorted.to_csv(entries_file, index=False)
        print(f"Entries exported to: {entries_file}")

    # Fetch treatments
    treatments_df = fetch_treatments(base_url, token, args.max_count, api_from_date, api_to_date,
                                     args.use_local_timezone)

    # Save treatments to CSV
    if not treatments_df.empty:
        treatments_file = output_folder / f"{date_range_suffix}-treatments.csv"
        treatments_df_sorted = treatments_df.sort_values('created_at') if 'created_at' in treatments_df.columns else treatments_df
        treatments_df_sorted.to_csv(treatments_file, index=False)
        print(f"Treatments exported to: {treatments_file}")

    # Fetch devicestatus
    devicestatus_df = fetch_devicestatus(base_url, token, args.max_count, api_from_date, api_to_date,
                                         args.use_local_timezone)

    # Save devicestatus to CSV
    if not devicestatus_df.empty:
        devicestatus_file = output_folder / f"{date_range_suffix}-devicestatus.csv"
        devicestatus_df_sorted = devicestatus_df.sort_values('created_at') if 'created_at' in devicestatus_df.columns else devicestatus_df
        devicestatus_df_sorted.to_csv(devicestatus_file, index=False)
        print(f"Devicestatus exported to: {devicestatus_file}")

    # Process and combine data
    combined_df = process_combined_data(entries_df, treatments_df, from_datetime,
                                        to_datetime if to_datetime else datetime.max,
                                        devicestatus_df)

    # Save combined data to CSV
    if not combined_df.empty:
        combined_file = output_folder / f"{date_range_suffix}-combined.csv"
        combined_df.to_csv(combined_file, index=False)

        print(f"\nCombined data exported to: {combined_file}")
        print(f"Total SGV entries: {len(combined_df[combined_df['eventType'] == 'SGV'])}")
        print(f"Total treatments: {len(combined_df[combined_df['eventType'] != 'SGV'])}")
        print(f"Total combined records: {len(combined_df)}")


if __name__ == "__main__":
    main()
