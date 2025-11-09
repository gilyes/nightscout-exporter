#!/usr/bin/env python3
"""
Nightscout Data Analyzer
Analyzes time-aligned CSV data to generate daily statistics and optional visualizations.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze Nightscout time-aligned data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Statistics only
  python analyze_nightscout_data.py -i output/20250601-20251108-time_aligned.csv

  # With graphs
  python analyze_nightscout_data.py -i output/20250601-20251108-time_aligned.csv --graphs

  # Date range + graphs
  python analyze_nightscout_data.py -i output/20250601-20251108-time_aligned.csv -g --from-date 2025-06-01 --to-date 2025-06-30
        '''
    )

    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Path to time-aligned CSV file')
    parser.add_argument('-g', '--graphs', action='store_true',
                        help='Generate daily graphs')
    parser.add_argument('-o', '--output-folder', type=str, default='./output',
                        help='Output folder for graphs (default: ./output)')
    parser.add_argument('--from-date', type=str, default=None,
                        help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=str, default=None,
                        help='End date filter (YYYY-MM-DD)')

    return parser.parse_args()


def load_data(file_path: str) -> pd.DataFrame:
    """Load time-aligned CSV data."""
    try:
        df = pd.read_csv(file_path)
        df['interval'] = pd.to_datetime(df['interval'])
        df['date'] = df['interval'].dt.date
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)


def filter_by_date_range(df: pd.DataFrame, from_date: str = None, to_date: str = None) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if from_date:
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d").date()
            df = df[df['date'] >= from_dt]
        except ValueError:
            print(f"Error: Invalid from-date format: {from_date}. Use YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    if to_date:
        try:
            to_dt = datetime.strptime(to_date, "%Y-%m-%d").date()
            df = df[df['date'] <= to_dt]
        except ValueError:
            print(f"Error: Invalid to-date format: {to_date}. Use YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)

    return df


def calculate_daily_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily insulin and carb statistics."""
    daily_stats = df.groupby('date').agg({
        'bolus': 'sum',
        'scheduled_basal': 'sum',
        'positive_temp': 'sum',
        'negative_temp': 'sum',
        'total_basal': 'sum',
        'carbs': 'sum'
    }).fillna(0)

    # Calculate total insulin (bolus + basal)
    daily_stats['total_insulin'] = daily_stats['bolus'] + daily_stats['total_basal']

    # Reorder columns for better readability
    daily_stats = daily_stats[[
        'bolus',
        'scheduled_basal',
        'positive_temp',
        'negative_temp',
        'total_basal',
        'total_insulin',
        'carbs'
    ]]

    return daily_stats


def print_statistics(stats: pd.DataFrame):
    """Print daily statistics to console."""
    print("\n" + "=" * 100)
    print("DAILY STATISTICS")
    print("=" * 100)
    print()

    # Check if scheduled basal is all zeros (no basal data)
    has_basal_data = stats['scheduled_basal'].sum() > 0

    # Format the dataframe for display
    pd.options.display.float_format = '{:.2f}'.format
    pd.options.display.max_rows = None
    pd.options.display.width = 150

    # Create a copy for display and replace basal columns with N/A if no data
    display_stats = stats.copy()
    if not has_basal_data:
        display_stats['scheduled_basal'] = 'N/A'
        display_stats['positive_temp'] = 'N/A'
        display_stats['negative_temp'] = 'N/A'
        display_stats['total_basal'] = 'N/A'

    print(display_stats.to_string())

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print()
    print(f"Total days: {len(stats)}")
    print(f"Date range: {stats.index.min()} to {stats.index.max()}")
    print()
    print("Averages per day:")
    print(f"  Bolus insulin:        {stats['bolus'].mean():>8.2f} units")

    if has_basal_data:
        print(f"  Scheduled basal:      {stats['scheduled_basal'].mean():>8.2f} units")
        print(f"  Positive temp:        {stats['positive_temp'].mean():>8.2f} units")
        print(f"  Negative temp:        {stats['negative_temp'].mean():>8.2f} units")
        print(f"  Total basal:          {stats['total_basal'].mean():>8.2f} units")
    else:
        print(f"  Scheduled basal:      {'N/A':>8} units")
        print(f"  Positive temp:        {'N/A':>8} units")
        print(f"  Negative temp:        {'N/A':>8} units")
        print(f"  Total basal:          {'N/A':>8} units")

    print(f"  Total insulin:        {stats['total_insulin'].mean():>8.2f} units")
    print(f"  Carbs:                {stats['carbs'].mean():>8.2f} grams")
    print()


def detect_glucose_unit(df: pd.DataFrame) -> tuple[str, float, float]:
    """
    Detect glucose unit from SGV data range.
    Returns: (unit_label, low_target, high_target)
    """
    sgv_data = df['sgv'].dropna()
    if sgv_data.empty:
        return ('mg/dL', 70, 180)  # Default to mg/dL if no data

    max_sgv = sgv_data.max()

    # If max SGV < 50, likely mmol/L; otherwise mg/dL
    if max_sgv < 50:
        return ('mmol/L', 3.9, 10.0)
    else:
        return ('mg/dL', 70, 180)


def create_daily_graph(day_data: pd.DataFrame, date: datetime.date, output_path: Path, glucose_unit: str, target_low: float, target_high: float):
    """Create visualization for a single day."""
    # Create figure with 5 subplots stacked vertically
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Nightscout Analysis - {date}', fontsize=16, fontweight='bold')

    # Prepare time axis
    times = day_data['interval']

    # 1. SGV (Glucose) - Line Chart
    ax1 = axes[0]
    sgv_data = day_data['sgv'].dropna()
    if not sgv_data.empty:
        ax1.plot(times[day_data['sgv'].notna()], sgv_data, 'o-', color='#1f77b4', linewidth=2, markersize=3)
        ax1.axhline(y=target_low, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Low ({target_low:.1f})')
        ax1.axhline(y=target_high, color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'High ({target_high:.1f})')
        ax1.fill_between(times[day_data['sgv'].notna()], target_low, target_high, alpha=0.1, color='green')
    ax1.set_ylabel(f'SGV ({glucose_unit})', fontsize=10, fontweight='bold')
    ax1.set_title('Blood Glucose', fontsize=11, loc='left')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # 2. Basal Insulin - Stacked Area Chart
    ax2 = axes[1]
    scheduled = day_data['scheduled_basal'].fillna(0)
    positive = day_data['positive_temp'].fillna(0)
    negative = day_data['negative_temp'].fillna(0)

    ax2.fill_between(times, 0, scheduled, label='Scheduled', alpha=0.7, color='#2ca02c')
    ax2.fill_between(times, scheduled, scheduled + positive, label='Positive Temp', alpha=0.7, color='#ff7f0e')
    ax2.fill_between(times, scheduled + negative, scheduled, label='Negative Temp', alpha=0.7, color='#d62728')

    ax2.set_ylabel('Basal (units)', fontsize=10, fontweight='bold')
    ax2.set_title('Basal Insulin Delivery', fontsize=11, loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    # 3. Bolus & Carbs - Bar Chart with Secondary Axis
    ax3 = axes[2]
    ax3_carbs = ax3.twinx()

    bolus_data = day_data[day_data['bolus'].notna()]
    carbs_data = day_data[day_data['carbs'].notna()]

    if not bolus_data.empty:
        ax3.bar(bolus_data['interval'], bolus_data['bolus'], width=0.003, color='#9467bd', alpha=0.7, label='Bolus')
    if not carbs_data.empty:
        ax3_carbs.bar(carbs_data['interval'], carbs_data['carbs'], width=0.003, color='#8c564b', alpha=0.5, label='Carbs')

    ax3.set_ylabel('Bolus (units)', fontsize=10, fontweight='bold', color='#9467bd')
    ax3_carbs.set_ylabel('Carbs (grams)', fontsize=10, fontweight='bold', color='#8c564b')
    ax3.set_title('Bolus & Carbs', fontsize=11, loc='left')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='#9467bd')
    ax3_carbs.tick_params(axis='y', labelcolor='#8c564b')

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_carbs.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # 4. IOB (Insulin on Board) - Line Chart
    ax4 = axes[3]
    iob_data = day_data['iob'].dropna()
    if not iob_data.empty:
        ax4.plot(times[day_data['iob'].notna()], iob_data, '-', color='#e377c2', linewidth=2)
        ax4.fill_between(times[day_data['iob'].notna()], 0, iob_data, alpha=0.3, color='#e377c2')
    ax4.set_ylabel('IOB (units)', fontsize=10, fontweight='bold')
    ax4.set_title('Insulin on Board', fontsize=11, loc='left')
    ax4.grid(True, alpha=0.3)

    # 5. COB (Carbs on Board) - Line Chart
    ax5 = axes[4]
    cob_data = day_data['cob'].dropna()
    if not cob_data.empty:
        ax5.plot(times[day_data['cob'].notna()], cob_data, '-', color='#bcbd22', linewidth=2)
        ax5.fill_between(times[day_data['cob'].notna()], 0, cob_data, alpha=0.3, color='#bcbd22')
    ax5.set_ylabel('COB (grams)', fontsize=10, fontweight='bold')
    ax5.set_title('Carbs on Board', fontsize=11, loc='left')
    ax5.set_xlabel('Time', fontsize=10, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Format x-axis to show time
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Generated: {output_path}")


def generate_graphs(df: pd.DataFrame, output_folder: str):
    """Generate daily graphs for all dates in the dataset."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print("GENERATING GRAPHS")
    print("=" * 100)
    print()

    # Detect glucose unit from entire dataset
    glucose_unit, target_low, target_high = detect_glucose_unit(df)
    print(f"Detected glucose unit: {glucose_unit} (targets: {target_low:.1f}-{target_high:.1f})")
    print()

    dates = sorted(df['date'].unique())

    for date in dates:
        day_data = df[df['date'] == date].copy()
        filename = f"{date}-analysis.png"
        file_path = output_path / filename
        create_daily_graph(day_data, date, file_path, glucose_unit, target_low, target_high)

    print()
    print(f"Graphs saved to: {output_path.absolute()}")
    print()


def main():
    """Main entry point."""
    args = parse_arguments()

    print()
    print("=" * 100)
    print("NIGHTSCOUT DATA ANALYZER")
    print("=" * 100)
    print()
    print(f"Input file: {args.input}")

    # Load data
    df = load_data(args.input)
    print(f"Loaded {len(df)} intervals")

    # Filter by date range if specified
    df = filter_by_date_range(df, args.from_date, args.to_date)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Calculate and print statistics
    stats = calculate_daily_statistics(df)
    print_statistics(stats)

    # Generate graphs if requested
    if args.graphs:
        generate_graphs(df, args.output_folder)

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
