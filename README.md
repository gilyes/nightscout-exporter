# Nightscout Data Exporter

A Python tool for exporting glucose monitoring data from Nightscout API to CSV.

## Features

- Fetches CGM entries, treatments, device status, and profile data from Nightscout API
- Exports data in multiple formats:
  - **Time-aligned:** Analysis-ready CSV combining glucose, insulin, carbs, basal, and IOB/COB in 5-minute intervals
  - **Raw:** Individual CSVs for entries, treatments, devicestatus, and profiles
- Data processing:
  - Automatic timezone conversion (UTC <-> local time)
  - Glucose unit conversion (mg/dL <-> mmol/L)
  - Calculates basal insulin from profiles (scheduled + temp basals)
  - Matches IOB/COB from devicestatus to timestamps (only supports Loop at this time)

## Prerequisites

- **Python 3.11 or higher**
- Access to a Nightscout instance
- Nightscout access token (if not a public instance)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/gilyes/nightscout-exporter.git
cd nightscout-exporter
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your Nightscout credentials:
```
NIGHTSCOUT_URL=https://your-nightscout-url.com
NIGHTSCOUT_TOKEN=your-token-here
```

Alternatively, you can include the token in the URL query string:
```
NIGHTSCOUT_URL=https://your-nightscout-url.com?token=your-token-here
```

**Note:** If both `NIGHTSCOUT_TOKEN` env var and URL token are present, the env var takes precedence.

Reference `.env.sample` for the expected format.

## Usage

Basic usage with defaults (30 days, mmol/L, local timezone):

```bash
python export_nightscout_data.py
```

Common examples:

```bash
# Custom date range
python export_nightscout_data.py --from-date "2024-01-01" --to-date "2024-12-31"

# Keep mg/dL units
python export_nightscout_data.py --convert-to-mmol false

# Work in UTC
python export_nightscout_data.py --use-local-timezone false

# Custom output folder
python export_nightscout_data.py --output-folder "C:/Data/Nightscout"
```

## Configuration

All parameters are optional. Defaults can be overridden via command-line arguments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--from-date` | string | 30 days ago | Start date in YYYY-MM-DD format |
| `--to-date` | string | None | End date in YYYY-MM-DD format (optional) |
| `--max-count` | int | 100000 | Maximum number of records to fetch |
| `--convert-to-mmol` | bool | true | Convert glucose values from mg/dL to mmol/L |
| `--use-local-timezone` | bool | true | Use local timezone for dates and timestamps |
| `--output-folder` | string | ./output | Output folder path (relative or absolute) |

## Output Files

The script creates an output folder (default: `output/`) with up to five CSV files named with the date range. If specific data is not available then no file is created (e.g. no treatments or devicestatus).

File naming format: `yyyyMMdd-yyyyMMdd-<type>.csv`

For example, if you export data from 2024-01-01 to 2024-12-31, the files will be:
- `20240101-20241231-time_aligned.csv`
- `20240101-20241231-entries.csv`
- `20240101-20241231-treatments.csv`
- `20240101-20241231-devicestatus.csv`
- `20240101-20241231-profiles.csv`

If no end date is specified, the current date is used.

### File Contents

1. **yyyyMMdd-yyyyMMdd-time_aligned.csv**
   - Processed and time-aligned dataset with interval-based structure
   - Data aligned into 5-minute intervals (288 per day)
   - Glucose values converted to mmol/L by default (set `--convert-to-mmol false` to keep mg/dL)
   - Timestamps in local timezone by default (set `--use-local-timezone false` to keep UTC)
   - Includes IOB/COB data matched from devicestatus (from Loop only for now)
   - Sorted by interval timestamp
   - Columns: interval, sgv, bolus, scheduled_basal, positive_temp, negative_temp, total_basal, carbs, iob, cob, events

2. **yyyyMMdd-yyyyMMdd-entries.csv**
   - Raw CGM entries from the Nightscout API
   - Contains sensor glucose values and timestamps
   - Glucose values converted to mmol/L by default (set `--convert-to-mmol false` to keep mg/dL)

3. **yyyyMMdd-yyyyMMdd-treatments.csv**
   - Raw treatments from the Nightscout API
   - Contains insulin, carbs, and other treatment data

4. **yyyyMMdd-yyyyMMdd-devicestatus.csv**
   - Raw devicestatus records from the Nightscout API

5. **yyyyMMdd-yyyyMMdd-profiles.csv**
   - Profile settings from the Nightscout API
   - Contains basal schedules, ISF, carb ratios, target ranges, DIA
   - Time-based schedules flattened with one row per time segment
   - Target ranges and ISF converted to mmol/L by default (when source units are mg/dL)
   - Columns: profile_name, start_date, time, basal_rate, isf, carb_ratio, target_low, target_high, dia, timezone, units

## Data Transformations

- **Glucose Units**: SGV values are converted from mg/dL to mmol/L by default. Set `--convert-to-mmol false` to keep mg/dL values.
- **Timezone**: When `--use-local-timezone` is `true` (default), input dates are treated as local time and converted to UTC for API queries, then response timestamps are converted back to local time for output. Set to `false` to work entirely in UTC.
- **IOB/COB Matching**: Devicestatus IOB/COB values are matched to timestamps within ±5 minutes. **Only Loop data is supported at this time**.
- **Time Alignment**: Data is aligned into fixed 5-minute intervals with an `events` column tracking what occurred in each interval.
- **Date Filtering**: Applied both at API level and post-processing after timezone conversion.

## Security

- The `.env` file containing your credentials is excluded from version control
- Never commit your `.env` file or share your API credentials
- Store your credentials securely

## Troubleshooting

### Missing Environment Variables
```
Error: Missing required environment variables
```
Ensure your `.env` file exists and contains both required variables:
- NIGHTSCOUT_URL
- NIGHTSCOUT_TOKEN

### Invalid Date Format
```
Error: Invalid FromDate format
```
Dates must be in `yyyy-MM-dd` format (e.g., `2024-01-01`)

## License

MIT.
