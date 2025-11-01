# Nightscout Data Exporter

A Python tool for exporting glucose monitoring data (CGM entries and treatments) from Nightscout API to CSV format.

## Features

- Fetches CGM entries, treatment and device status data from Nightscout
- Exports data in multiple formats:
  - Raw entries CSV (with converted values)
  - Raw treatments CSV (normalized columns)
  - Raw devicestatus CSV
  - Combined and processed CSV with unified schema including IOB/COB. **Note:** For IOB/COB, only Loop data is supported at this time.
- Configurable date ranges and record limits

## Prerequisites

- **Python 3.11 or higher**
- Access to a Nightscout instance
- Nightscout API token

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

Reference `.env.sample` for the expected format.

## Usage

### Basic Usage

Fetch the last month of data (default):

```bash
python export_nightscout_data.py
```

### Glucose Unit Conversion

**By default, glucose values are converted from mg/dL to mmol/L.** To keep original mg/dL values:

```bash
# Keep original mg/dL values
python export_nightscout_data.py --convert-to-mmol false

# Convert to mmol/L (default)
python export_nightscout_data.py
```

### Custom Date Range

Specify custom date range (dates are in your local timezone by default):

```bash
python export_nightscout_data.py --from-date "2024-01-01" --to-date "2024-12-31"
```

### Working with UTC Instead of Local Timezone

By default, input dates and output timestamps use your local timezone. To work entirely in UTC:

```bash
python export_nightscout_data.py --use-local-timezone false
```

### Custom Maximum Record Count

Limit the number of records fetched:

```bash
python export_nightscout_data.py --max-count 50000
```

### Custom Output Folder

Specify a custom output folder (defaults to `./Output`):

```bash
# Use custom output folder (relative path)
python export_nightscout_data.py --output-folder "./MyData"

# Use absolute path
python export_nightscout_data.py --output-folder "C:/Data/Nightscout"
```

### Combined Parameters

```bash
python export_nightscout_data.py --from-date "2024-06-01" --to-date "2024-06-30" --max-count 20000 --convert-to-mmol false --output-folder "./June2024"
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--from-date` | string | Last month | Start date in YYYY-MM-DD format (local timezone by default) |
| `--to-date` | string | None | End date in YYYY-MM-DD format (optional) |
| `--max-count` | int | 100000 | Maximum number of records to fetch |
| `--convert-to-mmol` | bool | **true** | **Convert glucose values from mg/dL to mmol/L** |
| `--use-local-timezone` | bool | true | Work in local timezone (input dates and output timestamps) |
| `--output-folder` | string | ./Output | Output folder path (relative or absolute) |

## Output Files

The script creates an output folder (default: `Output/`) with four CSV files named with the date range:

File naming format: `yyyyMMdd-yyyyMMdd-<type>.csv`

For example, if you export data from 2024-01-01 to 2024-12-31, the files will be:
- `20240101-20241231-entries.csv`
- `20240101-20241231-treatments.csv`
- `20240101-20241231-devicestatus.csv`
- `20240101-20241231-combined.csv`

If no end date is specified, the current date is used:
- `20240101-20251014-entries.csv`

### File Contents

1. **yyyyMMdd-yyyyMMdd-entries.csv**
   - Raw CGM entries from the Nightscout API
   - Contains sensor glucose values and timestamps
   - Glucose values converted to mmol/L by default (set `--convert-to-mmol false` to keep mg/dL)

2. **yyyyMMdd-yyyyMMdd-treatments.csv**
   - Raw treatments from the Nightscout API
   - Contains insulin, carbs, and other treatment data
   - Normalized to include both `insulin` and `amount` columns

3. **yyyyMMdd-yyyyMMdd-devicestatus.csv**
   - Raw devicestatus records from the Nightscout API
   - Contains Loop system data with full structure
   - Includes IOB and COB data from the Loop system

4. **yyyyMMdd-yyyyMMdd-combined.csv**
   - Processed and merged dataset with unified schema
   - Glucose values converted to mmol/L by default
   - Timestamps in local timezone by default
   - Includes IOB/COB data matched from devicestatus (Loop only)
   - Sorted by DateTime
   - Columns: DateTime, eventType, sgv, insulin, amount, carbs, iob, cob

## Data Transformations

- **Glucose Units**: SGV values are converted from mg/dL to mmol/L (÷ 18.0182) by default. Set `--convert-to-mmol false` to keep mg/dL values.
- **Timezone**: When `--use-local-timezone` is `true` (default), input dates are treated as local time and converted to UTC for API queries, then response timestamps are converted back to local time for output. Set to `false` to work entirely in UTC.
- **IOB/COB Matching**: Devicestatus IOB/COB values are matched to timestamps within ±5 minutes. Only Loop system data is supported.
- **Schema**: Combined CSV uses unified columns for easy analysis
- **Date Filtering**: Applied both at API level and post-processing after timezone conversion

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
