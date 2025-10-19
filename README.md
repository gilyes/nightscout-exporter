# Nightscout Data Exporter

A PowerShell tool for exporting glucose monitoring data (CGM entries and treatments) from Nightscout API to CSV format.

## Features

- Fetches CGM (Continuous Glucose Monitor) entries and treatment data from Nightscout
- Converts glucose values from mg/dL to mmol/L (enabled by default, configurable)
- Works in local timezone by default (automatically handles UTC conversion)
- Exports data in multiple formats:
  - Raw entries CSV (with converted values)
  - Raw treatments CSV (normalized columns)
  - Combined and processed CSV with unified schema
- Configurable date ranges and record limits

## Prerequisites

- **Windows**: PowerShell 5.1 or higher (pre-installed on Windows 10/11)
- **Linux**: PowerShell 7+ ([install guide](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-linux))
- **macOS**: PowerShell 7+ ([install guide](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-macos))
- Access to a Nightscout instance
- Nightscout API token


## Installation

1. Clone this repository:
```bash
git clone https://github.com/gilyes/nightscout-exporter.git
cd nightscout-exporter
```

2. Create a `.env` file in the root directory with your Nightscout credentials:
```
NIGHTSCOUT_URL=https://your-nightscout-url.com
NIGHTSCOUT_TOKEN=your-token-here
```

Reference `.env.sample` for the expected format.

## Usage

### Basic Usage

Fetch the last month of data (default):

```powershell
.\Export-NightscoutData.ps1
```

### Glucose Unit Conversion

**By default, glucose values are converted from mg/dL to mmol/L.** To keep original mg/dL values:

```powershell
# Keep original mg/dL values
.\Export-NightscoutData.ps1 -ConvertToMmol $false

# Convert to mmol/L (default)
.\Export-NightscoutData.ps1
```

### Custom Date Range

Specify custom date range (dates are in your local timezone by default):

```powershell
.\Export-NightscoutData.ps1 -FromDate "2024-01-01" -ToDate "2024-12-31"
```

### Working with UTC Instead of Local Timezone

By default, input dates and output timestamps use your local timezone. To work entirely in UTC:

```powershell
.\Export-NightscoutData.ps1 -UseLocalTimezone $false
```

### Custom Maximum Record Count

Limit the number of records fetched:

```powershell
.\Export-NightscoutData.ps1 -MaxCount 50000
```

### Custom Output Folder

Specify a custom output folder (defaults to `.\Output`):

```powershell
# Use custom output folder (relative path)
.\Export-NightscoutData.ps1 -OutputFolder ".\MyData"

# Use absolute path
.\Export-NightscoutData.ps1 -OutputFolder "C:\Data\Nightscout"
```

### Combined Parameters

```powershell
.\Export-NightscoutData.ps1 -FromDate "2024-06-01" -ToDate "2024-06-30" -MaxCount 20000 -ConvertToMmol $false -OutputFolder ".\June2024"
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `FromDate` | string | Last month | Start date in yyyy-MM-dd format (local timezone by default) |
| `ToDate` | string | null | End date in yyyy-MM-dd format (optional) |
| `MaxCount` | int | 100000 | Maximum number of records to fetch |
| `ConvertToMmol` | bool | **true** | **Convert glucose values from mg/dL to mmol/L** |
| `UseLocalTimezone` | bool | true | Work in local timezone (input dates and output timestamps) |
| `OutputFolder` | string | .\Output | Output folder path (relative or absolute) |

## Output Files

The script creates an output folder (default: `Output/`) with three CSV files named with the date range:

File naming format: `yyyyMMdd-yyyyMMdd-<type>.csv`

For example, if you export data from 2024-01-01 to 2024-12-31, the files will be:
- `20240101-20241231-entries.csv`
- `20240101-20241231-treatments.csv`
- `20240101-20241231-combined.csv`

If no end date is specified, the current date is used:
- `20240101-20251014-entries.csv`

### File Contents

1. **yyyyMMdd-yyyyMMdd-entries.csv**
   - Raw CGM entries from the Nightscout API
   - Contains sensor glucose values and timestamps
   - Glucose values converted to mmol/L by default (set `-ConvertToMmol $false` to keep mg/dL)

2. **yyyyMMdd-yyyyMMdd-treatments.csv**
   - Raw treatments from the Nightscout API
   - Contains insulin, carbs, and other treatment data
   - Normalized to include both `insulin` and `amount` columns

3. **yyyyMMdd-yyyyMMdd-combined.csv**
   - Processed and merged dataset with unified schema
   - Glucose values converted to mmol/L by default
   - Timestamps in local timezone by default
   - Sorted by DateTime
   - Columns: DateTime, eventType, sgv, insulin, amount, carbs

## Data Transformations

- **Glucose Units**: SGV values are converted from mg/dL to mmol/L (÷ 18.0182) by default. Set `-ConvertToMmol $false` to keep mg/dL values.
- **Timezone**: When `UseLocalTimezone` is `$true` (default), input dates are treated as local time and converted to UTC for API queries, then response timestamps are converted back to local time for output. Set to `$false` to work entirely in UTC.
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
