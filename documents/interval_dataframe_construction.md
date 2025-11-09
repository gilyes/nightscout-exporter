# Interval DataFrame Construction

## Overview

The interval DataFrame is the core data structure that combines glucose readings, treatments (insulin/carbs), basal insulin calculations, and loop system data (IOB/COB) into a unified time-aligned dataset. This document explains how the DataFrame is built from raw API data.

## Purpose

The interval DataFrame provides:
- **Fixed 5-minute intervals** (288 per day) for consistent time-based analysis
- **Time-aligned data** from multiple sources with different timestamps
- **Aggregated values** (averages for glucose, sums for insulin/carbs)
- **Calculated basal insulin** (scheduled + temp basals)
- **Matched IOB/COB** from devicestatus records

## Construction Process

The `build_interval_dataframe()` function (lines 1150-1267) orchestrates the construction process.

### Step 1: Generate Fixed 5-Minute Intervals

```python
INTERVAL_MINUTES = 5
intervals = pd.date_range(start=from_datetime, end=to_datetime, freq=f'{INTERVAL_MINUTES}min')
bins_for_cut = pd.date_range(start=from_datetime, periods=len(intervals) + 1, freq=f'{INTERVAL_MINUTES}min')
result_df = pd.DataFrame({'interval': intervals})
```

**What happens:**
- Creates a fixed grid of 5-minute intervals spanning the requested date range
- Example: 2024-01-01 00:00:00, 00:05:00, 00:10:00, ..., 23:55:00
- Creates bins for categorizing timestamps into intervals (one extra for the upper bound)
- Initializes result DataFrame with these intervals

**Why 5 minutes:**
- Matches CGM reading frequency (typically every 5 minutes)
- Balances granularity with data size
- Standard interval for diabetes data analysis

---

### Step 2: Process SGV (Sensor Glucose Values)

```python
if not entries_df.empty and 'sgv' in entries_df.columns:
    entries_df['DateTime'] = pd.to_datetime(entries_df['dateString'])
    entries_df['interval_bin'] = pd.cut(entries_df['DateTime'], bins=bins_for_cut,
                                        labels=intervals, include_lowest=True, right=False)

    sgv_intervals = entries_df.groupby('interval_bin', observed=True)['sgv'].mean().reset_index()
    sgv_intervals.columns = ['interval', 'sgv']
    result_df = result_df.merge(sgv_intervals, on='interval', how='left')
```

**What happens:**
1. Convert entry timestamps to datetime objects
2. **Bin assignment**: Use `pd.cut()` to assign each entry to its 5-minute interval
   - `include_lowest=True`: Include the lower bound of the first interval
   - `right=False`: Intervals are [start, end) - includes start, excludes end
3. **Aggregation**: Calculate mean SGV for each interval (handles multiple readings)
4. **Merge**: Left join to preserve all intervals (even those without readings)

**Example:**
- Raw entries: 10:02:30 (120 mg/dL), 10:03:45 (125 mg/dL)
- Both assigned to 10:00:00 interval
- Result: interval 10:00:00 has sgv = 122.5 (average)

**Edge cases:**
- No entries in interval → `sgv = NaN`
- Multiple entries → averaged
- Missing 'sgv' column → `sgv = None` for all intervals

---

### Step 3: Process Treatments - Bolus Insulin

```python
treatments_df['DateTime'] = pd.to_datetime(treatments_df['created_at'])
treatments_df['interval_bin'] = pd.cut(treatments_df['DateTime'], bins=bins_for_cut,
                                       labels=intervals, include_lowest=True, right=False)

bolus_df = treatments_df[treatments_df['eventType'] == 'Correction Bolus']
if not bolus_df.empty and 'insulin' in bolus_df.columns:
    bolus_intervals = bolus_df.groupby('interval_bin', observed=True)['insulin'].sum().reset_index()
    bolus_intervals.columns = ['interval', 'bolus']
    result_df = result_df.merge(bolus_intervals, on='interval', how='left')
```

**What happens:**
1. Convert treatment timestamps to datetime
2. Bin each treatment to its 5-minute interval
3. **Filter**: Only include "Correction Bolus" events
4. **Aggregation**: Sum insulin units within each interval
5. **Merge**: Left join to result DataFrame

**Why sum instead of average:**
- Multiple boluses in same interval should be totaled
- Example: 2 boluses of 3 units each → 6 units total

**Event type filtering:**
- Only "Correction Bolus" counted as bolus insulin
- Other event types (meals, temp targets) excluded
- This ensures accurate insulin delivery tracking

---

### Step 4: Process Treatments - Carbohydrates

```python
carb_df = treatments_df[treatments_df['eventType'] == 'Carb Correction']
if not carb_df.empty and 'carbs' in carb_df.columns:
    carb_intervals = carb_df.groupby('interval_bin', observed=True)['carbs'].sum().reset_index()
    carb_intervals.columns = ['interval', 'carbs']
    result_df = result_df.merge(carb_intervals, on='interval', how='left')
```

**What happens:**
1. Filter treatments for "Carb Correction" events
2. **Aggregation**: Sum carb grams within each interval
3. **Merge**: Left join to result DataFrame

**Similar logic to bolus:**
- Multiple carb entries summed
- Only "Carb Correction" events counted
- Preserves total carb intake per interval

---

### Step 4a: Track Other Treatment Events

```python
# Events dict - collect non-bolus/carb events
if 'eventType' in treatments_df.columns:
    events_dict = {}
    for interval_bin, group in treatments_df.groupby('interval_bin', observed=True):
        event_types = []
        for event_type in group['eventType'].dropna().unique():
            if event_type not in ['Correction Bolus', 'Carb Correction']:
                event_types.append(event_type)
        if event_types:
            events_dict[pd.to_datetime(interval_bin)] = event_types
    result_df['other_events'] = result_df['interval'].map(events_dict)
```

**What happens:**
1. Group treatments by interval
2. Collect event types that are NOT bolus or carb corrections
3. Store as temporary `other_events` column (list of event types)

**Examples of other events:**
- "Temp Target" - temporary glucose target changes
- "Profile Switch" - switching between different basal profiles
- "Announcement" - notes/annotations
- "Exercise" - activity logging
- "Site Change" - pump site changes

**Purpose:**
- Preserves information about non-insulin/carb events
- Used later in the comprehensive events column
- Helps track context around glucose/treatment changes

---

### Step 5: Add Basal Insulin Calculations

```python
if interval_data:
    basal_df = pd.DataFrame([
        {'interval': k, 'scheduled_basal': v['scheduled_basal'],
         'positive_temp': v['positive_temp'], 'negative_temp': v['negative_temp'],
         'total_basal': v['total_basal']}
        for k, v in interval_data.items()
    ])
    result_df = result_df.merge(basal_df, on='interval', how='left')

# Ensure columns exist
for col in ['scheduled_basal', 'positive_temp', 'negative_temp', 'total_basal']:
    if col not in result_df.columns:
        result_df[col] = 0
```

**What happens:**
1. Convert pre-calculated basal dict to DataFrame
2. **Merge**: Left join basal data on interval timestamp
3. Ensure all basal columns exist (default to 0 if missing)
4. Four basal columns added:
   - `scheduled_basal`: Basal rate from profile schedule
   - `positive_temp`: Additional insulin from temp basal > scheduled
   - `negative_temp`: Reduced insulin from temp basal < scheduled
   - `total_basal`: `scheduled_basal + positive_temp + negative_temp`

**Pre-calculation:**
- Basal data calculated separately using profile timeline + temp basal timeline
- Uses 5-minute intervals for precise calculation
- See `calculate_interval_based_basal()` for details

**Why separate:**
- Basal calculation is complex (profile switches, temp basals)
- Pre-calculating allows reuse and testing
- Keeps DataFrame construction logic cleaner

**Optimization:**
- DataFrame merge is faster than iterative row assignment
- Vectorized operation handles all intervals at once

---

### Step 6: Match IOB/COB from Devicestatus

```python
if devicestatus_df is not None and not devicestatus_df.empty:
    result_df['DateTime'] = pd.to_datetime(result_df['interval'])
    iob_values, cob_values = match_devicestatus_to_timestamps(
        result_df, devicestatus_df, max_time_diff_minutes=5
    )
    result_df['iob'] = iob_values
    result_df['cob'] = cob_values
    result_df = result_df.drop(columns=['DateTime'])
```

**What happens:**
1. Add temporary `DateTime` column for matching
2. Call `match_devicestatus_to_timestamps()` to find nearest IOB/COB values
3. Assign matched IOB/COB to each interval
4. Clean up temporary column

**Matching algorithm:**
- For each interval timestamp, find nearest devicestatus record within ±5 minutes
- Uses binary search for O(log n) performance
- If no match within window → `IOB/COB = None`
- See `match_devicestatus_to_timestamps()` documentation for details

**Data source:**
- IOB (Insulin On Board): `loop.iob.iob` from devicestatus
- COB (Carbs On Board): `loop.cob.cob` from devicestatus
- Only supports Loop system data structure

---

### Step 7: Generate Events Column

```python
def build_events_list(row):
    events = []
    if pd.notna(row.get('sgv')):
        events.append('SGV')
    if pd.notna(row.get('bolus')) and row['bolus'] > 0:
        events.append('Correction Bolus')
    if pd.notna(row.get('carbs')) and row['carbs'] > 0:
        events.append('Carb Correction')
    other_events = row.get('other_events')
    if isinstance(other_events, list) and len(other_events) > 0:
        events.extend(other_events)
    return ', '.join(events) if events else None

result_df['events'] = result_df.apply(build_events_list, axis=1)
result_df = result_df.drop(columns=['other_events'])
```

**What happens:**
1. For each interval row, build list of event types present
2. Include specific event type names (not just generic categories)
3. Append other treatment events from `other_events` list
4. Join as comma-separated string
5. Drop temporary `other_events` column

**Event types included:**
- `SGV` - glucose reading present
- `Correction Bolus` - insulin bolus administered
- `Carb Correction` - carbohydrates recorded
- Plus any other treatment events: `Temp Target`, `Profile Switch`, `Exercise`, etc.

**Purpose:**
- Detailed log of what happened in each interval
- Shows specific Nightscout event types
- Useful for filtering and analysis
- Debugging aid

**Example values:**
- `"SGV"` - only glucose reading
- `"SGV, Correction Bolus, Carb Correction"` - meal with insulin
- `"SGV, Temp Target, Exercise"` - glucose reading with context events
- `"Correction Bolus, Carb Correction, Profile Switch"` - multiple treatment events
- `None` - empty interval (no data)

**Note:** IOB/COB are not included in events column as they're continuous calculated values rather than discrete events

---

### Step 8: Handle Null/Zero Values

```python
# Replace 0 with None for bolus and carbs (distinguish absence from zero)
result_df['bolus'] = result_df['bolus'].apply(lambda x: None if pd.isna(x) or x == 0 else x)
result_df['carbs'] = result_df['carbs'].apply(lambda x: None if pd.isna(x) or x == 0 else x)
```

**What happens:**
1. Convert `0` values to `None` for bolus and carbs
2. Distinguishes "no data" from "explicitly zero"

**Why:**
- `None` = no treatment in this interval
- `0` could be ambiguous (was it recorded as 0, or missing?)
- Makes CSV output cleaner (empty cells vs "0")

**Not applied to:**
- SGV: 0 mg/dL is invalid but handled upstream
- Basal: 0 is valid (temp basal can be 0)
- IOB/COB: 0 is valid (no insulin/carbs remaining)

---

### Step 9: Sort and Finalize Column Order

```python
# Sort by interval timestamp
result_df = result_df.sort_values('interval')

# Ensure consistent column order
column_order = ['interval', 'sgv', 'bolus', 'scheduled_basal', 'positive_temp',
                'negative_temp', 'total_basal', 'carbs', 'iob', 'cob', 'events']
result_df = result_df[column_order]
```

**What happens:**
1. Sort rows by interval timestamp (ascending)
2. Reorder columns to consistent, logical sequence

**Column order rationale:**
1. `interval` - timestamp (key)
2. `sgv` - glucose reading (primary metric)
3. `bolus` - direct insulin delivery
4. `scheduled_basal`, `positive_temp`, `negative_temp`, `total_basal` - basal insulin breakdown
5. `carbs` - carbohydrate intake
6. `iob`, `cob` - calculated metrics from loop
7. `events` - metadata summary

---

## Final DataFrame Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `interval` | datetime | 5-minute interval timestamp | 2024-01-15 10:00:00 |
| `sgv` | float | Average glucose in mg/dL or mmol/L | 120.0 (or 6.7) |
| `bolus` | float | Total bolus insulin in units | 3.5 |
| `scheduled_basal` | float | Scheduled basal insulin in units | 0.05 |
| `positive_temp` | float | Additional insulin from high temp | 0.02 |
| `negative_temp` | float | Reduced insulin from low temp | -0.01 |
| `total_basal` | float | Total basal (scheduled + temps) | 0.06 |
| `carbs` | float | Total carbs in grams | 45.0 |
| `iob` | float | Insulin on board (from Loop) | 2.3 |
| `cob` | float | Carbs on board (from Loop) | 28.0 |
| `events` | string | Comma-separated event type list | "SGV, Correction Bolus, Carb Correction" |

---

## Key Design Decisions

### 1. Fixed Intervals vs Sparse Events
**Choice**: Fixed 5-minute intervals (even if empty)

**Rationale:**
- Consistent time grid for analysis and visualization
- Easy to identify gaps in data
- Predictable row count (288 per day)
- Simplifies time-series operations

**Alternative**: Only intervals with data
- Would reduce file size
- But complicates analysis (irregular timestamps)

### 2. Left Join Strategy
**Choice**: Left join all data sources to interval grid

**Rationale:**
- Preserves all intervals (even empty ones)
- `NaN` clearly indicates missing data
- Maintains chronological continuity

**Alternative**: Inner join (only intervals with data)
- Would lose empty intervals
- Could miss important gaps (sensor failures, etc.)

### 3. Aggregation Methods
**Choice**:
- SGV: **mean** (average)
- Bolus/Carbs: **sum** (total)

**Rationale:**
- SGV averaging handles multiple readings naturally
- Summing insulin/carbs preserves total delivery/intake
- Matches clinical interpretation

**Example**:
- 2 readings (120, 124) in interval → mean = 122
- 2 boluses (2u, 3u) in interval → sum = 5u

### 4. Timestamp Binning
**Choice**: `pd.cut()` with `right=False` (left-closed intervals)

**Rationale:**
- `[10:00, 10:05)` includes 10:00:00 but excludes 10:05:00
- 10:05:00 belongs to next interval `[10:05, 10:10)`
- Prevents double-counting at boundaries
- Standard convention in time-series analysis

---

## Performance Considerations

### Optimizations Applied

1. **Vectorized Operations**
   - Uses pandas groupby/merge instead of loops
   - Significantly faster for large datasets

2. **Binary Search for Devicestatus Matching**
   - O(log n) lookup vs O(n) linear search
   - Critical for large devicestatus datasets

3. **Single Pass Processing**
   - Each data source processed once
   - Results merged incrementally

### Typical Performance

For a 30-day export:
- Intervals: 8,640 rows (30 days × 288 intervals/day)
- Processing time: ~2-5 seconds (depending on data volume)
- Memory: Minimal (pandas handles efficiently)

---

## Error Handling

### Missing Data
- **No entries**: `sgv = NaN` for all intervals
- **No treatments**: `bolus/carbs = None` for all intervals
- **No devicestatus**: `iob/cob = None` for all intervals
- **No profiles**: Basal columns = `NaN` (warning logged)

### Invalid Data
- **Malformed timestamps**: Filtered out upstream during API fetch
- **Missing required columns**: Handled with column checks (`if 'column' in df.columns`)
- **Empty DataFrames**: Checked before processing each source

### Data Quality Warnings
Printed to console when issues detected:
- "Warning: No entries returned from API"
- "Warning: No devicestatus data available for IOB/COB matching"
- "Warning: No data to time-align"

---

## Example Output

```csv
interval,sgv,bolus,scheduled_basal,positive_temp,negative_temp,total_basal,carbs,iob,cob,events
2024-01-15 10:00:00,122.5,,0.05,0.0,0.0,0.05,,2.3,0.0,SGV
2024-01-15 10:05:00,125.0,,0.05,0.0,0.0,0.05,,2.2,0.0,SGV
2024-01-15 10:10:00,128.3,3.5,0.05,0.0,0.0,0.05,45.0,5.7,45.0,"SGV, Correction Bolus, Carb Correction"
2024-01-15 10:15:00,126.0,,0.05,0.0,0.0,0.05,,5.5,42.0,SGV
2024-01-15 10:20:00,124.5,,0.05,0.02,0.0,0.07,,5.3,39.0,"SGV, Temp Target"
```

**Interpretation:**
- 10:00-10:05: Normal glucose readings, basal only
- 10:10: Meal bolus (3.5u) + carbs (45g) entered
  - IOB increases to 5.7u, COB starts at 45g
  - Events show both bolus and carb correction
- 10:15: IOB/COB decreasing as insulin acts and carbs absorb
- 10:20: Temp target set (context event), positive temp basal active (+0.02u)

---

## Integration with Main Pipeline

The interval DataFrame is built within `process_time_aligned_data()`:

```python
def process_time_aligned_data(...):
    # 1. Calculate basal insulin for all intervals
    interval_data = calculate_interval_based_basal(...)

    # 2. Build interval DataFrame from entries/treatments
    df = build_interval_dataframe(
        from_datetime,
        to_datetime,
        entries_df,
        treatments_df,
        interval_data,
        devicestatus_df
    )

    # 3. Convert interval column to string for CSV export
    df['interval'] = df['interval'].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df
```

**Flow:**
1. Main script fetches raw data from API
2. `process_time_aligned_data()` orchestrates processing
3. `build_interval_dataframe()` constructs unified dataset
4. Result exported to `yyyyMMdd-yyyyMMdd-time_aligned.csv`

---

## Related Documentation

- **Basal Calculations**: See `basal_insulin_calculations.md` (if exists)
- **Devicestatus Matching**: See `devicestatus_matching_algorithm.md` (if exists)
- **API Data Fetching**: See `README.md` - Data Flow section

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-01-08 | 1.0 | Initial documentation |
