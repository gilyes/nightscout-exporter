"""Integration tests for process_time_aligned_data orchestration."""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import (
    process_time_aligned_data,
    build_profile_timeline,
    build_temp_basal_timeline
)


class TestProcessTimeAlignedData:
    """Test process_time_aligned_data function integration."""

    def test_full_integration_with_all_data(self, sample_profile_data):
        """Test full integration with entries, treatments, profiles, and devicestatus."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 15, 0)

        # Create sample data
        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:02:00', 'sgv': 120},
            {'dateString': '2024-01-15 10:07:00', 'sgv': 125},
            {'dateString': '2024-01-15 10:12:00', 'sgv': 130}
        ])

        treatments_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:03:00',
                'eventType': 'Correction Bolus',
                'insulin': 2.0
            },
            {
                'created_at': '2024-01-15 10:05:00',
                'eventType': 'Temp Basal',
                'duration': 30,
                'absolute': 0.8
            }
        ])

        devicestatus_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:02:00',
                'loop': {'iob': {'iob': 2.5}, 'cob': {'cob': 30.0}}
            }
        ])

        # Build timelines
        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_basal_timeline = build_temp_basal_timeline(treatments_df)

        # Process time-aligned data
        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            devicestatus_df, profile_timeline, temp_basal_timeline
        )

        # Verify structure
        assert not df.empty
        assert len(df) == 4  # 10:00, 10:05, 10:10, 10:15

        # Verify all columns exist
        expected_columns = ['interval', 'sgv', 'bolus', 'scheduled_basal',
                          'positive_temp', 'negative_temp', 'total_basal',
                          'carbs', 'iob', 'cob', 'events']
        for col in expected_columns:
            assert col in df.columns

        # Verify interval timestamps are strings (converted for CSV)
        assert isinstance(df.iloc[0]['interval'], str)
        assert df.iloc[0]['interval'] == '2024-01-15 10:00:00'

        # Verify data is present
        assert df['sgv'].notna().sum() > 0
        assert df['bolus'].notna().sum() > 0
        assert df['iob'].notna().sum() > 0

    def test_integration_without_profile_timeline(self):
        """Test integration when profile_timeline is None (should skip basal calculations)."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 15, 0)

        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:02:00', 'sgv': 120}
        ])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        # Process without profile timeline
        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, None, None
        )

        # Should still work but with zero basal values
        assert not df.empty
        assert 'scheduled_basal' in df.columns
        assert df['scheduled_basal'].iloc[0] == 0
        assert df['total_basal'].iloc[0] == 0

    def test_integration_with_empty_entries(self, sample_profile_data):
        """Test integration with empty entries DataFrame."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 15, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        profile_timeline = build_profile_timeline(sample_profile_data)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, []
        )

        # Should create intervals even with no data
        assert not df.empty
        assert len(df) == 4

        # SGV should be null
        assert df['sgv'].isna().all()

        # But basal should be calculated
        assert df['scheduled_basal'].notna().all()
        assert df['total_basal'].notna().all()

    def test_integration_with_empty_treatments(self, sample_profile_data):
        """Test integration with empty treatments DataFrame."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 15, 0)

        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:02:00', 'sgv': 120}
        ])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        profile_timeline = build_profile_timeline(sample_profile_data)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, []
        )

        # Should work with SGV but no treatments
        assert not df.empty
        assert df['sgv'].notna().sum() > 0
        assert df['bolus'].isna().all()
        assert df['carbs'].isna().all()

    def test_integration_basal_calculations(self, sample_profile_data):
        """Test that basal calculations are integrated correctly."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 15, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:00:00',
                'eventType': 'Temp Basal',
                'duration': 10,  # 10 minutes = 2 intervals
                'absolute': 1.2  # Higher than scheduled (0.6)
            }
        ])

        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_basal_timeline = build_temp_basal_timeline(treatments_df)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, temp_basal_timeline
        )

        # First interval (10:00-10:05) should have temp basal
        # Scheduled = 0.6 U/hr * 5 min / 60 = 0.05 U
        assert df.iloc[0]['scheduled_basal'] == pytest.approx(0.05, abs=0.001)

        # Active rate is 1.2, so positive temp = (1.2 - 0.6) * 5/60 = 0.05
        assert df.iloc[0]['positive_temp'] == pytest.approx(0.05, abs=0.001)

        # Total = scheduled + positive_temp = 0.10
        assert df.iloc[0]['total_basal'] == pytest.approx(0.10, abs=0.001)

        # Second interval (10:05-10:10) should also have temp basal
        assert df.iloc[1]['positive_temp'] > 0

        # Third interval (10:10-10:15) should NOT have temp basal (expired)
        assert df.iloc[2]['positive_temp'] == 0

    def test_integration_with_devicestatus_matching(self, sample_profile_data):
        """Test that devicestatus IOB/COB is integrated correctly."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 10, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        # Devicestatus at different times
        devicestatus_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:02:00',
                'loop': {'iob': {'iob': 2.5}, 'cob': {'cob': 30.0}}
            },
            {
                'created_at': '2024-01-15 10:07:00',
                'loop': {'iob': {'iob': 2.3}, 'cob': {'cob': 28.0}}
            }
        ])

        profile_timeline = build_profile_timeline(sample_profile_data)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            devicestatus_df, profile_timeline, []
        )

        # First interval (10:00) should match 10:02 devicestatus (within 5 min)
        assert df.iloc[0]['iob'] == 2.5
        assert df.iloc[0]['cob'] == 30.0

        # Second interval (10:05) should match 10:07 devicestatus
        assert df.iloc[1]['iob'] == 2.3
        assert df.iloc[1]['cob'] == 28.0

    def test_integration_timestamp_string_conversion(self, sample_profile_data):
        """Test that interval timestamps are converted to strings for CSV export."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        profile_timeline = build_profile_timeline(sample_profile_data)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, []
        )

        # All interval values should be strings
        for interval_val in df['interval']:
            assert isinstance(interval_val, str)
            # Verify format
            assert len(interval_val) == 19  # "YYYY-MM-DD HH:MM:SS"
            assert interval_val[4] == '-'
            assert interval_val[7] == '-'
            assert interval_val[10] == ' '
            assert interval_val[13] == ':'
            assert interval_val[16] == ':'

    def test_integration_with_large_date_range(self, sample_profile_data):
        """Test integration with a large date range (multiple days).

        Note: to_datetime=None is handled by main(), not by process_time_aligned_data.
        """
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 18, 10, 0, 0)  # 3 days

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        profile_timeline = build_profile_timeline(sample_profile_data)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, []
        )

        # Should work with large date range
        assert not df.empty
        # 3 days = 72 hours = 864 intervals (at 5 min each)
        assert len(df) > 800

    def test_integration_date_range_boundaries(self, sample_profile_data):
        """Test integration with data at exact boundaries."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        # Data exactly at boundaries
        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:00:00', 'sgv': 100},  # Start
            {'dateString': '2024-01-15 10:05:00', 'sgv': 110},  # End
        ])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        profile_timeline = build_profile_timeline(sample_profile_data)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, []
        )

        # Both boundary points should be included
        assert df.iloc[0]['sgv'] == 100  # First interval has first reading
        assert df.iloc[1]['sgv'] == 110  # Second interval has second reading

    def test_integration_with_complex_treatments(self, sample_profile_data):
        """Test integration with multiple treatment types."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 15, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:02:00',
                'eventType': 'Correction Bolus',
                'insulin': 2.0
            },
            {
                'created_at': '2024-01-15 10:02:30',
                'eventType': 'Carb Correction',
                'carbs': 30.0
            },
            {
                'created_at': '2024-01-15 10:00:00',  # Start at beginning of interval
                'eventType': 'Temp Basal',
                'duration': 30,
                'absolute': 0.8
            },
            {
                'created_at': '2024-01-15 10:04:00',
                'eventType': 'Note',
                'notes': 'Exercise'
            }
        ])

        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_basal_timeline = build_temp_basal_timeline(treatments_df)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, temp_basal_timeline
        )

        # First interval should have bolus, carbs, and note event
        assert df.iloc[0]['bolus'] == 2.0
        assert df.iloc[0]['carbs'] == 30.0
        assert 'Note' in df.iloc[0]['events']
        assert 'Correction Bolus' in df.iloc[0]['events']
        assert 'Carb Correction' in df.iloc[0]['events']

        # Should have temp basal adjustment (positive because 0.8 > 0.6 scheduled)
        assert df.iloc[0]['positive_temp'] > 0

    def test_integration_statistics_calculation(self, sample_profile_data, capsys):
        """Test that statistics are calculated and printed correctly."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 11, 0, 0)  # 1 hour = 12 intervals

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:00:00',
                'eventType': 'Temp Basal',
                'duration': 30,
                'absolute': 1.2  # Increase from 0.6 to 1.2
            }
        ])

        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_basal_timeline = build_temp_basal_timeline(treatments_df)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, temp_basal_timeline
        )

        # Capture printed output
        captured = capsys.readouterr()

        # Verify statistics are printed
        assert "Total basal insulin:" in captured.out
        assert "Scheduled basal:" in captured.out
        assert "Positive temp:" in captured.out
        assert "Negative temp:" in captured.out

        # Verify statistics are correct
        # 1 hour = 13 intervals (10:00, 10:05, 10:10, ..., 11:00)
        # Scheduled rate at hour 10 = 0.6 U/hr
        # Without temp: 13 * 0.05 = 0.65 U
        # With temp from 10:00 for 30 minutes = 6 intervals (10:00, 10:05, 10:10, 10:15, 10:20, 10:25)
        # Temp increase per interval = (1.2 - 0.6) * 5/60 = 0.05 U
        # Total temp increase = 6 * 0.05 = 0.3 U
        # Total basal = 0.65 + 0.3 = 0.95 U
        total_basal = df['total_basal'].sum()
        assert total_basal == pytest.approx(0.95, abs=0.01)

    def test_integration_with_negative_temp_basal(self, sample_profile_data):
        """Test integration with negative temp basal (rate lower than scheduled)."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 15, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:00:00',
                'eventType': 'Temp Basal',
                'duration': 10,
                'absolute': 0.3  # Lower than scheduled (0.6)
            }
        ])

        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_basal_timeline = build_temp_basal_timeline(treatments_df)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, profile_timeline, temp_basal_timeline
        )

        # First interval should have negative temp basal
        assert df.iloc[0]['positive_temp'] == 0
        assert df.iloc[0]['negative_temp'] < 0

        # Negative temp = (0.3 - 0.6) * 5/60 = -0.025
        assert df.iloc[0]['negative_temp'] == pytest.approx(-0.025, abs=0.001)

    def test_integration_empty_result_handling(self):
        """Test handling when time-aligned data is empty."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 0, 0)  # Same time = no intervals

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            None, None, None
        )

        # Should return empty DataFrame with warning
        # Actually, with same from/to, we get exactly 1 interval at 10:00
        assert len(df) == 1

    def test_integration_preserves_data_types(self, sample_profile_data):
        """Test that data types are preserved correctly after processing."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:02:00', 'sgv': 120}
        ])
        treatments_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:02:00',
                'eventType': 'Correction Bolus',
                'insulin': 2.5
            }
        ])

        devicestatus_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:02:00',
                'loop': {'iob': {'iob': 2.5}, 'cob': {'cob': 30.0}}
            }
        ])

        profile_timeline = build_profile_timeline(sample_profile_data)

        df = process_time_aligned_data(
            entries_df, treatments_df, from_datetime, to_datetime,
            devicestatus_df, profile_timeline, []
        )

        # interval should be string
        assert df['interval'].dtype == 'object'
        assert isinstance(df.iloc[0]['interval'], str)

        # Numeric fields should be float or None
        row = df.iloc[0]
        if pd.notna(row['sgv']):
            assert isinstance(row['sgv'], (int, float))
        if pd.notna(row['bolus']):
            assert isinstance(row['bolus'], (int, float))
        if pd.notna(row['iob']):
            assert isinstance(row['iob'], (int, float))
        if pd.notna(row['cob']):
            assert isinstance(row['cob'], (int, float))

        # Basal values should always be numeric
        assert isinstance(row['scheduled_basal'], (int, float))
        assert isinstance(row['total_basal'], (int, float))
