"""Tests for data time-alignment functions."""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import (
    build_interval_dataframe,
    parse_devicestatus_iob,
    parse_devicestatus_cob
)


class TestBuildIntervalDataframe:
    """Test build_interval_dataframe function."""

    def test_basic_interval_structure(self):
        """Test that intervals are created correctly."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 15, 0)  # 15 minutes = 3 intervals

        # Empty dataframes need columns to avoid implementation bug
        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Should have 4 intervals: 10:00, 10:05, 10:10, 10:15
        assert len(df) == 4
        assert 'interval' in df.columns

        # Check all expected columns exist
        expected_columns = ['interval', 'sgv', 'bolus', 'scheduled_basal',
                          'positive_temp', 'negative_temp', 'total_basal',
                          'carbs', 'iob', 'cob', 'events']
        for col in expected_columns:
            assert col in df.columns

    def test_sgv_averaging_single_reading(self):
        """Test that single SGV reading in interval is used."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:02:00', 'sgv': 120}
        ])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # First interval should have SGV = 120
        assert df.iloc[0]['sgv'] == 120

    def test_sgv_averaging_multiple_readings(self):
        """Test that multiple SGV readings in same interval are averaged."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:01:00', 'sgv': 110},
            {'dateString': '2024-01-15 10:02:00', 'sgv': 120},
            {'dateString': '2024-01-15 10:03:00', 'sgv': 130}
        ])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # First interval should have average of 110, 120, 130 = 120
        assert df.iloc[0]['sgv'] == pytest.approx(120.0, abs=0.1)

    def test_sgv_with_nan_values(self):
        """Test that NaN SGV values are excluded from averaging."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:01:00', 'sgv': 110},
            {'dateString': '2024-01-15 10:02:00', 'sgv': None},
            {'dateString': '2024-01-15 10:03:00', 'sgv': 130}
        ])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Should average only 110 and 130 = 120
        assert df.iloc[0]['sgv'] == pytest.approx(120.0, abs=0.1)

    def test_bolus_filtering_correction_bolus_only(self):
        """Test that only 'Correction Bolus' events are counted as bolus."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {'created_at': '2024-01-15 10:02:00', 'eventType': 'Correction Bolus', 'insulin': 2.0},
            {'created_at': '2024-01-15 10:03:00', 'eventType': 'Meal Bolus', 'insulin': 5.0},
            {'created_at': '2024-01-15 10:04:00', 'eventType': 'Temp Basal', 'insulin': 1.0}
        ])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Should only count Correction Bolus = 2.0
        assert df.iloc[0]['bolus'] == 2.0

    def test_bolus_summing_multiple_in_interval(self):
        """Test that multiple Correction Bolus events in same interval are summed."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {'created_at': '2024-01-15 10:01:00', 'eventType': 'Correction Bolus', 'insulin': 2.0},
            {'created_at': '2024-01-15 10:03:00', 'eventType': 'Correction Bolus', 'insulin': 1.5}
        ])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Should sum 2.0 + 1.5 = 3.5
        assert df.iloc[0]['bolus'] == 3.5

    def test_carb_filtering_carb_correction_only(self):
        """Test that only 'Carb Correction' events are counted as carbs."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {'created_at': '2024-01-15 10:02:00', 'eventType': 'Carb Correction', 'carbs': 30.0},
            {'created_at': '2024-01-15 10:03:00', 'eventType': 'Meal Bolus', 'carbs': 50.0},
            {'created_at': '2024-01-15 10:04:00', 'eventType': 'Snack Bolus', 'carbs': 15.0}
        ])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Should only count Carb Correction = 30.0
        assert df.iloc[0]['carbs'] == 30.0

    def test_carb_summing_multiple_in_interval(self):
        """Test that multiple Carb Correction events in same interval are summed."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {'created_at': '2024-01-15 10:01:00', 'eventType': 'Carb Correction', 'carbs': 20.0},
            {'created_at': '2024-01-15 10:03:00', 'eventType': 'Carb Correction', 'carbs': 15.0}
        ])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Should sum 20.0 + 15.0 = 35.0
        assert df.iloc[0]['carbs'] == 35.0

    def test_interval_boundaries(self):
        """Test that data is correctly assigned to 5-minute intervals."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 10, 0)

        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:00:00', 'sgv': 100},  # First interval
            {'dateString': '2024-01-15 10:04:59', 'sgv': 110},  # First interval (just before boundary)
            {'dateString': '2024-01-15 10:05:00', 'sgv': 120},  # Second interval (exactly at boundary)
            {'dateString': '2024-01-15 10:09:59', 'sgv': 130},  # Second interval
        ])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # First interval (10:00-10:05) should have average of 100 and 110 = 105
        assert df.iloc[0]['sgv'] == pytest.approx(105.0, abs=0.1)

        # Second interval (10:05-10:10) should have average of 120 and 130 = 125
        assert df.iloc[1]['sgv'] == pytest.approx(125.0, abs=0.1)

    def test_basal_data_integration(self):
        """Test that basal data from interval_data is correctly integrated."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])

        # Create interval_data with basal info
        interval_time = pd.Timestamp('2024-01-15 10:00:00')
        interval_data = {
            interval_time: {
                'scheduled_basal': 0.05,
                'positive_temp': 0.02,
                'negative_temp': -0.01,
                'total_basal': 0.06
            }
        }

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Check basal values are correctly assigned
        assert df.iloc[0]['scheduled_basal'] == 0.05
        assert df.iloc[0]['positive_temp'] == 0.02
        assert df.iloc[0]['negative_temp'] == -0.01
        assert df.iloc[0]['total_basal'] == 0.06

    def test_events_list_generation(self):
        """Test that events list is correctly generated."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame([
            {'dateString': '2024-01-15 10:02:00', 'sgv': 120}
        ])
        treatments_df = pd.DataFrame([
            {'created_at': '2024-01-15 10:02:00', 'eventType': 'Correction Bolus', 'insulin': 2.0},
            {'created_at': '2024-01-15 10:03:00', 'eventType': 'Carb Correction', 'carbs': 30.0},
            {'created_at': '2024-01-15 10:04:00', 'eventType': 'Note', 'notes': 'Exercise'}
        ])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Should have events: SGV, Correction Bolus, Carb Correction, Note
        events = df.iloc[0]['events']
        assert 'SGV' in events
        assert 'Correction Bolus' in events
        assert 'Carb Correction' in events
        assert 'Note' in events

    def test_events_list_no_duplicates(self):
        """Test that events list doesn't have duplicate entries."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {'created_at': '2024-01-15 10:02:00', 'eventType': 'Correction Bolus', 'insulin': 2.0},
            {'created_at': '2024-01-15 10:03:00', 'eventType': 'Correction Bolus', 'insulin': 1.0},
        ])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Should only have one 'Correction Bolus' entry in events
        events = df.iloc[0]['events']
        assert events == 'Correction Bolus'

    def test_empty_interval(self):
        """Test that intervals with no data have None/null values."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # All data fields should be None/null
        assert pd.isna(df.iloc[0]['sgv'])
        assert pd.isna(df.iloc[0]['bolus'])
        assert pd.isna(df.iloc[0]['carbs'])
        assert pd.isna(df.iloc[0]['iob'])
        assert pd.isna(df.iloc[0]['cob'])
        assert pd.isna(df.iloc[0]['events'])

        # Basal should be 0 (from empty interval_data)
        assert df.iloc[0]['scheduled_basal'] == 0
        assert df.iloc[0]['total_basal'] == 0

    def test_iob_cob_matching_with_devicestatus(self):
        """Test that IOB/COB from devicestatus is matched to intervals."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 10, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        # Devicestatus at 10:02 (within 5 min of 10:00 interval)
        devicestatus_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:02:00',
                'loop': {'iob': {'iob': 2.5}, 'cob': {'cob': 30.0}}
            }
        ])

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df,
            interval_data, devicestatus_df
        )

        # First interval (10:00) should have matched IOB/COB
        assert df.iloc[0]['iob'] == 2.5
        assert df.iloc[0]['cob'] == 30.0

    def test_iob_cob_matching_outside_window(self):
        """Test that IOB/COB outside 5-minute window is not matched."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        # Devicestatus at 10:10 (> 5 min from 10:00 interval)
        devicestatus_df = pd.DataFrame([
            {
                'created_at': '2024-01-15 10:10:00',
                'loop': {'iob': {'iob': 2.5}, 'cob': {'cob': 30.0}}
            }
        ])

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df,
            interval_data, devicestatus_df
        )

        # First interval (10:00) should not have matched IOB/COB
        assert pd.isna(df.iloc[0]['iob'])
        assert pd.isna(df.iloc[0]['cob'])

    def test_null_bolus_and_carbs_when_zero(self):
        """Test that bolus and carbs are None when zero."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame([
            {'created_at': '2024-01-15 10:02:00', 'eventType': 'Note', 'notes': 'Test'}
        ])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Bolus and carbs should be None (not 0) when no bolus/carb events
        assert pd.isna(df.iloc[0]['bolus'])
        assert pd.isna(df.iloc[0]['carbs'])

    def test_interval_timestamps_are_timestamps(self):
        """Test that interval timestamps are pandas Timestamp objects.

        Note: Conversion to strings happens later in process_time_aligned_data().
        """
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        # Interval should be a Timestamp object (strings conversion happens later)
        assert df.iloc[0]['interval'] == pd.Timestamp('2024-01-15 10:00:00')
        assert isinstance(df.iloc[0]['interval'], pd.Timestamp)

    def test_empty_dataframes(self):
        """Test handling of empty entries and treatments DataFrames."""
        from_datetime = datetime(2024, 1, 15, 10, 0, 0)
        to_datetime = datetime(2024, 1, 15, 10, 5, 0)

        entries_df = pd.DataFrame(columns=['dateString', 'sgv'])
        treatments_df = pd.DataFrame(columns=['created_at', 'eventType'])
        interval_data = {}

        # Should not raise error with empty DataFrames
        df = build_interval_dataframe(
            from_datetime, to_datetime, entries_df, treatments_df, interval_data
        )

        assert len(df) == 2  # 10:00 and 10:05
        assert not df.empty


class TestDevicestatusIOBCOBParsing:
    """Test devicestatus IOB/COB parsing helpers."""

    def test_parse_iob_loop_structure(self):
        """Test parsing IOB from Loop structure."""
        record = {
            'loop': {
                'iob': {'iob': 2.5, 'timestamp': '2024-01-15T10:00:00Z'}
            }
        }

        iob = parse_devicestatus_iob(record)
        assert iob == 2.5

    def test_parse_cob_loop_structure(self):
        """Test parsing COB from Loop structure."""
        record = {
            'loop': {
                'cob': {'cob': 30.0, 'timestamp': '2024-01-15T10:00:00Z'}
            }
        }

        cob = parse_devicestatus_cob(record)
        assert cob == 30.0

    def test_parse_iob_missing_returns_none(self):
        """Test that missing IOB returns None."""
        record = {
            'loop': {
                'cob': {'cob': 30.0}
            }
        }

        iob = parse_devicestatus_iob(record)
        assert iob is None

    def test_parse_cob_missing_returns_none(self):
        """Test that missing COB returns None."""
        record = {
            'loop': {
                'iob': {'iob': 2.5}
            }
        }

        cob = parse_devicestatus_cob(record)
        assert cob is None

    def test_parse_iob_malformed_structure(self):
        """Test that malformed structure returns None."""
        record = {
            'loop': {
                'iob': 'not a dict'
            }
        }

        iob = parse_devicestatus_iob(record)
        assert iob is None

    def test_parse_no_loop_data(self):
        """Test that missing loop data returns None."""
        record = {
            'created_at': '2024-01-15T10:00:00Z'
        }

        iob = parse_devicestatus_iob(record)
        cob = parse_devicestatus_cob(record)
        assert iob is None
        assert cob is None
