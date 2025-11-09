"""Tests for basal calculation functions."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import (
    parse_basal_array_to_hourly,
    build_profile_timeline,
    build_temp_basal_timeline,
    get_scheduled_basal_rate,
    get_active_basal_rate,
    calculate_interval_based_basal
)


class TestParseBasalArrayToHourly:
    """Test parse_basal_array_to_hourly function."""

    def test_parse_simple_basal_schedule(self, sample_basal_array):
        """Test parsing a simple basal schedule."""
        hourly = parse_basal_array_to_hourly(sample_basal_array)

        assert len(hourly) == 24
        # Hours 0-5 should be 0.5
        assert hourly[0] == 0.5
        assert hourly[5] == 0.5
        # Hours 6-11 should be 0.6
        assert hourly[6] == 0.6
        assert hourly[11] == 0.6
        # Hours 12-17 should be 0.55
        assert hourly[12] == 0.55
        assert hourly[17] == 0.55
        # Hours 18-23 should be 0.5
        assert hourly[18] == 0.5
        assert hourly[23] == 0.5

    def test_parse_basal_single_rate(self):
        """Test parsing a basal schedule with single rate."""
        basal_array = [{'time': '00:00', 'value': 0.75}]
        hourly = parse_basal_array_to_hourly(basal_array)

        assert len(hourly) == 24
        # All hours should be 0.75
        for hour in range(24):
            assert hourly[hour] == 0.75

    def test_parse_basal_midnight_wraparound(self):
        """Test that basal rate wraps around from end of day to beginning."""
        basal_array = [
            {'time': '06:00', 'value': 0.6},
            {'time': '12:00', 'value': 0.5}
        ]
        hourly = parse_basal_array_to_hourly(basal_array)

        # Hours 0-5 should use last rate (wraps from previous day)
        assert hourly[0] == 0.5
        assert hourly[5] == 0.5
        # Hours 6-11 should be 0.6
        assert hourly[6] == 0.6
        # Hours 12-23 should be 0.5
        assert hourly[12] == 0.5
        assert hourly[23] == 0.5

    def test_parse_basal_empty_array(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="Basal array is empty"):
            parse_basal_array_to_hourly([])

    def test_parse_basal_non_midnight_start(self):
        """Test basal schedule that doesn't start at midnight."""
        basal_array = [
            {'time': '03:00', 'value': 0.4},
            {'time': '09:00', 'value': 0.6}
        ]
        hourly = parse_basal_array_to_hourly(basal_array)

        # Hours before first entry should use last entry
        assert hourly[0] == 0.6
        assert hourly[2] == 0.6
        # Hours 3-8 should be 0.4
        assert hourly[3] == 0.4
        assert hourly[8] == 0.4
        # Hours 9-23 should be 0.6
        assert hourly[9] == 0.6
        assert hourly[23] == 0.6


class TestBuildProfileTimeline:
    """Test build_profile_timeline function."""

    def test_build_timeline_single_profile(self, sample_profile_data):
        """Test building timeline with single profile."""
        timeline = build_profile_timeline(sample_profile_data)

        assert len(timeline) == 1
        assert timeline[0]['start_date'] == pd.to_datetime('2024-01-01T00:00:00Z')
        assert timeline[0]['end_date'] is None
        assert len(timeline[0]['basal_schedule']) == 24
        assert timeline[0]['basal_schedule'][0] == 0.5
        assert timeline[0]['basal_schedule'][6] == 0.6

    def test_build_timeline_multiple_profiles(self, sample_profile_data_multiple):
        """Test building timeline with multiple profiles."""
        timeline = build_profile_timeline(sample_profile_data_multiple)

        assert len(timeline) == 2

        # First profile
        assert timeline[0]['start_date'] == pd.to_datetime('2024-01-01T00:00:00Z')
        assert timeline[0]['end_date'] == pd.to_datetime('2024-02-01T00:00:00Z')
        assert timeline[0]['basal_schedule'][0] == 0.5

        # Second profile
        assert timeline[1]['start_date'] == pd.to_datetime('2024-02-01T00:00:00Z')
        assert timeline[1]['end_date'] is None
        assert timeline[1]['basal_schedule'][0] == 0.45

    def test_build_timeline_empty_profiles(self):
        """Test that empty profiles raises ValueError."""
        with pytest.raises(ValueError, match="No profile data provided"):
            build_profile_timeline([])

    def test_build_timeline_missing_default_profile(self):
        """Test handling of profile with no defaultProfile."""
        bad_profile = [{'startDate': '2024-01-01T00:00:00Z'}]

        with pytest.raises(ValueError, match="No valid profiles could be parsed"):
            build_profile_timeline(bad_profile)

    def test_build_timeline_profile_with_single_value_basal(self):
        """Test profile with single value basal instead of array."""
        profile = [{
            'startDate': '2024-01-01T00:00:00Z',
            'defaultProfile': 'Default',
            'store': {
                'Default': {
                    'dia': 5,
                    'timezone': 'UTC',
                    'units': 'mg/dL',
                    'basal': 0.55  # Single value, not array
                }
            }
        }]

        timeline = build_profile_timeline(profile)

        assert len(timeline) == 1
        assert timeline[0]['basal_schedule'][0] == 0.55
        assert timeline[0]['basal_schedule'][12] == 0.55


class TestBuildTempBasalTimeline:
    """Test build_temp_basal_timeline function."""

    def test_build_temp_basal_timeline(self, sample_treatments_with_temp_basals):
        """Test building temp basal timeline from treatments."""
        timeline = build_temp_basal_timeline(sample_treatments_with_temp_basals)

        assert len(timeline) == 2  # Two temp basals

        # First temp basal
        start1, end1, rate1 = timeline[0]
        assert start1 == pd.to_datetime('2024-01-15 10:00:00')
        assert end1 == pd.to_datetime('2024-01-15 10:30:00')
        assert rate1 == 0.8

        # Second temp basal
        start2, end2, rate2 = timeline[1]
        assert start2 == pd.to_datetime('2024-01-15 11:00:00')
        assert end2 == pd.to_datetime('2024-01-15 12:00:00')
        assert rate2 == 0.3

    def test_build_temp_basal_timeline_empty(self):
        """Test building timeline with no temp basals."""
        df = pd.DataFrame({
            'created_at': [],
            'eventType': [],
            'duration': [],
            'absolute': []
        })

        timeline = build_temp_basal_timeline(df)
        assert len(timeline) == 0

    def test_build_temp_basal_timeline_missing_duration(self):
        """Test that temp basals without duration are skipped."""
        df = pd.DataFrame([
            {
                'created_at': '2024-01-15T10:00:00Z',
                'eventType': 'Temp Basal',
                'duration': None,  # Missing duration
                'absolute': 0.8
            }
        ])

        timeline = build_temp_basal_timeline(df)
        assert len(timeline) == 0

    def test_build_temp_basal_timeline_missing_rate(self):
        """Test that temp basals without rate are skipped."""
        df = pd.DataFrame([
            {
                'created_at': '2024-01-15T10:00:00Z',
                'eventType': 'Temp Basal',
                'duration': 30,
                'absolute': None  # Missing rate
            }
        ])

        timeline = build_temp_basal_timeline(df)
        assert len(timeline) == 0


class TestGetScheduledBasalRate:
    """Test get_scheduled_basal_rate function."""

    def test_get_scheduled_basal_single_profile(self, sample_profile_data):
        """Test getting scheduled basal from single profile."""
        timeline = build_profile_timeline(sample_profile_data)

        # Test various hours
        ts = pd.to_datetime('2024-01-15T03:00:00')
        assert get_scheduled_basal_rate(ts, timeline) == 0.5

        ts = pd.to_datetime('2024-01-15T08:00:00')
        assert get_scheduled_basal_rate(ts, timeline) == 0.6

        ts = pd.to_datetime('2024-01-15T14:00:00')
        assert get_scheduled_basal_rate(ts, timeline) == 0.55

    def test_get_scheduled_basal_multiple_profiles(self, sample_profile_data_multiple):
        """Test getting scheduled basal across profile changes."""
        timeline = build_profile_timeline(sample_profile_data_multiple)

        # January - first profile
        ts = pd.to_datetime('2024-01-15T03:00:00')
        assert get_scheduled_basal_rate(ts, timeline) == 0.5

        # February - second profile
        ts = pd.to_datetime('2024-02-15T03:00:00')
        assert get_scheduled_basal_rate(ts, timeline) == 0.45

    def test_get_scheduled_basal_at_profile_boundary(self, sample_profile_data_multiple):
        """Test getting basal rate exactly at profile change boundary."""
        timeline = build_profile_timeline(sample_profile_data_multiple)

        # Exactly at boundary (should use new profile)
        ts = pd.to_datetime('2024-02-01T00:00:00')
        assert get_scheduled_basal_rate(ts, timeline) == 0.45


class TestGetActiveBasalRate:
    """Test get_active_basal_rate function."""

    def test_get_active_basal_no_temp(self, sample_profile_data, sample_treatments_with_temp_basals):
        """Test getting active basal when no temp basal is active."""
        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_timeline = build_temp_basal_timeline(sample_treatments_with_temp_basals)

        # Time before any temp basal
        ts = pd.to_datetime('2024-01-15T09:00:00')
        rate = get_active_basal_rate(ts, profile_timeline, temp_timeline)
        assert rate == 0.6  # Scheduled rate at hour 9 (6AM-12PM = 0.6)

    def test_get_active_basal_with_temp(self, sample_profile_data, sample_treatments_with_temp_basals):
        """Test getting active basal when temp basal is active."""
        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_timeline = build_temp_basal_timeline(sample_treatments_with_temp_basals)

        # During first temp basal (10:00-10:30, rate 0.8)
        ts = pd.to_datetime('2024-01-15T10:15:00')
        rate = get_active_basal_rate(ts, profile_timeline, temp_timeline)
        assert rate == 0.8

    def test_get_active_basal_after_temp_expires(self, sample_profile_data, sample_treatments_with_temp_basals):
        """Test getting active basal after temp basal expires."""
        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_timeline = build_temp_basal_timeline(sample_treatments_with_temp_basals)

        # After first temp expires, before second starts (10:30-11:00)
        ts = pd.to_datetime('2024-01-15T10:45:00')
        rate = get_active_basal_rate(ts, profile_timeline, temp_timeline)
        assert rate == 0.6  # Back to scheduled rate


class TestCalculateIntervalBasedBasal:
    """Test calculate_interval_based_basal function."""

    def test_calculate_intervals_no_temp_basal(self, sample_profile_data):
        """Test interval calculation with no temp basals."""
        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_timeline = []

        from_dt = datetime(2024, 1, 15, 10, 0, 0)
        to_dt = datetime(2024, 1, 15, 10, 30, 0)

        intervals = calculate_interval_based_basal(from_dt, to_dt, profile_timeline, temp_timeline)

        # Should have 6 intervals (10:00, 10:05, 10:10, 10:15, 10:20, 10:25)
        assert len(intervals) >= 6

        # Check first interval
        first_interval = pd.to_datetime('2024-01-15T10:00:00')
        assert first_interval in intervals
        # Scheduled rate at 10AM is 0.6 U/hr, so 5 minutes = 0.6 * 5/60 = 0.05U
        assert intervals[first_interval]['scheduled_basal'] == pytest.approx(0.05, abs=0.001)
        assert intervals[first_interval]['positive_temp'] == 0
        assert intervals[first_interval]['negative_temp'] == 0
        assert intervals[first_interval]['total_basal'] == pytest.approx(0.05, abs=0.001)

    def test_calculate_intervals_with_temp_basal(self, sample_profile_data, sample_treatments_with_temp_basals):
        """Test interval calculation with temp basals."""
        profile_timeline = build_profile_timeline(sample_profile_data)
        temp_timeline = build_temp_basal_timeline(sample_treatments_with_temp_basals)

        from_dt = datetime(2024, 1, 15, 10, 0, 0)
        to_dt = datetime(2024, 1, 15, 10, 30, 0)

        intervals = calculate_interval_based_basal(from_dt, to_dt, profile_timeline, temp_timeline)

        # Check interval during temp basal
        temp_interval = pd.to_datetime('2024-01-15T10:05:00')
        assert temp_interval in intervals

        # Scheduled: 0.6 * 5/60 = 0.05U
        # Temp rate: 0.8 * 5/60 = 0.0667U
        # Positive temp adjustment: (0.8 - 0.6) * 5/60 = 0.0167U
        assert intervals[temp_interval]['scheduled_basal'] == pytest.approx(0.05, abs=0.001)
        assert intervals[temp_interval]['positive_temp'] == pytest.approx(0.0167, abs=0.001)
        assert intervals[temp_interval]['negative_temp'] == 0
        assert intervals[temp_interval]['total_basal'] == pytest.approx(0.0667, abs=0.001)

    def test_calculate_intervals_with_negative_temp(self, sample_profile_data):
        """Test interval calculation with negative temp basal adjustment."""
        profile_timeline = build_profile_timeline(sample_profile_data)

        # Create temp basal lower than scheduled
        temp_df = pd.DataFrame([{
            'created_at': '2024-01-15T10:00:00Z',
            'eventType': 'Temp Basal',
            'duration': 30,
            'absolute': 0.3  # Lower than scheduled 0.6
        }])
        temp_timeline = build_temp_basal_timeline(temp_df)

        from_dt = datetime(2024, 1, 15, 10, 0, 0)
        to_dt = datetime(2024, 1, 15, 10, 15, 0)

        intervals = calculate_interval_based_basal(from_dt, to_dt, profile_timeline, temp_timeline)

        temp_interval = pd.to_datetime('2024-01-15T10:00:00')
        # Negative adjustment: (0.3 - 0.6) * 5/60 = -0.025U
        assert intervals[temp_interval]['scheduled_basal'] == pytest.approx(0.05, abs=0.001)
        assert intervals[temp_interval]['positive_temp'] == 0
        assert intervals[temp_interval]['negative_temp'] == pytest.approx(-0.025, abs=0.001)
        assert intervals[temp_interval]['total_basal'] == pytest.approx(0.025, abs=0.001)

    def test_calculate_intervals_across_profile_boundary(self, sample_profile_data_multiple):
        """Test interval calculation across profile changes."""
        profile_timeline = build_profile_timeline(sample_profile_data_multiple)
        temp_timeline = []

        # Calculate across profile boundary (Jan 31 to Feb 1)
        from_dt = datetime(2024, 1, 31, 23, 50, 0)
        to_dt = datetime(2024, 2, 1, 0, 10, 0)

        intervals = calculate_interval_based_basal(from_dt, to_dt, profile_timeline, temp_timeline)

        # Interval before boundary (23:50 Jan 31)
        before_interval = pd.to_datetime('2024-01-31T23:50:00')
        # Old profile (Winter): at hour 23, rate is 0.6 U/hr (from 12:00 entry)
        assert intervals[before_interval]['scheduled_basal'] == pytest.approx(0.6 * 5/60, abs=0.001)

        # Interval after boundary (00:00 Feb 1)
        after_interval = pd.to_datetime('2024-02-01T00:00:00')
        # New profile (Spring): 0.45 U/hr at hour 0
        assert intervals[after_interval]['scheduled_basal'] == pytest.approx(0.45 * 5/60, abs=0.001)
