"""Tests for timezone conversion functions."""

import pytest
from datetime import datetime, timezone
from freezegun import freeze_time
import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import convert_to_utc, convert_from_utc


class TestTimezoneConversions:
    """Test timezone conversion functions."""

    def test_convert_to_utc_produces_utc_timezone(self):
        """Test that convert_to_utc always produces UTC timezone."""
        # January 15, 2024 10:00 AM local time
        local_dt = datetime(2024, 1, 15, 10, 0, 0)
        utc_dt = convert_to_utc(local_dt)

        assert utc_dt.tzinfo == timezone.utc
        assert utc_dt.year == 2024
        # Can't verify exact hour/day without knowing system timezone

    def test_convert_from_utc_produces_naive_datetime(self):
        """Test that convert_from_utc produces naive datetime (local time)."""
        # January 15, 2024 18:00 UTC
        utc_dt = datetime(2024, 1, 15, 18, 0, 0)
        local_dt = convert_from_utc(utc_dt)

        # Should produce naive datetime (no tzinfo)
        assert local_dt.tzinfo is None
        assert local_dt.year == 2024
        # Can't verify exact hour/day without knowing system timezone

    def test_convert_round_trip_standard_time(self):
        """Test round trip conversion during standard time."""
        original = datetime(2024, 1, 15, 10, 30, 45)

        utc = convert_to_utc(original)
        back_to_local = convert_from_utc(utc)

        # Should match original time
        assert back_to_local.year == original.year
        assert back_to_local.month == original.month
        assert back_to_local.day == original.day
        assert back_to_local.hour == original.hour
        assert back_to_local.minute == original.minute
        assert back_to_local.second == original.second

    def test_convert_round_trip_daylight_time(self):
        """Test round trip conversion during daylight time."""
        original = datetime(2024, 7, 15, 10, 30, 45)

        utc = convert_to_utc(original)
        back_to_local = convert_from_utc(utc)

        # Should match original time
        assert back_to_local.year == original.year
        assert back_to_local.month == original.month
        assert back_to_local.day == original.day
        assert back_to_local.hour == original.hour
        assert back_to_local.minute == original.minute
        assert back_to_local.second == original.second


    def test_convert_to_utc_with_timezone_aware_input(self):
        """Test that timezone-aware input is handled correctly."""
        # Create a timezone-aware datetime (shouldn't happen in normal usage,
        # but test defensive handling)
        aware_dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Function should handle this gracefully
        # Since input is already UTC, the timestamp conversion will keep it the same
        result = convert_to_utc(aware_dt)
        assert result.tzinfo == timezone.utc
