"""Tests for data parsing functions."""

import pytest
import responses
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import parse_date, fetch_entries, fetch_treatments


class TestParseDate:
    """Test parse_date function."""

    def test_parse_valid_date(self):
        """Test parsing a valid date string."""
        dt = parse_date("2024-01-15")

        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0

    def test_parse_leap_year_date(self):
        """Test parsing a leap year date."""
        dt = parse_date("2024-02-29")

        assert dt.year == 2024
        assert dt.month == 2
        assert dt.day == 29

    def test_parse_end_of_year(self):
        """Test parsing end of year date."""
        dt = parse_date("2024-12-31")

        assert dt.year == 2024
        assert dt.month == 12
        assert dt.day == 31

    def test_parse_beginning_of_year(self):
        """Test parsing beginning of year date."""
        dt = parse_date("2024-01-01")

        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1

    def test_parse_invalid_format_no_dashes(self):
        """Test that invalid format without dashes raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("20240115")

    def test_parse_invalid_format_wrong_separator(self):
        """Test that invalid format with wrong separator raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("2024/01/15")

    def test_parse_invalid_format_backwards(self):
        """Test that backwards date format raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("15-01-2024")

    def test_parse_invalid_month(self):
        """Test that invalid month raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("2024-13-01")

    def test_parse_invalid_day(self):
        """Test that invalid day raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("2024-01-32")

    def test_parse_invalid_leap_year(self):
        """Test that invalid leap year date raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("2023-02-29")  # 2023 is not a leap year

    def test_parse_empty_string(self):
        """Test that empty string raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("")

    def test_parse_invalid_year(self):
        """Test that invalid year format raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("24-01-15")

    def test_parse_too_many_parts(self):
        """Test that too many date parts raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("2024-01-15-12")

    def test_parse_non_numeric(self):
        """Test that non-numeric values raise SystemExit."""
        with pytest.raises(SystemExit):
            parse_date("2024-Jan-15")


class TestGlucoseConversion:
    """Test glucose unit conversions in fetch_entries()."""

    @responses.activate
    def test_mgdl_to_mmol_conversion_enabled(self):
        """Test mg/dL to mmol/L conversion when convert_to_mmol=True."""
        # Mock API response with mg/dL values
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json=[
                {"dateString": "2024-01-15T10:00:00Z", "sgv": 180, "type": "sgv"},
                {"dateString": "2024-01-15T10:05:00Z", "sgv": 90, "type": "sgv"},
            ],
            status=200
        )

        df = fetch_entries(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False,
            convert_to_mmol=True
        )

        # Verify conversion: 180 mg/dL = 10.0 mmol/L, 90 mg/dL = 5.0 mmol/L
        assert len(df) == 2
        assert 'sgv' in df.columns
        assert df.iloc[0]['sgv'] == pytest.approx(10.0, abs=0.1)
        assert df.iloc[1]['sgv'] == pytest.approx(5.0, abs=0.1)

    @responses.activate
    def test_mgdl_to_mmol_conversion_disabled(self):
        """Test that glucose values remain in mg/dL when convert_to_mmol=False."""
        # Mock API response with mg/dL values
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json=[
                {"dateString": "2024-01-15T10:00:00Z", "sgv": 180, "type": "sgv"},
                {"dateString": "2024-01-15T10:05:00Z", "sgv": 90, "type": "sgv"},
            ],
            status=200
        )

        df = fetch_entries(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False,
            convert_to_mmol=False
        )

        # Verify no conversion - values remain in mg/dL
        assert len(df) == 2
        assert 'sgv' in df.columns
        assert df.iloc[0]['sgv'] == 180
        assert df.iloc[1]['sgv'] == 90

    @responses.activate
    def test_glucose_conversion_rounding(self):
        """Test that conversion rounds to 1 decimal place."""
        # Mock API response with value that needs rounding
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json=[
                {"dateString": "2024-01-15T10:00:00Z", "sgv": 123, "type": "sgv"},
            ],
            status=200
        )

        df = fetch_entries(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False,
            convert_to_mmol=True
        )

        # 123 mg/dL = 6.82696... mmol/L, should round to 6.8
        sgv_value = df.iloc[0]['sgv']
        assert sgv_value == pytest.approx(6.8, abs=0.01)

        # Verify it's actually rounded to 1 decimal place (not just close)
        # Convert to string to check decimal places
        sgv_str = str(sgv_value)
        if '.' in sgv_str:
            decimal_places = len(sgv_str.split('.')[1])
            assert decimal_places <= 1

    @responses.activate
    def test_common_glucose_values(self):
        """Test conversion of common glucose values."""
        test_cases = [
            (70, 3.9),   # Low
            (100, 5.5),  # Normal
            (180, 10.0), # High
            (250, 13.9), # Very high
        ]

        for mgdl_value, expected_mmol in test_cases:
            responses.add(
                responses.GET,
                "https://test.nightscout.com/api/v1/entries",
                json=[{"dateString": "2024-01-15T10:00:00Z", "sgv": mgdl_value, "type": "sgv"}],
                status=200
            )

        # Fetch with conversion enabled
        for mgdl_value, expected_mmol in test_cases:
            responses.reset()
            responses.add(
                responses.GET,
                "https://test.nightscout.com/api/v1/entries",
                json=[{"dateString": "2024-01-15T10:00:00Z", "sgv": mgdl_value, "type": "sgv"}],
                status=200
            )

            df = fetch_entries(
                base_url="https://test.nightscout.com",
                token="test_token",
                max_count=100,
                api_from_date="2024-01-15",
                api_to_date=None,
                use_local_timezone=False,
                convert_to_mmol=True
            )

            assert df.iloc[0]['sgv'] == pytest.approx(expected_mmol, abs=0.1)

    @responses.activate
    def test_missing_sgv_column(self):
        """Test that missing SGV column doesn't cause error."""
        # Mock API response without sgv field
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json=[
                {"dateString": "2024-01-15T10:00:00Z", "type": "cal"},  # Calibration, no SGV
            ],
            status=200
        )

        df = fetch_entries(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False,
            convert_to_mmol=True
        )

        # Should not error - conversion only applies if 'sgv' column exists
        assert len(df) == 1
        assert 'sgv' not in df.columns or pd.isna(df.iloc[0].get('sgv'))


class TestTimestampConversion:
    """Test timestamp conversions in fetch functions."""

    @responses.activate
    def test_entries_timestamp_conversion_utc(self):
        """Test that entries timestamps remain UTC when use_local_timezone=False."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json=[
                {"dateString": "2024-01-15T10:30:45Z", "sgv": 120, "sysTime": "2024-01-15T10:30:50Z"},
            ],
            status=200
        )

        df = fetch_entries(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False,
            convert_to_mmol=False
        )

        # Timestamps should be formatted as strings in UTC
        assert df.iloc[0]['dateString'] == "2024-01-15 10:30:45"
        assert df.iloc[0]['sysTime'] == "2024-01-15 10:30:50"

    @responses.activate
    def test_treatments_timestamp_conversion_utc(self):
        """Test that treatment timestamps remain UTC when use_local_timezone=False."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/treatments",
            json=[
                {
                    "created_at": "2024-01-15T14:30:00Z",
                    "timestamp": "2024-01-15T14:30:00Z",
                    "eventType": "Meal Bolus",
                    "insulin": 5.0
                },
            ],
            status=200
        )

        df = fetch_treatments(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False
        )

        # Timestamps should be formatted as strings in UTC
        assert df.iloc[0]['created_at'] == "2024-01-15 14:30:00"
        assert df.iloc[0]['timestamp'] == "2024-01-15 14:30:00"

    @responses.activate
    def test_columns_sorted_alphabetically(self):
        """Test that columns are sorted alphabetically."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json=[
                {"sgv": 120, "dateString": "2024-01-15T10:00:00Z", "direction": "Flat", "type": "sgv"},
            ],
            status=200
        )

        df = fetch_entries(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False,
            convert_to_mmol=False
        )

        # Columns should be in alphabetical order
        columns = list(df.columns)
        assert columns == sorted(columns)


class TestTreatmentNormalization:
    """Test treatment data normalization."""

    @responses.activate
    def test_insulin_column_added_when_missing(self):
        """Test that insulin column is added if missing from treatment."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/treatments",
            json=[
                {
                    "created_at": "2024-01-15T10:00:00Z",
                    "eventType": "Carb Correction",
                    "carbs": 45
                    # No insulin field
                },
            ],
            status=200
        )

        df = fetch_treatments(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False
        )

        # insulin column should exist and be None/NaN
        assert 'insulin' in df.columns
        assert pd.isna(df.iloc[0]['insulin'])

    @responses.activate
    def test_amount_column_added_when_missing(self):
        """Test that amount column is added if missing from treatment."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/treatments",
            json=[
                {
                    "created_at": "2024-01-15T10:00:00Z",
                    "eventType": "Temp Basal",
                    "duration": 30,
                    "absolute": 0.8
                    # No amount field
                },
            ],
            status=200
        )

        df = fetch_treatments(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False
        )

        # amount column should exist and be None/NaN
        assert 'amount' in df.columns
        assert pd.isna(df.iloc[0]['amount'])

    @responses.activate
    def test_both_columns_preserved_when_present(self):
        """Test that insulin and amount columns are preserved when present."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/treatments",
            json=[
                {
                    "created_at": "2024-01-15T10:00:00Z",
                    "eventType": "Meal Bolus",
                    "insulin": 5.0,
                    "amount": 5.0,
                    "carbs": 50
                },
            ],
            status=200
        )

        df = fetch_treatments(
            base_url="https://test.nightscout.com",
            token="test_token",
            max_count=100,
            api_from_date="2024-01-15",
            api_to_date=None,
            use_local_timezone=False
        )

        # Both columns should exist with correct values
        assert 'insulin' in df.columns
        assert 'amount' in df.columns
        assert df.iloc[0]['insulin'] == 5.0
        assert df.iloc[0]['amount'] == 5.0


class TestProfileDataParsing:
    """Test profile data parsing."""

    def test_parse_profile_basal_array(self, sample_basal_array):
        """Test that basal array structure is correct."""
        assert len(sample_basal_array) == 4
        assert all('time' in entry for entry in sample_basal_array)
        assert all('value' in entry for entry in sample_basal_array)

    def test_parse_profile_times_format(self, sample_basal_array):
        """Test that time strings are in HH:MM format."""
        for entry in sample_basal_array:
            time_str = entry['time']
            parts = time_str.split(':')
            assert len(parts) == 2
            hour, minute = parts
            assert 0 <= int(hour) < 24
            assert 0 <= int(minute) < 60

    def test_parse_profile_rates_positive(self, sample_basal_array):
        """Test that basal rates are positive numbers."""
        for entry in sample_basal_array:
            assert entry['value'] > 0
            assert isinstance(entry['value'], (int, float))
