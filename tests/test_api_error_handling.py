"""Tests for API error handling."""

import pytest
import responses
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import fetch_entries, fetch_treatments, fetch_devicestatus, fetch_profiles


class TestFetchEntriesErrors:
    """Test error handling in fetch_entries."""

    @responses.activate
    def test_fetch_entries_network_error(self):
        """Test handling of network errors."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json={"error": "network error"},
            status=500
        )

        with pytest.raises(SystemExit):
            fetch_entries(
                "https://test.nightscout.com",
                "test-token",
                100000,
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z",
                use_local_timezone=False,
                convert_to_mmol=False
            )

    @responses.activate
    def test_fetch_entries_empty_response(self):
        """Test handling of empty API response returns empty DataFrame."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json=[],
            status=200
        )

        df = fetch_entries(
            "https://test.nightscout.com",
            "test-token",
            100000,
            "2024-01-01T00:00:00Z",
            "2024-01-31T23:59:59Z",
            use_local_timezone=False,
            convert_to_mmol=False
        )

        # Empty response returns empty DataFrame with warning
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    @responses.activate
    def test_fetch_entries_unauthorized(self):
        """Test handling of unauthorized access (401)."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json={"error": "Unauthorized"},
            status=401
        )

        with pytest.raises(SystemExit):
            fetch_entries(
                "https://test.nightscout.com",
                "test-token",
                100000,
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z",
                use_local_timezone=False,
                convert_to_mmol=False
            )

    @responses.activate
    def test_fetch_entries_success(self):
        """Test successful API response."""
        sample_data = [
            {
                "date": 1704067200000,
                "dateString": "2024-01-01T00:00:00Z",
                "sgv": 120,
                "direction": "Flat",
                "type": "sgv"
            }
        ]

        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/entries",
            json=sample_data,
            status=200
        )

        df = fetch_entries(
            "https://test.nightscout.com",
            "test-token",
            100000,
            "2024-01-01T00:00:00Z",
            "2024-01-31T23:59:59Z",
            use_local_timezone=False,
            convert_to_mmol=False
        )

        assert len(df) == 1
        assert df['sgv'].iloc[0] == 120


class TestFetchTreatmentsErrors:
    """Test error handling in fetch_treatments."""

    @responses.activate
    def test_fetch_treatments_network_error(self):
        """Test handling of network errors."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/treatments",
            json={"error": "network error"},
            status=500
        )

        with pytest.raises(SystemExit):
            fetch_treatments(
                "https://test.nightscout.com",
                "test-token",
                100000,
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z",
                use_local_timezone=False
            )

    @responses.activate
    def test_fetch_treatments_empty_response(self):
        """Test handling of empty API response returns empty DataFrame."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/treatments",
            json=[],
            status=200
        )

        df = fetch_treatments(
            "https://test.nightscout.com",
            "test-token",
            100000,
            "2024-01-01T00:00:00Z",
            "2024-01-31T23:59:59Z",
            use_local_timezone=False
        )

        # Empty response returns empty DataFrame with warning
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    @responses.activate
    def test_fetch_treatments_success(self):
        """Test successful API response."""
        sample_data = [
            {
                "created_at": "2024-01-01T12:00:00Z",
                "eventType": "Meal Bolus",
                "insulin": 5.0,
                "carbs": 50.0
            }
        ]

        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/treatments",
            json=sample_data,
            status=200
        )

        df = fetch_treatments(
            "https://test.nightscout.com",
            "test-token",
            100000,
            "2024-01-01T00:00:00Z",
            "2024-01-31T23:59:59Z",
            use_local_timezone=False
        )

        assert len(df) == 1
        assert df['insulin'].iloc[0] == 5.0


class TestFetchDevicestatusErrors:
    """Test error handling in fetch_devicestatus."""

    @responses.activate
    def test_fetch_devicestatus_network_error(self):
        """Test handling of network errors."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/devicestatus",
            json={"error": "network error"},
            status=500
        )

        with pytest.raises(SystemExit):
            fetch_devicestatus(
                "https://test.nightscout.com",
                "test-token",
                100000,
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z",
                use_local_timezone=False
            )

    @responses.activate
    def test_fetch_devicestatus_empty_response_ok(self):
        """Test that empty devicestatus doesn't cause error (it's optional)."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/devicestatus",
            json=[],
            status=200
        )

        # Empty devicestatus should return empty DataFrame, not raise error
        df = fetch_devicestatus(
            "https://test.nightscout.com",
            "test-token",
            100000,
            "2024-01-01T00:00:00Z",
            "2024-01-31T23:59:59Z",
            use_local_timezone=False
        )

        assert len(df) == 0

    @responses.activate
    def test_fetch_devicestatus_success(self):
        """Test successful API response."""
        sample_data = [
            {
                "created_at": "2024-01-01T12:00:00Z",
                "loop": {
                    "iob": {"iob": 2.5},
                    "cob": {"cob": 30.0}
                }
            }
        ]

        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/devicestatus",
            json=sample_data,
            status=200
        )

        df = fetch_devicestatus(
            "https://test.nightscout.com",
            "test-token",
            100000,
            "2024-01-01T00:00:00Z",
            "2024-01-31T23:59:59Z",
            use_local_timezone=False
        )

        assert len(df) == 1


class TestFetchProfilesErrors:
    """Test error handling in fetch_profiles."""

    @responses.activate
    def test_fetch_profiles_network_error(self):
        """Test handling of network errors."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/profile",
            json={"error": "network error"},
            status=500
        )

        with pytest.raises(SystemExit):
            fetch_profiles(
                "https://test.nightscout.com",
                "test-token",
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z"
            )

    @responses.activate
    def test_fetch_profiles_empty_response(self):
        """Test handling of empty profiles (should error - profiles required)."""
        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/profile",
            json=[],
            status=200
        )

        with pytest.raises(SystemExit):
            fetch_profiles(
                "https://test.nightscout.com",
                "test-token",
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z"
            )

    @responses.activate
    def test_fetch_profiles_success(self):
        """Test successful API response."""
        sample_data = [
            {
                "startDate": "2024-01-01T00:00:00Z",
                "defaultProfile": "Default",
                "store": {
                    "Default": {
                        "dia": 5,
                        "timezone": "UTC",
                        "units": "mg/dL",
                        "basal": [{"time": "00:00", "value": 0.5}],
                        "sens": [{"time": "00:00", "value": 50}],
                        "carbratio": [{"time": "00:00", "value": 10}],
                        "target_low": [{"time": "00:00", "value": 90}],
                        "target_high": [{"time": "00:00", "value": 120}]
                    }
                }
            }
        ]

        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/profile",
            json=sample_data,
            status=200
        )

        profiles = fetch_profiles(
            "https://test.nightscout.com",
            "test-token",
            "2024-01-01T00:00:00Z",
            "2024-01-31T23:59:59Z"
        )

        assert len(profiles) == 1
        assert profiles[0]['defaultProfile'] == 'Default'

    @responses.activate
    def test_fetch_profiles_timeout(self):
        """Test handling of request timeout."""
        import requests

        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/profile",
            body=requests.exceptions.Timeout()
        )

        with pytest.raises(SystemExit):
            fetch_profiles(
                "https://test.nightscout.com",
                "test-token",
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z"
            )

    @responses.activate
    def test_fetch_profiles_connection_error(self):
        """Test handling of connection error."""
        import requests

        responses.add(
            responses.GET,
            "https://test.nightscout.com/api/v1/profile",
            body=requests.exceptions.ConnectionError()
        )

        with pytest.raises(SystemExit):
            fetch_profiles(
                "https://test.nightscout.com",
                "test-token",
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z"
            )
