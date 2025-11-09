"""Tests for profile timeline and export functions."""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import export_profiles_to_csv


class TestExportProfilesToCSV:
    """Test export_profiles_to_csv function."""

    def test_export_single_profile(self, sample_profile_data):
        """Test exporting a single profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                sample_profile_data,
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=True
            )

            assert result is True
            assert output_path.exists()

            # Read and verify CSV
            df = pd.read_csv(output_path)
            assert len(df) == 4  # 4 time segments in basal schedule
            assert 'profile_name' in df.columns
            assert 'start_date' in df.columns
            assert 'basal_rate' in df.columns
            assert df['profile_name'].iloc[0] == 'Default'

    def test_export_multiple_profiles(self, sample_profile_data_multiple):
        """Test exporting multiple profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                sample_profile_data_multiple,
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=True
            )

            assert result is True
            df = pd.read_csv(output_path)

            # Should have rows for both profiles
            profiles = df['profile_name'].unique()
            assert 'Winter' in profiles
            assert 'Spring' in profiles

    def test_export_filters_profiles_by_date_range(self):
        """Test that profiles outside date range are filtered out."""
        # Create profiles: one inside range, one outside
        profiles = [
            {
                'startDate': '2024-01-01T00:00:00Z',
                'defaultProfile': 'InRange',
                'store': {
                    'InRange': {
                        'dia': 5,
                        'timezone': 'UTC',
                        'units': 'mg/dL',
                        'basal': [{'time': '00:00', 'value': 0.5}],
                        'sens': [{'time': '00:00', 'value': 50}],
                        'carbratio': [{'time': '00:00', 'value': 10}],
                        'target_low': [{'time': '00:00', 'value': 90}],
                        'target_high': [{'time': '00:00', 'value': 120}]
                    }
                }
            },
            {
                'startDate': '2025-01-01T00:00:00Z',  # Outside range
                'defaultProfile': 'OutOfRange',
                'store': {
                    'OutOfRange': {
                        'dia': 5,
                        'timezone': 'UTC',
                        'units': 'mg/dL',
                        'basal': [{'time': '00:00', 'value': 0.6}],
                        'sens': [{'time': '00:00', 'value': 50}],
                        'carbratio': [{'time': '00:00', 'value': 10}],
                        'target_low': [{'time': '00:00', 'value': 90}],
                        'target_high': [{'time': '00:00', 'value': 120}]
                    }
                }
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                profiles,
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=False
            )

            assert result is True
            df = pd.read_csv(output_path)

            # Should only have 'InRange' profile
            assert 'InRange' in df['profile_name'].values
            assert 'OutOfRange' not in df['profile_name'].values

    def test_export_converts_targets_to_mmol(self, sample_profile_data):
        """Test that target ranges are converted to mmol/L."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                sample_profile_data,
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=True
            )

            assert result is True
            df = pd.read_csv(output_path)

            # Target low 90 mg/dL ≈ 5.0 mmol/L
            # Target high 120 mg/dL ≈ 6.7 mmol/L
            assert df['target_low'].iloc[0] == pytest.approx(5.0, abs=0.1)
            assert df['target_high'].iloc[0] == pytest.approx(6.7, abs=0.1)
            assert df['units'].iloc[0] == 'mmol/L'

    def test_export_no_conversion_keeps_mgdl(self, sample_profile_data):
        """Test that targets stay in mg/dL when conversion is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                sample_profile_data,
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=False
            )

            assert result is True
            df = pd.read_csv(output_path)

            # Should keep original values
            assert df['target_low'].iloc[0] == 90
            assert df['target_high'].iloc[0] == 120
            assert df['units'].iloc[0] == 'mg/dL'

    def test_export_empty_profiles_returns_false(self):
        """Test that empty profiles returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                [],
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=True
            )

            assert result is False
            assert not output_path.exists()

    def test_export_profile_without_startdate_filtered(self):
        """Test that profiles without startDate are filtered out."""
        bad_profile = [
            {
                # No startDate
                'defaultProfile': 'Default',
                'store': {
                    'Default': {
                        'basal': [{'time': '00:00', 'value': 0.5}]
                    }
                }
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                bad_profile,
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=True
            )

            # Should fail because no valid profiles
            assert result is False

    def test_export_includes_all_schedule_types(self, sample_profile_data):
        """Test that export includes all time-based schedules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                sample_profile_data,
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=True
            )

            assert result is True
            df = pd.read_csv(output_path)

            # Should have all columns
            expected_columns = [
                'profile_name', 'start_date', 'time', 'basal_rate',
                'isf', 'carb_ratio', 'target_low', 'target_high',
                'dia', 'timezone', 'units'
            ]
            for col in expected_columns:
                assert col in df.columns

    def test_export_profile_with_single_value_settings(self):
        """Test exporting profile with single-value (non-array) settings."""
        profile = [{
            'startDate': '2024-01-01T00:00:00Z',
            'defaultProfile': 'Simple',
            'store': {
                'Simple': {
                    'dia': 5,
                    'timezone': 'UTC',
                    'units': 'mg/dL',
                    'basal': 0.5,  # Single value
                    'sens': 50,    # Single value
                    'carbratio': 10,  # Single value
                    'target_low': 90,  # Single value
                    'target_high': 120  # Single value
                }
            }
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            from_dt = datetime(2024, 1, 1)
            to_dt = datetime(2024, 12, 31)

            result = export_profiles_to_csv(
                profile,
                output_path,
                from_dt,
                to_dt,
                convert_to_mmol=False
            )

            assert result is True
            df = pd.read_csv(output_path)

            # Should have one row with 00:00 time
            assert len(df) == 1
            assert df['time'].iloc[0] == '00:00'
            assert df['basal_rate'].iloc[0] == 0.5
            assert df['isf'].iloc[0] == 50
