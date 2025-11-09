"""Test for handling duplicate devicestatus timestamps correctly."""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import match_devicestatus_to_timestamps


def test_carb_entry_duplicate_timestamps():
    """Test the specific case from 2025-06-17 23:55 with duplicate timestamps.

    Scenario: 20g carbs entered at 23:55:27, Loop posts two calculations at 23:55:34
    - First calculation: COB = 69.75 (after carbs processed)
    - Second calculation: COB = 49.73 (different/earlier calculation)

    Expected: Last (most recent) calculation should be used.
    """
    # Two devicestatus records at exactly the same timestamp
    devicestatus = pd.DataFrame({
        'created_at': [
            '2025-06-17 23:55:34',
            '2025-06-17 23:55:34',  # Duplicate timestamp
            '2025-06-18 00:09:45'
        ],
        'loop': [
            {'iob': {'iob': 10.656316249014193}, 'cob': {'cob': 69.75150415858155}},  # First
            {'iob': {'iob': 10.656316249014193}, 'cob': {'cob': 49.73275618700677}},  # Second (LAST)
            {'iob': {'iob': 10.172188160223811}, 'cob': {'cob': 70.0}}
        ]
    })

    # Intervals we're matching
    time_aligned_data = pd.DataFrame({
        'DateTime': pd.to_datetime([
            '2025-06-17 23:55:00',  # 34 seconds before duplicate
            '2025-06-18 00:00:00',  # 4m 26s after duplicate
            '2025-06-18 00:05:00'   # 4m 45s before next
        ])
    })

    iob_values, cob_values = match_devicestatus_to_timestamps(
        time_aligned_data, devicestatus
    )

    # All three intervals should use the LAST (most recent) record for duplicate timestamps
    # 23:55:00 -> matches to 23:55:34 (34s away) -> uses LAST duplicate = 49.73
    assert cob_values[0] == 49.73275618700677

    # 00:00:00 -> matches to 23:55:34 (4m 26s away) -> uses LAST duplicate = 49.73
    assert cob_values[1] == 49.73275618700677

    # 00:05:00 -> matches to 00:09:45 (4m 45s away) -> 70.0
    assert cob_values[2] == 70.0


def test_deduplication_order():
    """Test that deduplication keeps the last occurrence of each timestamp."""
    devicestatus = pd.DataFrame({
        'created_at': [
            '2024-01-15 10:00:00',
            '2024-01-15 10:00:00',
            '2024-01-15 10:00:00',
            '2024-01-15 10:05:00'
        ],
        'loop': [
            {'iob': {'iob': 1.0}, 'cob': {'cob': 10.0}},  # Will be discarded
            {'iob': {'iob': 2.0}, 'cob': {'cob': 20.0}},  # Will be discarded
            {'iob': {'iob': 3.0}, 'cob': {'cob': 30.0}},  # Will be kept (last at 10:00)
            {'iob': {'iob': 4.0}, 'cob': {'cob': 40.0}}   # Different timestamp
        ]
    })

    time_aligned_data = pd.DataFrame({
        'DateTime': pd.to_datetime([
            '2024-01-15 10:00:00',
            '2024-01-15 10:05:00'
        ])
    })

    iob_values, cob_values = match_devicestatus_to_timestamps(
        time_aligned_data, devicestatus
    )

    # 10:00 should match to the LAST record at that timestamp
    assert iob_values[0] == 3.0
    assert cob_values[0] == 30.0

    # 10:05 should match to the only record at that timestamp
    assert iob_values[1] == 4.0
    assert cob_values[1] == 40.0
