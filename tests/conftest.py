"""Pytest fixtures for nightscout-exporter tests."""

import pytest
import pandas as pd
from datetime import datetime, timezone


@pytest.fixture
def sample_profile_data():
    """Sample profile data with basal schedule."""
    return [
        {
            'startDate': '2024-01-01T00:00:00Z',
            'defaultProfile': 'Default',
            'store': {
                'Default': {
                    'dia': 5,
                    'timezone': 'America/Los_Angeles',
                    'units': 'mg/dL',
                    'basal': [
                        {'time': '00:00', 'value': 0.5},
                        {'time': '06:00', 'value': 0.6},
                        {'time': '12:00', 'value': 0.55},
                        {'time': '18:00', 'value': 0.5}
                    ],
                    'sens': [
                        {'time': '00:00', 'value': 50}
                    ],
                    'carbratio': [
                        {'time': '00:00', 'value': 10}
                    ],
                    'target_low': [
                        {'time': '00:00', 'value': 90}
                    ],
                    'target_high': [
                        {'time': '00:00', 'value': 120}
                    ]
                }
            }
        }
    ]


@pytest.fixture
def sample_profile_data_multiple():
    """Sample profile data with multiple profiles over time."""
    return [
        {
            'startDate': '2024-01-01T00:00:00Z',
            'defaultProfile': 'Winter',
            'store': {
                'Winter': {
                    'dia': 5,
                    'timezone': 'America/Los_Angeles',
                    'units': 'mg/dL',
                    'basal': [
                        {'time': '00:00', 'value': 0.5},
                        {'time': '12:00', 'value': 0.6}
                    ],
                    'sens': [{'time': '00:00', 'value': 50}],
                    'carbratio': [{'time': '00:00', 'value': 10}],
                    'target_low': [{'time': '00:00', 'value': 90}],
                    'target_high': [{'time': '00:00', 'value': 120}]
                }
            }
        },
        {
            'startDate': '2024-02-01T00:00:00Z',
            'defaultProfile': 'Spring',
            'store': {
                'Spring': {
                    'dia': 5,
                    'timezone': 'America/Los_Angeles',
                    'units': 'mg/dL',
                    'basal': [
                        {'time': '00:00', 'value': 0.45},
                        {'time': '12:00', 'value': 0.55}
                    ],
                    'sens': [{'time': '00:00', 'value': 50}],
                    'carbratio': [{'time': '00:00', 'value': 10}],
                    'target_low': [{'time': '00:00', 'value': 90}],
                    'target_high': [{'time': '00:00', 'value': 120}]
                }
            }
        }
    ]


@pytest.fixture
def sample_treatments_with_temp_basals():
    """Sample treatments DataFrame with temp basals."""
    data = [
        {
            'created_at': '2024-01-15 10:00:00',
            'eventType': 'Temp Basal',
            'duration': 30,
            'absolute': 0.8,
            'insulin': None,
            'amount': None,
            'carbs': None
        },
        {
            'created_at': '2024-01-15 11:00:00',
            'eventType': 'Temp Basal',
            'duration': 60,
            'absolute': 0.3,
            'insulin': None,
            'amount': None,
            'carbs': None
        },
        {
            'created_at': '2024-01-15 12:00:00',
            'eventType': 'Meal Bolus',
            'duration': None,
            'absolute': None,
            'insulin': 5.0,
            'amount': 5.0,
            'carbs': 50.0
        }
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_devicestatus_data():
    """Sample devicestatus DataFrame with Loop IOB/COB data."""
    data = [
        {
            'created_at': '2024-01-15 10:00:00',
            'loop': {
                'iob': {'iob': 2.5},
                'cob': {'cob': 30.0}
            }
        },
        {
            'created_at': '2024-01-15 10:05:00',
            'loop': {
                'iob': {'iob': 2.3},
                'cob': {'cob': 28.0}
            }
        },
        {
            'created_at': '2024-01-15 10:10:00',
            'loop': {
                'iob': {'iob': 2.1},
                'cob': {'cob': 25.0}
            }
        }
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_entries_data():
    """Sample CGM entries DataFrame."""
    data = [
        {
            'date': 1705315200000,  # 2024-01-15T10:00:00Z in milliseconds
            'dateString': '2024-01-15T10:00:00Z',
            'sgv': 120,
            'direction': 'Flat',
            'type': 'sgv'
        },
        {
            'date': 1705315500000,  # 2024-01-15T10:05:00Z
            'dateString': '2024-01-15T10:05:00Z',
            'sgv': 125,
            'direction': 'FortyFiveUp',
            'type': 'sgv'
        },
        {
            'date': 1705315800000,  # 2024-01-15T10:10:00Z
            'dateString': '2024-01-15T10:10:00Z',
            'sgv': 130,
            'direction': 'FortyFiveUp',
            'type': 'sgv'
        }
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_basal_array():
    """Sample basal schedule array."""
    return [
        {'time': '00:00', 'value': 0.5},
        {'time': '06:00', 'value': 0.6},
        {'time': '12:00', 'value': 0.55},
        {'time': '18:00', 'value': 0.5}
    ]


@pytest.fixture
def sample_devicestatus_record_with_iob_cob():
    """Sample devicestatus record with Loop IOB/COB structure."""
    return {
        'created_at': '2024-01-15T10:00:00Z',
        'loop': {
            'iob': {'iob': 2.5, 'timestamp': '2024-01-15T10:00:00Z'},
            'cob': {'cob': 30.0, 'timestamp': '2024-01-15T10:00:00Z'},
            'predicted': {
                'values': [120, 122, 125, 128]
            }
        }
    }


@pytest.fixture
def sample_devicestatus_record_missing_iob_cob():
    """Sample devicestatus record without IOB/COB data."""
    return {
        'created_at': '2024-01-15T10:00:00Z',
        'loop': {
            'predicted': {
                'values': [120, 122, 125, 128]
            }
        }
    }
