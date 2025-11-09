"""Tests for environment loading and token extraction."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_nightscout_data import load_environment


class TestLoadEnvironment:
    """Test load_environment function with token extraction."""

    @patch('export_nightscout_data.load_dotenv')
    def test_extract_token_from_url_simple(self, mock_load_dotenv, monkeypatch):
        """Test extracting token from URL with only token parameter."""
        monkeypatch.setenv("NIGHTSCOUT_URL", "https://my.nightscout.com?token=abc123")
        monkeypatch.delenv("NIGHTSCOUT_TOKEN", raising=False)

        base_url, token = load_environment()

        assert base_url == "https://my.nightscout.com"
        assert token == "abc123"

    @patch('export_nightscout_data.load_dotenv')
    def test_extract_token_from_url_with_other_params(self, mock_load_dotenv, monkeypatch):
        """Test extracting token from URL with multiple query parameters."""
        monkeypatch.setenv("NIGHTSCOUT_URL", "https://my.nightscout.com?foo=bar&token=xyz789&baz=qux")
        monkeypatch.delenv("NIGHTSCOUT_TOKEN", raising=False)

        base_url, token = load_environment()

        # Token should be extracted and removed
        assert token == "xyz789"
        # URL should still have other params
        assert "foo=bar" in base_url
        assert "baz=qux" in base_url
        assert "token" not in base_url

    @patch('export_nightscout_data.load_dotenv')
    def test_env_var_takes_precedence_over_url_token(self, mock_load_dotenv, monkeypatch):
        """Test that NIGHTSCOUT_TOKEN env var takes priority over URL token."""
        monkeypatch.setenv("NIGHTSCOUT_URL", "https://my.nightscout.com?token=url_token")
        monkeypatch.setenv("NIGHTSCOUT_TOKEN", "env_token")

        base_url, token = load_environment()

        # Env var token should be used
        assert token == "env_token"
        # URL should still be cleaned
        assert "token" not in base_url
        assert base_url == "https://my.nightscout.com"

    @patch('export_nightscout_data.load_dotenv')
    def test_no_token_in_url(self, mock_load_dotenv, monkeypatch):
        """Test that URL without token works normally."""
        monkeypatch.setenv("NIGHTSCOUT_URL", "https://my.nightscout.com")
        monkeypatch.setenv("NIGHTSCOUT_TOKEN", "my_token")

        base_url, token = load_environment()

        assert base_url == "https://my.nightscout.com"
        assert token == "my_token"

    @patch('export_nightscout_data.load_dotenv')
    def test_no_token_at_all(self, mock_load_dotenv, monkeypatch):
        """Test that no token (neither env var nor URL) returns None for token."""
        monkeypatch.setenv("NIGHTSCOUT_URL", "https://my.nightscout.com")
        monkeypatch.delenv("NIGHTSCOUT_TOKEN", raising=False)

        base_url, token = load_environment()

        assert base_url == "https://my.nightscout.com"
        assert token is None

    @patch('export_nightscout_data.load_dotenv')
    def test_url_with_path_and_token(self, mock_load_dotenv, monkeypatch):
        """Test extracting token from URL with path."""
        monkeypatch.setenv("NIGHTSCOUT_URL", "https://my.nightscout.com/api?token=path_token")
        monkeypatch.delenv("NIGHTSCOUT_TOKEN", raising=False)

        base_url, token = load_environment()

        assert base_url == "https://my.nightscout.com/api"
        assert token == "path_token"

    @patch('export_nightscout_data.load_dotenv')
    def test_url_with_fragment_and_token(self, mock_load_dotenv, monkeypatch):
        """Test extracting token from URL with fragment."""
        monkeypatch.setenv("NIGHTSCOUT_URL", "https://my.nightscout.com?token=frag_token#section")
        monkeypatch.delenv("NIGHTSCOUT_TOKEN", raising=False)

        base_url, token = load_environment()

        assert "token" not in base_url
        assert token == "frag_token"
        assert "#section" in base_url

    @patch('export_nightscout_data.load_dotenv')
    def test_missing_url_exits(self, mock_load_dotenv, monkeypatch, capsys):
        """Test that missing NIGHTSCOUT_URL causes exit."""
        monkeypatch.delenv("NIGHTSCOUT_URL", raising=False)
        monkeypatch.delenv("NIGHTSCOUT_TOKEN", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            load_environment()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Missing required environment variables" in captured.err
