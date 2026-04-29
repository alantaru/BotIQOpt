import os
import pytest
import configparser
from unittest.mock import patch, mock_open
from utils.ConfigManager import ConfigManager

@pytest.fixture
def mock_config_content():
    return """
[General]
operation_mode = TEST
assets = eurusd, gbpusd
timeframe_type = Minutes
timeframe_value = 1

[Trading]
amount = 10.0
stop_loss = 0.5
take_profit = 0.5
max_daily_loss = 100.0
max_consecutive_losses = 5
min_confidence = 0.7

[AutoSwitchCriteria]
min_accuracy = 0.8
min_precision = 0.75
"""

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reseta o Singleton antes de cada teste."""
    ConfigManager._instance = None
    yield

def test_singleton():
    cm1 = ConfigManager()
    cm2 = ConfigManager()
    assert cm1 is cm2

def test_load_config_success(mock_config_content):
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_config_content)):
            cm = ConfigManager("dummy_config.ini")
            assert cm.get_value("General", "operation_mode") == "test" # ConfigManager converts to lower
            assert cm.get_value("General", "assets") == "eurusd, gbpusd"

def test_get_value_types(mock_config_content):
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_config_content)):
            cm = ConfigManager("dummy_config.ini")
            
            # Test float conversion
            assert cm.get_value("Trading", "amount", value_type=float) == 10.0
            
            # Test boolean conversion
            # Mocking a boolean value
            with patch.object(cm.config, 'get', return_value='true'):
                assert cm.get_value("General", "some_bool", value_type=bool) is True
                
            # Test int conversion
            assert cm.get_value("Trading", "max_consecutive_losses", value_type=int) == 5

def test_get_list(mock_config_content):
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_config_content)):
            cm = ConfigManager("dummy_config.ini")
            assets = cm.get_assets()
            assert isinstance(assets, list)
            assert "eurusd" in assets
            assert "gbpusd" in assets

def test_get_auto_switch_criteria(mock_config_content):
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_config_content)):
            cm = ConfigManager("dummy_config.ini")
            criteria = cm.get_auto_switch_criteria()
            assert criteria['min_accuracy'] == 0.8
            assert criteria['min_precision'] == 0.75
            # Check default for value not in mock
            assert criteria['min_win_rate'] == 0.60

def test_get_risk_params(mock_config_content):
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_config_content)):
            cm = ConfigManager("dummy_config.ini")
            risk = cm.get_risk_params()
            assert risk['amount'] == 10.0
            assert risk['stop_loss'] == 0.5
