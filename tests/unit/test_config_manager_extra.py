import os
import pytest
import configparser
from unittest.mock import patch, MagicMock
from utils.ConfigManager import ConfigManager

@pytest.fixture(autouse=True)
def reset_singleton():
    ConfigManager._instance = None
    yield

def test_load_config_missing_file():
    # Line 54-55: File not found
    cm = ConfigManager(config_file="non_existent.ini")
    assert cm.load_config() is False

def test_load_config_exception():
    # Line 60-62: Exception during read
    with patch('configparser.ConfigParser.read', side_effect=Exception("Read error")):
        cm = ConfigManager(config_file="dummy.ini")
        # Since __init__ calls load_config, we check the initialization
        # but let's call it explicitly to be sure
        assert cm.load_config() is False

def test_get_value_missing_section():
    # Line 82-83: Section not found
    cm = ConfigManager()
    with patch.object(cm.config, '__contains__', return_value=False):
        assert cm.get_value("MissingSection", "key", default="def") == "def"

def test_get_value_exception():
    # Line 104-106: Exception in get_value
    cm = ConfigManager()
    if 'Exist' not in cm.config:
        cm.config.add_section('Exist')
    
    # Mocking the section's get method to raise an exception
    with patch.object(cm.config['Exist'], 'get', side_effect=Exception("Value error")):
        assert cm.get_value("Exist", "key", default="def") == "def"



def test_get_list_empty():
    # Line 128: Empty value
    cm = ConfigManager()
    with patch.object(cm, 'get_value', return_value=""):
        assert cm.get_list("Section", "ListKey", default=["a"]) == ["a"]

def test_get_list_exception():
    # Line 138-140: Exception in get_list
    cm = ConfigManager()
    with patch.object(cm, 'get_value', side_effect=Exception("List error")):
        assert cm.get_list("Section", "ListKey", default=["err"]) == ["err"]

def test_singleton_reset():
    # Ensure Singleton behavior and reset
    cm1 = ConfigManager()
    ConfigManager._instance = None
    cm2 = ConfigManager()
    assert cm1 is not cm2

def test_get_config_parser():
    # Line 288
    cm = ConfigManager()
    assert isinstance(cm.get_config_parser(), configparser.ConfigParser)

def test_missing_and_special_keys():
    cm = ConfigManager()
    # Line 91: Value is None
    # Use a real section that exists
    if 'General' not in cm.config:
        cm.config.add_section('General')
    
    with patch.object(cm.config['General'], 'get', return_value=None):
        assert cm.get_value("General", "NonExistentKey") is None
        
    # Line 96: auto_switch_modes
    cm.config['General']['auto_switch_modes'] = 'true'
    # We need to make sure we don't return from the 'if value_type is bool' (line 98)
    # but hit line 96. Line 96 is hit IF key == 'auto_switch_modes'.
    assert cm.get_value('General', 'auto_switch_modes') is True
    
    # Line 135: item_type conversion in get_list
    cm.config['General']['int_list'] = '1, 2, 3'
    assert cm.get_list('General', 'int_list', item_type=int) == [1, 2, 3]

def test_all_getters():
    cm = ConfigManager()
    # Call all getters to cover lines like 185, 203-206, etc.
    cm.get_operation_mode() # Line 185
    cm.get_assets()
    cm.get_timeframe()
    cm.get_risk_params()
    cm.get_learning_params()
    cm.get_download_params()
    cm.get_logging_params()
    cm.get_api_params()
    
def test_load_config_exception_direct():
    # Line 60-62: Direct exception in load_config
    cm = ConfigManager()
    with patch.object(cm.config, 'read', side_effect=Exception("Forced error")):
        assert cm.load_config() is False

