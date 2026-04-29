import pytest
import time
from unittest.mock import MagicMock, patch
from tests.integration.mock_api import MockIQOptionAPI

# Mocking the entire module before Ferramental imports it
with patch('iqoptionapi.stable_api.IQ_Option', new=MockIQOptionAPI):
    from ferramental.Ferramental import Ferramental
    from utils.ConfigManager import ConfigManager
    from utils.ErrorTracker import ErrorTracker

@pytest.fixture
def ferramental_setup():
    ConfigManager._instance = None
    Ferramental._instance = None
    config = MagicMock(spec=ConfigManager)
    error_tracker = MagicMock(spec=ErrorTracker)
    
    # Setup default config mock returns
    def get_val(s, k, d=None, t=None):
        if k == 'assets': return ['eurusd', 'gbpusd']
        if k == 'retry_count': return 1
        if k == 'timeout': return 1
        return d
    config.get_value.side_effect = get_val
    config.get_list.side_effect = get_val
    
    f = Ferramental(config, error_tracker)
    return f

def test_ferramental_connection_success(ferramental_setup):
    f = ferramental_setup
    # Mock iq_option instance creation
    with patch('ferramental.Ferramental.IQ_Option', new=MockIQOptionAPI):
        # Configure credentials in mock config first
        f.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "test@test.com" if k in ['email', 'password'] else d
        success, reason = f.connect()
        assert success is True
        assert f.connected is True
        assert f.iq_option.email == "test@test.com"

def test_ferramental_connection_failure(ferramental_setup):
    f = ferramental_setup
    with patch('ferramental.Ferramental.IQ_Option', new=MockIQOptionAPI):
        f.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "error@test.com" if k in ['email', 'password'] else d
        success, reason = f.connect()
        assert success is False
        assert f.connected is False

def test_get_balance_integration(ferramental_setup):
    f = ferramental_setup
    with patch('ferramental.Ferramental.IQ_Option', new=MockIQOptionAPI):
        f.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "test@test.com" if k in ['email', 'password'] else d
        f.connect()
        balance = f.get_balance()
        assert balance == 10000.0

def test_check_asset_open_integration(ferramental_setup):
    f = ferramental_setup
    with patch('ferramental.Ferramental.IQ_Option', new=MockIQOptionAPI):
        f.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "test@test.com" if k in ['email', 'password'] else d
        f.connect()
        # According to mock_api: eurusd is open
        assert f.check_asset_open("eurusd") is True

def test_buy_integration_success(ferramental_setup):
    f = ferramental_setup
    with patch('ferramental.Ferramental.IQ_Option', new=MockIQOptionAPI):
        f.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "test@test.com" if k in ['email', 'password'] else d
        f.connect()
        
        with patch.object(f.iq_option, 'buy', return_value=(True, 123)):
             # buy(self, asset: str, amount: float, action: str, expiration: int)
             status, order_id = f.buy('eurusd', 10.0, 'call', 1)
             assert status is True
             assert order_id == 123

def test_get_trade_results_integration(ferramental_setup):
    f = ferramental_setup
    with patch('ferramental.Ferramental.IQ_Option', new=MockIQOptionAPI):
        f.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "test@test.com" if k in ['email', 'password'] else d
        f.connect()
        
        # Execute a trade first in the mock
        f.iq_option.buy(10, 'eurusd', 'call', 1)
        
        results = f.get_trade_results()
        assert len(results) >= 1
        assert results[0]['asset'] == 'EURUSD'
        assert 'profit' in results[0]

def test_get_historical_data_integration(ferramental_setup):
    f = ferramental_setup
    with patch('ferramental.Ferramental.IQ_Option', new=MockIQOptionAPI):
        f.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "test@test.com" if k in ['email', 'password'] else d
        f.connect()
        
        df = f.get_historical_data("EURUSD", timeframe=60, count=10)
        assert df is not None
        assert len(df) == 10
        assert 'open' in df.columns

