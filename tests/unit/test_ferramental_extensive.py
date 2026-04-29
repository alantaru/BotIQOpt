import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from ferramental.Ferramental import Ferramental, FerramentalError, ConnectionError, InvalidAssetError

@pytest.fixture
def mock_deps():
    config = MagicMock()
    # Setup some default config values
    config.get_list.return_value = ['EURUSD', 'GBPUSD']
    # Fix lambda to support calls with 2 or more args
    config.get_value.side_effect = lambda s, k, d=None, t=None: {
        ('API', 'retry_count'): 3,
        ('API', 'timeout'): 5,
        ('Trading', 'daily_loss_limit'): 5.0,
        ('Trading', 'risk_per_trade'): 1.0,
        ('Trading', 'max_consecutive_losses'): 3,
        ('General', 'simulation_mode'): False,
        ('Credentials', 'email'): 'test@example.com',
        ('Credentials', 'password'): 'password'
    }.get((s, k), d)
    
    errors = MagicMock()
    return config, errors

@pytest.fixture
def mock_iq_class():
    with patch('ferramental.Ferramental.IQ_Option') as mock:
        yield mock

@pytest.fixture
def ferramental(mock_deps, mock_iq_class):
    Ferramental._instance = None
    config, errors = mock_deps
    f = Ferramental(config, errors)
    # Initialize iq_option instance mock
    f.iq_option = mock_iq_class.return_value
    f.connected = True
    return f

def test_initialization(mock_deps):
    Ferramental._instance = None
    config, errors = mock_deps
    f = Ferramental(config, errors)
    assert f.initialized
    assert f.asset_pairs == ['EURUSD', 'GBPUSD']
    assert f.max_retries == 3

def test_connect_success(ferramental, mock_iq_class):
    instance = mock_iq_class.return_value
    instance.connect.return_value = (True, None)
    status, reason = ferramental.connect()
    assert status is True
    assert reason is None
    assert ferramental.connected is True

def test_connect_failure(ferramental, mock_iq_class):
    instance = mock_iq_class.return_value
    # Test invalid credentials
    instance.connect.return_value = (False, "invalid_credentials")
    status, reason = ferramental.connect()
    assert status is False
    assert reason == "Credenciais inválidas"
    
    # Test 2FA
    instance.connect.return_value = (False, "2FA")
    status, reason = ferramental.connect()
    assert status is False
    assert reason == "2FA"

def test_connect_2fa(ferramental):
    ferramental.iq_option.connect_2fa.return_value = (True, None)
    status, reason = ferramental.connect_2fa("123456")
    assert status is True
    assert reason is None
    
    ferramental.iq_option.connect_2fa.return_value = (False, "Invalid code")
    status, reason = ferramental.connect_2fa("000000")
    assert status is False
    assert reason == "Invalid code"

def test_get_balance_v2_and_fallback(ferramental):
    # Test simulation
    ferramental._simulation_mode = True
    ferramental.simulation_balance = 500.0
    assert ferramental.get_balance() == 500.0
    
    # Test real v2 success
    ferramental._simulation_mode = False
    ferramental.iq_option.get_balance_v2.return_value = 1000.0
    assert ferramental.get_balance() == 1000.0
    
    # Test real v2 failure, fallback success
    ferramental.iq_option.get_balance_v2.side_effect = Exception("error")
    ferramental.iq_option.get_balance.return_value = 900.0
    assert ferramental.get_balance() == 900.0
    
    # Test total failure
    ferramental.iq_option.get_balance.side_effect = Exception("total error")
    assert ferramental.get_balance() == 0.0

def test_buy_risk_limits(ferramental):
    ferramental.connected = True
    ferramental.iq_option.get_balance_v2.return_value = 1000.0
    
    # 1. Max trade risk (Line 463)
    status, order_id = ferramental.buy("EURUSD", 20.0, "call", 1)
    assert status is False
    assert order_id is None
    
    # 2. Daily loss limit
    ferramental.risk_metrics.daily_loss = 0.06 # 6% > 5%
    status, _ = ferramental.buy("EURUSD", 5.0, "call", 1)
    assert status is False
    
    # 3. Consecutive losses
    ferramental.risk_metrics.daily_loss = 0.0
    ferramental.risk_metrics.consecutive_losses = 3
    status, _ = ferramental.buy("EURUSD", 5.0, "call", 1)
    assert status is False

def test_buy_asset_and_action_validation(ferramental):
    ferramental.connected = True
    ferramental.iq_option.get_balance_v2.return_value = 1000.0
    ferramental.risk_metrics.consecutive_losses = 0
    
    # Invalid asset
    status, _ = ferramental.buy("INVALID", 5.0, "call", 1)
    assert status is False
    
    # Invalid action
    status, _ = ferramental.buy("EURUSD", 5.0, "jump", 1)
    assert status is False
    
    # Amount too low
    status, _ = ferramental.buy("EURUSD", 0.5, "call", 1)
    assert status is False

def test_buy_execution_real_and_sim(ferramental):
    ferramental.connected = True
    ferramental.iq_option.get_balance_v2.return_value = 1000.0
    ferramental.risk_metrics.consecutive_losses = 0
    # Mocking check_asset_open for speed
    ferramental.check_asset_open = MagicMock(return_value=True)
    
    # Real success
    ferramental.iq_option.buy.return_value = (True, 12345)
    status, order_id = ferramental.buy("EURUSD", 10.0, "call", 1)
    assert status is True
    assert order_id == 12345
    
    # Real failure
    ferramental.iq_option.buy.return_value = (False, None)
    status, _ = ferramental.buy("EURUSD", 10.0, "put", 1)
    assert status is False
    
    # Asset closed
    ferramental.check_asset_open.return_value = False
    status, _ = ferramental.buy("EURUSD", 10.0, "call", 1)
    assert status is False
    ferramental.check_asset_open.return_value = True

    # Exception during buy
    ferramental.iq_option.buy.side_effect = Exception("buy error")
    status, _ = ferramental.buy("EURUSD", 10.0, "call", 1)
    assert status is False
    ferramental.iq_option.buy.side_effect = None

    # Simulation
    ferramental.simulation_mode = True
    status, order_id = ferramental.buy("GBPUSD", 10.0, "call", 5)
    assert status is True
    assert order_id is not None
    assert len(ferramental.simulation_trades) == 1

def test_get_historical_data_sim_and_real(ferramental):
    ferramental.simulation_mode = True
    ferramental.simulation_assets = {'EURUSD': {}}
    ferramental.simulation_historical_data = {'EURUSD': pd.DataFrame([{'open': 1.0}])}
    
    # Sim success
    df = ferramental.get_historical_data("EURUSD")
    assert not df.empty
    
    # Sim fail
    assert ferramental.get_historical_data("GBPUSD") is None
    
    # Real success (mocking API)
    ferramental.simulation_mode = False
    ferramental.connected = True
    # different timeframes coverage
    for tf in [60, 300, 900, 3600, 7200]:
        ferramental.iq_option.get_candles.return_value = [{'from': 1000, 'to': 1060, 'id': 1, 'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.15, 'min': 1.0, 'max': 1.2, 'volume': 100}]
        df = ferramental.get_historical_data("EURUSD", timeframe=tf)
        assert df is not None
    
def test_check_asset_open(ferramental):
    ferramental.simulation_mode = True
    assert ferramental.check_asset_open("EURUSD") is True
    
    ferramental.simulation_mode = False
    ferramental.iq_option.get_all_open_time.return_value = {
        'binary': {'EURUSD': {'open': True}, 'GBPUSD': {'open': False}}
    }
    assert ferramental.check_asset_open("EURUSD") is True
    assert ferramental.check_asset_open("GBPUSD") is False

def test_get_candles_validation_fails(ferramental):
    # Missing keys
    ferramental.iq_option.get_candles.return_value = [{'open': 1.1}]
    assert ferramental.get_candles("EURUSD", "Minutes", 1, 10) is None
    
    # Empty result
    ferramental.iq_option.get_candles.return_value = []
    assert ferramental.get_candles("EURUSD", "Minutes", 1, 10) is None
    
    # Exception
    ferramental.iq_option.get_candles.side_effect = Exception("error")
    assert ferramental.get_candles("EURUSD", "Minutes", 1, 10) is None

def test_enable_methods(ferramental):
    # enable_auto_reconnect
    ferramental.enable_auto_reconnect(60)
    ferramental.iq_option.enable_auto_reconnect.assert_called_with(60)
    
    # enable_two_factor_auth
    ferramental.iq_option.enable_two_factor_auth.return_value = True
    assert ferramental.enable_two_factor_auth("CODE") is True
    ferramental.iq_option.enable_two_factor_auth.assert_called_with("CODE")

def test_get_realtime_data(ferramental):
    # Mocking internal methods
    ferramental.start_candles_stream = MagicMock()
    ferramental.get_realtime_candles = MagicMock(return_value={0: {'from': 1000, 'open': 1.1, 'close': 1.2}})
    
    df = ferramental.get_realtime_data("EURUSD")
    assert df is not None
    assert len(df) >= 1
    
    # Case with no data found
    ferramental.get_realtime_candles.return_value = {}
    ferramental.get_candles = MagicMock(return_value=None)
    assert ferramental.get_realtime_data("EURUSD") is None

def test_get_current_price(ferramental):
    # Real path
    ferramental.simulation_mode = False
    ferramental.check_asset_open = MagicMock(return_value=True)
    # Mock the internal call to self.get_candles
    ferramental.get_candles = MagicMock(return_value=[{'close': 1.123}])
    
    # We also need to mock iq_option.get_all_realtime_candles if it's used elsewhere, 
    # but in current refactored version it might not be.
    # Actually, get_current_price used to use get_all_realtime_candles, 
    # but I refactored it to use get_candles.
    
    assert ferramental.get_current_price("EURUSD") == 1.123

    
    # Sim path
    ferramental.simulation_mode = True
    # The code expects simulation_assets with a 'price' key
    ferramental.simulation_assets = {'EURUSD': {'price': 1.125}}
    
    # We need to mock numpy.random.normal to be 0 for exact match
    with patch('numpy.random.normal', return_value=0.0):
        assert ferramental.get_current_price("EURUSD") == 1.125


def test_get_trade_results(ferramental):
    # Real path
    ferramental.simulation_mode = False
    ferramental.iq_option.get_optioninfo.return_value = {'msg': {'result': 'win', 'profit_amount': 8.5}}
    # Note: get_trade_results might not take args or handle it differently
    res = ferramental.get_trade_results()
    assert res is not None
    
    # Sim path
    ferramental.simulation_mode = True
    # Mocking random to be deterministic
    with patch('random.random', return_value=0.1): # < win_rate
        # We need to ensure simulation_trades is not empty if the method uses it
        ferramental.simulation_trades = [12345]
        res = ferramental.get_trade_results()
        assert res is not None


def test_reconnect(ferramental):
    ferramental.connect = MagicMock(return_value=(True, None))
    status, reason = ferramental.reconnect()
    assert status is True
    ferramental.connect.assert_called_once()
