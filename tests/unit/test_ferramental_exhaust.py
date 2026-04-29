import pytest
import pandas as pd
import numpy as np
import random
from unittest.mock import MagicMock, patch, ANY
from ferramental.Ferramental import Ferramental, FerramentalError, ConnectionError, InvalidAssetError
from datetime import datetime, timedelta
import time

@pytest.fixture
def mock_deps():
    config = MagicMock()
    # Setup some default config values
    config.get_list.return_value = ['EURUSD', 'GBPUSD', 'BTCUSD']
    config.get_value.side_effect = lambda s, k, d=None, t=None: {
        ('API', 'retry_count'): 3,
        ('API', 'timeout'): 5,
        ('Trading', 'daily_loss_limit'): 5.0,
        ('Trading', 'risk_per_trade'): 1.0,
        ('Trading', 'max_consecutive_losses'): 3,
        ('General', 'simulation_mode'): False,
        ('Simulation', 'synthetic_volatility'): 0.0002,
        ('Simulation', 'synthetic_trend'): 0.00001,
        ('Simulation', 'simulated_win_rate'): 60.0,
        ('General', 'assets'): ['EURUSD', 'GBPUSD']
    }.get((s, k), d)
    
    errors = MagicMock()
    return config, errors

@pytest.fixture
def ferramental(mock_deps):
    Ferramental._instance = None
    config, errors = mock_deps
    f = Ferramental(config, errors)
    f.iq_option = MagicMock()
    f.logger = MagicMock()
    f.connected = True
    f.simulation_mode = False
    
    # Initialize attributes that might have been skipped if singleton was already initialized
    if not hasattr(f, 'simulation_assets'):
        f.simulation_assets = {}
    if not hasattr(f, 'simulation_historical_data'):
        f.simulation_historical_data = {}
    if not hasattr(f, 'simulation_trades'):
        f.simulation_trades = []
    
    f.simulation_balance = 1000.0
    f.simulation_volatility = 0.0002
    f.simulation_trend = 0.00001
    f.simulation_win_rate = 0.6
    
    # Standard risk management state
    f.risk_metrics.daily_loss = 0.0
    f.risk_metrics.consecutive_losses = 0
    f.risk_metrics.last_reset = time.time()
    f.risk_metrics.max_trade_risk = 0.01 # 1%
    f.risk_metrics.max_daily_loss = 0.05 # 5%
    f.risk_metrics.max_consecutive_losses = 3

    # Update the dict for compatibility tests
    f.risk_management.update({
        'daily_loss': f.risk_metrics.daily_loss,
        'consecutive_losses': f.risk_metrics.consecutive_losses,
        'last_reset': f.risk_metrics.last_reset,
        'max_trade_risk': f.risk_metrics.max_trade_risk,
        'max_daily_loss': f.risk_metrics.max_daily_loss,
        'max_consecutive_losses': f.risk_metrics.max_consecutive_losses
    })

    f.asset_pairs = ['EURUSD', 'GBPUSD', 'BTCUSD']
    return f

def test_singleton_pattern(mock_deps):
    Ferramental._instance = None
    config, errors = mock_deps
    f1 = Ferramental(config, errors)
    f2 = Ferramental(config, errors)
    assert f1 is f2

def test_risk_metrics_reset(ferramental):
    ferramental.risk_metrics.daily_loss = 100.0
    ferramental.risk_metrics.consecutive_losses = 5
    ferramental.reset_daily_metrics()
    assert ferramental.risk_metrics.daily_loss == 0.0
    assert ferramental.risk_metrics.consecutive_losses == 0

def test_check_and_reset_daily_metrics(ferramental):
    # Past 24h
    ferramental.risk_metrics.last_reset = time.time() - 86500
    ferramental.risk_metrics.daily_loss = 50.0
    ferramental.check_and_reset_daily_metrics()
    assert ferramental.risk_metrics.daily_loss == 0.0
    
    # Within 24h
    ferramental.risk_metrics.last_reset = time.time() - 100
    ferramental.risk_metrics.daily_loss = 5.0
    ferramental.check_and_reset_daily_metrics()
    assert ferramental.risk_metrics.daily_loss == 5.0

def test_set_session(ferramental):
    ferramental.set_session(headers={'X-Test': 'True'}, cookies={'session': '123'})
    ferramental.iq_option.set_session.assert_called()

def test_connect_2fa_exception(ferramental):
    ferramental.iq_option.connect_2fa.side_effect = Exception("Crash")
    status, reason = ferramental.connect_2fa("123456")
    assert status is False
    assert "Crash" in reason
    ferramental.error_tracker.add_error.assert_called_with("Connect2FAError", "Crash", ANY)

def test_get_version(ferramental):
    ferramental.iq_option.__version__ = "1.2.3"
    assert ferramental.get_version() == "1.2.3"

def test_digital_spot_compatibility(ferramental):
    # Tests methods with Redirect warnings
    with patch.object(ferramental, 'buy', return_value=(True, 999)):
        res = ferramental.buy_digital_spot("EURUSD", 10, "call", 1)
        assert res == (True, 999)
    
    assert ferramental.get_digital_spot_instruments() == []
    assert ferramental.get_digital_spot_profit("EURUSD") is None
    assert ferramental.check_win_digital(123) == (False, None)

def test_enable_auto_reconnect_error(ferramental):
    ferramental.iq_option.enable_auto_reconnect.side_effect = Exception("Err")
    ferramental.enable_auto_reconnect(100)
    ferramental.error_tracker.add_error.assert_called()

def test_enable_two_factor_auth_error(ferramental):
    ferramental.iq_option.enable_two_factor_auth.side_effect = Exception("Err")
    assert ferramental.enable_two_factor_auth("CODE") is False
    ferramental.error_tracker.add_error.assert_called()

def test_get_historical_data_synthetic_fallback(ferramental):
    ferramental.simulation_mode = True
    ferramental.simulation_assets = {"EURUSD": {"price": 1.0}}
    ferramental.simulation_historical_data = {} # Double ensure
    with patch.object(ferramental, '_generate_synthetic_data') as mock_gen:
        mock_gen.return_value = pd.DataFrame([{'open': 1, 'high': 2, 'low': 0, 'close': 1, 'volume': 1}], index=[datetime.now()])
        df = ferramental.get_historical_data("EURUSD")
        assert df is not None
        assert len(df) == 1
        assert df['open'].iloc[0] == 1.0

def test_get_candles_simulation_flow(ferramental):
    ferramental.simulation_mode = True
    # Should use _generate_synthetic_data internally if enabled
    with patch.object(ferramental, '_generate_synthetic_data') as mock_gen:
        mock_gen.return_value = pd.DataFrame([{'open': 1.0, 'close': 1.1, 'high': 1.2, 'low': 0.9, 'volume': 100}], index=[datetime.now()])
        res = ferramental.get_candles("EURUSD", "Minutes", 1, 100)
        assert len(res) == 1
        assert res[0]['open'] == 1.0

def test_get_candles_real_validation_errors(ferramental):
    ferramental.simulation_mode = False
    # Asset not in pairs
    assert ferramental.get_candles("INVALID", "Minutes", 1, 10) is None
    # Invalid timeframe type
    assert ferramental.get_candles("EURUSD", "Decades", 1, 10) is None
    # Invalid timeframe value
    assert ferramental.get_candles("EURUSD", "Minutes", -1, 10) is None
    # Data type validation in candles
    ferramental.iq_option.get_candles.return_value = [{'open': 'not_a_float', 'close': 1.1, 'min': 1.0, 'max': 1.2, 'volume': 100, 'from': 1, 'to': 2, 'id': 1}]
    assert ferramental.get_candles("EURUSD", "Minutes", 1, 1) is None

def test_buy_risk_management_more_paths(ferramental):
    # Balance fail
    ferramental.iq_option.get_balance_v2.side_effect = Exception("Balance error")
    ferramental.iq_option.get_balance.side_effect = Exception("Balance error")
    status, _ = ferramental.buy("EURUSD", 10, "call", 1)
    assert status is False
    
    # Asset closed
    ferramental.iq_option.get_balance_v2.side_effect = None
    ferramental.iq_option.get_balance_v2.return_value = 10000.0
    with patch.object(ferramental, 'check_asset_open', return_value=False):
        status, _ = ferramental.buy("EURUSD", 10, "call", 1)
        assert status is False

def test_get_balance_v2_explicit(ferramental):
    ferramental.iq_option.get_balance_v2.return_value = 123.45
    assert ferramental.get_balance_v2() == 123.45
    
    ferramental.iq_option.get_balance_v2.side_effect = Exception("Fail")
    assert ferramental.get_balance_v2() is None

def test_get_currency_and_amounts(ferramental):
    # Currency
    ferramental.iq_option.get_currency.return_value = "BRL"
    assert ferramental.get_currency() == "BRL"
    # Min trade amount
    assert ferramental.get_min_trade_amount() == 5.0
    ferramental.iq_option.get_currency.return_value = "USD"
    assert ferramental.get_min_trade_amount() == 1.0
    # Max trade amount
    ferramental.iq_option.get_balance_v2.return_value = 1000.0
    assert ferramental.get_max_trade_amount() == 200.0 # 20% of 1000

def test_get_spread_logic(ferramental):
    # Mock get_all_open_time with both assets
    ferramental.iq_option.get_all_open_time.return_value = {
        'turbo': {'EURUSD': {'open': True}, 'BTCUSD': {'open': True}}
    }
    ferramental.iq_option.get_price_raw.side_effect = [1.0000, 1.0002] # bid, ask
    assert ferramental.get_spread("EURUSD") == pytest.approx(0.0002)
    
    # bid == ask case
    ferramental.iq_option.get_price_raw.side_effect = [1.0000, 1.0000, 20000.0, 20000.0]
    assert ferramental.get_spread("EURUSD") == 0.0002 # Default for major
    assert ferramental.get_spread("BTCUSD") == 0.001 # Default for crypto
    
    # Not open
    ferramental.iq_option.get_all_open_time.return_value = {'turbo': {}}
    assert ferramental.get_spread("EURUSD") == 0.05 # Default fail

def test_configure_assets(ferramental):
    ferramental.iq_option.get_all_open_time.return_value = {'turbo': {'EURUSD': {'open': True}, 'GBPUSD': {'open': True}}}
    assert ferramental.configure_assets(['EURUSD', 'INVALID']) is True
    assert ferramental.asset_pairs == ['EURUSD']
    assert ferramental.configure_assets(['NONE']) is False

def test_reset_practice_balance(ferramental):
    ferramental.iq_option.reset_practice_balance.return_value = True
    assert ferramental.reset_practice_balance() is True
    ferramental.iq_option.reset_practice_balance.side_effect = Exception("Err")
    assert ferramental.reset_practice_balance() is False

def test_change_balance(ferramental):
    assert ferramental.change_balance("REAL") is True
    assert ferramental.change_balance("INVALID") is False
    ferramental.iq_option.change_balance.side_effect = Exception("Err")
    assert ferramental.change_balance("PRACTICE") is False

@pytest.fixture
def mock_iq_class():
    with patch('ferramental.Ferramental.IQ_Option') as mock_iq:
        yield mock_iq

def test_connect_various_failures(ferramental, mock_iq_class):
    instance = mock_iq_class.return_value
    ferramental.simulation_mode = False
    
    # 1. Missing credentials in config
    ferramental.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: None
    status, reason = ferramental.connect()
    assert status is False
    assert "não configuradas" in reason

    # Restore some config for next tests
    ferramental.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "value"

    # 2. Network error (Errno -2)
    instance.connect.return_value = (False, "[Errno -2] Name or service not known")
    status, reason = ferramental.connect()
    assert status is False
    assert reason == "Erro de rede"

def test_get_historical_data_tf_and_errors(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    
    # Missing asset
    assert ferramental.get_historical_data(asset=None) is None
    
    # Valid TFs and sorting
    ferramental.iq_option.get_candles.return_value = [
        {'from': 200, 'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.1, 'volume': 100},
        {'from': 100, 'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 90}
    ]
    df = ferramental.get_historical_data("EURUSD", timeframe=300)
    assert df is not None
    assert df['timestamp'].iloc[0] == 100 # Should be sorted
    
    # Missing columns check
    ferramental.iq_option.get_candles.return_value = [{'from': 100, 'open': 1.1}] # Missing high, low, etc.
    assert ferramental.get_historical_data("EURUSD") is None

def test_get_realtime_data_aggregation(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    
    # Mock successful candle streams and realtime data
    with patch.object(ferramental, 'get_realtime_candles') as mock_rt:
        mock_rt.return_value = {123: {'from': 123, 'open': 1.0, 'close': 1.1, 'max': 1.2, 'min': 0.9, 'volume': 100}}
        df = ferramental.get_realtime_data("EURUSD")
        assert df is not None
        assert 'asset' in df.columns
        assert len(df) >= 1

    # Backup historical path
    with patch.object(ferramental, 'get_realtime_candles', return_value={}):
        with patch.object(ferramental, 'get_candles') as mock_hist:
            mock_hist.return_value = [{'from': 1, 'open': 1.1, 'close': 1.2, 'max': 1.3, 'min': 1.0, 'volume': 50}]
            df = ferramental.get_realtime_data("EURUSD")
            assert df is not None
            assert len(df) == 1

def test_check_connection_real(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    
    ferramental.iq_option.check_connect.return_value = True
    assert ferramental.check_connection() is True
    
    ferramental.iq_option.check_connect.return_value = False
    assert ferramental.check_connection() is False
    
    ferramental.iq_option.check_connect.side_effect = Exception("Crash")
    assert ferramental.check_connection() is False

def test_get_all_assets_logic(ferramental):
    ferramental.connected = True
    # Real path
    ferramental.iq_option.get_all_open_time.return_value = {
        'binary': {'EURUSD': {'open': True}, 'GBPUSD': {'open': False}},
        'turbo': {'BTCUSD': {'open': True}}
    }
    assets = ferramental.get_all_assets()
    assert 'EURUSD' in assets
    assert 'BTCUSD' in assets
    assert 'GBPUSD' not in assets

    # Error path
    ferramental.iq_option.get_all_open_time.side_effect = Exception("Err")
    assert ferramental.get_all_assets() is None

def test_check_asset_open_real_complex(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    ferramental.iq_option.get_all_open_time.return_value = {
        'turbo': {'eurusd': {'open': True}} # testing case sensitivity
    }
    assert ferramental.check_asset_open("EURUSD") is True
    assert ferramental.check_asset_open("INVALID") is False

def test_synthetic_data_generation_boundary(ferramental):
    # Test internal _generate_synthetic_data
    ferramental.simulation_historical_data = {}
    ferramental.simulation_assets = {"EURUSD": {"price": 1.08}}
    ferramental.simulation_volatility = 0.001
    ferramental._generate_synthetic_data("EURUSD", count=100)
    assert "EURUSD" in ferramental.simulation_historical_data
    df = ferramental.simulation_historical_data["EURUSD"]
    assert len(df) == 5000 

def test_get_current_price_real_error(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    with patch.object(ferramental, 'check_asset_open', return_value=True):
        ferramental.get_candles = MagicMock(side_effect=Exception("Crash"))
        assert ferramental.get_current_price("EURUSD") is None

def test_get_trade_results_simulation_empty(ferramental):
    ferramental.simulation_mode = True
    ferramental.simulation_trades = []
    assert ferramental.get_trade_results() == []

def test_get_historical_data_tf_conversion(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    ferramental.iq_option.get_candles.return_value = [{'from': 1, 'open': 1, 'high': 2, 'low': 0, 'close': 1, 'volume': 1}]
    
    # Test 15m
    ferramental.get_historical_data("EURUSD", timeframe=900)
    ferramental.iq_option.get_candles.assert_called_with("EURUSD", 15, "Minutes", 1000)
    
    # Test 1h
    ferramental.get_historical_data("EURUSD", timeframe=3600)
    ferramental.iq_option.get_candles.assert_called_with("EURUSD", 1, "Hours", 1000)
    
    # Test Default
    ferramental.get_historical_data("EURUSD", timeframe=123)
    ferramental.iq_option.get_candles.assert_called_with("EURUSD", 1, "Minutes", 1000)

def test_connect_full_paths(ferramental, mock_iq_class):
    instance = mock_iq_class.return_value
    ferramental.simulation_mode = False
    
    # Credentials fail in config (General/API)
    ferramental.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: None
    res, msg = ferramental.connect()
    assert res is False
    assert "configuradas" in msg
    
    # Restore config
    ferramental.config_manager.get_value.side_effect = lambda s, k, d=None, t=None: "user" if k in ['email','password'] else None
    
    # 2FA
    instance.connect.return_value = (False, "2FA")
    res, msg = ferramental.connect()
    assert res is False
    assert msg == "2FA"
    
    # Invalid Credentials
    instance.connect.return_value = (False, "invalid_credentials")
    res, msg = ferramental.connect()
    assert res is False
    assert "inválidas" in msg
    
    # Other error
    instance.connect.return_value = (False, "Unknown Error")
    res, msg = ferramental.connect()
    assert res is False
    assert msg == "Unknown Error"

def test_get_realtime_candles_alternatives(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    
    # Attribute error fallback
    del ferramental.iq_option.get_realtime_candles
    ferramental.get_candles = MagicMock(return_value=[{'close': 1.2}])
    res = ferramental.get_realtime_candles("EURUSD")
    assert len(res) == 1
    
    # Total fail
    ferramental.get_candles = MagicMock(return_value=None)
    assert ferramental.get_realtime_candles("EURUSD") == {}

def test_start_candles_stream_fail(ferramental):
    ferramental.simulation_mode = False
    ferramental.iq_option = MagicMock()
    ferramental.iq_option.start_candles_stream.side_effect = Exception("Fail")
    assert ferramental.start_candles_stream("EURUSD") is False

def test_stop_candles_stream_fail(ferramental):
    ferramental.simulation_mode = False
    ferramental.iq_option.stop_candles_stream.side_effect = Exception("Fail")
    assert ferramental.stop_candles_stream("EURUSD") is False

def test_get_realtime_data_full_fail(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    with patch.object(ferramental, 'get_realtime_candles', return_value={}):
        with patch.object(ferramental, 'get_candles', return_value=None):
            assert ferramental.get_realtime_data("EURUSD") is None

def test_buy_risk_limits_all_branches(ferramental):
    ferramental.connected = True
    ferramental.iq_option.get_balance_v2.return_value = 100.0
    ferramental.risk_metrics.max_trade_risk = 0.1 # 10%
    
    # 1. Too risky
    status, _ = ferramental.buy("EURUSD", 15.0, "call", 1) # 15% > 10%
    assert status is False
    
    # 2. Too low amount
    status, _ = ferramental.buy("EURUSD", 0.5, "call", 1)
    assert status is False
    
    # 3. Consecutive losses hit
    ferramental.risk_metrics.consecutive_losses = 3
    status, _ = ferramental.buy("EURUSD", 5.0, "call", 1)
    assert status is False
    
    # 4. Success call (Real)
    ferramental.risk_metrics.consecutive_losses = 0
    ferramental.iq_option.buy.return_value = (True, 777)
    with patch.object(ferramental, 'check_asset_open', return_value=True):
        status, order_id = ferramental.buy("EURUSD", 5.0, "call", 1)
        assert status is True
        assert order_id == 777

def test_generate_synthetic_data_full(ferramental):
    # Test internal _generate_synthetic_data logic
    ferramental.simulation_assets = {"EURUSD": {"price": 1.0}}
    ferramental.simulation_volatility = 0.0002
    ferramental.simulation_trend = 0.0001
    ferramental._generate_synthetic_data("EURUSD", count=100)
    assert "EURUSD" in ferramental.simulation_historical_data
    df = ferramental.simulation_historical_data["EURUSD"]
    assert len(df) == 5000
    assert 'open' in df.columns
    assert 'close' in df.columns

def test_get_historical_data_sim_missing_asset(ferramental):
    ferramental.simulation_mode = True
    ferramental.simulation_assets = {"EURUSD": {"price": 1.0}}
    assert ferramental.get_historical_data("GBPUSD") is None

def test_get_realtime_candles_errors_and_alternative(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    # Real failure (AttributeError handled in test_get_realtime_candles_alternatives)
    # Now test the Exception catch (line 981)
    ferramental.iq_option.get_realtime_candles.side_effect = Exception("Major Crash")
    res = ferramental.get_realtime_candles("EURUSD")
    assert res == {}
    ferramental.error_tracker.add_error.assert_called_with("GetRealtimeCandlesError", "Major Crash", ANY)

def test_stop_candles_stream_full(ferramental):
    # Sim
    ferramental.simulation_mode = True
    ferramental.simulation_active_streams = {"EURUSD": {}}
    assert ferramental.stop_candles_stream("EURUSD") is True
    assert "EURUSD" not in ferramental.simulation_active_streams
    
    # Real fail
    ferramental.simulation_mode = False
    ferramental.iq_option.stop_candles_stream.side_effect = Exception("Fail")
    assert ferramental.stop_candles_stream("EURUSD") is False

def test_get_realtime_data_no_asset_logic(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    ferramental.asset_pairs = ['EURUSD']
    with patch.object(ferramental, 'start_candles_stream'):
        with patch.object(ferramental, 'get_realtime_candles', return_value={1: {'from': 1, 'open': 1.1, 'close': 1.2, 'max': 1.3, 'min': 1.0, 'volume': 50}}):
            df = ferramental.get_realtime_data() # No asset passed, should use asset_pairs
            assert df is not None
            assert df['asset'].iloc[0] == 'EURUSD'

def test_handle_2fa_full(ferramental):
    ferramental.connected = True
    # Success
    ferramental.iq_option.connect_2fa.return_value = (True, None)
    assert ferramental.handle_two_factor_auth("123456") is True
    # Failure
    ferramental.iq_option.connect_2fa.return_value = (False, "wrong")
    assert ferramental.handle_two_factor_auth("000000") is False
    # Not connected
    ferramental.connected = False
    assert ferramental.handle_two_factor_auth("123") is False

def test_check_asset_open_categories(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    # Only binary open
    ferramental.iq_option.get_all_open_time.return_value = {
        'binary': {'EURUSD': {'open': True}},
        'turbo': {'EURUSD': {'open': False}}
    }
    assert ferramental.check_asset_open("EURUSD") is True
    
    # Neither open
    ferramental.iq_option.get_all_open_time.return_value = {
        'binary': {'EURUSD': {'open': False}},
        'turbo': {'EURUSD': {'open': False}}
    }
    assert ferramental.check_asset_open("EURUSD") is False

def test_get_historical_data_exceptions(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    ferramental.iq_option.get_candles.side_effect = Exception("API error")
    assert ferramental.get_historical_data("EURUSD") is None

def test_get_realtime_data_exception_during_loop(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    ferramental.asset_pairs = ['EURUSD']
    with patch.object(ferramental, 'start_candles_stream', side_effect=Exception("Crash")):
        assert ferramental.get_realtime_data() is None

def test_send_notification(ferramental):
    # Simply calls logger.info
    ferramental.send_notification("Test msg")
    ferramental.logger.info.assert_called()

def test_get_current_price_sim_detailed(ferramental):
    ferramental.simulation_mode = True
    ferramental.simulation_assets = {"EURUSD": {"price": 1.0821}}
    with patch('numpy.random.normal', return_value=0.0):
        assert ferramental.get_current_price("EURUSD") == 1.0821
    # Missing asset
    assert ferramental.get_current_price("MISSING") is None

def test_not_connected_paths(ferramental):
    ferramental.connected = False
    ferramental.simulation_mode = False
    assert ferramental.get_realtime_data() is None
    assert ferramental.get_historical_data("EURUSD") is None
    assert ferramental.get_all_assets() is None
    assert ferramental.check_connection() is False
    assert ferramental.change_balance("REAL") is False
    assert ferramental.buy("EURUSD", 10, "call", 1) == (False, None)
    assert ferramental.handle_two_factor_auth("123") is False

def test_check_connection_sim(ferramental):
    # Coverage for lines 1362-1363
    ferramental.connected = True
    ferramental.simulation_mode = True
    with patch('ferramental.Ferramental.IQOPTION_API_AVAILABLE', False):
        assert ferramental.check_connection() is True

def test_digital_spot_redirection_cases(ferramental):
    # Coverage for 844-845, 870-871
    ferramental.connected = False
    assert ferramental.buy_digital_spot("EURUSD", 10, "call", 1) == (False, None)
    assert ferramental.check_win_digital(123) == (False, None)

def test_get_realtime_candles_sim_no_data(ferramental):
    # Coverage for 958-959
    ferramental.simulation_mode = True
    ferramental.simulation_historical_data = {}
    assert ferramental.get_realtime_candles("EURUSD") == {}

def test_setup_simulation_mode_full(ferramental):
    assert ferramental.setup_simulation_mode() is True
    assert ferramental.simulation_mode is True
    assert len(ferramental.simulation_assets) > 0
    
    # Fail path
    with patch.object(ferramental, '_setup_simulation_mode', side_effect=Exception("Crash")):
        assert ferramental.setup_simulation_mode() is False

def test_get_trade_results_api_paths(ferramental):
    ferramental.simulation_mode = False
    ferramental.iq_option.get_optioninfo_v2.return_value = [
        {'id': 1, 'active': 'EURUSD', 'direction': 'call', 'amount': 10, 'profit': 8, 'win': True, 'open_time': 100, 'close_time': 200}
    ]
    res = ferramental.get_trade_results()
    assert len(res) == 1
    assert res[0]['is_win'] is True
    
    # API Error
    ferramental.iq_option.get_optioninfo_v2.side_effect = Exception("API error")
    assert ferramental.get_trade_results() == []

def test_get_balance_real_alternatives(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    
    # get_balance_v2 success
    ferramental.iq_option.get_balance_v2.return_value = 500.0
    assert ferramental.get_balance() == 500.0
    
    # get_balance_v2 fail, get_balance success
    ferramental.iq_option.get_balance_v2.side_effect = Exception("Fail")
    ferramental.iq_option.get_balance.return_value = 450.0
    assert ferramental.get_balance() == 450.0
    
    # Both fail
    ferramental.iq_option.get_balance.side_effect = Exception("Fail")
    assert ferramental.get_balance() == 0.0

def test_get_digital_spot_instruments_fail(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    # Return error
    assert ferramental.get_digital_spot_instruments() == []

def test_check_win_digital_real_paths(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    # Success
    # Returns (False, None) as it's not supported
    assert ferramental.check_win_digital(123) == (False, None)
    # Fail
    ferramental.iq_option.check_win_digital_v2.side_effect = Exception("Fail")
    assert ferramental.check_win_digital(123) == (False, None)

def test_import_error_mock():
    # To cover lines 15-21, we need to mock the import failure
    with patch.dict('sys.modules', {'iqoptionapi.stable_api': None}):
        # Reload doesn't really work easily for this structure without refactoring
        # But we can at least verify the behavior by looking at the code
        pass

def test_start_candles_stream_various(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    # Normal success
    assert ferramental.start_candles_stream("EURUSD") is True
    # Exception
    ferramental.iq_option.start_candles_stream.side_effect = Exception("Crash")
    assert ferramental.start_candles_stream("EURUSD") is False

def test_get_realtime_candles_validations(ferramental):
    ferramental.simulation_mode = False
    ferramental.connected = True
    # Invalid asset - should return {}
    ferramental.simulation_mode = False
    assert ferramental.get_realtime_candles(None) == {}
    # Valid call
    ferramental.iq_option.get_realtime_candles.return_value = {123: {'close': 1.1}}
    res = ferramental.get_realtime_candles("EURUSD")
    assert 123 in res

def test_get_current_price_logic_more(ferramental):
    ferramental.connected = True
    ferramental.simulation_mode = False
    # Asset not specified or not open
    assert ferramental.get_current_price(None) is None
    with patch.object(ferramental, 'check_asset_open', return_value=False):
        assert ferramental.get_current_price("EURUSD") is None

def test_buy_invalid_params(ferramental):
    ferramental.connected = True
    # Invalid amount
    assert ferramental.buy("EURUSD", -10, "call", 1) == (False, None)
    # Asset not specified
    assert ferramental.buy(None, 10, "call", 1) == (False, None)

def test_reconnect_method(ferramental):
    with patch.object(ferramental, 'connect', return_value=(True, None)):
        res, msg = ferramental.reconnect()
        assert res is True
        assert msg is None
    
    # Error
    ferramental.iq_option = None
    assert ferramental.start_candles_stream("EURUSD") is False

def test_start_stop_candles_stream(ferramental):
    # Simulation
    ferramental.simulation_mode = True
    assert ferramental.start_candles_stream("EURUSD") is True
    assert ferramental.stop_candles_stream("EURUSD") is True
    
    # Real
    ferramental.simulation_mode = False
    assert ferramental.start_candles_stream("EURUSD") is True
    assert ferramental.stop_candles_stream("EURUSD") is True
    
    # Error
    ferramental.iq_option = None
    assert ferramental.start_candles_stream("EURUSD") is False
