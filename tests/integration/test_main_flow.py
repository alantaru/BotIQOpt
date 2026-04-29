import pytest
from unittest.mock import MagicMock, patch
import sys
import os
import pandas as pd

# Patch sys.argv before importing main
with patch('sys.argv', ['main.py', '--mode', 'test']):
    import main

@pytest.fixture
def mock_components():
    with patch('main.ConfigManager') as mock_cm_class, \
         patch('ferramental.Ferramental.Ferramental') as mock_ferramental_class, \
         patch('inteligencia.Inteligencia.Inteligencia') as mock_inteligencia_class, \
         patch('main.display_banner'), \
         patch('main.check_dependencies', return_value=True), \
         patch('main.create_directories'), \
         patch('main.time.sleep') as mock_sleep, \
         patch('main.parse_args') as mock_args_func:
        
        # Setup mocks
        mock_args = MagicMock()
        mock_args_func.return_value = mock_args
        mock_args.config = 'config.ini'
        mock_args.mode = 'test'
        mock_args.assets = 'EURUSD'
        mock_args.duration = None
        mock_args.max_cycles = 1 # Default to 1 to avoid infinite loop
        mock_args.debug = False
        
        mock_cm = mock_cm_class.return_value
        # Default side effect for get_value
        def get_val(section, key, default=None, type=None):
            if key == 'email': return 'test@example.com'
            if key == 'password': return 'pass'
            if key == 'assets': return 'EURUSD'
            if key == 'mode': return 'test'
            if key == 'enable_real_mode': return True
            if type == float: return float(default) if default is not None else 0.0
            if type == int: return int(default) if default is not None else 0
            if type == bool: return bool(default) if default is not None else False
            return default if default is not None else 'test'
            
        mock_cm.get_value.side_effect = get_val
        mock_cm.get_list.return_value = ['EURUSD']
        
        mock_ferramental = mock_ferramental_class.return_value
        mock_ferramental.check_connection.return_value = True
        mock_ferramental.connect.return_value = (True, None)
        mock_ferramental.get_realtime_data.return_value = pd.DataFrame([{'close': 1.1}])
        mock_ferramental.buy.return_value = (True, 12345)
        mock_ferramental.get_trade_results.return_value = []
        mock_ferramental.get_balance.return_value = 1000.0
        mock_ferramental.risk_management = {'consecutive_losses': 0, 'daily_loss': 0}
        
        mock_inteligencia = mock_inteligencia_class.return_value
        mock_inteligencia.predict.return_value = {'action': 'BUY', 'confidence': 0.8}
        mock_inteligencia.should_switch_to_real_mode.return_value = False
        mock_inteligencia.should_switch_to_test_mode.return_value = False
        
        # Break loop after one sleep
        mock_sleep.side_effect = lambda x: main.stop_event.set()
        
        yield {
            'cm': mock_cm,
            'ferramental': mock_ferramental,
            'inteligencia': mock_inteligencia,
            'args': mock_args,
            'sleep': mock_sleep
        }

def test_main_flow_test_mode(mock_components):
    main.stop_event.clear()
    mock_components['args'].mode = 'test'
    mock_components['args'].config = 'config.ini'
    mock_components['args'].assets = 'EURUSD'
    
    main.main()
    
    mock_components['ferramental'].get_realtime_data.assert_called()
    mock_components['inteligencia'].predict.assert_called()
    mock_components['ferramental'].buy.assert_called()

def test_main_flow_real_mode(mock_components):
    main.stop_event.clear()
    mock_components['args'].mode = 'real'
    # Ensure real mode is "enabled" in mock_cm
    mock_components['cm'].get_value.side_effect = lambda s, k, d=None, t=None: \
        True if k == 'enable_real_mode' else (d if d is not None else (10.0 if t == float else 'test'))
    
    main.main()
    
    # In real mode it should call buy if signal is BUY
    mock_components['ferramental'].buy.assert_called()

def test_main_flow_learning_mode(mock_components):
    main.stop_event.clear()
    mock_components['args'].mode = 'learning'
    
    # Mock historical data for learning
    # We need at least 100 rows to satisfy _split_data(0.2) and sequence lengths
    data = []
    for i in range(110):
        data.append({
            'open': 1.1 + i*0.001,
            'high': 1.2 + i*0.001,
            'low': 1.0 + i*0.001,
            'close': 1.1 + i*0.001,
            'volume': 1000 + i*10
        })
    df = pd.DataFrame(data)
    # Ensure index is datetime for indicators
    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='min')
    
    # Mock Inteligencia to return the same DF or processed one
    mock_components['ferramental'].get_historical_data.return_value = df
    mock_components['inteligencia'].process_historical_data.return_value = df
    mock_components['inteligencia'].train.return_value = {'accuracy': 0.8, 'loss': 0.1}
    mock_components['inteligencia'].evaluate_model.return_value = {'accuracy': 0.8}
    
    # Mock _split_data since main.py calls it and it might fail with mocks
    mock_components['inteligencia']._split_data.return_value = (df, df)
    mock_components['inteligencia'].generate_training_report.return_value = "Report Content"
    
    # Make it switch to test mode after learning
    mock_components['inteligencia'].should_switch_to_test_mode.return_value = True
    
    # We need multiple iterations to see the switch, but let's just check learning call
    main.main()
    
    mock_components['inteligencia'].train.assert_called_once()
    mock_components['inteligencia'].evaluate_model.assert_called_once()

def test_main_connection_fail_simulation_fallback(mock_components):
    main.stop_event.clear()
    mock_components['ferramental'].check_connection.return_value = False
    mock_components['ferramental'].connect.return_value = (False, "Timeout")
    
    main.main()
    
    mock_components['ferramental'].setup_simulation_mode.assert_called()

def test_main_signal_handler_integration():
    with patch('main.logger'):
        main.stop_event.clear()
        main.signal_handler(None, None)
        assert main.stop_event.is_set()
