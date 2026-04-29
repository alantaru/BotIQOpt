import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from inteligencia.Inteligencia import Inteligencia
from utils.ConfigManager import ConfigManager
from utils.ErrorTracker import ErrorTracker

@pytest.fixture
def intelligence():
    config = MagicMock(spec=ConfigManager)
    error_tracker = MagicMock(spec=ErrorTracker)
    # Mock get_value to return some defaults
    config.get_value.side_effect = lambda s, k, d=None, t=None: d
    return Inteligencia(config_manager=config, error_tracker=error_tracker)

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=200, freq='T'),
        'open': np.random.uniform(1.0, 1.1, 200),
        'high': np.random.uniform(1.1, 1.2, 200),
        'low': np.random.uniform(0.9, 1.0, 200),
        'close': np.random.uniform(1.0, 1.1, 200),
        'volume': np.random.uniform(100, 1000, 200)
    })
    return df

def test_preprocess_data_length_validation(intelligence):
    # Test failure with short data
    short_df = pd.DataFrame({'close': [1, 2, 3]})
    assert intelligence.preprocess_data(short_df) is None

def test_add_technical_indicators(intelligence, sample_data):
    # Test if indicators are added
    df_with_indicators = intelligence._add_technical_indicators(sample_data)
    assert 'sma_20' in df_with_indicators.columns
    assert 'sma_200' in df_with_indicators.columns
    # In newer Inteligencia, we fillna(method='bfill').fillna(method='ffill')
    # So the first row should NOT be NaN if there's enough data later.
    assert not pd.isna(df_with_indicators['sma_200'].iloc[0])

def test_should_switch_to_test_mode(intelligence):
    # Mock evaluation metrics
    intelligence.last_evaluation_metrics = {
        'accuracy': 0.85,
        'precision': 0.80,
        'recall': 0.75,
        'f1': 0.78
    }
    intelligence.min_accuracy = 0.7
    intelligence.min_precision = 0.7
    intelligence.min_recall = 0.7
    intelligence.min_f1_score = 0.7
    
    assert intelligence.should_switch_to_test_mode() is True

def test_should_switch_to_real_mode(intelligence):
    intelligence.auto_switch_to_real = True
    intelligence.min_win_rate = 0.6
    intelligence.min_profit = 10.0
    intelligence.min_trades_count = 5
    
    perf_tracker = MagicMock()
    perf_tracker.metrics = {
        'win_rate': 0.7,
        'total_profit': 50.0,
        'total_trades': 10
    }
    
    assert intelligence.should_switch_to_real_mode(perf_tracker) is True
    
    # Test failure case
    perf_tracker.metrics['win_rate'] = 0.5
    assert intelligence.should_switch_to_real_mode(perf_tracker) is False

def test_predict_feature_filtering(intelligence):
    # This tests the fix I made earlier: excluding metadata columns
    df = pd.DataFrame({
        'close': [1.0] * 20,
        'sma_20': [1.1] * 20,
        'asset': ['eurusd'] * 20, # Should be ignored
        'timeframe': [60] * 20 # Should be ignored
    })
    
    # Mock model
    intelligence.model = MagicMock()
    intelligence.model.input_size = 2
    intelligence.model.eval = MagicMock()
    
    # Mock model call
    intelligence.model.side_effect = lambda x: MagicMock(return_value=MagicMock())
    
    with patch('torch.FloatTensor') as mock_tensor:
        with patch('torch.no_grad'):
            with patch('torch.nn.functional.softmax', return_value=MagicMock(cpu=lambda: MagicMock(numpy=lambda: [np.array([0,1,0])]))):
                with patch('torch.max', return_value=(MagicMock(item=lambda: 0.9), MagicMock(item=lambda: 1))):
                    intelligence.predict(df)
                    # The first argument to FloatTensor should have shape (1, 20, 2) 
                    # because it should only include 'close' and 'sma_20'
                    args, _ = mock_tensor.call_args
                    input_data = args[0]
                    assert input_data.shape == (20, 2)
