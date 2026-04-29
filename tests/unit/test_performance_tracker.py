import os
import json
import pytest
from datetime import datetime
from utils.PerformanceTracker import PerformanceTracker

@pytest.fixture
def pt():
    return PerformanceTracker()

def test_add_trade_inference(pt):
    # Test inferring 'win' result
    pt.add_trade({'asset': 'EURUSD', 'profit': 10.0})
    assert pt.trades[0]['result'] == 'win'
    
    # Test inferring 'loss' result
    pt.add_trade({'asset': 'EURUSD', 'profit': -5.0})
    assert pt.trades[1]['result'] == 'loss'
    
    # Test inferring 'draw' result
    pt.add_trade({'asset': 'EURUSD', 'profit': 0.0})
    assert pt.trades[2]['result'] == 'draw'

def test_basic_metrics(pt):
    trades = [
        {'asset': 'EURUSD', 'profit': 10.0, 'result': 'win'},
        {'asset': 'EURUSD', 'profit': -5.0, 'result': 'loss'},
        {'asset': 'GBPUSD', 'profit': 20.0, 'result': 'win'}
    ]
    pt.update(trades)
    
    metrics = pt.get_metrics()
    assert metrics['total_trades'] == 3
    assert metrics['win_count'] == 2
    assert metrics['loss_count'] == 1
    assert metrics['total_profit'] == 25.0
    assert metrics['win_rate'] == pytest.approx(0.6666, 0.001)
    assert metrics['profit_factor'] == 30.0 / 5.0

def test_equity_curve(pt):
    trades = [
        {'asset': 'EURUSD', 'profit': 10.0, 'timestamp': '2023-01-01T10:00:00'},
        {'asset': 'EURUSD', 'profit': -5.0, 'timestamp': '2023-01-01T10:05:00'},
        {'asset': 'GBPUSD', 'profit': 5.0, 'timestamp': '2023-01-01T10:10:00'}
    ]
    pt.update(trades)
    
    curve = pt.get_equity_curve()
    assert len(curve) == 3
    assert curve[0]['equity'] == 10.0
    assert curve[1]['equity'] == 5.0
    assert curve[2]['equity'] == 10.0

def test_max_drawdown(pt):
    trades = [
        {'asset': 'EURUSD', 'profit': 100.0, 'timestamp': '2023-01-01T10:00:00'},
        {'asset': 'EURUSD', 'profit': -20.0, 'timestamp': '2023-01-01T10:05:00'},
        {'asset': 'EURUSD', 'profit': -10.0, 'timestamp': '2023-01-01T10:10:00'},
        {'asset': 'EURUSD', 'profit': 40.0, 'timestamp': '2023-01-01T10:15:00'}
    ]
    pt.update(trades)
    
    # Peak was 100, then dropped to 80, then 70. 
    # Max DD = (100 - 70) / 100 = 0.3
    metrics = pt.get_metrics()
    assert metrics['max_drawdown'] == 0.3

def test_asset_performance(pt):
    trades = [
        {'asset': 'EURUSD', 'profit': 10.0, 'result': 'win'},
        {'asset': 'GBPUSD', 'profit': -5.0, 'result': 'loss'},
        {'asset': 'EURUSD', 'profit': 5.0, 'result': 'win'}
    ]
    pt.update(trades)
    
    eur_perf = pt.get_asset_performance('EURUSD')
    assert eur_perf['total_trades'] == 2
    assert eur_perf['win_rate'] == 1.0
    
    gbp_perf = pt.get_asset_performance('GBPUSD')
    assert gbp_perf['loss_count'] == 1

def test_daily_performance(pt):
    trades = [
        {'asset': 'EURUSD', 'profit': 10.0, 'timestamp': '2023-01-01T10:00:00'},
        {'asset': 'GBPUSD', 'profit': 5.0, 'timestamp': '2023-01-01T11:00:00'},
        {'asset': 'EURUSD', 'profit': -2.0, 'timestamp': '2023-01-02T10:00:00'}
    ]
    pt.update(trades)
    
    daily = pt.get_daily_performance()
    assert '2023-01-01' in daily
    assert '2023-01-02' in daily
    assert daily['2023-01-01']['total_trades'] == 2
    assert daily['2023-01-02']['total_profit'] == -2.0

def test_persistence(pt, tmp_path):
    test_file = tmp_path / "perf.json"
    pt.add_trade({'asset': 'EURUSD', 'profit': 10.0, 'result': 'win'})
    pt.save_to_file(str(test_file))
    
    new_pt = PerformanceTracker()
    assert new_pt.load_from_file(str(test_file))
    metrics = new_pt.get_metrics()
    assert metrics['total_trades'] == 1
    assert new_pt.trades[0]['asset'] == 'EURUSD'
