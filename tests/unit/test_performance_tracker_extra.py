import os
import json
import pytest
from datetime import datetime, timedelta
from utils.PerformanceTracker import PerformanceTracker

@pytest.fixture
def tracker():
    return PerformanceTracker()

def test_add_trade_invalid_format(tracker):
    # Line 54-55: Invalid format (not a dict or missing keys)
    tracker.add_trade("not a dict")
    assert len(tracker.trades) == 0
    tracker.add_trade({'only_profit': 10})
    assert len(tracker.trades) == 0

def test_update_metrics_empty(tracker):
    # Line 90: _update_metrics with no trades
    tracker._update_metrics()
    assert tracker.metrics['total_trades'] == 0

def test_avg_trade_duration_edge_cases(tracker):
    # Line 121-127: Duration with mixed formats and errors
    now = datetime.now()
    # Case: datetime objects
    tracker.add_trade({'asset': 'EURUSD', 'profit': 10, 'open_time': now, 'close_time': now + timedelta(minutes=1)})
    # Case: ISO strings
    tracker.add_trade({'asset': 'EURUSD', 'profit': -5, 'open_time': now.isoformat(), 'close_time': (now + timedelta(minutes=2)).isoformat()})
    # Case: ValueError/TypeError (Line 126-127)
    tracker.add_trade({'asset': 'EURUSD', 'profit': 0, 'open_time': 'invalid', 'close_time': 'date'})
    
    assert tracker.metrics['avg_trade_duration'] == 90.0 # (60 + 120) / 2

def test_best_worst_metrics(tracker):
    # Multiple assets and timeframes to test max/min (Lines 137-138, 145-146, 251-275)
    tracker.add_trade({'asset': 'WINNER', 'profit': 10, 'timeframe': 60, 'result': 'win'})
    tracker.add_trade({'asset': 'LOSER', 'profit': -10, 'timeframe': 300, 'result': 'loss'})
    
    assert tracker.metrics['best_asset'][0] == 'WINNER'
    assert tracker.metrics['worst_asset'][0] == 'LOSER'
    assert tracker.metrics['best_timeframe'][0] == 60
    assert tracker.metrics['worst_timeframe'][0] == 300

def test_daily_performance_edge_cases(tracker):
    # Line 291: No timestamp
    # We cheat a bit because add_trade adds a timestamp if missing. 
    # Let's bypass add_trade to test the internal method directly or mock add_trade.
    tracker.trades.append({'asset': 'X', 'profit': 10}) # No timestamp
    res = tracker._calculate_daily_performance()
    assert len(res) == 0 # Line 291 hit
    
    # Line 330-331: ValueError in datetime parsing
    tracker.trades.append({'asset': 'Y', 'profit': 10, 'timestamp': 'garbage'})
    res = tracker._calculate_daily_performance()
    assert len(res) == 0

def test_getters(tracker):
    stable_now = datetime(2026, 3, 9, 12, 0, 0)
    tracker.add_trade({'asset': 'EURUSD', 'profit': 0, 'timeframe': 60, 'timestamp': stable_now, 'result': 'draw'})

    # Draw logic (Line 268-269)
    assert tracker.get_timeframe_performance(60)['draw_count'] == 1

    # Int/float timestamp logic (Line 296-297)
    now_ts = stable_now.timestamp()
    tracker.trades.append({'asset': 'TS', 'profit': 5, 'timestamp': now_ts})
    res = tracker._calculate_daily_performance()
    assert len(res) >= 1

    # Else logic for dt (Line 299)
    now_dt = stable_now
    tracker.trades.append({'asset': 'DT', 'profit': 5, 'timestamp': now_dt})
    res = tracker._calculate_daily_performance()
    assert len(res) >= 1

    # Specific asset (Line 353-354)
    assert tracker.get_asset_performance('EURUSD')['total_trades'] == 1
    assert isinstance(tracker.get_asset_performance(), dict)  # Line 354
    assert tracker.get_asset_performance('NON_EXISTENT') is None

    # Specific timeframe (Line 365-367)
    assert tracker.get_timeframe_performance(60)['total_trades'] == 1
    assert isinstance(tracker.get_timeframe_performance(), dict)  # Line 367
    assert tracker.get_timeframe_performance(300) is None

    # Specific date (Line 379-380)
    assert tracker.get_daily_performance('2026-03-09')['total_trades'] == 1
    assert tracker.get_daily_performance('2000-01-01') is None


def test_persistence_failures(tracker, tmp_path):
    # Line 417-419: Save failure
    test_dir = tmp_path / "protected_dir"
    test_dir.mkdir()
    assert tracker.save_to_file(str(test_dir)) is False
    
    # Line 432-433: Load not found
    assert tracker.load_from_file("ghost.json") is False
    
    # Line 450-452: Load failure (corrupt JSON)
    corrupt = tmp_path / "corrupt.json"
    with open(corrupt, 'w') as f:
        f.write("{corrupt...")
    assert tracker.load_from_file(str(corrupt)) is False

def test_max_drawdown_empty(tracker):
    # Line 180: empty equity curve
    tracker.equity_curve = []
    assert tracker._calculate_max_drawdown() == 0.0

def test_asset_performance_no_asset_logic(tracker):
    # Line 208: trade with no asset
    tracker.trades.append({'profit': 10}) # Bypass validation
    res = tracker._calculate_asset_performance()
    assert len(res) == 0
