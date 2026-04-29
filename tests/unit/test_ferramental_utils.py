import pytest
from ferramental.Ferramental import PerformanceMetrics, RiskMetrics, Ferramental
from unittest.mock import MagicMock

def test_performance_metrics():
    pm = PerformanceMetrics(window_size=5)
    assert pm.win_rate() == 0.0
    assert pm.avg_execution_time() == 0.0
    assert pm.execution_time_stddev() == 0.0
    
    pm.add_trade_result(True)
    pm.add_trade_result(False)
    assert pm.win_rate() == 0.5
    
    pm.add_execution_time(1.0)
    pm.add_execution_time(2.0)
    assert pm.avg_execution_time() == 1.5
    assert pm.execution_time_stddev() > 0
    
    # Test window size
    for _ in range(10):
        pm.add_trade_result(True)
    assert len(pm.trade_results) == 5

def test_risk_metrics_default():
    rm = RiskMetrics()
    assert rm.max_daily_loss == 0.05
    assert rm.consecutive_losses == 0

def test_ferramental_singleton_reset():
    # Helper to test singleton behavior
    Ferramental._instance = None
    config = MagicMock()
    errors = MagicMock()
    f1 = Ferramental(config, errors)
    f2 = Ferramental(config, errors)
    assert f1 is f2
