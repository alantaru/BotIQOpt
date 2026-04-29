import pytest
from unittest.mock import MagicMock, patch
from ferramental.Ferramental import PerformanceMetrics, FerramentalError, ConnectionError, InvalidAssetError

def test_performance_metrics():
    metrics = PerformanceMetrics(window_size=5)
    
    # Empty stats
    assert metrics.win_rate() == 0.0
    assert metrics.avg_execution_time() == 0.0
    assert metrics.execution_time_stddev() == 0.0
    
    # Add data
    metrics.add_trade_result(True)
    metrics.add_trade_result(False)
    metrics.add_trade_result(True)
    assert metrics.win_rate() == pytest.approx(0.666, 0.01)
    
    metrics.add_execution_time(0.1)
    metrics.add_execution_time(0.2)
    assert metrics.avg_execution_time() == pytest.approx(0.15)
    assert metrics.execution_time_stddev() > 0
    
    # Single execution time stddev
    metrics2 = PerformanceMetrics(window_size=5)
    metrics2.add_execution_time(0.1)
    assert metrics2.execution_time_stddev() == 0.0

def test_custom_exceptions():
    with pytest.raises(FerramentalError):
        raise ConnectionError("conn")
    with pytest.raises(FerramentalError):
        raise InvalidAssetError("asset")
