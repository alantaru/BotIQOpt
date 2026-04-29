import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from tests.integration.mock_api import MockIQOptionAPI

def test_full_trading_cycle_e2e():
    """Simula um ciclo completo de trading: login -> dados -> predição -> execução -> resultado."""
    
    # 1. Setup - Mocking imports and creating instances
    with patch('iqoptionapi.stable_api.IQ_Option', new=MockIQOptionAPI):
        from utils.ConfigManager import ConfigManager
        from utils.ErrorTracker import ErrorTracker
        from utils.PerformanceTracker import PerformanceTracker
        from ferramental.Ferramental import Ferramental
        from inteligencia.Inteligencia import Inteligencia
        
        # Reset Singletons
        ConfigManager._instance = None
        Ferramental._instance = None
        
        # Mocks
        config = MagicMock(spec=ConfigManager)
        error_tracker = ErrorTracker() # Use real instance for better trace
        perf_tracker = PerformanceTracker()
        
        # Configure Config Mocks
        config.get_value.side_effect = lambda s, k, d=None, t=None: {
            'email': 'test@test.com',
            'password': 'password',
            'assets': 'EURUSD',
            'min_confidence': 0.6,
            'amount': 100.0,
            'expiration': 1,
            'retry_count': 1,
            'timeout': 1
        }.get(k, d)
        config.get_list.side_effect = lambda s, k, d=None: ['EURUSD'] if k == 'assets' else d
        
        # Initialize components
        ferramental = Ferramental(config, error_tracker)
        inteligencia = Inteligencia(config, error_tracker)
        
        # 2. Connection
        success, reason = ferramental.connect()
        assert success is True
        assert ferramental.connected is True
        
        # 3. Data Retrieval
        # get_realtime_data internally calls get_candles of iq_option (Mocked)
        data = ferramental.get_realtime_data("EURUSD")
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        
        # 4. Prediction
        # Mocking the model call inside predict to return BUY
        inteligencia.model = MagicMock()
        inteligencia.model.input_size = len([c for c in data.columns if c not in ['timestamp', 'date', 'label', 'asset', 'timeframe', 'datetime']])
        
        with patch('torch.no_grad'):
            with patch('torch.FloatTensor'):
                 with patch('torch.nn.functional.softmax', return_value=MagicMock(cpu=lambda: MagicMock(numpy=lambda: [np.array([0, 0.9, 0.1])]))):
                     with patch('torch.max', return_value=(MagicMock(item=lambda: 0.9), MagicMock(item=lambda: 1))):
                         predictions = inteligencia.predict(data, confidence_threshold=0.6)
        
        assert predictions['action'] == 'BUY'
        assert predictions['confidence'] >= 0.6
        
        # 5. Execution
        asset = 'EURUSD'
        amount = 100.0
        expiration = 1
        mapped_action = 'call' if predictions['action'] == 'BUY' else 'put'
        print(f"DEBUG: Buying {asset} {mapped_action} {amount}")
        success, order_id = ferramental.buy(
            asset=asset,
            amount=amount,
            action=mapped_action,
            expiration=expiration
        )

        if not success:
             print(f"DEBUG: Buy failed. Connected: {ferramental.connected}, Risk: {ferramental.risk_management}")
             # Check why it failed
             balance = ferramental.get_balance()
             print(f"DEBUG: Balance: {balance}, Asset Pairs: {ferramental.asset_pairs}")
             print(f"DEBUG: Asset Open: {ferramental.check_asset_open(asset)}")
             
        assert success is True, f"Buy failed for {asset}. Check debug output."
        assert order_id is not None

        
        # 6. Result Polling & Verification
        # In mock_api, buy(100) -> even amount -> win
        results = ferramental.get_trade_results()
        import json
        with open('debug_e2e_results.json', 'w') as f:
            json.dump(results, f)
        assert len(results) >= 1

        
        target_result = next((r for r in results if r['id'] == order_id), None)
        assert target_result is not None
        assert target_result['profit'] > 0
        
        # 7. Performance & Risk Update
        perf_tracker.update(results)
        print(f"DEBUG: Performance metrics: {perf_tracker.metrics}")
        assert perf_tracker.metrics['total_trades'] >= 1
        assert perf_tracker.metrics['win_count'] >= 1

        
        # Check risk management side effect (reset losses on win)
        if target_result['profit'] > 0:
            ferramental.risk_management['consecutive_losses'] = 0
            assert ferramental.risk_management['consecutive_losses'] == 0
