import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Patch sys.argv before importing main
with patch('sys.argv', ['main.py']):
    import main
    import importlib
    importlib.reload(main)

def test_check_dependencies():
    # Test with all dependencies present
    with patch.dict('sys.modules', {
        'pandas': MagicMock(),
        'numpy': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'talib': MagicMock(),
        'sklearn': MagicMock(),
        'torch': MagicMock(),
        'iqoptionapi': MagicMock()
    }):
        # Ensure we reload or re-import if necessary, 
        # but check_dependencies does imports inside
        assert main.check_dependencies() is True

def test_create_directories():
    with patch('os.makedirs') as mock_makedirs:
        main.create_directories()
        assert mock_makedirs.call_count >= 6

def test_display_banner():
    # Just ensure it doesn't crash
    with patch('builtins.print') as mock_print:
        main.display_banner()
        mock_print.assert_called()

def test_signal_handler():
    with patch('main.logger') as mock_logger:
        main.stop_event.clear()
        main.signal_handler(None, None)
        assert main.stop_event.is_set()
        mock_logger.info.assert_called_with("Recebido sinal de interrupção. Encerrando...")

def test_cleanup():
    with patch('main.performance_tracker') as mock_pt, \
         patch('main.error_tracker') as mock_et, \
         patch('main.logger') as mock_logger:
        main.cleanup()
        mock_pt.save_to_file.assert_called_with('performance_metrics.json')
        mock_et.save_to_file.assert_called_with('error_log.json')
