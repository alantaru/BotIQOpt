import matplotlib
matplotlib.use('Agg')

import pytest
from unittest.mock import MagicMock
from ferramental.Ferramental import Ferramental
from inteligencia.Inteligencia import Inteligencia
from utils.ConfigManager import ConfigManager
from utils.ErrorTracker import ErrorTracker

@pytest.fixture
def mock_ferramental():
    config = MagicMock(spec=ConfigManager)
    errors = MagicMock(spec=ErrorTracker)
    ferramental = Ferramental(config, errors)
    ferramental.iq_option = MagicMock()
    return ferramental

@pytest.fixture
def mock_inteligencia():
    config = MagicMock(spec=ConfigManager)
    errors = MagicMock(spec=ErrorTracker)
    inteligencia = Inteligencia(config, errors)
    return inteligencia

def test_initial_state(mock_ferramental, mock_inteligencia):
    # Ferramental uses simulation_mode (bool)
    assert hasattr(mock_ferramental, 'simulation_mode')
    # Inteligencia uses mode string, defaults to 'LEARNING'
    assert mock_inteligencia.mode == "LEARNING"
