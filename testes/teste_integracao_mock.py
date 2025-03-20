import sys
import os
import logging
from datetime import datetime
import pandas as pd
import unittest
from unittest.mock import MagicMock, patch

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teste_integracao_mock.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TesteIntegracaoMock')

class TestIntegracaoMock(unittest.TestCase):
    """Testes de integração usando mocks para evitar dependências externas"""
    
    def setUp(self):
        """Configura os mocks e patches necessários para os testes"""
        # Criamos patches para as importações problemáticas
        self.patches = []
        
        # Mock para o módulo tulipy
        tulipy_mock = MagicMock()
        tulipy_patch = patch.dict('sys.modules', {'tulipy': tulipy_mock})
        tulipy_patch.start()
        self.patches.append(tulipy_patch)
        
        # Mock para IQ_Option
        iq_option_mock = MagicMock()
        iq_option_patch = patch.dict('sys.modules', {'iqoptionapi.stable_api': MagicMock()})
        iq_option_patch.start()
        self.patches.append(iq_option_patch)
        
        # Agora podemos importar os módulos que dependem dessas bibliotecas
        from inteligencia.Inteligencia import Inteligencia
        from ferramental.Ferramental import Ferramental
        
        self.Inteligencia = Inteligencia
        self.Ferramental = Ferramental
    
    def tearDown(self):
        """Limpa os patches após os testes"""
        for patch in self.patches:
            patch.stop()
    
    def test_get_historical_data_integration(self):
        """Testa a integração entre get_historical_data das classes Inteligencia e Ferramental"""
        # Cria um DataFrame de exemplo que seria retornado pelo Ferramental
        mock_data = pd.DataFrame({
            'open': [1.1, 1.2, 1.3],
            'close': [1.2, 1.3, 1.4],
            'high': [1.3, 1.4, 1.5],
            'low': [1.0, 1.1, 1.2],
            'volume': [100, 200, 300],
            'asset': ['EURUSD', 'EURUSD', 'EURUSD'],
            'timeframe': [60, 60, 60]
        })
        
        # Cria instâncias das classes com mocks
        ferramental = self.Ferramental()
        inteligencia = self.Inteligencia(historical_data_filename="teste_mock.csv")
        
        # Configura o mock para retornar os dados de exemplo
        ferramental.get_historical_data = MagicMock(return_value=mock_data)
        
        # Testa a integração
        resultado = inteligencia.get_historical_data(ferramental)
        
        # Verifica se o método do Ferramental foi chamado
        ferramental.get_historical_data.assert_called_once()
        
        # Verifica se os dados foram retornados corretamente
        self.assertIsNotNone(resultado)
        self.assertEqual(len(resultado), 3)
        self.assertEqual(resultado['asset'].iloc[0], 'EURUSD')
    
    def test_update_historical_data_integration(self):
        """Testa a integração entre update_historical_data e get_historical_data"""
        # Cria um DataFrame de exemplo
        mock_data = pd.DataFrame({
            'open': [1.1, 1.2, 1.3],
            'close': [1.2, 1.3, 1.4],
            'high': [1.3, 1.4, 1.5],
            'low': [1.0, 1.1, 1.2],
            'volume': [100, 200, 300],
            'asset': ['EURUSD', 'EURUSD', 'EURUSD'],
            'timeframe': [60, 60, 60]
        })
        
        # Cria instâncias das classes com mocks
        ferramental = self.Ferramental()
        inteligencia = self.Inteligencia(historical_data_filename="teste_update_mock.csv")
        
        # Configura o mock para retornar os dados de exemplo
        ferramental.get_historical_data = MagicMock(return_value=mock_data)
        
        # Substitui o método os.path.exists para simular que o arquivo não existe
        with patch('os.path.exists', return_value=False):
            # Testa a atualização de dados
            resultado = inteligencia.update_historical_data(ferramental)
            
            # Verifica se o método get_historical_data do Ferramental foi chamado
            ferramental.get_historical_data.assert_called_once()
            
            # Verifica se os dados foram retornados corretamente
            self.assertIsNotNone(resultado)
            self.assertEqual(len(resultado), 3)
    
    def test_preprocess_data_integration(self):
        """Testa a integração entre preprocess_data e os métodos de processamento"""
        # Cria um DataFrame de exemplo
        mock_data = pd.DataFrame({
            'open': [1.1, 1.2, 1.3],
            'close': [1.2, 1.3, 1.4],
            'high': [1.3, 1.4, 1.5],
            'low': [1.0, 1.1, 1.2],
            'volume': [100, 200, 300]
        })
        
        # Cria uma instância da classe Inteligencia
        inteligencia = self.Inteligencia()
        
        # Configura mocks para os métodos de processamento
        inteligencia._add_technical_indicators = MagicMock(return_value=mock_data)
        inteligencia._add_candle_patterns = MagicMock(return_value=mock_data)
        inteligencia._add_derived_features = MagicMock(return_value=mock_data)
        inteligencia._normalize_data = MagicMock(return_value=mock_data)
        
        # Define os dados históricos
        inteligencia.historical_data = mock_data
        
        # Testa o pré-processamento
        resultado = inteligencia.preprocess_data()
        
        # Verifica se todos os métodos de processamento foram chamados
        inteligencia._add_technical_indicators.assert_called_once()
        inteligencia._add_candle_patterns.assert_called_once()
        inteligencia._add_derived_features.assert_called_once()
        inteligencia._normalize_data.assert_called_once()
        
        # Verifica se os dados foram retornados corretamente
        self.assertIsNotNone(resultado)
        self.assertEqual(len(resultado), 3)

if __name__ == "__main__":
    logger.info("==== INICIANDO TESTES DE INTEGRAÇÃO COM MOCKS ====")
    unittest.main()
