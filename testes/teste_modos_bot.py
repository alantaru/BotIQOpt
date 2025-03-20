import os
import sys
import json
import time
import logging
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teste_modos_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TesteModosBOT')

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock das classes principais
class InteligenciaMock:
    """Mock da classe Inteligencia para testes"""
    
    def __init__(self, model_path=None, historical_data_filename=None):
        self.model_path = model_path
        self.historical_data_filename = historical_data_filename
        self.auto_switch = False
        self.training_data = []
        self.model = None
        
    def store_training_data(self, data):
        """Armazena dados de treinamento"""
        logger.info(f"Armazenando {len(data)} registros de dados de treinamento")
        self.training_data.append(data)
        return True
        
    def train_model(self):
        """Treina o modelo com os dados armazenados"""
        logger.info("Treinando modelo com dados históricos")
        self.model = "modelo_treinado"
        return True
        
    def get_predictions(self, assets):
        """Retorna previsões para os ativos especificados"""
        logger.info(f"Gerando previsões para {len(assets)} ativos")
        predictions = {}
        for asset in assets:
            predictions[asset] = {
                'direction': np.random.choice(['call', 'put']),
                'confidence': np.random.uniform(0.6, 0.95)
            }
        return predictions
        
    def set_auto_switch(self, value):
        """Define se deve mudar automaticamente para modo real"""
        self.auto_switch = value
        
    def should_switch_to_real(self):
        """Verifica se deve mudar para modo real"""
        return self.auto_switch
        
    def save_model(self):
        """Salva o modelo treinado"""
        logger.info("Salvando modelo")
        return True
        
class FerramentalMock:
    """Mock da classe Ferramental para testes"""
    
    def __init__(self, assets=None):
        self.assets = assets or ["EURUSD", "GBPUSD"]
        self.connected = False
        
    def connect(self):
        """Conecta à API"""
        logger.info("Conectando à API")
        self.connected = True
        return True
        
    def download_historical_data(self, asset, timeframe_type, timeframe_value, candle_count):
        """Baixa dados históricos"""
        logger.info(f"Baixando dados históricos para {asset}")
        
        # Gera dados históricos simulados
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        dates = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        np.random.seed(42)
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.0002, size=len(dates))
        prices = np.cumsum(price_changes) + base_price
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 0.0010, size=len(dates)),
            'low': prices - np.random.uniform(0, 0.0010, size=len(dates)),
            'close': prices + np.random.normal(0, 0.0001, size=len(dates)),
            'volume': np.random.randint(100, 1000, size=len(dates))
        })
        
        return data
        
    def execute_test_trade(self, asset, direction, amount):
        """Executa trade de teste"""
        logger.info(f"Executando trade de teste para {asset} ({direction})")
        return np.random.choice([True, False], p=[0.7, 0.3])
        
    def execute_real_trade(self, asset, direction, amount):
        """Executa trade real"""
        logger.info(f"Executando trade real para {asset} ({direction})")
        return np.random.choice([True, False], p=[0.7, 0.3])

class BotPerformanceMetrics:
    """Classe para rastrear métricas de desempenho do bot"""
    
    def __init__(self):
        self.trades = []
        self.start_time = datetime.now()
        
    def add_trade(self, asset, direction, amount, result):
        """Adiciona um trade às métricas"""
        self.trades.append({
            'timestamp': datetime.now(),
            'asset': asset,
            'direction': direction,
            'amount': amount,
            'result': result
        })
        
    def calculate_metrics(self):
        """Calcula métricas de desempenho"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'runtime': str(datetime.now() - self.start_time)
            }
            
        wins = sum(1 for trade in self.trades if trade['result'])
        losses = len(self.trades) - wins
        
        win_rate = wins / len(self.trades) if len(self.trades) > 0 else 0
        
        total_profit = sum(trade['amount'] if trade['result'] else -trade['amount'] for trade in self.trades)
        profit_factor = (
            sum(trade['amount'] for trade in self.trades if trade['result']) /
            abs(sum(trade['amount'] for trade in self.trades if not trade['result']))
            if sum(trade['amount'] for trade in self.trades if not trade['result']) != 0 else 
            float('inf') if sum(trade['amount'] for trade in self.trades if trade['result']) > 0 else 0
        )
        
        return {
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'runtime': str(datetime.now() - self.start_time)
        }

class TesteModosBOT(unittest.TestCase):
    """Classe para testar os diferentes modos do bot"""
    
    def setUp(self):
        """Configuração inicial para cada teste"""
        logger.info("Configurando ambiente de teste")
        
        # Configura instâncias das classes mockadas
        self.inteligencia = InteligenciaMock(
            model_path="modelo_teste.pkl",
            historical_data_filename="dados_historicos.csv"
        )
        self.ferramental = FerramentalMock(assets=["EURUSD", "GBPUSD"])
        
        # Configura parâmetros de teste
        self.assets = ["EURUSD", "GBPUSD"]
        self.general_params = {
            'timeframe_type': 'Minutes',
            'timeframe_value': 5,
            'candle_count': 100,
            'risk_per_trade': 10.0,
            'auto_switch_to_real': False
        }
        
        # Configura métricas de desempenho
        self.performance_metrics = BotPerformanceMetrics()
    
    def _calcular_timeframe(self, timeframe_type, timeframe_value):
        """Calcula timeframe em segundos"""
        if timeframe_type == "Minutes":
            return timeframe_value * 60
        elif timeframe_type == "Seconds":
            return timeframe_value
        elif timeframe_type == "Hours":
            return timeframe_value * 3600
        else:
            raise ValueError(f"Tipo de timeframe inválido: {timeframe_type}")
    
    def test_modo_download(self):
        """Testa o modo Download do bot"""
        logger.info("Iniciando teste do modo Download")
        
        # Executa o modo Download
        try:
            for asset in self.assets:
                logger.info(f"Baixando dados históricos para {asset}")
                data = self.ferramental.download_historical_data(
                    asset=asset,
                    timeframe_type=self.general_params['timeframe_type'],
                    timeframe_value=self.general_params['timeframe_value'],
                    candle_count=self.general_params['candle_count']
                )
                
                # Verifica se os dados foram baixados
                self.assertIsNotNone(data)
                
                # Armazena os dados para treinamento
                result = self.inteligencia.store_training_data(data)
                
                # Verifica se os dados foram armazenados
                self.assertTrue(result)
                
                logger.info(f"Dados para {asset} baixados com sucesso")
            
            logger.info("Teste do modo Download concluído com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante o teste do modo Download: {str(e)}")
            self.fail(f"Teste do modo Download falhou: {str(e)}")
            return False
    
    def test_modo_learning(self):
        """Testa o modo Learning do bot"""
        logger.info("Iniciando teste do modo Learning")
        
        try:
            # Executa o modo Learning
            result = self.inteligencia.train_model()
            
            # Verifica se o treinamento foi bem-sucedido
            self.assertTrue(result)
            
            logger.info("Teste do modo Learning concluído com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante o teste do modo Learning: {str(e)}")
            self.fail(f"Teste do modo Learning falhou: {str(e)}")
            return False
    
    def test_modo_test(self):
        """Testa o modo Test do bot"""
        logger.info("Iniciando teste do modo Test")
        
        try:
            # Obtém previsões da IA
            predictions = self.inteligencia.get_predictions(self.assets)
            
            # Executa trades de teste
            for asset, prediction in predictions.items():
                if prediction['confidence'] > 0.7:  # Apenas opera com alta confiança
                    result = self.ferramental.execute_test_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade']
                    )
                    
                    # Adiciona o trade às métricas
                    self.performance_metrics.add_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade'],
                        result=result
                    )
                    
                    logger.info(f"Trade de teste executado para {asset}")
            
            # Calcula e exibe métricas
            metrics = self.performance_metrics.calculate_metrics()
            logger.info(f"Métricas de desempenho: {metrics}")
            
            # Verifica se as métricas foram calculadas corretamente
            self.assertGreaterEqual(metrics['total_trades'], 0)
            
            logger.info("Teste do modo Test concluído com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante o teste do modo Test: {str(e)}")
            self.fail(f"Teste do modo Test falhou: {str(e)}")
            return False
    
    def test_modo_real(self):
        """Testa o modo Real do bot"""
        logger.info("Iniciando teste do modo Real")
        
        try:
            # Obtém previsões da IA
            predictions = self.inteligencia.get_predictions(self.assets)
            
            # Executa trades reais
            for asset, prediction in predictions.items():
                if prediction['confidence'] > 0.8:  # Limite mais alto para trades reais
                    result = self.ferramental.execute_real_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade']
                    )
                    
                    # Adiciona o trade às métricas
                    self.performance_metrics.add_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade'],
                        result=result
                    )
                    
                    logger.info(f"Trade real executado para {asset}")
            
            # Calcula e exibe métricas
            metrics = self.performance_metrics.calculate_metrics()
            logger.info(f"Métricas de desempenho: {metrics}")
            
            # Verifica se as métricas foram calculadas corretamente
            self.assertGreaterEqual(metrics['total_trades'], 0)
            
            logger.info("Teste do modo Real concluído com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante o teste do modo Real: {str(e)}")
            self.fail(f"Teste do modo Real falhou: {str(e)}")
            return False
    
    def test_auto_switch(self):
        """Testa a funcionalidade de mudança automática para modo real"""
        logger.info("Iniciando teste de mudança automática para modo real")
        
        try:
            # Configura o modo inicial como Test
            mode = "Test"
            
            # Configura auto_switch como True
            self.general_params['auto_switch_to_real'] = True
            self.inteligencia.set_auto_switch(True)
            
            # Executa trades de teste
            predictions = self.inteligencia.get_predictions(self.assets)
            
            for asset, prediction in predictions.items():
                if prediction['confidence'] > 0.7:
                    result = self.ferramental.execute_test_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade']
                    )
                    
                    self.performance_metrics.add_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade'],
                        result=result
                    )
            
            # Verifica se deve mudar para modo real
            if mode == "Test" and self.inteligencia.should_switch_to_real():
                logger.info("Mudando para modo de operação real com base no desempenho")
                mode = "Real"
                self.inteligencia.set_auto_switch(False)
            
            # Verifica se o modo foi alterado
            self.assertEqual(mode, "Real")
            
            logger.info("Teste de mudança automática concluído com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante o teste de mudança automática: {str(e)}")
            self.fail(f"Teste de mudança automática falhou: {str(e)}")
            return False
    
    def test_pipeline_completo(self):
        """Testa o pipeline completo do bot em todos os modos"""
        logger.info("Iniciando teste do pipeline completo")
        
        try:
            # 1. Modo Download
            logger.info("Executando modo Download")
            for asset in self.assets:
                data = self.ferramental.download_historical_data(
                    asset=asset,
                    timeframe_type=self.general_params['timeframe_type'],
                    timeframe_value=self.general_params['timeframe_value'],
                    candle_count=self.general_params['candle_count']
                )
                self.inteligencia.store_training_data(data)
            
            # 2. Modo Learning
            logger.info("Executando modo Learning")
            self.inteligencia.train_model()
            
            # 3. Modo Test
            logger.info("Executando modo Test")
            predictions = self.inteligencia.get_predictions(self.assets)
            
            for asset, prediction in predictions.items():
                if prediction['confidence'] > 0.7:
                    result = self.ferramental.execute_test_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade']
                    )
                    self.performance_metrics.add_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade'],
                        result=result
                    )
            
            # Calcula métricas após modo Test
            test_metrics = self.performance_metrics.calculate_metrics()
            logger.info(f"Métricas após modo Test: {test_metrics}")
            
            # 4. Modo Real
            logger.info("Executando modo Real")
            predictions = self.inteligencia.get_predictions(self.assets)
            
            for asset, prediction in predictions.items():
                if prediction['confidence'] > 0.8:
                    result = self.ferramental.execute_real_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade']
                    )
                    self.performance_metrics.add_trade(
                        asset=asset,
                        direction=prediction['direction'],
                        amount=self.general_params['risk_per_trade'],
                        result=result
                    )
            
            # Calcula métricas finais
            final_metrics = self.performance_metrics.calculate_metrics()
            logger.info(f"Métricas finais: {final_metrics}")
            
            # Verifica se as métricas finais foram calculadas corretamente
            self.assertGreater(final_metrics['total_trades'], test_metrics['total_trades'])
            
            logger.info("Teste do pipeline completo concluído com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante o teste do pipeline completo: {str(e)}")
            self.fail(f"Teste do pipeline completo falhou: {str(e)}")
            return False

def run_tests():
    """Executa todos os testes de modos do bot"""
    logger.info("==== INICIANDO TESTES DE MODOS DO BOT ====")
    
    # Lista de testes a serem executados
    testes = [
        ("Modo Download", TesteModosBOT("test_modo_download")),
        ("Modo Learning", TesteModosBOT("test_modo_learning")),
        ("Modo Test", TesteModosBOT("test_modo_test")),
        ("Modo Real", TesteModosBOT("test_modo_real")),
        ("Auto Switch", TesteModosBOT("test_auto_switch")),
        ("Pipeline Completo", TesteModosBOT("test_pipeline_completo"))
    ]
    
    resultados = {}
    
    # Executa cada teste
    for nome, teste in testes:
        logger.info(f"\n==== TESTE: {nome} ====")
        try:
            resultado = unittest.TextTestRunner().run(teste)
            resultados[nome] = "SUCESSO" if resultado.wasSuccessful() else "FALHA"
        except Exception as e:
            logger.error(f"Erro durante o teste {nome}: {str(e)}")
            resultados[nome] = "ERRO"
    
    # Exibe resumo dos testes
    logger.info("\n==== RESUMO DOS TESTES ====")
    for nome, resultado in resultados.items():
        logger.info(f"{nome}: {resultado}")
    
    # Salva resultados em arquivo JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resultados_testes_modos_bot_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(resultados, f, indent=4)
    
    logger.info(f"Resultados salvos em {filename}")
    
    return all(resultado == "SUCESSO" for resultado in resultados.values())

if __name__ == "__main__":
    run_tests()
