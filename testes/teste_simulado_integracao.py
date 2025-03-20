import os
import sys
import time
import pandas as pd
import logging
from datetime import datetime

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teste_simulado.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TesteSimulado')

# Simulação simplificada da classe Ferramental
class FerramentalSimulado:
    """Versão simplificada da classe Ferramental para testes"""
    
    def __init__(self, asset_pairs=None):
        self.asset_pairs = asset_pairs or ["EURUSD", "USDJPY"]
        self.connected = True
        logger.info(f"Ferramental simulado inicializado com ativos: {self.asset_pairs}")
    
    def get_historical_data(self):
        """Simula a obtenção de dados históricos"""
        logger.info("Obtendo dados históricos simulados")
        
        # Cria dados simulados
        dados = []
        
        for asset in self.asset_pairs:
            for tf in [60, 300, 900]:  # 1m, 5m, 15m
                # Gera 100 registros para cada ativo e timeframe
                for i in range(100):
                    base_price = 1.1 if asset == "EURUSD" else 110.0
                    timestamp = int(time.time()) - (i * tf)
                    
                    # Simula variação de preço
                    open_price = base_price + (i * 0.001)
                    close_price = open_price + (0.002 * (i % 3 - 1))
                    high_price = max(open_price, close_price) + 0.001
                    low_price = min(open_price, close_price) - 0.001
                    
                    dados.append({
                        'id': i,
                        'from': timestamp,
                        'to': timestamp + tf,
                        'open': open_price,
                        'close': close_price,
                        'min': low_price,
                        'max': high_price,
                        'volume': 100 + (i * 10),
                        'asset': asset,
                        'timeframe': tf
                    })
        
        # Converte para DataFrame
        df = pd.DataFrame(dados)
        logger.info(f"Gerados {len(df)} registros de dados históricos simulados")
        
        # Renomeia colunas para corresponder ao formato esperado
        df = df.rename(columns={'min': 'low', 'max': 'high'})
        
        return df

# Simulação simplificada da classe Inteligencia
class InteligenciaSimulado:
    """Versão simplificada da classe Inteligencia para testes"""
    
    def __init__(self, historical_data_filename="dados_historicos_simulados.csv"):
        self.historical_data_filename = historical_data_filename
        self.historical_data = None
        self.logger = logger
        logger.info(f"Inteligencia simulada inicializada com arquivo: {historical_data_filename}")
    
    def get_historical_data(self, ferramental_instance):
        """Obtém dados históricos para treinamento utilizando a instância do Ferramental.
        
        Args:
            ferramental_instance: Instância da classe Ferramental
            
        Returns:
            DataFrame com dados históricos ou None em caso de erro
        """
        try:
            # Utiliza o método get_historical_data da classe Ferramental
            self.logger.info("Obtendo dados históricos através do Ferramental")
            historical_data = ferramental_instance.get_historical_data()
            
            if historical_data is None or historical_data.empty:
                self.logger.error("Não foi possível obter dados históricos")
                return None
                
            self.logger.info(f"Obtidos {len(historical_data)} registros de dados históricos")
            
            # Salva os dados históricos em um arquivo para uso futuro
            os.makedirs(os.path.dirname(self.historical_data_filename) if os.path.dirname(self.historical_data_filename) else '.', exist_ok=True)
            historical_data.to_csv(self.historical_data_filename, index=False)
            self.logger.info(f"Dados históricos salvos em {self.historical_data_filename}")
            
            self.historical_data = historical_data
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Erro ao obter dados históricos: {str(e)}")
            return None
    
    def update_historical_data(self, ferramental_instance, force_update=False):
        """Atualiza dados históricos periodicamente.
        
        Args:
            ferramental_instance: Instância da classe Ferramental
            force_update: Se True, força a atualização mesmo que o cache seja recente
            
        Returns:
            DataFrame com dados históricos atualizados ou None em caso de erro
        """
        try:
            # Verifica se já temos dados em cache e se são recentes
            cache_file = self.historical_data_filename
            cache_is_recent = False
            
            if os.path.exists(cache_file) and not force_update:
                # Verifica a idade do arquivo de cache
                file_time = os.path.getmtime(cache_file)
                current_time = time.time()
                hours_old = (current_time - file_time) / 3600
                
                # Cache é considerado recente se tiver menos de 24 horas
                cache_is_recent = hours_old < 24
                
                if cache_is_recent:
                    self.logger.info(f"Usando cache de dados históricos (atualizado há {hours_old:.1f} horas)")
                    try:
                        self.historical_data = pd.read_csv(cache_file)
                        return self.historical_data
                    except Exception as e:
                        self.logger.error(f"Erro ao ler cache: {str(e)}")
                        # Se falhar na leitura do cache, continuamos para baixar novos dados
            
            # Se não temos cache recente ou a leitura falhou, baixamos novos dados
            return self.get_historical_data(ferramental_instance)
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar dados históricos: {str(e)}")
            return None
    
    def preprocess_data(self, data=None):
        """Versão simplificada do pré-processamento de dados históricos.
        
        Args:
            data: DataFrame com dados históricos (opcional, usa self.historical_data se None)
            
        Returns:
            DataFrame pré-processado ou None em caso de erro
        """
        try:
            if data is None:
                if self.historical_data is None:
                    self.logger.error("Nenhum dado histórico disponível para pré-processamento")
                    return None
                data = self.historical_data.copy()
                
            self.logger.info(f"Iniciando pré-processamento de {len(data)} registros")
            
            # Verifica se as colunas necessárias estão presentes
            required_columns = ['open', 'close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Colunas obrigatórias ausentes: {missing_columns}")
                return None
                
            # Preenchimento de valores ausentes
            for col in data.columns:
                if data[col].isna().any():
                    if col in ['open', 'close', 'high', 'low']:
                        # Para dados de preço, preenchemos com o valor anterior
                        data[col] = data[col].fillna(method='ffill')
                    elif col == 'volume':
                        # Para volume, preenchemos com 0
                        data[col] = data[col].fillna(0)
                    else:
                        # Para outras colunas, preenchemos com a média
                        data[col] = data[col].fillna(data[col].mean())
            
            # Adiciona indicadores técnicos simulados
            data['sma_20'] = data['close'].rolling(window=20).mean().fillna(0)
            data['rsi_14'] = (data['close'] - data['open']).rolling(window=14).mean().fillna(0)
            
            self.logger.info(f"Pré-processamento concluído. Resultado: {len(data)} registros com {len(data.columns)} features")
            return data
            
        except Exception as e:
            self.logger.error(f"Erro no pré-processamento de dados: {str(e)}")
            return None

def teste_obter_dados_historicos():
    """Testa a obtenção de dados históricos através da integração Inteligencia-Ferramental"""
    logger.info("Iniciando teste de obtenção de dados históricos")
    
    # Inicializa as instâncias simuladas
    ferramental = FerramentalSimulado(asset_pairs=["EURUSD", "USDJPY"])
    inteligencia = InteligenciaSimulado(historical_data_filename="teste_dados_historicos.csv")
    
    # Obtém dados históricos
    logger.info("Obtendo dados históricos")
    dados_historicos = inteligencia.get_historical_data(ferramental)
    
    if dados_historicos is None or dados_historicos.empty:
        logger.error("Falha ao obter dados históricos")
        return False
    
    logger.info(f"Obtidos {len(dados_historicos)} registros de dados históricos")
    logger.info(f"Colunas disponíveis: {dados_historicos.columns.tolist()}")
    
    return True

def teste_atualizar_dados_historicos():
    """Testa a atualização de dados históricos"""
    logger.info("Iniciando teste de atualização de dados históricos")
    
    # Inicializa as instâncias simuladas
    ferramental = FerramentalSimulado(asset_pairs=["EURUSD", "USDJPY"])
    inteligencia = InteligenciaSimulado(historical_data_filename="teste_dados_historicos.csv")
    
    # Atualiza dados históricos
    logger.info("Atualizando dados históricos")
    dados_atualizados = inteligencia.update_historical_data(ferramental)
    
    if dados_atualizados is None or dados_atualizados.empty:
        logger.error("Falha ao atualizar dados históricos")
        return False
    
    logger.info(f"Atualizados {len(dados_atualizados)} registros de dados históricos")
    
    # Testa a funcionalidade de cache
    logger.info("Testando cache de dados históricos")
    dados_cache = inteligencia.update_historical_data(ferramental)
    
    if dados_cache is None or dados_cache.empty:
        logger.error("Falha ao obter dados do cache")
        return False
    
    logger.info(f"Obtidos {len(dados_cache)} registros do cache")
    
    return True

def teste_preprocessamento_dados():
    """Testa o pré-processamento de dados históricos"""
    logger.info("Iniciando teste de pré-processamento de dados")
    
    # Inicializa as instâncias simuladas
    ferramental = FerramentalSimulado(asset_pairs=["EURUSD", "USDJPY"])
    inteligencia = InteligenciaSimulado(historical_data_filename="teste_dados_historicos.csv")
    
    # Obtém dados históricos
    logger.info("Obtendo dados históricos")
    dados_historicos = inteligencia.get_historical_data(ferramental)
    
    if dados_historicos is None or dados_historicos.empty:
        logger.error("Falha ao obter dados históricos")
        return False
    
    # Pré-processa os dados
    logger.info("Pré-processando dados históricos")
    dados_preprocessados = inteligencia.preprocess_data(dados_historicos)
    
    if dados_preprocessados is None or dados_preprocessados.empty:
        logger.error("Falha ao pré-processar dados históricos")
        return False
    
    logger.info(f"Pré-processados {len(dados_preprocessados)} registros")
    logger.info(f"Colunas após pré-processamento: {dados_preprocessados.columns.tolist()}")
    
    # Verifica se os indicadores técnicos foram adicionados
    indicadores = ['sma_20', 'rsi_14']
    for indicador in indicadores:
        if indicador not in dados_preprocessados.columns:
            logger.error(f"Indicador {indicador} não encontrado após pré-processamento")
            return False
    
    logger.info("Todos os indicadores técnicos foram adicionados com sucesso")
    
    return True

def teste_pipeline_completo():
    """Testa o pipeline completo de obtenção, atualização e pré-processamento de dados"""
    logger.info("Iniciando teste do pipeline completo")
    
    # Inicializa as instâncias simuladas
    ferramental = FerramentalSimulado(asset_pairs=["EURUSD", "USDJPY"])
    inteligencia = InteligenciaSimulado(historical_data_filename="teste_pipeline_completo.csv")
    
    # Atualiza dados históricos
    logger.info("Atualizando dados históricos")
    dados_atualizados = inteligencia.update_historical_data(ferramental)
    
    if dados_atualizados is None or dados_atualizados.empty:
        logger.error("Falha ao atualizar dados históricos")
        return False
    
    # Pré-processa os dados
    logger.info("Pré-processando dados históricos")
    dados_preprocessados = inteligencia.preprocess_data()
    
    if dados_preprocessados is None or dados_preprocessados.empty:
        logger.error("Falha ao pré-processar dados históricos")
        return False
    
    logger.info(f"Pipeline completo executado com sucesso. Resultado: {len(dados_preprocessados)} registros processados")
    
    # Salva os dados processados para análise
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dados_preprocessados.to_csv(f"dados_processados_{timestamp}.csv", index=False)
    logger.info(f"Dados processados salvos em dados_processados_{timestamp}.csv")
    
    return True

if __name__ == "__main__":
    logger.info("==== INICIANDO TESTES DE INTEGRAÇÃO SIMULADOS ====")
    
    testes = [
        ("Obtenção de Dados Históricos", teste_obter_dados_historicos),
        ("Atualização de Dados Históricos", teste_atualizar_dados_historicos),
        ("Pré-processamento de Dados", teste_preprocessamento_dados),
        ("Pipeline Completo", teste_pipeline_completo)
    ]
    
    resultados = {}
    
    for nome, teste in testes:
        logger.info(f"\n==== TESTE: {nome} ====")
        try:
            resultado = teste()
            resultados[nome] = "SUCESSO" if resultado else "FALHA"
        except Exception as e:
            logger.error(f"Erro durante o teste {nome}: {str(e)}")
            resultados[nome] = "ERRO"
    
    logger.info("\n==== RESUMO DOS TESTES ====")
    for nome, resultado in resultados.items():
        logger.info(f"{nome}: {resultado}")
