import sys
import os
import pandas as pd
import logging
from datetime import datetime

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa as classes necessárias
from inteligencia.Inteligencia import Inteligencia
from ferramental.Ferramental import Ferramental

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teste_integracao.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TesteIntegracao')

def teste_obter_dados_historicos():
    """Testa a obtenção de dados históricos através da integração Inteligencia-Ferramental"""
    logger.info("Iniciando teste de obtenção de dados históricos")
    
    # Inicializa as instâncias
    ferramental = Ferramental(asset_pairs=["EURUSD", "USDJPY"])
    inteligencia = Inteligencia(historical_data_filename="teste_dados_historicos.csv")
    
    # Tenta conectar ao IQ Option
    logger.info("Conectando à API do IQ Option")
    status, msg = ferramental.connect()
    
    if not status:
        logger.error(f"Falha na conexão: {msg}")
        return False
    
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
    
    # Inicializa as instâncias
    ferramental = Ferramental(asset_pairs=["EURUSD", "USDJPY"])
    inteligencia = Inteligencia(historical_data_filename="teste_dados_historicos.csv")
    
    # Tenta conectar ao IQ Option
    logger.info("Conectando à API do IQ Option")
    status, msg = ferramental.connect()
    
    if not status:
        logger.error(f"Falha na conexão: {msg}")
        return False
    
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
    
    # Inicializa as instâncias
    ferramental = Ferramental(asset_pairs=["EURUSD", "USDJPY"])
    inteligencia = Inteligencia(historical_data_filename="teste_dados_historicos.csv")
    
    # Tenta conectar ao IQ Option
    logger.info("Conectando à API do IQ Option")
    status, msg = ferramental.connect()
    
    if not status:
        logger.error(f"Falha na conexão: {msg}")
        return False
    
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
    indicadores = ['sma_20', 'bollinger_upper', 'bollinger_lower', 'rsi_14', 'macd', 'stoch_k']
    for indicador in indicadores:
        if indicador not in dados_preprocessados.columns:
            logger.error(f"Indicador {indicador} não encontrado após pré-processamento")
            return False
    
    logger.info("Todos os indicadores técnicos foram adicionados com sucesso")
    
    return True

def teste_pipeline_completo():
    """Testa o pipeline completo de obtenção, atualização e pré-processamento de dados"""
    logger.info("Iniciando teste do pipeline completo")
    
    # Inicializa as instâncias
    ferramental = Ferramental(asset_pairs=["EURUSD", "USDJPY"])
    inteligencia = Inteligencia(historical_data_filename="teste_pipeline_completo.csv")
    
    # Tenta conectar ao IQ Option
    logger.info("Conectando à API do IQ Option")
    status, msg = ferramental.connect()
    
    if not status:
        logger.error(f"Falha na conexão: {msg}")
        return False
    
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
    
    # Cria rótulos para treinamento
    logger.info("Criando rótulos para treinamento")
    dados_rotulados = inteligencia.create_labels(dados_preprocessados)
    
    if dados_rotulados is None or dados_rotulados.empty:
        logger.error("Falha ao criar rótulos para treinamento")
        return False
    
    if 'label' not in dados_rotulados.columns:
        logger.error("Coluna de rótulos não encontrada após processamento")
        return False
    
    logger.info(f"Pipeline completo executado com sucesso. Resultado: {len(dados_rotulados)} registros com rótulos")
    
    # Salva os dados processados para análise
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dados_rotulados.to_csv(f"dados_processados_{timestamp}.csv", index=False)
    logger.info(f"Dados processados salvos em dados_processados_{timestamp}.csv")
    
    return True

if __name__ == "__main__":
    logger.info("==== INICIANDO TESTES DE INTEGRAÇÃO ====")
    
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
