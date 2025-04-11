#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import queue
import signal
import traceback
from sklearn.model_selection import train_test_split # Adicionado para divisão treino/avaliação

from dotenv import load_dotenv # Importar dotenv
# Importações locais
from utils.ConfigManager import ConfigManager
from utils.Logger import setup_logger
from utils.ErrorTracker import ErrorTracker
from utils.PerformanceTracker import PerformanceTracker
from ferramental.NovoFerramental import NovoFerramental # Alterado para usar a nova classe
from inteligencia.Inteligencia import Inteligencia

# logging.basicConfig removido para evitar duplicação com setup_logger

# Configuração de argumentos da linha de comando
parser = argparse.ArgumentParser(description='Bot IQ Option')
parser.add_argument('--config', type=str, default='config.ini', help='Caminho para o arquivo de configuração')
parser.add_argument('--mode', type=str, choices=['download', 'learning', 'test', 'real'], help='Modo de operação')
parser.add_argument('--assets', type=str, help='Lista de ativos separados por vírgula (ex: EURUSD,GBPUSD)')
parser.add_argument('--debug', action='store_true', help='Ativa modo de debug')
args = parser.parse_args()

# Configuração de logging
# Passa o nível DEBUG como override se args.debug for True, senão passa None (usará config)
# Configura o logger principal
logger = setup_logger('main', override_level=logging.DEBUG if args.debug else None)
# Configura também o logger da Inteligencia para garantir que o nível seja aplicado
setup_logger('Inteligencia', override_level=logging.DEBUG if args.debug else None)

# Carrega variáveis de ambiente do arquivo .env (se existir)
load_dotenv()

# Variáveis globais
stop_event = threading.Event()
performance_tracker = PerformanceTracker()
error_tracker = ErrorTracker()

def signal_handler(sig, frame):
    """Manipulador de sinais para interrupção do programa."""
    logger.info("Recebido sinal de interrupção. Encerrando...")
    stop_event.set()

# Registra manipuladores de sinais
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def display_banner():
    """Exibe o banner do bot."""
    banner = """
    ██████╗  ██████╗ ████████╗    ██╗ ██████╗      ██████╗ ██████╗ ████████╗    
    ██╔══██╗██╔═══██╗╚══██╔══╝    ██║██╔═══██╗    ██╔═══██╗██╔══██╗╚══██╔══╝    
    ██████╔╝██║   ██║   ██║       ██║██║   ██║    ██║   ██║██████╔╝   ██║       
    ██╔══██╗██║   ██║   ██║       ██║██║▄▄ ██║    ██║   ██║██╔═══╝    ██║       
    ██████╔╝╚██████╔╝   ██║       ██║╚██████╔╝    ╚██████╔╝██║        ██║       
    ╚═════╝  ╚═════╝    ╚═╝       ╚═╝ ╚══▀▀═╝      ╚═════╝ ╚═╝        ╚═╝       
"""
    print(banner)

def check_dependencies():
    """Verifica se todas as dependências estão instaladas."""
    missing_deps = []
    try:
        import pandas as pd
    except ImportError as e:
        missing_deps.append("pandas")
    
    try:
        import numpy as np
    except ImportError as e:
        missing_deps.append("numpy")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        missing_deps.append("matplotlib")
    
    try:
        import talib
    except ImportError as e:
        logger.warning("talib não está instalado. Usando alternativas para indicadores técnicos.")
        # Não adicionamos talib à lista de dependências ausentes, pois temos alternativas
    
    try:
        import sklearn
    except ImportError as e:
        missing_deps.append("scikit-learn")
    
    try:
        import torch
    except ImportError as e:
        missing_deps.append("torch")
    
    try:
        import iqoptionapi
    except ImportError as e:
        missing_deps.append("iqoptionapi")
    
    if missing_deps:
        logger.error(f"Dependências ausentes: {', '.join(missing_deps)}")
        logger.error("Dependências ausentes. Por favor, instale-as (ex: pip install -r requirements.txt).") # Mensagem mais clara
        return False
    else:
        logger.info("Todas as dependências estão instaladas.") # Traduzido
        return True

def create_directories():
    """Cria os diretórios necessários para o funcionamento do bot."""
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    logger.debug("Diretórios criados ou já existentes.") # Traduzido

def cleanup():
    """Realiza a limpeza antes de encerrar o programa."""
    try:
        logger.info("Realizando limpeza...") # Traduzido
        
        # Salva métricas de desempenho
        performance_tracker.save_to_file('performance_metrics.json')
        logger.info("Métricas de desempenho salvas em performance_metrics.json") # Traduzido
        
        # Salva erros registrados
        error_tracker.save_to_file('error_log.json')
        logger.info("Log de erros salvo em error_log.json") # Traduzido
        
    except Exception as e:
        logger.error(f"Erro durante a limpeza: {str(e)}") # Traduzido

def main():
    """Função principal do bot."""
    try:
        # Exibe o banner
        display_banner()
        
        # Registra início da execução
        logger.info("Bot IQ Option iniciado") # Traduzido
        
        # Verifica dependências
        if not check_dependencies():
             # Mensagem de erro já é dada em check_dependencies
             sys.exit("Encerrando devido a dependências ausentes.") # Traduzido
        
        # Cria diretórios necessários
        create_directories()
        
        # Carrega configurações
        config_file = args.config
        config_manager = ConfigManager(config_file)
        
        # --- Carregamento de Credenciais ---
        use_env = config_manager.get_value('Credentials', 'use_env_file', True, bool)
        iqoption_email = None
        iqoption_password = None

        # A lógica agora sempre tenta carregar do ambiente (que pode ter sido populado pelo .env ou externamente)
        logger.info("Tentando carregar credenciais das variáveis de ambiente (IQ_OPTION_EMAIL, IQ_OPTION_PASSWORD)") # Traduzido
        iqoption_email = os.getenv('IQ_OPTION_EMAIL')
        iqoption_password = os.getenv('IQ_OPTION_PASSWORD')

        if not iqoption_email or not iqoption_password:
            logger.critical("Credenciais IQ_OPTION_EMAIL ou IQ_OPTION_PASSWORD não encontradas nas variáveis de ambiente.") # Traduzido
            logger.critical("Verifique se o arquivo .env existe, está correto e contém as variáveis, ou se as variáveis de ambiente foram definidas externamente.") # Traduzido
            sys.exit("Erro Crítico: Credenciais ausentes. Encerrando.") # Traduzido
        else:
            # Log de sucesso, mas sem expor o email completo
            email_masked = iqoption_email[:3] + '***' + iqoption_email[iqoption_email.find('@'):] if '@' in iqoption_email else iqoption_email[:3] + '***'
            logger.info(f"Credenciais carregadas com sucesso para o usuário: {email_masked}") # Traduzido
            # Não logar a senha ou partes dela

        # O parâmetro 'use_env_file' no config.ini agora serve apenas como documentação/lembrete.
        # A remoção do bloco 'else' garante que o config.ini nunca seja usado para credenciais.
        # --- Inicializa componentes ---
        # Passa as credenciais carregadas para o Ferramental
        # Instancia o NovoFerramental (não precisa mais de config_manager e error_tracker no init)
        ferramental = NovoFerramental(email=iqoption_email, password=iqoption_password)
        inteligencia = Inteligencia(config_manager, error_tracker) # Passa o error_tracker global

        # --- Inicialização para aumento progressivo de dados ---
        initial_data_count = config_manager.get_value('Download', 'initial_historical_data_count', 1000, int)
        # max_data_count removido para permitir aumento indefinido
        current_historical_data_count = initial_data_count
        logger.info(f"Contagem inicial de dados históricos: {current_historical_data_count}") # Traduzido
        # --- Fim da inicialização ---

        # --- Inicialização para relaxamento de critérios ---
        learning_failures = 0 # Contador de falhas consecutivas no aprendizado
        min_accuracy = config_manager.get_value('AutoSwitch', 'min_accuracy', 0.75, float) # Usando valor atualizado
        min_precision = config_manager.get_value('AutoSwitch', 'min_precision', 0.75, float) # Usando valor atualizado
        min_recall = config_manager.get_value('AutoSwitch', 'min_recall', 0.75, float) # Usando valor atualizado
        min_f1_score = config_manager.get_value('AutoSwitch', 'min_f1_score', 0.75, float) # Usando valor atualizado
        min_trades_count = config_manager.get_value('AutoSwitch', 'min_trades_count', 20, int)
        min_win_rate = config_manager.get_value('AutoSwitch', 'min_win_rate', 0.75, float) # Usando valor atualizado
        min_profit = config_manager.get_value('AutoSwitch', 'min_profit', 0.0, float)
        relax_after_failures = config_manager.get_value('Learning', 'relax_after_failures', 5, int)
        criteria_relax_factor = config_manager.get_value('Learning', 'criteria_relax_factor', 0.95, float)
        max_learning_cycles = config_manager.get_value('Learning', 'max_learning_cycles', 5, int) # Lê o novo parâmetro
        learning_cycles_count = 0 # Inicializa o contador de ciclos
        # --- Fim da inicialização ---

        # Verifica conexão com a API
        logger.info("Verificando conexão inicial estabelecida pelo NovoFerramental...") # Traduzido
        if not ferramental.connected:
             logger.critical("Falha na conexão inicial com a API ao instanciar NovoFerramental.") # Traduzido
             logger.critical("Verifique as credenciais, conexão com a internet, status da conta ou necessidade de 2FA (não suportado automaticamente).") # Traduzido
             sys.exit("Erro Crítico: Falha na conexão inicial. Encerrando.") # Traduzido
        else:
            logger.info("Conexão inicial verificada com sucesso.") # Traduzido

        # Determina o modo de operação
        operation_mode = args.mode or config_manager.get_value('General', 'mode', 'test') # 'download', 'learning', 'test', 'real'
        
        # Determina os ativos para negociação
        # Determina os ativos para negociação (lógica movida para após a verificação de conexão)
        # A lista de ativos agora é gerenciada pelo main.py, não pelo ferramental.
        # Leitura dos ativos da config ou args
        assets_str = args.assets or config_manager.get_value('General', 'assets', 'eurusd')
        assets = [asset.strip().upper() for asset in assets_str.split(',') if asset.strip()] # Garante uppercase e remove vazios

        # Verifica se a lista ficou vazia após a leitura e processamento
        if not assets:
            logger.warning("Nenhum ativo válido encontrado na configuração ou argumentos. Usando 'EURUSD' como padrão.") # Traduzido
            assets = ['EURUSD']
        
        logger.info(f"Lista final de ativos para operação: {assets}") # Traduzido
        
        logger.info(f"Modo inicial definido como: {operation_mode}") # Traduzido
        logger.info(f"Ativos para negociação: {', '.join(assets)}") # Traduzido
        operation_mode = 'learning' # FORÇANDO MODO LEARNING TEMPORARIAMENTE
        logger.info(f"Forçando modo de operação para: {operation_mode}")

        # Loop principal
        while not stop_event.is_set():
            try:
                # Modo de Download de Dados Históricos
                if operation_mode == 'download':
                    logger.info("Iniciando modo Download...") # Traduzido
                    download_success_count = 0
                    for asset in assets:
                        logger.info(f"Baixando dados históricos para {asset}...") # Traduzido
                        try:
                            # Define parâmetros para get_historical_data
                            timeframe_seconds = config_manager.get_value('General', 'timeframe', 60, int)
                            if timeframe_seconds == 60: tf_type, tf_value = "Minutes", 1
                            elif timeframe_seconds == 300: tf_type, tf_value = "Minutes", 5
                            elif timeframe_seconds == 900: tf_type, tf_value = "Minutes", 15
                            elif timeframe_seconds == 3600: tf_type, tf_value = "Hours", 1
                            else: tf_type, tf_value = "Seconds", timeframe_seconds; logger.warning(f"Timeframe {timeframe_seconds}s não mapeado, usando Seconds.")

                            data_count = config_manager.get_value('Download', 'historical_data_count', 1000, int) # Quantidade de velas

                            # 1. Baixar dados
                            raw_data_list = ferramental.get_historical_data(
                                asset=asset,
                                timeframe_type=tf_type,
                                timeframe_value=tf_value,
                                count=data_count
                            )

                            logger.debug(f"Dados brutos recebidos (download) para {asset}: {raw_data_list}") # Log adicionado
                            if raw_data_list is None:
                                logger.error(f"Falha ao baixar dados para {asset}. API retornou None.") # Traduzido
                                continue
                            if not raw_data_list:
                                logger.warning(f"Nenhum dado histórico retornado para {asset}.") # Traduzido
                                continue

                            # 2. Converter para DataFrame e Salvar Dados Brutos
                            df_raw = pd.DataFrame(raw_data_list)
                            if 'from' in df_raw.columns and 'timestamp' not in df_raw.columns:
                                df_raw['timestamp'] = pd.to_datetime(df_raw['from'], unit='s')
                            elif 'id' in df_raw.columns and 'timestamp' not in df_raw.columns:
                                 df_raw['timestamp'] = pd.to_datetime(df_raw['id'], unit='s')

                            df_raw.rename(columns={'min': 'low', 'max': 'high'}, inplace=True)
                            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            if not all(col in df_raw.columns for col in required_cols):
                                logger.error(f"Dados brutos para {asset} não contêm todas as colunas necessárias ({required_cols}). Colunas presentes: {df_raw.columns.tolist()}") # Traduzido
                                continue

                            raw_filename = f"{asset}_{tf_value}{tf_type}_raw.csv"
                            raw_filepath = os.path.join('data', 'raw', raw_filename)
                            os.makedirs(os.path.dirname(raw_filepath), exist_ok=True)
                            df_raw.to_csv(raw_filepath, index=False)
                            logger.info(f"Dados brutos para {asset} salvos em {raw_filepath}") # Traduzido

                            # 3. Processar Dados (chamar Inteligencia)
                            logger.info(f"Processando dados para {asset}...") # Traduzido
                            df_processed = inteligencia.process_historical_data(
                                data=df_raw.copy(),
                                asset_name=asset,
                                timeframe=timeframe_seconds
                            )

                            # 4. Salvar Dados Processados
                            if df_processed is not None and not df_processed.empty:
                                processed_filename = f"{asset}_{tf_value}{tf_type}_processed.csv"
                                processed_filepath = os.path.join('data', 'processed', processed_filename)
                                os.makedirs(os.path.dirname(processed_filepath), exist_ok=True)
                                df_processed.to_csv(processed_filepath, index=False)
                                logger.info(f"Dados processados para {asset} salvos em {processed_filepath}") # Traduzido
                                download_success_count += 1
                            else:
                                logger.error(f"Falha ao processar dados para {asset}.") # Traduzido

                        except Exception as download_err:
                            logger.exception(f"Erro durante o download/processamento para {asset}: {download_err}") # Traduzido
                            error_tracker.add_error(f"DownloadError_{asset}", str(download_err), traceback.format_exc())

                    if download_success_count > 0:
                        logger.info(f"Download e processamento concluídos para {download_success_count} ativo(s). Mudando para modo Aprendizado.") # Traduzido
                        operation_mode = 'learning' # Muda para o modo de aprendizado
                    else:
                        logger.error("Nenhum dado foi baixado ou processado com sucesso. Verifique os logs. O bot permanecerá em modo Download ou será encerrado.") # Traduzido
                        stop_event.set()
                
                # Modo de Aprendizado
                elif operation_mode == 'learning':
                    logger.info("Iniciando modo Aprendizado...") # Traduzido
                    
                    try:
                        # Lista para armazenar dados processados de todos os ativos
                        all_processed_data = []
                        
                        # Process historical data for each asset
                        for asset in assets:
                            logger.info(f"Processando dados históricos para {asset}") # Traduzido
                            
                            # --- Otimização: Verificar se dados processados já existem ---
                            timeframe_seconds = config_manager.get_value('General', 'timeframe', 60, int)
                            processed_dir = os.path.join('data', 'processed')
                            # Usa timeframe_seconds no nome para consistência com como é salvo em Inteligencia.py
                            processed_filename = f"{asset.lower()}_{timeframe_seconds}_processed.csv"
                            processed_filepath = os.path.join(processed_dir, processed_filename)

                            data_processed = None # Inicializa data_processed

                            if os.path.exists(processed_filepath):
                                try:
                                    logger.info(f"Carregando dados processados existentes de {processed_filepath}")
                                    # Garante que o timestamp seja parseado corretamente ao carregar
                                    data_processed = pd.read_csv(processed_filepath, parse_dates=['timestamp'])
                                    # Verifica se carregou corretamente e tem as colunas esperadas (opcional, mas bom)
                                    if data_processed.empty or 'timestamp' not in data_processed.columns:
                                         logger.warning(f"Arquivo processado {processed_filepath} está vazio ou inválido. Tentando baixar e processar novamente.")
                                         data_processed = None # Força o reprocessamento
                                    else:
                                         logger.info(f"Dados processados para {asset} carregados com sucesso.")
                                except Exception as load_err:
                                    logger.error(f"Erro ao carregar arquivo processado {processed_filepath}: {load_err}. Tentando baixar e processar novamente.")
                                    data_processed = None # Força o reprocessamento

                            # Se os dados processados não foram carregados (não existem ou erro ao carregar), baixa e processa
                            if data_processed is None:
                                logger.info(f"Arquivo processado não encontrado ou inválido para {asset}. Baixando e processando...")
                                # Define parâmetros para get_historical_data no modo learning
                                if timeframe_seconds == 60: tf_type, tf_value = "Minutes", 1
                                elif timeframe_seconds == 300: tf_type, tf_value = "Minutes", 5
                                elif timeframe_seconds == 900: tf_type, tf_value = "Minutes", 15
                                elif timeframe_seconds == 3600: tf_type, tf_value = "Hours", 1
                                else: tf_type, tf_value = "Seconds", timeframe_seconds; logger.warning(f"Timeframe {timeframe_seconds}s não mapeado, usando Seconds.")

                                # Usa a contagem atual que aumenta progressivamente
                                data_count = current_historical_data_count
                                logger.info(f"Tentando baixar {data_count} velas para {asset}...") # Traduzido

                                # Chama get_historical_data com os argumentos corretos
                                historical_data_list = ferramental.get_historical_data(
                                    asset=asset,
                                    timeframe_type=tf_type,
                                    timeframe_value=tf_value,
                                    count=data_count
                                )
                                # Converte a lista retornada para DataFrame (ou None/vazio se falhar)
                                # --- DEBUG LOGS: Início ---
                                logger.debug(f"Dados brutos recebidos (learning) para {asset}: {historical_data_list}") # Log adicionado
                                logger.debug(f"[DEBUG] Tipo de historical_data_list para {asset}: {type(historical_data_list)}")
                                if historical_data_list:
                                    logger.debug(f"[DEBUG] Primeiro elemento de historical_data_list: {historical_data_list[0]}")
                                    historical_data = pd.DataFrame(historical_data_list)
                                    # Garante que as colunas 'min' e 'max' sejam renomeadas se existirem
                                    historical_data.rename(columns={'min': 'low', 'max': 'high'}, inplace=True)
                                    # --- DEBUG LOGS: Fim --- # Movido para o final do bloco if
                                else:
                                    historical_data = pd.DataFrame() # Cria DataFrame vazio se a lista for None ou vazia

                                # --- DEBUG LOGS: Início --- # Movido para antes da verificação de erro
                                if not historical_data.empty:
                                    logger.debug(f"[DEBUG] Colunas do DataFrame 'historical_data' para {asset} APÓS conversão inicial: {historical_data.columns.tolist()}")
                                    logger.debug(f"[DEBUG] Primeiros 3 registros de 'historical_data' para {asset}:\\n{historical_data.head(3).to_string()}")
                                # --- DEBUG LOGS: Fim ---

                                if historical_data.empty: # Verifica se está vazio após as tentativas
                                    logger.error(f"Falha ao obter dados históricos para {asset} (DataFrame vazio).") # Traduzido e ajustado
                                    continue # Pula para o próximo ativo

                                # --- Tratamento de Timestamp e Colunas Essenciais ---
                                if 'from' in historical_data.columns and 'timestamp' not in historical_data.columns:
                                    historical_data['timestamp'] = pd.to_datetime(historical_data['from'], unit='s')
                                elif 'id' in historical_data.columns and 'timestamp' not in historical_data.columns:
                                     historical_data['timestamp'] = pd.to_datetime(historical_data['id'], unit='s')

                                # Renomeia min/max ANTES da verificação de colunas essenciais
                                historical_data.rename(columns={'min': 'low', 'max': 'high'}, inplace=True)

                                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                                if not all(col in historical_data.columns for col in required_cols):
                                    logger.error(f"Dados históricos para {asset} (modo learning) não contêm todas as colunas necessárias ({required_cols}) após tentativa de criação do timestamp. Colunas presentes: {historical_data.columns.tolist()}") # Traduzido
                                    continue # Pula para o próximo ativo
                                # --- Fim Tratamento ---

                                # Processar os dados históricos (agora com save_processed=True por padrão na função)
                                # A função process_historical_data já salva o arquivo se tiver sucesso
                                data_processed = inteligencia.process_historical_data(
                                    data=historical_data.copy(), # Passa cópia para evitar modificar o original aqui
                                    asset_name=asset,
                                    timeframe=timeframe_seconds # Passa timeframe em segundos
                                )
                            
                            # Verificar se os dados foram processados corretamente e adicionar à lista
                            if data_processed is not None and not data_processed.empty:
                                # Adicionar coluna 'asset' para diferenciar no treinamento, se necessário
                                data_processed['asset'] = asset
                                all_processed_data.append(data_processed)
                                logger.info(f"Dados processados para {asset} adicionados para treinamento.") # Traduzido
                            else:
                                logger.error(f"Falha ao processar dados históricos para {asset} ou dados estão vazios.") # Traduzido
                                continue # Pula para o próximo ativo se o processamento falhar

                        # Verificar se há dados processados para treinar
                        if not all_processed_data:
                            logger.error("Nenhum dado processado disponível para treinar o modelo. Pulando treinamento.") # Traduzido
                            time.sleep(60) # Espera antes de tentar o ciclo de aprendizado novamente
                            continue

                        # Concatenar dados de todos os ativos
                        logger.info(f"Concatenando dados processados de {len(all_processed_data)} ativo(s).") # Traduzido
                        full_processed_data = pd.concat(all_processed_data, ignore_index=True)

                        # Divide os dados concatenados em treino e avaliação
                        test_split_size = config_manager.get_value('Learning', 'test_size', 0.2, float)
                        logger.info(f"Dividindo dados em conjuntos de treinamento ({1-test_split_size:.0%}) e avaliação ({test_split_size:.0%}).") # Traduzido
                        # Usando divisão cronológica (shuffle=False) para séries temporais
                        train_set, eval_set = train_test_split(full_processed_data, test_size=test_split_size, shuffle=False)

                        if train_set.empty or eval_set.empty:
                             logger.error("Falha ao dividir dados ou um dos conjuntos está vazio. Pulando treinamento.") # Traduzido
                             time.sleep(60)
                             continue

                        # Atualiza os critérios na instância da Inteligencia antes de treinar/avaliar
                        inteligencia.update_auto_switch_criteria(
                            min_accuracy=min_accuracy,
                            min_precision=min_precision,
                            min_recall=min_recall,
                            min_f1_score=min_f1_score,
                            min_trades_count=min_trades_count,
                            min_win_rate=min_win_rate,
                            min_profit=min_profit,
                            auto_switch_to_real=config_manager.get_value('AutoSwitch', 'auto_switch_to_real', False, bool)
                        )

                        # Train model using the actual 'train' function name and split data
                        logger.info(f"Treinando modelo com dados de treinamento ({len(train_set)} registros)") # Traduzido
                        training_history = inteligencia.train(
                            train_data=train_set,
                            val_data=eval_set, # Passa o conjunto de validação explicitamente
                            epochs=config_manager.get_value('Model', 'epochs', 100, int),
                            learning_rate=config_manager.get_value('Model', 'learning_rate', 0.001, float),
                            sequence_length=config_manager.get_value('Learning', 'sequence_length', 20, int),
                            test_size=0, # Define test_size como 0 pois já passamos val_data
                            early_stopping=config_manager.get_value('Learning', 'early_stopping', True, bool),
                            patience=config_manager.get_value('Learning', 'patience', 10, int)
                        )

                        if training_history is None: # A função train retorna None em caso de erro
                            logger.error("Falha ao treinar o modelo") # Traduzido
                            time.sleep(60)  # Espera antes de tentar novamente
                            continue

                        # Avalia o modelo no conjunto de avaliação
                        logger.info(f"Avaliando desempenho do modelo com dados de avaliação ({len(eval_set)} registros)") # Traduzido
                        evaluation_metrics = inteligencia.evaluate_model(test_data=eval_set) # Passa eval_set para avaliação
                        
                        if evaluation_metrics:
                            # Logar métricas como dicionário (sem json.dumps direto para ndarray)
                            logger.info(f"Métricas de avaliação do modelo: {evaluation_metrics}") # Traduzido
                            
                            # Verifica se deve mudar para modo de teste
                            if inteligencia.should_switch_to_test_mode():
                                logger.info("Desempenho do modelo é bom. Mudando para modo Teste") # Traduzido
                                operation_mode = 'test'
                                learning_failures = 0 # Reseta contador de falhas
                            else:
                                logger.info("Desempenho do modelo não é bom o suficiente para o modo Teste. Continuando aprendizado") # Traduzido
                                learning_failures += 1
                                logger.info(f"Contagem de falhas no aprendizado: {learning_failures}") # Traduzido
                                learning_cycles_count += 1 # Incrementa o contador de ciclos
                                logger.info(f"Ciclo de aprendizado {learning_cycles_count}/{max_learning_cycles if max_learning_cycles > 0 else 'ilimitados'} concluído (sem sucesso).") # Log ajustado

                                # Verifica se é hora de relaxar os critérios
                                if learning_failures >= relax_after_failures:
                                    logger.warning(f"Modelo falhou {learning_failures} vezes. Relaxando critérios de avaliação em {(1-criteria_relax_factor)*100:.1f}%.") # Traduzido
                                    min_accuracy *= criteria_relax_factor
                                    min_precision *= criteria_relax_factor
                                    min_recall *= criteria_relax_factor
                                    min_f1_score *= criteria_relax_factor
                                    min_win_rate *= criteria_relax_factor
                                    # min_profit pode ser mantido ou ajustado de outra forma, por exemplo, para aceitar pequenas perdas
                                    # min_profit *= criteria_relax_factor # Exemplo: Reduzir o lucro mínimo esperado
                                    logger.info(f"Novos critérios - Acurácia: {min_accuracy:.3f}, Precisão: {min_precision:.3f}, Recall: {min_recall:.3f}, F1: {min_f1_score:.3f}, Taxa de Acerto: {min_win_rate:.3f}") # Traduzido
                                    learning_failures = 0 # Reseta o contador após relaxar

                                # Lê o intervalo de nova tentativa do config, default 60s
                                retry_interval = config_manager.get_value('Learning', 'learning_retry_interval', 60, int)
                                logger.info(f"Pausando por {retry_interval} segundos antes de tentar o próximo ciclo de aprendizado...") # Traduzido
                                time.sleep(retry_interval)
                                logger.info("Pausa concluída. Iniciando novo ciclo de aprendizado.") # Traduzido
                                # Dobra a quantidade de dados para o próximo ciclo (sem limite máximo)
                                current_historical_data_count *= 2
                                logger.info(f"Dobrando quantidade de dados históricos para o próximo ciclo: {current_historical_data_count} velas") # Traduzido
                                # Verifica se o limite de ciclos foi atingido (REMOVIDO PARA CICLOS ILIMITADOS)
                                # if max_learning_cycles > 0 and learning_cycles_count >= max_learning_cycles:
                                #     logger.warning(f"Número máximo de ciclos de aprendizado ({max_learning_cycles}) atingido. Encerrando.") # Traduzido
                                #     stop_event.set()
                        else:
                            logger.error("Falha ao avaliar o modelo") # Traduzido
                            time.sleep(60)
                    
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.error(f"Erro durante o aprendizado: {str(e)}\n{tb_str}") # Traduzido
                        error_tracker.add_error("LearningModeError", str(e), tb_str)
                        time.sleep(60)
                
                # Modo de Teste
                elif operation_mode == 'test':
                    logger.info("Iniciando modo Teste...") # Traduzido
                    
                    # Modo Teste opera na conta PRACTICE se conectado, ou em simulação se a conexão falhou.
                    # A simulação é configurada na falha de conexão inicial.
                    
                    # Configura a inteligência para o modo de teste
                    inteligencia.setup_test_mode()
                    
                    # Logar o contexto de execução do modo Teste
                    if hasattr(ferramental, 'simulation_mode') and ferramental.simulation_mode:
                        logger.info("Modo Teste operando em MODO DE SIMULAÇÃO (conexão API falhou).") # Traduzido
                    elif ferramental.connected:
                         logger.info("Modo Teste operando conectado à conta PRACTICE/DEMO.") # Traduzido
                    else:
                         logger.warning("Modo Teste em estado inesperado (nem simulação, nem conectado).") # Traduzido
                    
                    # Executa negociações em modo de teste
                    try:
                        # Obtém dados em tempo real
                        for asset in assets:
                            logger.info(f"Obtendo dados históricos recentes para {asset} (para previsão)") # Traduzido
                            # Adaptado: Usar get_historical_data para obter dados recentes para previsão
                            sequence_length = config_manager.get_value('Learning', 'sequence_length', 20, int)
                            # Aumenta a quantidade de velas para garantir dados suficientes para preprocessamento
                            candles_to_fetch = max(250, sequence_length + 50) # Pega pelo menos 250 velas
                            timeframe_seconds = config_manager.get_value('General', 'timeframe', 60, int)
                            if timeframe_seconds == 60: tf_type, tf_value = "Minutes", 1
                            elif timeframe_seconds == 300: tf_type, tf_value = "Minutes", 5
                            else: tf_type, tf_value = "Seconds", timeframe_seconds; logger.warning(f"Timeframe {timeframe_seconds}s não mapeado, usando Seconds.")

                            recent_candles_list = ferramental.get_historical_data(asset, tf_type, tf_value, candles_to_fetch)
                            realtime_data = pd.DataFrame(recent_candles_list) if recent_candles_list else pd.DataFrame() # Converte para DataFrame

                            if realtime_data.empty: # Verifica se o DataFrame está vazio
                                logger.error(f"Falha ao obter dados em tempo real para {asset} (modo teste)") # Traduzido
                                continue
                            
                            # Renomeia colunas e garante timestamp (como nos modos download/learning)
                            if 'from' in realtime_data.columns and 'timestamp' not in realtime_data.columns:
                                realtime_data['timestamp'] = pd.to_datetime(realtime_data['from'], unit='s')
                            elif 'id' in realtime_data.columns and 'timestamp' not in realtime_data.columns:
                                 realtime_data['timestamp'] = pd.to_datetime(realtime_data['id'], unit='s')
                            realtime_data.rename(columns={'min': 'low', 'max': 'high'}, inplace=True)
                            
                            # Pré-processa os dados recentes da mesma forma que os dados de treinamento
                            # CORRIGIDO: Chama process_historical_data em vez de preprocess_data
                            processed_realtime_data = inteligencia.process_historical_data(realtime_data.copy(), asset_name=asset, timeframe=timeframe_seconds, save_processed=False)
                            if processed_realtime_data is None or processed_realtime_data.empty:
                                logger.error(f"Falha ao pré-processar dados em tempo real para {asset}") # Traduzido
                                continue
                            
                            # Faz previsões
                            logger.info(f"Realizando previsões para {asset}") # Traduzido
                            # Lê o confidence_threshold da configuração
                            confidence_threshold = config_manager.get_value('Trading', 'confidence_threshold', 0.75, float) # Usando 0.75 como padrão
                            # Usa os dados processados para previsão
                            predictions = inteligencia.predict(processed_realtime_data, confidence_threshold=confidence_threshold)
                            
                            if predictions is None:
                                logger.error(f"Falha ao realizar previsões para {asset}") # Traduzido
                                continue
                            
                            # Processa previsões e executa operações
                            action = predictions.get('action')
                            confidence = predictions.get('confidence', 0)
                            
                            if action == 'BUY' or action == 'SELL':
                                mapped_action = 'call' if action == 'BUY' else 'put'
                                amount = config_manager.get_value('Trading', 'amount', 10.0, float)
                                # Ler a duração da expiração em minutos do config
                                duration_minutes = config_manager.get_value('Trading', 'expiration', 1, int)
                                
                                logger.info(f"Executando operação de TESTE para {asset}: {mapped_action} ${amount}, Duração: {duration_minutes} min (Confiança: {confidence:.2f})") # Log atualizado
                                
                                # Chama a função 'buy' que é a correta para executar ordens binárias
                                # No modo teste, Ferramental.buy deve usar a conta DEMO ou simulação interna
                                success, order_id = ferramental.buy(
                                    asset=asset,
                                    amount=amount,
                                    action=mapped_action,
                                    duration=duration_minutes # Passa a duração em minutos
                                )
                                
                                if not success:
                                    logger.error(f"Falha ao executar operação de TESTE para {asset}") # Traduzido
                                    # Não necessariamente 'continue', pode ser interessante logar a falha e prosseguir
                            elif action == 'HOLD':
                                logger.info(f"Previsão para {asset} é MANTER (Confiança: {confidence:.2f}). Nenhuma operação executada.") # Traduzido
                            else:
                                logger.warning(f"Ação de previsão desconhecida para {asset}: {action}") # Traduzido
                        
                        # Verifica resultados das operações (método removido do NovoFerramental)
                        # trade_results = ferramental.get_trade_results() # Comentado
                        
                        # Bloco if trade_results: comentado pois a variável foi removida
                        # if trade_results:
                        #     # Processar resultados para atualizar gerenciamento de risco
                        #     current_balance = ferramental.get_balance()
                        #     if current_balance and current_balance > 0:
                        #         for result in trade_results:
                        #             # Verifica se a operação resultou em perda
                        #             # Usamos 'profit' < 0 como indicador primário, ou 'is_win' se disponível
                        #             is_loss = result.get('profit', 0) < 0
                        #             # Fallback para 'is_win' se 'profit' não for conclusivo (ex: 0.0)
                        #             if result.get('profit') == 0 and 'is_win' in result:
                        #                 is_loss = not result.get('is_win')
                        #
                        #             if is_loss:
                        #                 # A lógica de risk_management foi removida do NovoFerramental
                        #                 # ferramental.risk_management['consecutive_losses'] += 1
                        #                 # loss_amount = abs(result.get('profit', 0)) if result.get('profit', 0) < 0 else result.get('amount', 0)
                        #                 # ferramental.risk_management['daily_loss'] += loss_amount / current_balance
                        #                 # logger.warning(f"Loss detected. Consecutive losses: {ferramental.risk_management['consecutive_losses']}, Daily loss: {ferramental.risk_management['daily_loss']:.2%}")
                        #                 pass # Placeholder para lógica futura se necessário
                        #             elif result.get('profit', -1) > 0 or result.get('is_win', False):
                        #                 # ferramental.risk_management['consecutive_losses'] = 0
                        #                 pass # Placeholder
                        #     else:
                        #         logger.error("Não foi possível obter saldo para atualizar métricas de risco diário.")
                        #
                        #
                        #     # Atualiza métricas de desempenho
                        #     performance_tracker.update(trade_results) # Precisa ser adaptado para usar check_win
                        #
                        #     # Verifica se deve mudar para modo real
                        #     if inteligencia.should_switch_to_real_mode(performance_tracker):
                        #         if config_manager.get_value('AutoSwitch', 'auto_switch_to_real', False, bool):
                        #             logger.info("Performance is good. Switching to real mode")
                        #             operation_mode = 'real'
                        #         else:
                        #             logger.info("Performance is good for real mode, but auto-switch is disabled")
                        
                        # Espera antes da próxima iteração de teste
                        test_interval = config_manager.get_value('General', 'test_interval', 60, int)
                        logger.info(f"Aguardando {test_interval} segundos para o próximo ciclo de teste...") # Traduzido
                        time.sleep(test_interval)
                    
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.error(f"Erro durante o modo de teste: {str(e)}\n{tb_str}") # Traduzido
                        error_tracker.add_error("TestModeError", str(e), tb_str)
                        time.sleep(60)
                
                # Modo Real
                elif operation_mode == 'real':
                    logger.info("Iniciando modo Real...") # Traduzido
                    
                    # Verifica se o modo real está habilitado nas configurações
                    if not config_manager.get_value('General', 'enable_real_mode', False, bool):
                        logger.warning("Modo Real está desabilitado na configuração. Mudando para modo Teste") # Traduzido
                        operation_mode = 'test'
                        continue
                    
                    # Verifica conexão com a API
                    if not ferramental.check_connection():
                        logger.warning("Não conectado à API IQ Option. Não é possível operar em modo Real") # Traduzido
                        logger.info("Mudando para modo Teste") # Traduzido
                        operation_mode = 'test'
                        continue
                    
                    # Executa negociações em modo real
                    try:
                        # Obtém dados em tempo real
                        for asset in assets:
                            logger.info(f"Obtendo dados históricos recentes para {asset} (para previsão)") # Traduzido
                            # Adaptado: Usar get_historical_data para obter dados recentes para previsão
                            sequence_length = config_manager.get_value('Learning', 'sequence_length', 20, int)
                            # Aumenta a quantidade de velas para garantir dados suficientes para preprocessamento
                            candles_to_fetch = max(250, sequence_length + 50) # Pega pelo menos 250 velas
                            timeframe_seconds = config_manager.get_value('General', 'timeframe', 60, int)
                            if timeframe_seconds == 60: tf_type, tf_value = "Minutes", 1
                            elif timeframe_seconds == 300: tf_type, tf_value = "Minutes", 5
                            else: tf_type, tf_value = "Seconds", timeframe_seconds; logger.warning(f"Timeframe {timeframe_seconds}s não mapeado, usando Seconds.")

                            recent_candles_list = ferramental.get_historical_data(asset, tf_type, tf_value, candles_to_fetch)
                            realtime_data = pd.DataFrame(recent_candles_list) if recent_candles_list else pd.DataFrame() # Converte para DataFrame

                            if realtime_data.empty: # Verifica se o DataFrame está vazio
                                logger.error(f"Falha ao obter dados em tempo real para {asset} (modo real)") # Traduzido
                                continue

                            # Renomeia colunas e garante timestamp (como nos modos download/learning)
                            if 'from' in realtime_data.columns and 'timestamp' not in realtime_data.columns:
                                realtime_data['timestamp'] = pd.to_datetime(realtime_data['from'], unit='s')
                            elif 'id' in realtime_data.columns and 'timestamp' not in realtime_data.columns:
                                 realtime_data['timestamp'] = pd.to_datetime(realtime_data['id'], unit='s')
                            realtime_data.rename(columns={'min': 'low', 'max': 'high'}, inplace=True)
                            
                            # Pré-processa os dados recentes da mesma forma que os dados de treinamento
                            # CORRIGIDO: Chama process_historical_data em vez de preprocess_data
                            processed_realtime_data = inteligencia.process_historical_data(realtime_data.copy(), asset_name=asset, timeframe=timeframe_seconds, save_processed=False)
                            if processed_realtime_data is None or processed_realtime_data.empty:
                                logger.error(f"Falha ao pré-processar dados em tempo real para {asset}") # Traduzido
                                continue
                                
                            # Faz previsões
                            logger.info(f"Realizando previsões para {asset}") # Traduzido
                            # Usa os dados processados para previsão (passando confidence_threshold)
                            confidence_threshold = config_manager.get_value('Trading', 'confidence_threshold', 0.75, float) # Usa o threshold configurado
                            predictions = inteligencia.predict(processed_realtime_data, confidence_threshold=confidence_threshold)
                            
                            if predictions is None:
                                logger.error(f"Falha ao realizar previsões para {asset}") # Traduzido
                                continue
                            
                            # Processa previsões e executa operações REAIS
                            action = predictions.get('action')
                            confidence = predictions.get('confidence', 0)

                            if action == 'BUY' or action == 'SELL':
                                mapped_action = 'call' if action == 'BUY' else 'put'
                                amount = config_manager.get_value('Trading', 'amount', 10.0, float)
                                # Ler a duração da expiração em minutos do config
                                duration_minutes = config_manager.get_value('Trading', 'expiration', 1, int)

                                logger.info(f"Executando operação REAL para {asset}: {mapped_action} ${amount}, Duração: {duration_minutes} min (Confiança: {confidence:.2f})") # Log atualizado
                                
                                # Chama a função 'buy' para executar ordens binárias reais
                                success, order_id = ferramental.buy(
                                    asset=asset,
                                    amount=amount,
                                    action=mapped_action,
                                    duration=duration_minutes # Passa a duração em minutos
                                )

                                if not success:
                                     logger.error(f"Falha ao executar operação REAL para {asset}") # Traduzido
                                     # Não necessariamente 'continue', pode ser interessante logar a falha e prosseguir
                            elif action == 'HOLD':
                                logger.info(f"Previsão para {asset} é MANTER (Confiança: {confidence:.2f}). Nenhuma operação executada.") # Traduzido
                            else:
                                logger.warning(f"Ação de previsão desconhecida para {asset}: {action}") # Traduzido
                        
                        # Verifica resultados das operações (método removido do NovoFerramental)
                        # trade_results = ferramental.get_trade_results() # Comentado
                        # Removido: Lógica de get_trade_results e gerenciamento de risco interno do ferramental antigo.
                        # A verificação de resultados (check_win) precisaria ser implementada aqui
                        # de forma assíncrona ou em um loop separado, aguardando a expiração.
                        # O PerformanceTracker também precisaria ser atualizado com base nos resultados do check_win.
                        # Exemplo de como poderia ser feito (requer mais lógica):
                        # if success and order_id:
                        #    # Armazenar order_id para verificar depois
                        #    # Em outro loop/thread:
                        #    # time.sleep(duration * 60 + 10) # Espera expiração + margem
                        #    # status, profit = ferramental.check_win(order_id)
                        #    # if status:
                        #    #     performance_tracker.add_trade(profit > 0, profit) # Exemplo
                        #    #     # Lógica de gerenciamento de risco aqui
                        #    #     # Verificar should_stay_in_real_mode
                        
                        # Espera antes da próxima iteração real
                        real_interval = config_manager.get_value('General', 'real_interval', 60, int)
                        logger.info(f"Aguardando {real_interval} segundos para o próximo ciclo real...") # Traduzido
                        time.sleep(real_interval)
                    
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.error(f"Erro durante o modo real: {str(e)}\n{tb_str}") # Traduzido
                        error_tracker.add_error("RealModeError", str(e), tb_str)
                        time.sleep(60)
                
                else:
                    logger.error(f"Modo de operação desconhecido: {operation_mode}") # Traduzido
                    break
            
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Erro no loop principal: {str(e)}\n{tb_str}") # Traduzido
                error_tracker.add_error("MainLoopError", str(e), tb_str)
                time.sleep(60)
        
        logger.info("Bot encerrado graciosamente") # Traduzido
    
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.critical(f"Erro crítico: {str(e)}\n{tb_str}") # Traduzido
        # Tenta registrar o erro crítico, mas pode falhar se o tracker não foi inicializado
        try:
            error_tracker.add_error("CriticalError", str(e), tb_str, critical=True)
        except NameError: # Caso o erro ocorra antes da inicialização do error_tracker
             logger.error("Rastreador de erros não inicializado, não foi possível registrar o erro crítico.") # Traduzido
        except Exception as tracker_err:
             logger.error(f"Falha ao registrar erro crítico no rastreador: {tracker_err}") # Traduzido
    
    finally:
        # Realiza limpeza final
        cleanup()

if __name__ == "__main__":
    main()
