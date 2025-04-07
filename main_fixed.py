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

# Importações locais
from utils.ConfigManager import ConfigManager
from utils.Logger import setup_logger
from utils.ErrorTracker import ErrorTracker
from utils.PerformanceTracker import PerformanceTracker
from ferramental.Ferramental import Ferramental
from inteligencia.Inteligencia import Inteligencia

# Configuração de argumentos da linha de comando
parser = argparse.ArgumentParser(description='Bot IQ Option')
parser.add_argument('--config', type=str, default='config.ini', help='Caminho para o arquivo de configuração')
parser.add_argument('--mode', type=str, choices=['download', 'learning', 'test', 'real'], help='Modo de operação')
parser.add_argument('--assets', type=str, help='Lista de ativos separados por vírgula (ex: EURUSD,GBPUSD)')
parser.add_argument('--debug', action='store_true', help='Ativa modo de debug')
args = parser.parse_args()

# Configuração de logging
logger = setup_logger('main', level=logging.DEBUG if args.debug else logging.INFO)

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
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import talib
        import sklearn
        import torch
        import iqoptionapi
        logger.info("All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False

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
        
    logger.debug("Directories created")

def cleanup():
    """Realiza a limpeza antes de encerrar o programa."""
    try:
        logger.info("Performing cleanup")
        
        # Salva métricas de desempenho
        performance_tracker.save_to_file('performance_metrics.json')
        logger.info("Performance metrics saved to performance_metrics.json")
        
        # Salva erros registrados
        error_tracker.save_to_file('error_log.json')
        logger.info("Error log saved to error_log.json")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Função principal do bot."""
    try:
        # Exibe o banner
        display_banner()
        
        # Registra início da execução
        logger.info("Bot IQ Option iniciado")
        
        # Verifica dependências
        if not check_dependencies():
            logger.error("Dependências ausentes. Instalando...")
            # TODO: Implementar instalação automática de dependências
        
        # Cria diretórios necessários
        create_directories()
        
        # Carrega configurações
        config_file = args.config
        config_manager = ConfigManager(config_file)
        
        # Obtém credenciais da IQ Option
        iqoption_username = config_manager.get_value('IQOption', 'username')
        iqoption_password = config_manager.get_value('IQOption', 'password')
        
        logger.info(f"Username from config: {iqoption_username}")
        logger.info(f"Password from config: {'*' * len(iqoption_password)}")
        
        # Inicializa componentes
        ferramental = Ferramental(config_manager)
        inteligencia = Inteligencia(config_manager)
        
        # Atualiza critérios de mudança automática
        inteligencia.update_auto_switch_criteria(
            min_accuracy=config_manager.get_value('AutoSwitch', 'min_accuracy', 0.7, float),
            min_precision=config_manager.get_value('AutoSwitch', 'min_precision', 0.65, float),
            min_recall=config_manager.get_value('AutoSwitch', 'min_recall', 0.65, float),
            min_f1_score=config_manager.get_value('AutoSwitch', 'min_f1_score', 0.65, float),
            min_trades_count=config_manager.get_value('AutoSwitch', 'min_trades_count', 20, int),
            min_win_rate=config_manager.get_value('AutoSwitch', 'min_win_rate', 0.6, float),
            min_profit=config_manager.get_value('AutoSwitch', 'min_profit', 0.0, float),
            auto_switch_to_real=config_manager.get_value('AutoSwitch', 'auto_switch_to_real', False, bool)
        )
        
        # Verifica conexão com a API
        logger.info("Verificando conexão com a API do IQ Option...")
        if not ferramental.check_connection():
            logger.warning("Falha na conexão inicial. Tentando reconectar...")
            
            # Tenta conectar com as credenciais do arquivo de configuração
            connection_success = ferramental.connect(
                username=iqoption_username,
                password=iqoption_password
            )
            
            if not connection_success:
                # Se falhar, tenta entrar em modo de simulação
                logger.warning("Falha na conexão com a API. Entrando em modo de simulação.")
                ferramental.setup_simulation_mode(
                    volatility=config_manager.get_value('Simulation', 'volatility', 0.01, float),
                    trend=config_manager.get_value('Simulation', 'trend', 0.0, float),
                    win_rate=config_manager.get_value('Simulation', 'win_rate', 0.6, float)
                )
        
        # Determina o modo de operação
        operation_mode = args.mode or config_manager.get_value('General', 'mode', 'test')
        
        # Determina os ativos para negociação
        assets_str = args.assets or config_manager.get_value('General', 'assets', 'eurusd')
        assets = [asset.strip().lower() for asset in assets_str.split(',')]
        
        logger.info(f"Starting in {operation_mode} mode")
        logger.info(f"Trading assets: {', '.join(assets)}")
        
        # Loop principal
        while not stop_event.is_set():
            try:
                # Modo de Download de Dados Históricos
                if operation_mode == 'download':
                    logger.info("Starting download mode")
                    
                    # Download de dados históricos para cada ativo
                    for asset in assets:
                        logger.info(f"Downloading historical data for {asset}")
                        historical_data = ferramental.download_historical_data(
                            asset=asset,
                            timeframe=config_manager.get_value('General', 'timeframe', 60, int),
                            start_date=config_manager.get_value('General', 'start_date'),
                            end_date=config_manager.get_value('General', 'end_date')
                        )
                        
                        if historical_data is None:
                            logger.error(f"Failed to download historical data for {asset}")
                            continue
                    
                    # Após download, muda para modo de aprendizado
                    logger.info("Download completed. Switching to learning mode")
                    operation_mode = 'learning'
                
                # Modo de Aprendizado
                elif operation_mode == 'learning':
                    logger.info("Starting learning mode")
                    
                    try:
                        # Process historical data
                        for asset in assets:
                            logger.info(f"Processing historical data for {asset}")
                            
                            # Obter dados históricos do ativo
                            historical_data = ferramental.get_historical_data(
                                asset=asset,
                                timeframe=config_manager.get_value('General', 'timeframe', 60, int),
                                count=config_manager.get_value('General', 'historical_data_count', 5000, int)
                            )
                            
                            if historical_data is None or len(historical_data) == 0:
                                logger.error(f"Failed to get historical data for {asset}")
                                continue
                                
                            # Processar os dados históricos
                            data_processed = inteligencia.process_historical_data(
                                data=historical_data,
                                asset_name=asset,
                                timeframe=config_manager.get_value('General', 'timeframe', 60, int)
                            )
                            
                            # Verificar se os dados foram processados corretamente
                            if data_processed is None:
                                logger.error(f"Failed to process historical data for {asset}")
                                continue
                                
                            # Verificar se o DataFrame está vazio
                            if isinstance(data_processed, pd.DataFrame) and data_processed.empty:
                                logger.error(f"Processed data is empty for {asset}")
                                continue
                        
                        # Train model
                        logger.info("Training model on historical data")
                        training_success = inteligencia.train_model(assets)
                        
                        if not training_success:
                            logger.error("Failed to train model")
                            time.sleep(60)  # Espera antes de tentar novamente
                            continue
                        
                        # Avalia o modelo
                        logger.info("Evaluating model performance")
                        evaluation_metrics = inteligencia.evaluate_model()
                        
                        if evaluation_metrics:
                            logger.info(f"Model evaluation metrics: {json.dumps(evaluation_metrics, indent=2)}")
                            
                            # Verifica se deve mudar para modo de teste
                            if inteligencia.should_switch_to_test_mode():
                                logger.info("Model performance is good. Switching to test mode")
                                operation_mode = 'test'
                            else:
                                logger.info("Model performance is not good enough for test mode. Continuing learning")
                                time.sleep(300)  # Espera 5 minutos antes de tentar novamente
                        else:
                            logger.error("Failed to evaluate model")
                            time.sleep(60)
                    
                    except Exception as e:
                        logger.error(f"Error during learning: {str(e)}")
                        logger.error(traceback.format_exc())
                        time.sleep(60)
                
                # Modo de Teste
                elif operation_mode == 'test':
                    logger.info("Starting test mode")
                    
                    # Configura modo de simulação para testes
                    if not ferramental.simulation_mode:
                        ferramental.setup_simulation_mode(
                            volatility=config_manager.get_value('Simulation', 'volatility', 0.01, float),
                            trend=config_manager.get_value('Simulation', 'trend', 0.0, float),
                            win_rate=config_manager.get_value('Simulation', 'win_rate', 0.6, float)
                        )
                    
                    # Executa negociações em modo de teste
                    try:
                        # Obtém dados em tempo real
                        for asset in assets:
                            logger.info(f"Getting real-time data for {asset}")
                            realtime_data = ferramental.get_realtime_data(asset)
                            
                            if realtime_data is None:
                                logger.error(f"Failed to get real-time data for {asset}")
                                continue
                            
                            # Faz previsões
                            logger.info(f"Making predictions for {asset}")
                            predictions = inteligencia.predict(realtime_data, asset)
                            
                            if predictions is None:
                                logger.error(f"Failed to make predictions for {asset}")
                                continue
                            
                            # Executa operações baseadas nas previsões
                            logger.info(f"Executing trades for {asset} based on predictions")
                            trades_executed = ferramental.execute_trades(
                                asset=asset,
                                predictions=predictions,
                                amount=config_manager.get_value('Trading', 'amount', 10.0, float),
                                expiration=config_manager.get_value('Trading', 'expiration', 1, int)
                            )
                            
                            if not trades_executed:
                                logger.error(f"Failed to execute trades for {asset}")
                                continue
                        
                        # Verifica resultados das operações
                        trade_results = ferramental.get_trade_results()
                        
                        if trade_results:
                            # Atualiza métricas de desempenho
                            performance_tracker.update(trade_results)
                            
                            # Verifica se deve mudar para modo real
                            if inteligencia.should_switch_to_real_mode(performance_tracker):
                                if config_manager.get_value('AutoSwitch', 'auto_switch_to_real', False, bool):
                                    logger.info("Performance is good. Switching to real mode")
                                    operation_mode = 'real'
                                else:
                                    logger.info("Performance is good for real mode, but auto-switch is disabled")
                        
                        # Espera antes da próxima iteração
                        time.sleep(config_manager.get_value('General', 'test_interval', 60, int))
                    
                    except Exception as e:
                        logger.error(f"Error during test mode: {str(e)}")
                        logger.error(traceback.format_exc())
                        time.sleep(60)
                
                # Modo Real
                elif operation_mode == 'real':
                    logger.info("Starting real mode")
                    
                    # Verifica se o modo real está habilitado nas configurações
                    if not config_manager.get_value('General', 'enable_real_mode', False, bool):
                        logger.warning("Real mode is disabled in config. Switching to test mode")
                        operation_mode = 'test'
                        continue
                    
                    # Verifica conexão com a API
                    if not ferramental.check_connection() or ferramental.simulation_mode:
                        logger.warning("Not connected to IQ Option API or in simulation mode. Cannot operate in real mode")
                        logger.info("Switching to test mode")
                        operation_mode = 'test'
                        continue
                    
                    # Executa negociações em modo real
                    try:
                        # Obtém dados em tempo real
                        for asset in assets:
                            logger.info(f"Getting real-time data for {asset}")
                            realtime_data = ferramental.get_realtime_data(asset)
                            
                            if realtime_data is None:
                                logger.error(f"Failed to get real-time data for {asset}")
                                continue
                            
                            # Faz previsões
                            logger.info(f"Making predictions for {asset}")
                            predictions = inteligencia.predict(realtime_data, asset)
                            
                            if predictions is None:
                                logger.error(f"Failed to make predictions for {asset}")
                                continue
                            
                            # Executa operações baseadas nas previsões
                            logger.info(f"Executing REAL trades for {asset} based on predictions")
                            trades_executed = ferramental.execute_trades(
                                asset=asset,
                                predictions=predictions,
                                amount=config_manager.get_value('Trading', 'amount', 10.0, float),
                                expiration=config_manager.get_value('Trading', 'expiration', 1, int),
                                is_demo=False  # Modo real
                            )
                            
                            if not trades_executed:
                                logger.error(f"Failed to execute real trades for {asset}")
                                continue
                        
                        # Verifica resultados das operações
                        trade_results = ferramental.get_trade_results()
                        
                        if trade_results:
                            # Atualiza métricas de desempenho
                            performance_tracker.update(trade_results)
                            
                            # Verifica se deve voltar para modo de teste
                            if not inteligencia.should_stay_in_real_mode(performance_tracker):
                                logger.warning("Performance is not good enough for real mode. Switching back to test mode")
                                operation_mode = 'test'
                        
                        # Espera antes da próxima iteração
                        time.sleep(config_manager.get_value('General', 'real_interval', 60, int))
                    
                    except Exception as e:
                        logger.error(f"Error during real mode: {str(e)}")
                        logger.error(traceback.format_exc())
                        time.sleep(60)
                
                else:
                    logger.error(f"Unknown operation mode: {operation_mode}")
                    break
            
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(60)
        
        logger.info("Bot stopped gracefully")
    
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        logger.error(traceback.format_exc())
    
    finally:
        # Realiza limpeza final
        cleanup()

if __name__ == "__main__":
    main()
