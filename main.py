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
# Passa o nível DEBUG como override se args.debug for True, senão passa None (usará config)
logger = setup_logger('main', override_level=logging.DEBUG if args.debug else None)

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
        logger.error("Dependências ausentes. Instalando...")
        return False
    else:
        logger.info("Todas as dependências estão instaladas")
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
            # Nota: A instalação automática de dependências não pode ser implementada aqui.
            # Garanta que todas as dependências em requirements.txt estejam instaladas.
        
        # Cria diretórios necessários
        create_directories()
        
        # Carrega configurações
        config_file = args.config
        config_manager = ConfigManager(config_file)
        
        # Obtém credenciais da IQ Option
        # Busca credenciais na seção correta [Credentials]
        # Corrigido para usar a chave 'email' definida no config.ini
        iqoption_username = config_manager.get_value('Credentials', 'email')
        iqoption_password = config_manager.get_value('Credentials', 'password')
        
        logger.info(f"Username from config: {iqoption_username}")
        logger.info(f"Password from config: {'*' * len(iqoption_password)}")
        
        # Inicializa componentes
        ferramental = Ferramental(config_manager, error_tracker) # Passa o error_tracker global
        inteligencia = Inteligencia(config_manager, error_tracker) # Passa o error_tracker global
        
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
            connection_success = ferramental.connect()
            
            if not connection_success:
                # Se falhar, tenta entrar em modo de simulação
                logger.warning("Falha na conexão com a API. Entrando em modo de simulação.")
                # Passa os parâmetros de simulação lidos da config
                sim_volatility = config_manager.get_value('Simulation', 'synthetic_volatility', 0.0002, float)
                sim_trend = config_manager.get_value('Simulation', 'synthetic_trend', 0.00001, float)
                sim_win_rate = config_manager.get_value('Simulation', 'simulated_win_rate', 60.0, float) / 100.0
                ferramental.setup_simulation_mode(
                    volatility=sim_volatility,
                    trend=sim_trend,
                    win_rate=sim_win_rate
                )
        
        # Determina o modo de operação
        operation_mode = args.mode or config_manager.get_value('General', 'mode', 'test')
        
        # Determina os ativos para negociação
        assets_str = args.assets or config_manager.get_value('General', 'assets', 'eurusd')
        # Manter o caso original dos ativos (geralmente MAIÚSCULAS para APIs financeiras)
        assets = [asset.strip() for asset in assets_str.split(',')]
        
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
                        # Lista para armazenar dados processados de todos os ativos
                        all_processed_data = []
                        
                        # Process historical data for each asset
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
                            
                            # Verificar se os dados foram processados corretamente e adicionar à lista
                            if data_processed is not None and not data_processed.empty:
                                # Adicionar coluna 'asset' para diferenciar no treinamento, se necessário
                                data_processed['asset'] = asset
                                all_processed_data.append(data_processed)
                                logger.info(f"Processed data for {asset} added for training.")
                            else:
                                logger.error(f"Failed to process historical data for {asset} or data is empty.")
                                continue # Pula para o próximo ativo se o processamento falhar

                        # Verificar se há dados processados para treinar
                        if not all_processed_data:
                            logger.error("No processed data available to train the model. Skipping training.")
                            time.sleep(60) # Espera antes de tentar o ciclo de aprendizado novamente
                            continue

                        # Concatenar dados de todos os ativos
                        logger.info(f"Concatenating processed data from {len(all_processed_data)} assets.")
                        training_data = pd.concat(all_processed_data, ignore_index=True)
                        
                        # Train model using the actual 'train' function name and concatenated data
                        logger.info(f"Training model on aggregated historical data ({len(training_data)} records)")
                        # Passar parâmetros de treinamento da configuração, se necessário
                        training_history = inteligencia.train(
                            train_data=training_data,
                            epochs=config_manager.get_value('Model', 'epochs', 100, int),
                            learning_rate=config_manager.get_value('Model', 'learning_rate', 0.001, float),
                            sequence_length=config_manager.get_value('Learning', 'sequence_length', 20, int), # Exemplo, buscar config
                            test_size=config_manager.get_value('Learning', 'test_size', 0.2, float) # Exemplo, buscar config
                        )
                        
                        if training_history is None: # A função train retorna None em caso de erro
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
                        tb_str = traceback.format_exc()
                        logger.error(f"Error during learning: {str(e)}\n{tb_str}")
                        error_tracker.add_error("LearningModeError", str(e), tb_str)
                        time.sleep(60)
                
                # Modo de Teste
                elif operation_mode == 'test':
                    logger.info("Starting test mode")
                    
                    # Modo Teste opera na conta PRACTICE se conectado, ou em simulação se a conexão falhou.
                    # A simulação é configurada na falha de conexão inicial.
                    
                    # Configura a inteligência para o modo de teste
                    inteligencia.setup_test_mode()
                    
                    # Logar o contexto de execução do modo Teste
                    if hasattr(ferramental, 'simulation_mode') and ferramental.simulation_mode:
                        logger.info("Modo Teste operando em MODO DE SIMULAÇÃO (conexão API falhou).")
                    elif ferramental.connected:
                         logger.info("Modo Teste operando conectado à conta PRACTICE/DEMO.")
                    else:
                         logger.warning("Modo Teste em estado inesperado (nem simulação, nem conectado).")
                    
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
                            
                            # Processa previsões e executa operações
                            action = predictions.get('action')
                            confidence = predictions.get('confidence', 0)
                            
                            if action == 'BUY' or action == 'SELL':
                                mapped_action = 'call' if action == 'BUY' else 'put'
                                amount = config_manager.get_value('Trading', 'amount', 10.0, float)
                                # Usar expiration_mode (turbo/binary) em vez de expiration (minutos)
                                exp_mode = config_manager.get_value('Trading', 'trade_type', 'turbo')
                                
                                logger.info(f"Executing TEST trade for {asset}: {mapped_action} ${amount} (Confidence: {confidence:.2f})")
                                
                                # Chama a função 'buy' que é a correta para executar ordens binárias
                                # No modo teste, Ferramental.buy deve usar a conta DEMO ou simulação interna
                                success, order_id = ferramental.buy(
                                    asset=asset,
                                    amount=amount,
                                    action=mapped_action,
                                    expiration_mode=exp_mode
                                )
                                
                                if not success:
                                    logger.error(f"Failed to execute TEST trade for {asset}")
                                    # Não necessariamente 'continue', pode ser interessante logar a falha e prosseguir
                            elif action == 'HOLD':
                                logger.info(f"Prediction for {asset} is HOLD (Confidence: {confidence:.2f}). No trade executed.")
                            else:
                                logger.warning(f"Unknown prediction action for {asset}: {action}")
                        
                        # Verifica resultados das operações
                        trade_results = ferramental.get_trade_results()
                        
                        if trade_results:
                            # Processar resultados para atualizar gerenciamento de risco
                            current_balance = ferramental.get_balance()
                            if current_balance and current_balance > 0:
                                for result in trade_results:
                                    # Verifica se a operação resultou em perda
                                    # Usamos 'profit' < 0 como indicador primário, ou 'is_win' se disponível
                                    is_loss = result.get('profit', 0) < 0 
                                    # Fallback para 'is_win' se 'profit' não for conclusivo (ex: 0.0)
                                    if result.get('profit') == 0 and 'is_win' in result:
                                        is_loss = not result.get('is_win')

                                    if is_loss:
                                        ferramental.risk_management['consecutive_losses'] += 1
                                        # Calcula a perda: valor absoluto do profit negativo ou o amount total
                                        loss_amount = abs(result.get('profit', 0)) if result.get('profit', 0) < 0 else result.get('amount', 0)
                                        ferramental.risk_management['daily_loss'] += loss_amount / current_balance
                                        logger.warning(f"Loss detected. Consecutive losses: {ferramental.risk_management['consecutive_losses']}, Daily loss: {ferramental.risk_management['daily_loss']:.2%}")
                                    elif result.get('profit', -1) > 0 or result.get('is_win', False): 
                                        # Reseta perdas consecutivas em caso de ganho ou empate (profit >= 0 ou is_win=True)
                                        ferramental.risk_management['consecutive_losses'] = 0
                            else:
                                logger.error("Não foi possível obter saldo para atualizar métricas de risco diário.")


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
                        tb_str = traceback.format_exc()
                        logger.error(f"Error during test mode: {str(e)}\n{tb_str}")
                        error_tracker.add_error("TestModeError", str(e), tb_str)
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
                    if not ferramental.check_connection():
                        logger.warning("Not connected to IQ Option API. Cannot operate in real mode")
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
                            
                            # Processa previsões e executa operações REAIS
                            action = predictions.get('action')
                            confidence = predictions.get('confidence', 0)

                            if action == 'BUY' or action == 'SELL':
                                mapped_action = 'call' if action == 'BUY' else 'put'
                                amount = config_manager.get_value('Trading', 'amount', 10.0, float)
                                exp_mode = config_manager.get_value('Trading', 'trade_type', 'turbo')

                                logger.info(f"Executing REAL trade for {asset}: {mapped_action} ${amount} (Confidence: {confidence:.2f})")
                                
                                # Chama a função 'buy' para executar ordens binárias reais
                                success, order_id = ferramental.buy(
                                    asset=asset,
                                    amount=amount,
                                    action=mapped_action,
                                    expiration_mode=exp_mode
                                )

                                if not success:
                                     logger.error(f"Failed to execute REAL trade for {asset}")
                                     # Não necessariamente 'continue', pode ser interessante logar a falha e prosseguir
                            elif action == 'HOLD':
                                logger.info(f"Prediction for {asset} is HOLD (Confidence: {confidence:.2f}). No trade executed.")
                            else:
                                logger.warning(f"Unknown prediction action for {asset}: {action}")
                        
                        # Verifica resultados das operações
                        trade_results = ferramental.get_trade_results()
                        
                        if trade_results:
                            # Processar resultados para atualizar gerenciamento de risco
                            current_balance = ferramental.get_balance()
                            if current_balance and current_balance > 0:
                                for result in trade_results:
                                    # Verifica se a operação resultou em perda
                                    # Usamos 'profit' < 0 como indicador primário, ou 'is_win' se disponível
                                    is_loss = result.get('profit', 0) < 0 
                                    # Fallback para 'is_win' se 'profit' não for conclusivo (ex: 0.0)
                                    if result.get('profit') == 0 and 'is_win' in result:
                                        is_loss = not result.get('is_win')

                                    if is_loss:
                                        ferramental.risk_management['consecutive_losses'] += 1
                                        # Calcula a perda: valor absoluto do profit negativo ou o amount total
                                        loss_amount = abs(result.get('profit', 0)) if result.get('profit', 0) < 0 else result.get('amount', 0)
                                        ferramental.risk_management['daily_loss'] += loss_amount / current_balance
                                        logger.warning(f"Loss detected. Consecutive losses: {ferramental.risk_management['consecutive_losses']}, Daily loss: {ferramental.risk_management['daily_loss']:.2%}")
                                    elif result.get('profit', -1) > 0 or result.get('is_win', False): 
                                        # Reseta perdas consecutivas em caso de ganho ou empate (profit >= 0 ou is_win=True)
                                        ferramental.risk_management['consecutive_losses'] = 0
                            else:
                                logger.error("Não foi possível obter saldo para atualizar métricas de risco diário.")


                            # Atualiza métricas de desempenho
                            performance_tracker.update(trade_results)
                            
                            # Verifica se deve voltar para modo de teste
                            if not inteligencia.should_stay_in_real_mode(performance_tracker):
                                logger.warning("Performance is not good enough for real mode. Switching back to test mode")
                                operation_mode = 'test'
                        
                        # Espera antes da próxima iteração
                        time.sleep(config_manager.get_value('General', 'real_interval', 60, int))
                    
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.error(f"Error during real mode: {str(e)}\n{tb_str}")
                        error_tracker.add_error("RealModeError", str(e), tb_str)
                        time.sleep(60)
                
                else:
                    logger.error(f"Unknown operation mode: {operation_mode}")
                    break
            
            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"Error in main loop: {str(e)}\n{tb_str}")
                error_tracker.add_error("MainLoopError", str(e), tb_str)
                time.sleep(60)
        
        logger.info("Bot stopped gracefully")
    
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.critical(f"Critical error: {str(e)}\n{tb_str}")
        # Tenta registrar o erro crítico, mas pode falhar se o tracker não foi inicializado
        try:
            error_tracker.add_error("CriticalError", str(e), tb_str, critical=True)
        except NameError: # Caso o erro ocorra antes da inicialização do error_tracker
             logger.error("Error tracker not initialized, could not record critical error.")
        except Exception as tracker_err:
             logger.error(f"Failed to record critical error in tracker: {tracker_err}")
    
    finally:
        # Realiza limpeza final
        cleanup()

if __name__ == "__main__":
    main()
