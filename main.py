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
# Importações locais (serão importadas sob demanda se necessário para evitar conflitos de teste)
# from ferramental.Ferramental import Ferramental
# from inteligencia.Inteligencia import Inteligencia

# Variáveis globais (serão inicializadas no main ou sob demanda)
stop_event = threading.Event()
performance_tracker = PerformanceTracker()
error_tracker = ErrorTracker()
logger = logging.getLogger('main')

def parse_args():
    """Configura e processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Bot IQ Option')
    parser.add_argument('--config', type=str, default='config.ini', help='Caminho para o arquivo de configuração')
    parser.add_argument('--mode', type=str, choices=['download', 'learning', 'test', 'real'], help='Modo de operação')
    parser.add_argument('--assets', type=str, help='Lista de ativos separados por vírgula (ex: EURUSD,GBPUSD)')
    parser.add_argument('--debug', action='store_true', help='Ativa modo de debug')
    parser.add_argument('--duration', type=int, help='Tempo máximo de execução em segundos (para testes)')
    parser.add_argument('--max-cycles', type=int, help='Número máximo de ciclos no loop principal (para testes)')
    
    # Se estiver rodando em pytest, use argumentos vazios para evitar erro
    if 'pytest' in sys.modules or 'tox' in sys.modules:
        return parser.parse_args([])
    return parser.parse_args()

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
    try:
        print(banner)
    except UnicodeEncodeError:
        # Fallback para sistemas que não suportam UTF-8 no terminal
        print("    BOT IQ OPTION - INICIADO")
        print("    (Banner original contém caracteres não suportados por este terminal)")

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
    global logger
    args = parse_args()
    logger = setup_logger('main', override_level=logging.DEBUG if args.debug else None)
    
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
        
        # Inicializa componentes (Importação tardia para evitar conflitos)
        from ferramental.Ferramental import Ferramental
        from inteligencia.Inteligencia import Inteligencia
        
        ferramental = Ferramental(config_manager, error_tracker) # Passa o error_tracker global
        inteligencia = Inteligencia(config_manager, error_tracker) # Passa o error_tracker global
        
        # Opcional: Carregar dados históricos processados para "esquentar" o scaler caso ele não venha no modelo
        # Isso garante que a normalização funcione mesmo com modelos antigos
        data_dir = config_manager.get_value('General', 'data_directory', 'data')
        processed_dir = os.path.join(data_dir, 'processed')
        if os.path.exists(processed_dir):
            import glob
            processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
            if processed_files:
                logger.info(f"Carregando {len(processed_files)} arquivos de dados históricos para inicializar o scaler")
                all_data = []
                for pf in processed_files:
                    try:
                        all_data.append(pd.read_csv(pf))
                    except Exception as e:
                        logger.warning(f"Erro ao carregar {pf}: {e}")
                if all_data:
                    inteligencia.historical_data = pd.concat(all_data, ignore_index=True)
        
        # Tenta carregar o melhor modelo disponível
        best_model_path = os.path.join(inteligencia.model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            logger.info(f"Carregando o melhor modelo: {best_model_path}")
            inteligencia.load_model(best_model_path)
        else:
            logger.info("Tentando carregar o modelo mais recente...")
            inteligencia.load_model()
        
        # Log available methods of Ferramental to debug AttributeError
        logger.debug(f"Ferramental methods: {[m for m in dir(ferramental) if not m.startswith('_')]}")
        
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
            connection_success, reason = ferramental.connect()
            
            if not connection_success:
                # Se falhar, tenta entrar em modo de simulação
                logger.warning(f"Falha na conexão com a API ({reason}). Entrando em modo de simulação.")
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
        start_time = time.time()
        cycle_count = 0
        streams_initialized = False
        
        while not stop_event.is_set():
            cycle_count += 1
            
            # Verificação de limites de execução
            if args.duration and (time.time() - start_time) > args.duration:
                logger.info(f"Limite de tempo de execução atingido ({args.duration}s). Encerrando...")
                break
                
            if args.max_cycles and cycle_count > args.max_cycles:
                logger.info(f"Limite de ciclos de execução atingido ({args.max_cycles}). Encerrando...")
                break
                
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
                                count=config_manager.get_value('Data', 'historical_candle_count', 1000, int)
                            )
                            
                            if historical_data is None or len(historical_data) == 0:
                                logger.error(f"Failed to get historical data for {asset}")
                                continue
                                
                            # Processar os dados históricos (Sem normalizar aqui, o treino faz isso)
                            processed_data = inteligencia.process_historical_data(
                                data=historical_data,
                                asset_name=asset,
                                timeframe=config_manager.get_value('General', 'timeframe', 60, int),
                                normalize=False, # Importante: Normalização correta no treino
                                window=config_manager.get_value('Learning', 'lookahead_periods', 5, int),
                                threshold=config_manager.get_value('Learning', 'prediction_threshold', 0.001, float)
                            )
                            
                            if processed_data is not None and not processed_data.empty:
                                # Adicionar coluna 'asset' para diferenciar no treinamento, se necessário
                                processed_data['asset'] = asset
                                all_processed_data.append(processed_data)
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
                        
                        # Treinamento do modelo com divisão interna e normalização robusta
                        logger.info(f"Training model on aggregated historical data ({len(training_data)} records)")
                        training_history = inteligencia.train(
                            train_data=training_data,
                            epochs=config_manager.get_value('Learning', 'epochs', 50, int),
                            learning_rate=config_manager.get_value('Model', 'learning_rate', 0.001, float),
                            sequence_length=config_manager.get_value('General', 'sequence_length', 20, int),
                            test_size=config_manager.get_value('Learning', 'test_size', 0.2, float)
                        )
                        
                        if training_history:
                            logger.info("Model training completed successfully.")
                            
                            # Gera o Relatório de Qualidade Absoluto (Check Adversário e Financeiro)
                            logger.info("Generating Training Quality Report (Adversarial Check)...")
                            # Pequena amostra de validação para o report
                            train_data_rep, val_data_rep = inteligencia._split_data(training_data, test_size=0.2)
                            report = inteligencia.generate_training_report(train_data_rep, val_data_rep)
                            
                            logger.info("\n" + "="*40 + "\nTRAINING QUALITY REPORT\n" + "="*40 + "\n" + report + "\n" + "="*40)
                            
                            # Salva o modelo
                            inteligencia.save_model()
                            
                            # Verifica se deve mudar para modo de teste
                            if inteligencia.should_switch_to_test_mode():
                                logger.info("Performance criteria met. Switching to TEST mode.")
                                operation_mode = 'test'
                            else:
                                logger.info("Performance not sufficient for TEST mode. Continuing learning.")
                                time.sleep(60)
                        else:
                            logger.error("Failed to train model")
                            time.sleep(60)
                    
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.error(f"Error during learning: {str(e)}\n{tb_str}")
                        error_tracker.add_error("LearningModeError", str(e), tb_str)
                        time.sleep(60)
                
                # Modo de Teste
                elif operation_mode == 'test':
                    logger.info("Starting test mode")
                    # Configura a inteligência para o modo de teste
                    inteligencia.setup_test_mode()
                    
                    # Garante que estamos na conta PRACTICE para o modo de teste
                    if ferramental.connected:
                        ferramental.change_balance('PRACTICE')
                        logger.info("Modo Teste operando conectado à conta PRACTICE/DEMO.")
                    
                    if hasattr(ferramental, 'simulation_mode') and ferramental.simulation_mode:
                        logger.info("Modo Teste operando em MODO DE SIMULAÇÃO (conexão API falhou).")
                    elif not ferramental.connected:
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
                            confidence_threshold = config_manager.get_value('Trading', 'min_confidence', 0.6, float)
                            predictions = inteligencia.predict(realtime_data, confidence_threshold=confidence_threshold)
                            
                            if predictions is None:
                                logger.error(f"Failed to make predictions for {asset}")
                                continue
                            
                            # Processa previsões e executa operações
                            action = predictions.get('action')
                            confidence = predictions.get('confidence', 0)
                            
                            if action == 'BUY' or action == 'SELL':
                                mapped_action = 'call' if action == 'BUY' else 'put'
                                amount = config_manager.get_value('Trading', 'amount', 10.0, float)
                                expiration = config_manager.get_value('Trading', 'expiration', 1, int)
                                
                                logger.info(f"Executing TEST trade for {asset}: {mapped_action} ${amount} (Confidence: {confidence:.2f})")
                                
                                # Chama a função 'buy' que é a correta para executar ordens binárias
                                # No modo teste, Ferramental.buy deve usar a conta DEMO ou simulação interna
                                success, order_id = ferramental.buy(
                                    asset=asset,
                                    amount=amount,
                                    action=mapped_action,
                                    expiration=expiration
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
                
                # Modo Real - Ative Scanner Mode
                elif operation_mode == 'real':
                    # Garante que estamos na conta REAL
                    if ferramental.connected:
                        ferramental.change_balance('REAL')
                        
                    # Inicialização UNA: Ativa streams para todos os ativos uma única vez
                    if not streams_initialized:
                        logger.info(f"Initializing High-Frequency Scanner Streams for: {', '.join(assets)}")
                        for asset in assets:
                            ferramental.start_candles_stream(asset, timeframe=60)
                        streams_initialized = True
                    
                    # Verifica se o modo real está habilitado
                    if not config_manager.get_value('General', 'enable_real_mode', False, bool):
                        logger.warning("Real mode is disabled in config. Switching to test mode")
                        operation_mode = 'test'
                        continue
                    
                    if not ferramental.check_connection():
                        logger.warning("Not connected to IQ Option API. Switching to test mode")
                        operation_mode = 'test'
                        continue
                    
                    # SCANNER LOOP: Varredura rápida de todos os ativos
                    try:
                        opportunities = []
                        confidence_threshold = config_manager.get_value('Trading', 'min_confidence', 0.6, float)
                        
                        for asset in assets:
                            # Obtém dados via STREAMING (rápido, sem polling de histórico completo)
                            realtime_data = ferramental.get_realtime_data(asset)
                            if realtime_data is None: continue
                            
                            # Faz previsão
                            predictions = inteligencia.predict(realtime_data, confidence_threshold=confidence_threshold)
                            if predictions and predictions.get('action') in ['BUY', 'SELL']:
                                predictions['asset'] = asset
                                opportunities.append(predictions)
                            elif predictions and predictions.get('action') == 'HOLD':
                                # Log periódico para mostrar que o bot está vivo e caçando (a cada ~1 min a 2s/cycle)
                                if cycle_count % 30 == 0:
                                    logger.info(f"Asset {asset} scanner: HOLD (Conf: {predictions.get('confidence', 0):.2f})")

                        # RANKING: Executa a melhor oportunidade encontrada nesta varredura
                        if opportunities:
                            # Ordena por confiança decrescente
                            opportunities.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                            best_opp = opportunities[0]
                            
                            asset = best_opp['asset']
                            action = best_opp['action']
                            confidence = best_opp['confidence']
                            
                            mapped_action = 'call' if action == 'BUY' else 'put'
                            amount = config_manager.get_value('Trading', 'amount', 10.0, float)
                            expiration = config_manager.get_value('Trading', 'expiration', 1, int)

                            logger.info(f"🚀 OPPORTUNITY FOUND! Asset: {asset} | Action: {mapped_action} | Conf: {confidence:.2f}")
                            success, order_id = ferramental.buy(
                                asset=asset,
                                amount=amount,
                                action=mapped_action,
                                expiration=expiration
                            )
                            if not success:
                                logger.error(f"Failed to execute trade for {asset}")
                        
                        # Verifica resultados das operações a cada 10 ciclos para economizar chamadas de API
                        if cycle_count % 10 == 0:
                            trade_results = ferramental.get_trade_results()
                            if trade_results:
                                current_balance = ferramental.get_balance()
                                if current_balance:
                                    performance_tracker.update(trade_results)
                                    if not inteligencia.should_stay_in_real_mode(performance_tracker):
                                        logger.warning("Performance degraded. Switching back to test mode.")
                                        operation_mode = 'test'

                        # Intervalo de varredura ativa (Scanner Tick)
                        scan_interval = config_manager.get_value('General', 'real_mode_scan_interval', 2, int)
                        time.sleep(scan_interval)
                    
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.error(f"Error during Scanner Mode: {str(e)}\n{tb_str}")
                        error_tracker.add_error("ScannerModeError", str(e), tb_str)
                        time.sleep(5)
                
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
