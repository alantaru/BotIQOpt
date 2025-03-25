#!/usr/bin/env python3
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
from utils.LogManager import LogManager, setup_logger
from ferramental.ErrorTracker import ErrorTracker
from ferramental.PerformanceMetrics import BotPerformanceMetrics
from inteligencia.Inteligencia import Inteligencia
from ferramental.Ferramental import Ferramental

# Configuração inicial de logging
logger = setup_logger('main')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Bot IQ Option')
    parser.add_argument('--mode', type=str, choices=['DOWNLOAD', 'LEARNING', 'TEST', 'REAL'],
                        help='Mode of operation')
    parser.add_argument('--config', type=str, default='config.ini',
                        help='Path to configuration file')
    parser.add_argument('--assets', type=str,
                        help='Comma-separated list of assets to trade')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()


def setup_signal_handlers(stop_event):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received. Cleaning up...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def display_welcome_message():
    """Display welcome message with ASCII art."""
    welcome_message = """
    ██████╗  ██████╗ ████████╗    ██╗ ██████╗      ██████╗ ██████╗ ████████╗
    ██╔══██╗██╔═══██╗╚══██╔══╝    ██║██╔═══██╗    ██╔═══██╗██╔══██╗╚══██╔══╝
    ██████╔╝██║   ██║   ██║       ██║██║   ██║    ██║   ██║██████╔╝   ██║   
    ██╔══██╗██║   ██║   ██║       ██║██║▄▄ ██║    ██║   ██║██╔═══╝    ██║   
    ██████╔╝╚██████╔╝   ██║       ██║╚██████╔╝    ╚██████╔╝██║        ██║   
    ╚═════╝  ╚═════╝    ╚═╝       ╚═╝ ╚══▀▀═╝      ╚═════╝ ╚═╝        ╚═╝   
    """
    print(welcome_message)
    logger.info("Bot IQ Option iniciado")


def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        logger.info("All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False


def check_api_connection(ferramental):
    """Check connection to IQ Option API."""
    try:
        connected = ferramental.check_connection()
        if connected:
            logger.info("Successfully connected to IQ Option API")
            return True
        else:
            logger.error("Failed to connect to IQ Option API")
            return False
    except Exception as e:
        logger.error(f"Error checking API connection: {str(e)}")
        return False


def initialize_components(config_manager, args):
    """Initialize all components of the bot."""
    try:
        # Carrega as credenciais do arquivo de configuração
        username = config_manager.get_value('Credentials', 'username')
        password = config_manager.get_value('Credentials', 'password')
        
        logger.info(f"Username from config: {username}")
        logger.info(f"Password from config: {password}")
        
        # Initialize error tracker
        error_tracker = ErrorTracker()
        
        # Initialize performance metrics
        performance_metrics = BotPerformanceMetrics()
        
        # Initialize intelligence module
        inteligencia = Inteligencia(config_manager)
        
        # Inicializa o objeto Ferramental com o config_manager
        ferramental = Ferramental(config_manager=config_manager)
        
        # Update auto switch criteria from config manager
        inteligencia.update_auto_switch_criteria_from_config_manager(config_manager)
        
        return inteligencia, ferramental, error_tracker, performance_metrics
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        return None, None, None, None


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Display welcome message
    display_welcome_message()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please install required packages.")
        sys.exit(1)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Initialize components
    inteligencia, ferramental, error_tracker, performance_metrics = initialize_components(config_manager, args)
    
    if not inteligencia or not ferramental:
        logger.error("Failed to initialize components")
        sys.exit(1)
    
    # Check API connection
    if not check_api_connection(ferramental):
        logger.error("Failed to connect to IQ Option API. Please check your credentials and internet connection.")
        sys.exit(1)
    
    # Create stop event for graceful shutdown
    stop_event = threading.Event()
    setup_signal_handlers(stop_event)
    
    # Determine initial mode
    if args.mode:
        mode = args.mode.upper()
    else:
        mode = config_manager.get_value('General', 'operation_mode', 'TEST')
    
    logger.info(f"Starting in {mode} mode")
    
    # Get assets to trade
    if args.assets:
        assets = [asset.strip() for asset in args.assets.split(',')]
    else:
        assets_str = config_manager.get_value('General', 'assets', 'EURUSD')
        assets = [asset.strip() for asset in assets_str.split(',')]
    
    logger.info(f"Trading assets: {', '.join(assets)}")
    
    # Main loop
    try:
        while not stop_event.is_set():
            try:
                if mode.upper() == "DOWNLOAD":
                    logger.info("Starting historical data download")
                    try:
                        # Get date range for download
                        start_date_str = config_manager.get_value('Download', 'start_date', '2023-01-01')
                        end_date_str = config_manager.get_value('Download', 'end_date', datetime.now().strftime('%Y-%m-%d'))
                        
                        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                        
                        # Get timeframe
                        timeframe_type = config_manager.get_value('General', 'timeframe_type', 'Minutes')
                        timeframe_value = config_manager.get_value('General', 'timeframe_value', 1, int)
                        
                        # Download data for each asset
                        for asset in assets:
                            logger.info(f"Downloading historical data for {asset}")
                            success = ferramental.download_historical_data(
                                asset=asset,
                                start_date=start_date,
                                end_date=end_date,
                                timeframe_type=timeframe_type,
                                timeframe_value=timeframe_value
                            )
                            
                            if success:
                                logger.success(f"Successfully downloaded historical data for {asset}")
                            else:
                                logger.error(f"Failed to download historical data for {asset}")
                        
                        # After download, switch to LEARNING mode
                        logger.info("Download completed. Switching to LEARNING mode")
                        mode = "LEARNING"
                    except Exception as e:
                        logger.error(f"Error during data download: {str(e)}")
                        error_tracker.log_error('DOWNLOAD_ERROR', str(e))
                        break
                
                elif mode.upper() == "LEARNING":
                    logger.info("Starting learning mode")
                    try:
                        # Process historical data
                        for asset in assets:
                            logger.info(f"Processing historical data for {asset}")
                            data_processed = inteligencia.process_historical_data(asset)
                            
                            if not data_processed:
                                logger.error(f"Failed to process historical data for {asset}")
                                continue
                        
                        # Train model
                        logger.info("Training model on historical data")
                        training_success = inteligencia.train_model(assets)
                        
                        if training_success:
                            logger.success("Model training completed successfully")
                            
                            # Evaluate model
                            evaluation_metrics = inteligencia.evaluate_model(assets)
                            logger.info(f"Model evaluation metrics: {evaluation_metrics}")
                            
                            # Save model
                            model_saved = inteligencia.save_model()
                            if model_saved:
                                logger.success("Model saved successfully")
                            else:
                                logger.error("Failed to save model")
                            
                            # After learning, switch to TEST mode
                            logger.info("Learning completed. Switching to TEST mode")
                            mode = "TEST"
                        else:
                            logger.error("Model training failed")
                            # If learning fails, try again later
                            time.sleep(60)
                    except Exception as e:
                        logger.error(f"Error during learning: {str(e)}")
                        error_tracker.log_error('LEARNING_ERROR', str(e))
                        break
                
                elif mode.upper() == "TEST":
                    logger.info("Starting test trading")
                    try:
                        # Load model if not already loaded
                        if not inteligencia.is_model_loaded():
                            model_loaded = inteligencia.load_model()
                            if not model_loaded:
                                logger.error("Failed to load model. Switching to LEARNING mode")
                                mode = "LEARNING"
                                continue
                        
                        # Get current market data
                        market_data = {}
                        for asset in assets:
                            data = ferramental.get_current_candles(
                                asset=asset,
                                count=config_manager.get_value('General', 'candle_count', 100, int)
                            )
                            if data is not None:
                                market_data[asset] = data
                        
                        if not market_data:
                            logger.error("Failed to get market data for any asset")
                            time.sleep(60)
                            continue
                        
                        # Get predictions from AI
                        predictions = inteligencia.get_predictions(assets, market_data)
                        
                        # Execute test trades
                        for asset, prediction in predictions.items():
                            if prediction['confidence'] > config_manager.get_value('Trading', 'test_confidence_threshold', 0.6, float):
                                logger.info(f"Executing test trade for {asset} with direction {prediction['direction']} and confidence {prediction['confidence']:.2f}")
                                
                                # Simulate trade
                                result = ferramental.simulate_trade(
                                    asset=asset,
                                    direction=prediction['direction'],
                                    amount=config_manager.get_value('Trading', 'amount', 1.0, float)
                                )
                                
                                # Record trade result
                                inteligencia.record_trade(
                                    asset=asset,
                                    direction=prediction['direction'],
                                    amount=config_manager.get_value('Trading', 'amount', 1.0, float),
                                    result=result
                                )
                                
                                performance_metrics.add_trade(
                                    asset=asset,
                                    direction=prediction['direction'],
                                    amount=config_manager.get_value('Trading', 'amount', 1.0, float),
                                    result=result
                                )
                                
                                logger.info(f"Test trade result: {result}")
                        
                        # Check if we should switch to REAL mode
                        auto_switch_enabled = config_manager.is_auto_switch_enabled()
                        if auto_switch_enabled and inteligencia.should_switch_to_real():
                            logger.info("Performance criteria met for real trading. Switching to REAL mode")
                            mode = "REAL"
                        
                        # Wait before next cycle
                        time.sleep(config_manager.get_value('Trading', 'cycle_interval_seconds', 60, int))
                    
                    except Exception as e:
                        logger.error(f"Error during test trading: {str(e)}")
                        error_tracker.log_error('TEST_TRADING_ERROR', str(e))
                        break
                
                elif mode.upper() == "REAL":
                    logger.info("Starting real trading")
                    try:
                        # Verifica se é seguro operar no modo real
                        if not inteligencia.should_switch_to_real():
                            logger.warning("Condições para operação real não atingidas. Voltando para modo de TESTE.")
                            mode = "TEST"
                            continue
                        
                        # Verifica se o horário atual é adequado para operar
                        current_hour = datetime.now().hour
                        trading_hours_start = config_manager.get_value('Trading', 'trading_hours_start', 9, int)
                        trading_hours_end = config_manager.get_value('Trading', 'trading_hours_end', 17, int)
                        
                        if not (trading_hours_start <= current_hour <= trading_hours_end):
                            logger.warning(f"Fora do horário de operação ({trading_hours_start}h-{trading_hours_end}h). Aguardando próximo ciclo.")
                            time.sleep(60)  # Aguarda 1 minuto antes de verificar novamente
                            continue
                        
                        # Obtém o saldo atual para ajustar o valor das operações se necessário
                        balance = ferramental.get_balance_v2()
                        if balance is None:
                            logger.error("Não foi possível obter o saldo. Voltando para modo de TESTE.")
                            mode = "TEST"
                            continue
                        
                        # Verifica limites de risco
                        if not ferramental.check_risk_limits(balance):
                            logger.warning("Limites de risco excedidos. Voltando para modo de TESTE.")
                            mode = "TEST"
                            continue
                        
                        # Verifica a volatilidade atual do mercado
                        market_volatility = ferramental.check_market_volatility(assets)
                        max_volatility = config_manager.get_value('Trading', 'max_volatility_for_real', 0.4, float)
                        
                        if market_volatility > max_volatility:
                            logger.warning(f"Volatilidade do mercado muito alta: {market_volatility:.2%} > {max_volatility:.2%}. Voltando para modo de TESTE.")
                            mode = "TEST"
                            continue
                        
                        # Obtém estatísticas de operações recentes
                        trade_stats = inteligencia.get_trade_statistics(days=7)
                        consecutive_losses = trade_stats['max_consecutive_losses']
                        win_rate = trade_stats['win_rate']
                        
                        # Ajusta o valor da operação com base no saldo atual
                        risk_params = config_manager.get_risk_params()
                        risk_percentage = risk_params.get('risk_percentage', 0.02)  # 2% do saldo por padrão
                        base_risk_per_trade = balance * risk_percentage
                        
                        # Limita o valor máximo por operação
                        max_amount_per_trade = risk_params.get('max_amount_per_trade', 100)
                        base_risk_per_trade = min(base_risk_per_trade, max_amount_per_trade)
                        
                        logger.info(f"Saldo atual: {balance:.2f}, Valor base por operação: {base_risk_per_trade:.2f}")
                        
                        # Get predictions from AI
                        predictions = inteligencia.get_predictions(assets)
                        
                        # Execute real trades
                        trades_executed = 0
                        max_trades_per_cycle = config_manager.get_value('Trading', 'max_trades_per_cycle', 3, int)
                        confidence_threshold = config_manager.get_value('Trading', 'real_confidence_threshold', 0.8, float)
                        
                        logger.info(f"Configurações de operação: Max trades={max_trades_per_cycle}, Limiar de confiança={confidence_threshold:.2f}")
                        
                        # Ordena as previsões por confiança (da maior para a menor)
                        sorted_predictions = sorted(
                            [(asset, pred) for asset, pred in predictions.items()],
                            key=lambda x: x[1]['confidence'],
                            reverse=True
                        )
                        
                        for asset, prediction in sorted_predictions:
                            # Verifica se já atingimos o número máximo de operações por ciclo
                            if trades_executed >= max_trades_per_cycle:
                                logger.info(f"Máximo de operações por ciclo atingido ({max_trades_per_cycle})")
                                break
                                
                            if prediction['confidence'] > confidence_threshold:
                                # Verifica se já temos operações recentes neste ativo
                                recent_trades = inteligencia.get_recent_trades_for_asset(asset, hours=2)
                                if len(recent_trades) > 0:
                                    logger.info(f"Já existem operações recentes para {asset}. Pulando.")
                                    continue
                                
                                # Ajusta o valor da operação com base nos fatores de risco
                                adjusted_amount = ferramental.adjust_trade_amount_based_on_risk(
                                    base_amount=base_risk_per_trade,
                                    asset=asset,
                                    win_rate=win_rate,
                                    consecutive_losses=consecutive_losses,
                                    market_volatility=market_volatility
                                )
                                
                                # Verifica o payout atual antes de executar a operação
                                payout = ferramental.get_payout(asset)
                                if payout < config_manager.get_value('Trading', 'min_payout_for_real', 0.7, float):
                                    logger.warning(f"Payout muito baixo para {asset}: {payout:.2%}. Pulando.")
                                    continue
                                
                                # Executa a operação com valor ajustado
                                result = ferramental.execute_real_trade(
                                    asset=asset,
                                    direction=prediction['direction'],
                                    amount=adjusted_amount
                                )
                                
                                # Verifica se a ordem foi realmente executada
                                if result.get('status') == 'success' and 'order_id' in result:
                                    order_verified = ferramental.verify_order_execution(result['order_id'])
                                    if not order_verified:
                                        logger.warning(f"Não foi possível verificar a execução da ordem {result['order_id']}. Considere como não executada.")
                                        continue
                                
                                # Registra a operação no histórico para análise posterior
                                inteligencia.record_trade(
                                    asset=asset,
                                    direction=prediction['direction'],
                                    amount=adjusted_amount,
                                    result=result
                                )
                                
                                performance_metrics.add_trade(
                                    asset=asset,
                                    direction=prediction['direction'],
                                    amount=adjusted_amount,
                                    result=result
                                )
                                trades_executed += 1
                                logger.success(f"Operação real executada para {asset} com confiança {prediction['confidence']:.2f}")
                                
                                # Aguarda um intervalo entre operações para evitar sobrecarga
                                time.sleep(config_manager.get_value('Trading', 'trade_interval_seconds', 30, int))
                            else:
                                logger.info(f"Confiança insuficiente para {asset}: {prediction['confidence']:.2f} < {confidence_threshold:.2f}")
                        
                        if trades_executed == 0:
                            logger.warning("Nenhuma operação executada neste ciclo. Considere ajustar o limiar de confiança.")
                        
                        # Verifica se devemos continuar no modo REAL
                        metrics = performance_metrics.calculate_metrics()
                        if metrics['win_rate'] < config_manager.get_value('Trading', 'min_win_rate_real', 0.55, float) or metrics['profit_factor'] < config_manager.get_value('Trading', 'min_profit_factor_real', 1.2, float):
                            logger.warning(f"Métricas abaixo do esperado (Win Rate: {metrics['win_rate']:.2f}, Profit Factor: {metrics['profit_factor']:.2f}). Voltando para modo de TESTE.")
                            mode = "TEST"
                    
                    except Exception as e:
                        logger.error(f"Error during real trading: {str(e)}")
                        error_tracker.log_error('REAL_TRADING_ERROR', str(e))
                        # Em caso de erro no modo REAL, voltamos para o modo de TESTE por segurança
                        logger.warning("Voltando para modo de TESTE devido a erro no modo REAL")
                        mode = "TEST"
                else:
                    logger.error(f"Invalid mode: {mode}")
                    # Definir um modo válido para evitar loop infinito
                    mode = "TEST"
                    logger.warning("Redefinindo para modo TEST devido a modo inválido")
                    # Continuamos com o próximo ciclo do loop
                    continue
                
                # Display performance metrics
                metrics = performance_metrics.calculate_metrics()
                logger.info(f"Performance metrics: {metrics}")
                
                # Check for errors and report
                error_count = error_tracker.get_error_count()
                if error_count > 0:
                    logger.warning(f"Detected {error_count} errors in this session")
                    if error_count > 10:
                        logger.error("Too many errors. Consider restarting the bot")
                
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {str(e)}")
                error_tracker.log_error('MAIN_LOOP_ERROR', str(e))
                # Aguarda um tempo antes de tentar novamente
                time.sleep(60)
        
        logger.info("Bot stopped gracefully")
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        error_tracker.log_error('CRITICAL_ERROR', str(e))
    finally:
        # Cleanup
        logger.info("Performing cleanup")
        try:
            # Save performance metrics
            metrics_file = config_manager.get_value('Logging', 'metrics_file', 'performance_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(performance_metrics.calculate_metrics(), f, indent=4)
            logger.info(f"Performance metrics saved to {metrics_file}")
            
            # Save error log
            error_log_file = config_manager.get_value('Logging', 'error_log_file', 'error_log.json')
            error_tracker.save_to_file(error_log_file)
            logger.info(f"Error log saved to {error_log_file}")
            
            # Disconnect from API
            if ferramental:
                ferramental.disconnect()
                logger.info("Disconnected from API")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()
