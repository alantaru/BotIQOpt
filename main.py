import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

import os
logger.debug("Importing os")  # Added debug log

import argparse
logger.debug("Importing argparse")  # Added debug log

import time
logger.debug("Importing time")  # Added debug log

import signal
logger.debug("Importing signal")  # Added debug log

import sys
logger.debug("Importing sys")  # Added debug log

from contextlib import contextmanager
logger.debug("Importing contextlib")  # Added debug log

from typing import Dict, List, Optional
logger.debug("Importing typing")  # Added debug log

from datetime import datetime
logger.debug("Importing datetime")  # Added debug log

import json
logger.debug("Importing json")  # Added debug log

logger.debug("Importing pandas")
import pandas as pd
logger.debug("Pandas imported successfully")  # Added debug log

logger.debug("Importing numpy")
import numpy as np
logger.debug("Numpy imported successfully")  # Added debug log

logger.debug("Importing dotenv")
from dotenv import load_dotenv
logger.debug("Dotenv imported successfully")  # Added debug log

logger.debug("Importing configparser")
import configparser
logger.debug("Configparser imported successfully")  # Added debug log

logger.debug("Importing Inteligencia")
from inteligencia.Inteligencia import Inteligencia
logger.debug("Inteligencia imported successfully")  # Added debug log

logger.debug("Importing Ferramental")
from ferramental.Ferramental import Ferramental
logger.debug("Ferramental imported successfully")  # Added debug log

logger.debug("Importing ErrorTracker")
from ferramental.ErrorTracker import ErrorTracker
logger.debug("ErrorTracker imported successfully")  # Added debug log

logger.debug("Importing BotPerformanceMetrics")
from ferramental.PerformanceMetrics import BotPerformanceMetrics
logger.debug("BotPerformanceMetrics imported successfully")  # Added debug log

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Graceful shutdown handler
def signal_handler(sig, frame):
    logger.info("Received shutdown signal. Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@contextmanager
def resource_manager(resources: Dict[str, any]):
    """Context manager for resource cleanup"""
    try:
        yield resources
    finally:
        logger.info("Cleaning up resources...")
        for name, resource in resources.items():
            if hasattr(resource, 'close'):
                resource.close()
                logger.info(f"Closed resource: {name}")

def get_params(config, args, section, param_names):
    """Gets and validates parameters from command line arguments or config file."""
    params = {}
    type_mapping = {
        'timeframe_value': int,
        'candle_count': int,
        'max_trades': int,
        'risk_per_trade': float,
        'auto_switch_to_real': lambda x: x.lower() == 'true'
    }
    
    for param_name in param_names:
        try:
            arg_value = getattr(args, param_name, None)
            config_value = config.get(section, param_name, fallback=None)
            
            if arg_value is not None:
                value = arg_value
            elif config_value is not None:
                value = config_value
            else:
                if param_name != "auto_switch_to_real":
                    raise ValueError(f"Missing required parameter: {param_name}")
                continue
                
            # Apply type conversion if specified
            if param_name in type_mapping:
                converter = type_mapping[param_name]
                try:
                    params[param_name] = converter(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for {param_name}: {value}. Error: {str(e)}")
            else:
                params[param_name] = value
                
        except Exception as e:
            logging.error(f"Error processing parameter {param_name}: {str(e)}")
            raise
            
    return params

def calculate_timeframe(timeframe_type, timeframe_value):
    """Calculates timeframe in seconds."""
    if timeframe_type == "Minutes":
        return timeframe_value * 60
    elif timeframe_type == "Seconds":
        return timeframe_value
    elif timeframe_type == "Hours":
        return timeframe_value * 3600
    else:
        raise ValueError(f"Invalid timeframe type: {timeframe_type}")

def progress_bar(value: float, width: int = 40) -> str:
    """Generate ASCII progress bar"""
    filled = int(round(value * width))
    return '[' + '=' * filled + ' ' * (width - filled) + ']'

def plot_equity_curve(metrics) -> str:
    """Generate enhanced ASCII equity curve with markers and trend lines"""
    equity_curve = []
    current_equity = 0.0
    
    for trade in metrics.trades:
        current_equity += trade['amount'] if trade['result'] else -trade['amount']
        equity_curve.append(current_equity)
    
    if not equity_curve:
        return "No trades yet"
        
    max_equity = max(equity_curve)
    min_equity = min(equity_curve)
    scale = max_equity - min_equity
    
    if scale == 0:
        return "Flat equity"
        
    height = 10
    width = 60
    chart = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot equity points
    for i, value in enumerate(equity_curve):
        x = min(i * width // len(equity_curve), width - 1)
        y = int(((value - min_equity) / scale) * (height - 1))
        chart[height - 1 - y][x] = '●'
    
    # Add trend line
    if len(equity_curve) > 1:
        start_x = 0
        start_y = height - 1 - int((equity_curve[0] - min_equity) / scale * (height - 1))
        end_x = width - 1
        end_y = height - 1 - int((equity_curve[-1] - min_equity) / scale * (height - 1))
        
        # Bresenham's line algorithm
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        sx = 1 if start_x < end_x else -1
        sy = 1 if start_y < end_y else -1
        err = dx - dy
        
        while True:
            if 0 <= start_x < width and 0 <= start_y < height:
                if chart[start_y][start_x] == ' ':
                    chart[start_y][start_x] = '─'
            if start_x == end_x and start_y == end_y:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                start_x += sx
            if e2 < dx:
                err += dx
                start_y += sy
    
    # Add markers for key points
    max_idx = equity_curve.index(max_equity)
    min_idx = equity_curve.index(min_equity)
    
    max_x = min(max_idx * width // len(equity_curve), width - 1)
    max_y = height - 1 - int((max_equity - min_equity) / scale * (height - 1))
    min_x = min(min_idx * width // len(equity_curve), width - 1)
    min_y = height - 1 - int((min_equity - min_equity) / scale * (height - 1))
    
    if 0 <= max_x < width and 0 <= max_y < height:
        chart[max_y][max_x] = '▲'
            
        # Basic metrics
        successful_trades = [t for t in metrics.trades if t['result'] is True]
        win_rate = len(successful_trades) / len(metrics.trades)
        
        # Advanced metrics
        max_drawdown = metrics.calculate_max_drawdown()
        sortino_ratio = metrics.calculate_sortino_ratio()
        expectancy = metrics.calculate_expectancy()
        risk_of_ruin = metrics.calculate_risk_of_ruin(win_rate)
        
        # Trade statistics
        avg_win = metrics.calculate_average_win()
        avg_loss = metrics.calculate_average_loss()
        profit_factor = metrics.calculate_profit_factor()
        
        # Risk alerts
        risk_alerts = []
        if max_drawdown > 0.2:
            risk_alerts.append("High drawdown detected")
        if risk_of_ruin > 0.1:
            risk_alerts.append("High risk of ruin")
        if expectancy < 0:
            risk_alerts.append("Negative expectancy")
            
        return {
            'total_trades': len(metrics.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': metrics.calculate_sharpe_ratio(),
            'sortino_ratio': sortino_ratio,
            'expectancy': expectancy,
            'risk_of_ruin': risk_of_ruin,
            'max_drawdown': max_drawdown,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'risk_alerts': risk_alerts
        }
        
class BotPerformanceMetrics:
    def __init__(self):
        self.trades = []

    def add_trade(self, asset, direction, amount, result):
        self.trades.append({
            'asset': asset,
            'direction': direction,
            'amount': amount,
            'result': result
        })

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        equity_curve = []
        current_equity = 0.0
        
        for trade in self.trades:
            current_equity += trade['amount'] if trade['result'] else -trade['amount']
            equity_curve.append(current_equity)
            
        peak = -float('inf')
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
        
    def calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (risk-adjusted return using downside deviation)"""
        returns = [t['amount'] if t['result'] else -t['amount'] for t in self.trades]
        if not returns:
            return 0.0
            
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf')
            
        downside_deviation = np.std(downside_returns)
        return mean_return / downside_deviation if downside_deviation != 0 else 0.0
        
    def calculate_expectancy(self) -> float:
        """Calculate trading system expectancy"""
        wins = [t['amount'] for t in self.trades if t['result']]
        losses = [-t['amount'] for t in self.trades if not t['result']]
        
        if not wins or not losses:
            return 0.0
            
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        win_rate = len(wins) / len(self.trades)
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
    def calculate_risk_of_ruin(self, win_rate: float) -> float:
        """Calculate probability of losing entire account"""
        if not self.trades:
            return 0.0
            
        risk_per_trade = np.mean([t['amount'] for t in self.trades])
        account_size = sum(t['amount'] for t in self.trades if t['result']) - sum(t['amount'] for t in self.trades if not t['result'])
        
        if account_size <= 0:
            return 1.0
            
        num_trades = len(self.trades)
        risk_percent = risk_per_trade / account_size
        return ((1 - win_rate) / win_rate) ** (account_size / risk_per_trade)
        
    def calculate_average_win(self) -> float:
        """Calculate average winning trade amount"""
        wins = [t['amount'] for t in self.trades if t['result']]
        return np.mean(wins) if wins else 0.0
        
    def calculate_average_loss(self) -> float:
        """Calculate average losing trade amount"""
        losses = [t['amount'] for t in self.trades if not t['result']]
        return np.mean(losses) if losses else 0.0
        
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        profits = sum(t['amount'] for t in self.trades if t['result'] is True)
        losses = sum(t['amount'] for t in self.trades if t['result'] is False)
        return profits / losses if losses != 0 else float('inf')
        
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for risk-adjusted returns"""
        returns = [t['amount'] if t['result'] else -t['amount'] for t in self.trades]
        if not returns:
            return 0.0
            
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        risk_free_rate = 0.0  # Could be configurable
        
        return (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0.0

    def calculate_metrics(self):
        successful_trades = [t for t in self.trades if t['result'] is True]
        win_rate = len(successful_trades) / len(self.trades)
        
        # Advanced metrics
        max_drawdown = self.calculate_max_drawdown()
        sortino_ratio = self.calculate_sortino_ratio()
        expectancy = self.calculate_expectancy()
        risk_of_ruin = self.calculate_risk_of_ruin(win_rate)
        
        # Trade statistics
        avg_win = self.calculate_average_win()
        avg_loss = self.calculate_average_loss()
        profit_factor = self.calculate_profit_factor()
        
        # Risk alerts
        risk_alerts = []
        if max_drawdown > 0.2:
            risk_alerts.append("High drawdown detected")
        if risk_of_ruin > 0.1:
            risk_alerts.append("High risk of ruin")
        if expectancy < 0:
            risk_alerts.append("Negative expectancy")
            
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': sortino_ratio,
            'expectancy': expectancy,
            'risk_of_ruin': risk_of_ruin,
            'max_drawdown': max_drawdown,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'risk_alerts': risk_alerts
        }

def validate_config(config):
    """Validate configuration parameters"""
    required_sections = ['General', 'Learning', 'Trading', 'Logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")
            
    # Validate General section
    if not config['General'].get('operation_mode'):
        raise ValueError("operation_mode is required in General section")
        
    # Validate Trading section
    if not config['Trading'].get('amount'):
        raise ValueError("amount is required in Trading section")
        
    return True

def main():
    logger.debug("Entering main function")  # Added debug log
    load_dotenv()  # Load environment variables from .env file

    # Initialize logging and error tracking
    error_tracker = ErrorTracker()
    
    parser = argparse.ArgumentParser(description="IQ Option Bot")
    parser.add_argument("--mode", help="Bot mode (Download, Learning, Test, Real)")
    parser.add_argument("--asset", help="Asset to trade")
    parser.add_argument("--timeframe_type", help="Timeframe type (Minutes, Seconds, Hours)")
    parser.add_argument("--timeframe_value", type=int, help="Timeframe value")
    parser.add_argument("--candle_count", type=int, help="Number of candles to download/use")
    parser.add_argument("--max_trades", type=int, help="Maximum number of trades per asset")
    parser.add_argument("--risk_per_trade", type=float, help="Risk percentage per trade")
    args = parser.parse_args()

    # Initialize performance tracking
    performance_metrics = BotPerformanceMetrics()
    
    logger.info("IQ Option Bot started")
    logger.info(f"Initializing in {args.mode or 'default'} mode")

    config = configparser.ConfigParser()
    config.read('conf.ini')
    logger.debug("Config file read successfully")  # Added debug log

    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        error_tracker.log_error('CONFIG_ERROR', str(e))
        sys.exit(1)

    mode = args.mode if args.mode else config['General']['operation_mode']
    logger.debug(f"Selected mode: {mode}")  # Added debug log

    # Get general parameters with asset handling
    general_params = get_params(config, args, 'General', ['timeframe_type', 'timeframe_value', 'candle_count', 'auto_switch_to_real'])
    
    # Handle assets separately
    if args.asset:
        assets = [args.asset]
    else:
        assets = [asset.strip() for asset in config['General']['assets'].split(',') if asset.strip()]
        if not assets:
            raise ValueError("No valid assets configured in config file")
        general_params['asset'] = assets[0]  # Use first asset as default

    try:
        # Initialize components with error handling
        logger.debug("Initializing Inteligencia")  # Added debug log
        inteligencia = Inteligencia(
            model_path=config['General']['model_filename'],
            historical_data_filename=config['General']['historical_data_filename']
        )
        logger.debug("Inteligencia initialized successfully")  # Added debug log

        # Set auto_switch_to_real in Inteligencia
        inteligencia.set_auto_switch(general_params.get('auto_switch_to_real', False))

        assets = [asset.strip() for asset in config['General']['assets'].split(',') if asset.strip()]

        if not assets:
            raise ValueError("No valid assets configured")

        logger.debug("Initializing Ferramental")  # Added debug log
        ferramental = Ferramental(assets)
        logger.debug("Ferramental initialized successfully")  # Added debug log

        # Set timezone from config if specified
        timezone = config['General'].get('timezone', 'UTC')
        os.environ['TZ'] = timezone
        if hasattr(time, 'tzset'):
            time.tzset()

    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        error_tracker.log_error('INIT_ERROR', str(e))
        sys.exit(1)

    logger.info(f"Bot running in {mode} mode")
    logger.debug("Initialization complete")  # Added debug log
    logger.info(f"Timezone: {time.tzname[0]}")
    logger.info(f"Available assets: {', '.join(assets)}")

    # Validate environment variables
    required_env_vars = ["IQ_OPTION_EMAIL", "IQ_OPTION_PASSWORD"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Please set the following environment variables in .env file: {', '.join(missing_vars)}")
        return

    # Check Python version and dependencies
    if sys.version_info < (3, 8):
        logging.error("Python 3.8 or higher is required")
        return

    try:
        import numpy as np
        import pandas as pd
        from dotenv import load_dotenv
    except ImportError as e:
        logging.error(f"Missing required dependency: {str(e)}")
        print(f"Please install missing dependencies: {str(e)}")
        return

    # Connect to API with retries
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            if ferramental.connect():
                break
            else:
                logging.warning(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"Connection error: {str(e)}")
            error_tracker.log_error('CONNECTION_ERROR', str(e))
            if attempt == max_retries - 1:
                logging.error("Max connection attempts reached. Exiting...")
                sys.exit(1)
            time.sleep(retry_delay)

    # Main bot loop
    try:
        logger.info("Starting main bot loop")

        while True:
            # Check current mode and execute appropriate logic
            logger.debug(f"Current mode: {mode}")  # Added debug log
            if mode == "Download":
                logger.info("Starting data download")
                try:
                    for asset in assets:
                        logger.info(f"Downloading historical data for {asset}")
                        data = ferramental.download_historical_data(
                            asset=asset,
                            timeframe_type=general_params['timeframe_type'],
                            timeframe_value=general_params['timeframe_value'],
                            candle_count=general_params['candle_count']
                        )
                        if data is not None:
                            inteligencia.store_training_data(data)
                            logger.info(f"Successfully downloaded data for {asset}")
                        else:
                            logger.error(f"Failed to download data for {asset}")
                            error_tracker.log_error('DATA_DOWNLOAD_ERROR', f"Failed to download data for {asset}")

                except Exception as e:
                    logger.error(f"Error during data download: {str(e)}")
                    error_tracker.log_error('DATA_DOWNLOAD_ERROR', str(e))
                    break

            elif mode == "Learning":
                logger.info("Starting learning process")
                try:
                    inteligencia.train_model()
                    logger.info("Model training completed successfully")
                except Exception as e:
                    logger.error(f"Error during model training: {str(e)}")
                    error_tracker.log_error('TRAINING_ERROR', str(e))
                    break

            elif mode == "Test":
                logger.info("Starting test trading")
                try:
                    # Get predictions from AI
                    predictions = inteligencia.get_predictions(assets)

                    # Execute test trades
                    for asset, prediction in predictions.items():
                        if prediction['confidence'] > 0.7:  # Only trade with high confidence
                            result = ferramental.execute_test_trade(
                                asset=asset,
                                direction=prediction['direction'],
                                amount=general_params['risk_per_trade']
                            )
                            performance_metrics.add_trade(
                                asset=asset,
                                direction=prediction['direction'],
                                amount=general_params['risk_per_trade'],
                                result=result
                            )
                            logger.info(f"Test trade executed for {asset}")

                except Exception as e:
                    logger.error(f"Error during test trading: {str(e)}")
                    error_tracker.log_error('TEST_TRADING_ERROR', str(e))
                    break

            elif mode == "Real":
                logger.info("Starting real trading")
                try:
                    # Get predictions from AI
                    predictions = inteligencia.get_predictions(assets)

                    # Execute real trades
                    for asset, prediction in predictions.items():
                        if prediction['confidence'] > 0.8:  # Higher threshold for real trades
                            result = ferramental.execute_real_trade(
                                asset=asset,
                                direction=prediction['direction'],
                                amount=general_params['risk_per_trade']
                            )
                            performance_metrics.add_trade(
                                asset=asset,
                                direction=prediction['direction'],
                                amount=general_params['risk_per_trade'],
                                result=result
                            )
                            logger.info(f"Real trade executed for {asset}")

                except Exception as e:
                    logger.error(f"Error during real trading: {str(e)}")
                    error_tracker.log_error('REAL_TRADING_ERROR', str(e))
                    break

            else:
                logger.error(f"Invalid mode: {mode}")
                break

            # Display performance metrics
            metrics = performance_metrics.calculate_metrics()
            logger.info(f"Performance metrics: {metrics}")

            # Check for auto-switch conditions
            if mode == "Test" and inteligencia.should_switch_to_real():
                logger.info("Switching to real trading mode based on performance")
                mode = "Real"
                inteligencia.set_auto_switch(False)  # Disable auto-switch after first switch
                continue

            # Sleep between iterations
            timeframe = calculate_timeframe(
                general_params['timeframe_type'],
                general_params['timeframe_value']
            )
            logger.debug(f"Sleeping for {timeframe} seconds")  # Added debug log
            time.sleep(timeframe)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {str(e)}")
        error_tracker.log_error('MAIN_LOOP_ERROR', str(e))
    finally:
        logger.info("Shutting down bot")
        # Save final model state
        try:
            inteligencia.save_model()
            logger.info("Model state saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            error_tracker.log_error('MODEL_SAVE_ERROR', str(e))

        # Generate final performance report
        try:
            metrics = performance_metrics.calculate_metrics()
            logger.info("Final performance metrics:")
            logger.info(json.dumps(metrics, indent=2))
            logger.info("\nEquity curve:")
            logger.info(plot_equity_curve(performance_metrics))
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            error_tracker.log_error('FINAL_REPORT_ERROR', str(e))
        logger.debug("Exiting main function")  # Added debug log

if __name__ == "__main__":
    main()
