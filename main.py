import os
import argparse
import time
import logging
import signal
import sys
from contextlib import contextmanager
from typing import Dict, List, Optional
from datetime import datetime
import json

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import configparser

from inteligencia.Inteligencia import Inteligencia
from ferramental.Ferramental import Ferramental
from ferramental.ErrorTracker import ErrorTracker

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
    """Gets parameters from command line arguments or config file."""
    params = {}
    for param_name in param_names:
        arg_value = getattr(args, param_name, None)
        config_value = config.get(section, param_name, fallback=None)
        if arg_value is not None:
            params[param_name] = arg_value
        elif config_value is not None:
            if param_name in ("timeframe_value", "candle_count"):
                params[param_name] = int(config_value)
            elif param_name == "auto_switch_to_real":
                params[param_name] = config_value.lower() == 'true'
            else:
                params[param_name] = config_value
        else:
            # Handle missing required parameters (except for auto_switch_to_real, which has a default)
            if param_name != "auto_switch_to_real":
                raise ValueError(f"Missing required parameter: {param_name}")
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
    if 0 <= min_x < width and 0 <= min_y < height:
        chart[min_y][min_x] = '▼'
    
    # Add legend
    legend = [
        "┌────────────────────────────────────────────────────────────┐",
        "│ Legend:                                                    │",
        "│ ● - Equity point                                           │",
        "│ ▲ - Maximum equity                                         │",
        "│ ▼ - Minimum equity                                         │",
        "│ ─ - Trend line                                             │",
        "└────────────────────────────────────────────────────────────┘"
    ]
    
    chart_str = '\n'.join(''.join(row) for row in chart)
    return f"{chart_str}\n\n{'\n'.join(legend)}"

class BotPerformanceMetrics:
    """Class to track and analyze bot performance"""
    def __init__(self):
        self.start_time = datetime.now()
        self.trades = []
        self.performance_history = []
        
    def add_trade(self, asset: str, direction: str, amount: float, result: Optional[bool]):
        """Record trade details"""
        trade = {
            'timestamp': datetime.now(),
            'asset': asset,
            'direction': direction,
            'amount': amount,
            'result': result
        }
        self.trades.append(trade)
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        if not self.trades:
            return {}
            
        # Basic metrics
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
    
    try:
        validate_config(config)
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        error_tracker.log_error('CONFIG_ERROR', str(e))
        sys.exit(1)

    mode = args.mode if args.mode else config['General']['operation_mode']
    
    # Add auto_switch_to_real to general parameters
    general_params = get_params(config, args, 'General', ['asset', 'timeframe_type', 'timeframe_value', 'candle_count', 'auto_switch_to_real'])

    try:
        # Initialize components with error handling
        inteligencia = Inteligencia(
            model_path=config['General']['model_filename'],
            historical_data_filename=config['General']['historical_data_filename']
        )
        
        # Set auto_switch_to_real in Inteligencia
        inteligencia.set_auto_switch(general_params.get('auto_switch_to_real', False))
        
        assets = [asset.strip() for asset in config['General']['assets'].split(',') if asset.strip()]

        if not assets:
            raise ValueError("No valid assets configured")

        ferramental = Ferramental(assets)

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
    logger.info(f"Timezone: {time.tzname[0]}")
    logger.info(f"Available assets: {', '.join(assets)}")

    # Check if environment variables are loaded
    if not os.getenv("IQ_OPTION_EMAIL") or not os.getenv("IQ_OPTION_PASSWORD"):
        logging.error("IQ_OPTION_EMAIL and IQ_OPTION_PASSWORD environment variables must be set in .env file.")
        print("IQ_OPTION_EMAIL and IQ_OPTION_PASSWORD environment variables must be set in .env file.")
        return

    if not ferramental.connect():
        logging.error("Failed to connect to IQ Option API")
        print("Failed to connect to IQ Option API")
        return

    inteligencia.set_mode(mode)

    # Only load model if not in LEARNING mode
    if mode != "LEARNING":
        inteligencia.load_model()

    if mode == "DOWNLOAD":
        params = get_params(config, args, 'General', ['asset', 'timeframe_type', 'timeframe_value', 'candle_count'])
        print(f"Downloading historical data for {params['asset']}...")
        inteligencia.download_historical_data(ferramental, params['asset'], params['timeframe_type'], params['timeframe_value'], params['candle_count'])

    elif mode == "LEARNING":
        params = get_params(config, args, 'General', ['asset', 'timeframe_type', 'timeframe_value', 'candle_count'])
        test_size = float(config['Learning']['test_size'])
        print(f"Learning with historical data for {params['asset']}...")
        inteligencia.load_historical_data()
        if inteligencia.historical_data:
            inteligencia.train(inteligencia.historical_data, test_size)
        else:
            candles = ferramental.get_candles(params['asset'], params['timeframe_type'], params['timeframe_value'], params['candle_count'])
            if candles:
                inteligencia.train(candles, test_size)
                inteligencia.save_model()
            else:
                print("No candles to train on.")

    elif mode in ("TEST", "REAL"):
        params = get_params(config, args, 'General', ['timeframe_type', 'timeframe_value', 'candle_count'])
        timeframe = calculate_timeframe(params['timeframe_type'], params['timeframe_value'])

        for asset in assets:
            asset = asset.strip()
            ferramental.stop_candles_stream(asset, timeframe)
            if ferramental.iq_option is not None:
                ferramental.start_candles_stream(asset, timeframe, 100)
            else:
                print("Not connected to IQ Option API")
                return

        try:
            while True:
                for asset in assets:
                    asset = asset.strip()
                    candles = ferramental.get_realtime_candles(asset, timeframe)
                    if candles:
                        if len(candles) >= 14:
                            prediction_data = pd.DataFrame(list(candles.values())[-14:])
                        else:
                            prediction_data = pd.DataFrame(list(candles.values()))
                            logging.warning(f"Not enough candles for prediction for {asset}. Using all {len(candles)} available candles.")

                        if not prediction_data.empty:
                            prediction = inteligencia.predict(prediction_data)
                            logging.info(f"Prediction for {asset}: {prediction}")
                            print(f"Prediction for {asset}: {prediction}")

                            last_candle = list(candles.values())[-1]
                            amount = float(config['Trading']['amount'])

                            if prediction is not None:
                                # Calculate position size based on risk management
                                balance = ferramental.get_balance()
                                risk_per_trade = float(config['Trading'].get('risk_per_trade', 0.01))
                                position_size = balance * risk_per_trade
                                
                                # Get trade direction and confidence
                                direction = "call" if prediction > last_candle['close'] else "put"
                                confidence = abs(prediction - last_candle['close']) / last_candle['close']
                                
                                # Validate trade conditions
                                min_trade_amount = ferramental.get_min_trade_amount()
                                max_trade_amount = ferramental.get_max_trade_amount()
                                
                                if position_size < min_trade_amount:
                                    logger.warning(f"Position size {position_size:.2f} below minimum {min_trade_amount:.2f}")
                                    position_size = min_trade_amount
                                elif position_size > max_trade_amount:
                                    logger.warning(f"Position size {position_size:.2f} above maximum {max_trade_amount:.2f}")
                                    position_size = max_trade_amount
                                
                                # Check available balance
                                if position_size > balance:
                                    logger.error(f"Insufficient balance for trade: {position_size:.2f} > {balance:.2f}")
                                    continue
                                    
                                # Execute trade with position sizing
                                try:
                                    # Get current price and validate spread
                                    current_price = ferramental.get_current_price(asset)
                                    spread = ferramental.get_spread(asset)
                                    
                                    if spread > float(config['Trading'].get('max_spread', 0.05)):
                                        logger.warning(f"Spread too high: {spread:.2%} for {asset}")
                                        continue
                                        
                                    # Execute trade with additional validations
                                    status, id = ferramental.buy_digital_spot(
                                        asset, 
                                        position_size, 
                                        direction, 
                                        1,  # 1 minute expiry
                                        current_price
                                    )
                                    
                                    # Record trade in performance metrics
                                    performance_metrics.add_trade(
                                        asset=asset,
                                        direction=direction,
                                        amount=position_size,
                                        result=status
                                    )
                                    
                                    if status:
                                        logger.info(f"Trade executed: {asset} {direction} {position_size:.2f}")
                                        logger.info(f"Trade ID: {id}")
                                        
                                        # Update performance metrics
                                        metrics = performance_metrics.calculate_metrics()
                                        logger.info(f"Performance Metrics: {metrics}")
                                        
                                        # Dynamic risk adjustment
                                        if metrics['win_rate'] < 0.5:
                                            new_risk = risk_per_trade * 0.9
                                            if new_risk >= float(config['Trading'].get('min_risk', 0.005)):
                                                risk_per_trade = new_risk
                                                logger.info(f"Reducing risk per trade to {risk_per_trade:.2%}")
                                            else:
                                                logger.warning("Minimum risk level reached")
                                        elif metrics['win_rate'] > 0.7 and metrics['profit_factor'] > 1.5:
                                            new_risk = risk_per_trade * 1.1
                                            if new_risk <= float(config['Trading'].get('max_risk', 0.05)):
                                                risk_per_trade = new_risk
                                                logger.info(f"Increasing risk per trade to {risk_per_trade:.2%}")
                                            else:
                                                logger.warning("Maximum risk level reached")
                                            
                                    else:
                                        logger.error(f"Trade failed: {asset} {direction}")
                                        # Implement cooldown after failed trade
                                        time.sleep(float(config['Trading'].get('cooldown_after_fail', 5)))
                                except Exception as e:
                                    logger.error(f"Trade error: {e}")
                                    performance_metrics.add_trade(
                                        asset=asset,
                                        direction=direction,
                                        amount=position_size,
                                        result=False
                                    )
                            else:
                                logger.warning(f"No prediction available for {asset}")
                        else:
                            logging.warning(f"Not enough candles for prediction for {asset}")
                    else:
                        logging.warning(f"Could not retrieve candles for {asset}")

                time.sleep(1)
                
                # Analyze performance and adjust strategy
                performance_data = inteligencia.analyze_performance()
                metrics = performance_metrics.calculate_metrics()
                
                # Clear console and display performance dashboard
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Display performance dashboard with enhanced ASCII art
                print("""
                ╔════════════════════════════════════════════════════════════════╗
                ║                📊 IQ OPTION BOT PERFORMANCE DASHBOARD          ║
                ╠════════════════════════════════════════════════════════════════╣
                ║                                                                ║
                ║  ██████╗ ██╗   ██╗    ██████╗  ██████╗ ████████╗              ║
                ║ ██╔═══██╗╚██╗ ██╔╝    ██╔══██╗██╔═══██╗╚══██╔══╝              ║
                ║ ██║   ██║ ╚████╔╝     ██████╔╝██║   ██║   ██║                 ║
                ║ ██║   ██║  ╚██╔╝      ██╔═══╝ ██║   ██║   ██║                 ║
                ║ ╚██████╔╝   ██║       ██║     ╚██████╔╝   ██║                 ║
                ║  ╚═════╝    ╚═╝       ╚═╝      ╚═════╝    ╚═╝                 ║
                ╚════════════════════════════════════════════════════════════════╝
                """)
                
                # Display metrics with enhanced formatting
                print("\n┌──────────────────────────────────────────────────────┐")
                print("│ Performance Metrics:                                │")
                print("├──────────────────────────────────────────────────────┤")
                print(f"│ Total Trades: {metrics['total_trades']:>40} │")
                print("├──────────────────────────────────────────────────────┤")
                
                print(f"│ Win Rate: {metrics['win_rate']:.2%}")
                print(f"{progress_bar(metrics['win_rate'], 40)} │")
                
                print(f"│ Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"{progress_bar(min(metrics['profit_factor']/5, 1), 40)} │")
                
                print(f"│ Expectancy: ${metrics['expectancy']:.2f}")
                print(f"{progress_bar((metrics['expectancy'] + 10)/20, 40)} │")
                
                print(f"│ Max Drawdown: {metrics['max_drawdown']:.2%}")
                print(f"{progress_bar(1 - metrics['max_drawdown'], 40)} │")
                
                print(f"│ Risk of Ruin: {metrics['risk_of_ruin']:.2%}")
                print(f"{progress_bar(1 - metrics['risk_of_ruin'], 40)} │")
                print("└──────────────────────────────────────────────────────┘")
                
                # Display enhanced risk alerts with color coding
                if metrics['risk_alerts']:
                    print("\n┌──────────────────────────────────────────────────────┐")
                    print("│ Risk Alerts:                                       │")
                    print("├──────────────────────────────────────────────────────┤")
                    
                    for alert in metrics['risk_alerts']:
                        if "High drawdown" in alert:
                            color_code = "\033[91m"  # Red
                            symbol = "▼▼▼"
                        elif "High risk of ruin" in alert:
                            color_code = "\033[93m"  # Yellow
                            symbol = "⚠️⚠️"
                        elif "Negative expectancy" in alert:
                            color_code = "\033[91m"  # Red
                            symbol = "❌❌"
                        else:
                            color_code = "\033[93m"  # Yellow
                            symbol = "⚠️"
                            
                        print(f"│ {color_code}{symbol} {alert}\033[0m")
                        if os.name == 'nt':  # Windows
                            import winsound
                            winsound.Beep(1000, 500)
                        else:  # Linux/Mac
                            os.system('afplay /System/Library/Sounds/Ping.aiff')
                    
                    print("└──────────────────────────────────────────────────────┘\n")
                    
                # Display system status and equity curve
                print("\n┌──────────────────────────────────────────────────────┐")
                print("│ System Status:                                      │")
                print(f"│ Uptime: {str(datetime.now() - performance_metrics.start_time)[:-7]:>40} │")
                print(f"│ Active Assets: {len(assets):>36} │")
                print(f"│ Current Mode: {mode:>37} │")
                print("├──────────────────────────────────────────────────────┤")
                print("│ Equity Curve:                                       │")
                print("│ " + plot_equity_curve(performance_metrics).replace("\n", "\n│ ") + " │")
                print("└──────────────────────────────────────────────────────┘")
                
                # Adjust strategy based on performance
                if metrics['win_rate'] < 0.4:
                    logger.warning("Low win rate detected - adjusting strategy")
                    inteligencia.adjust_strategy_parameters({
                        'risk_multiplier': 0.8,
                        'min_confidence': 0.6
                    })
                elif metrics['win_rate'] > 0.7:
                    logger.info("High win rate detected - optimizing strategy")
                    inteligencia.adjust_strategy_parameters({
                        'risk_multiplier': 1.2,
                        'min_confidence': 0.4
                    })
                
                # Check mode switch conditions
                if inteligencia.should_switch_to_real():
                    mode = "REAL"
                    logging.info(f"Switching to {mode} mode")
                    print(f"\n🚀 Switching to {mode} mode")

        except KeyboardInterrupt:
            print("Exiting.")

        finally:
            for asset in assets:
                ferramental.stop_candles_stream(asset.strip(), timeframe)

if __name__ == "__main__":
    main()
