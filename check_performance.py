import sys
import os
import time

# Adiciona o diretório atual ao path para importar os módulos
sys.path.append(os.getcwd())

from utils.ConfigManager import ConfigManager
from utils.ErrorTracker import ErrorTracker
from ferramental.Ferramental import Ferramental

def check():
    config_manager = ConfigManager()
    error_tracker = ErrorTracker()
    ferramental = Ferramental(config_manager, error_tracker)
    
    print("--- PERFORMANCE AUDIT ---")
    print("Connecting to API...")
    success, error = ferramental.connect()
    
    if not success:
        print(f"Failed to connect: {error}")
        return
    
    print("Connected successfully.")
    
    # Check Balance
    balance = ferramental.get_balance()
    currency = ferramental.get_currency()
    print(f"Current Balance: {balance} {currency}")
    
    # Check Results
    results = ferramental.get_trade_results()
    if not results:
        print("No trade results found in recent history.")
    else:
        print(f"\nRecent Trades (Last {len(results[:10])}):")
        wins = 0
        total_profit = 0
        for trade in results[:10]:
            print(f"ID: {trade['id']} | Asset: {trade['asset']} | Result: {'WIN' if trade['is_win'] else 'LOSS'} | Profit: {trade['profit']:.2f}")
            if trade['is_win']: wins += 1
            total_profit += trade['profit']
            
        print(f"\nSummary (Last 10):")
        print(f"Win Rate: {(wins/len(results[:10]))*100:.1f}%")
        print(f"Total Session Profit: {total_profit:.2f}")

if __name__ == "__main__":
    check()
