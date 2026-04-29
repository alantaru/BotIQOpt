import sys
import os

# Adiciona o diretório atual ao path para importar os módulos
sys.path.append(os.getcwd())

from utils.ConfigManager import ConfigManager
from utils.ErrorTracker import ErrorTracker
from ferramental.Ferramental import Ferramental

def check():
    config_manager = ConfigManager()
    error_tracker = ErrorTracker()
    ferramental = Ferramental(config_manager, error_tracker)
    
    print("Connecting...")
    success, error = ferramental.connect()
    
    if not success:
        print(f"Connection failed: {error}")
        return
    
    # Force switch to REAL
    print("Switching to REAL account...")
    ferramental.change_balance('REAL')
    
    # Get balance directly from iq_option
    try:
        balance = ferramental.iq_option.get_balance()
        currency = ferramental.iq_option.get_currency()
        print(f"REAL BALANCE: {balance} {currency}")
        
        # Get candles for a recent minute to confirm activity
        print("Checking scanner data...")
        candles = ferramental.iq_option.get_candles("EURUSD", 60, 1, 1000)
        if candles:
            print(f"Successfully retrieved {len(candles)} candles.")
        
        # Get results
        print("Fetching trade history...")
        history = ferramental.iq_option.get_optioninfo_v2(10)
        if history:
            print(f"Recent History Found: {len(history)} trades.")
            for h in history:
                print(f"ID: {h.get('id')} | Win: {h.get('win')} | Profit: {h.get('profit')}")
        else:
            print("No recent trade history found.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check()
