import sys
import os

sys.path.append(os.getcwd())

from utils.ConfigManager import ConfigManager
from utils.ErrorTracker import ErrorTracker
from ferramental.Ferramental import Ferramental

def check():
    config_manager = ConfigManager()
    error_tracker = ErrorTracker()
    ferramental = Ferramental(config_manager, error_tracker)
    
    ferramental.connect()
    ferramental.change_balance('REAL')
    
    # Simple check via ferramental's own method
    balance = ferramental.get_balance()
    currency = ferramental.get_currency()
    print(f"REAL_BALANCE_REPORTED: {balance} {currency}")

if __name__ == "__main__":
    check()
