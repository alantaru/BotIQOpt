import os
import time
from iqoptionapi.stable_api import IQ_Option

class Ferramental:
    def __init__(self, asset_pairs):
        self.iq_option = None
        self.asset_pairs = asset_pairs

    def connect(self):
        email = os.getenv("IQ_OPTION_EMAIL")
        password = os.getenv("IQ_OPTION_PASSWORD")
        try:
            self.iq_option = IQ_Option(email, password)
            self.iq_option.connect()
            if self.iq_option.check_connect():
                print("Connected to IQ Option API")
                return True
            else:
                print("Failed to connect to IQ Option API")
                return False
        except Exception as e:
            print(f"Error connecting to IQ Option API: {e}")
            return False

    def get_candles(self, asset, timeframe_type, timeframe_value, count):
        if self.iq_option is None:
            print("Not connected to IQ Option API")
            return None

        # Map timeframe_type to API-compatible value
        if timeframe_type == "Minutes":
            timeframe = timeframe_value * 60
        elif timeframe_type == "Seconds":
            timeframe = timeframe_value
        elif timeframe_type == "Hours":
            timeframe = timeframe_value * 3600
        else:
            print(f"Invalid timeframe type: {timeframe_type}")
            return None

        try:
            print(f"Getting candles for {asset} with timeframe {timeframe} and count {count}")
            candles = self.iq_option.get_candles(asset, timeframe, count, time.time())
            print(f"Candles: {candles}")
            return candles
        except Exception as e:
            print(f"Error getting candles: {e}")
            return None

    def buy(self, asset, amount):
        if self.iq_option is None:
            print("Not connected to IQ Option API")
            return None
        try:
            status,id = self.iq_option.buy(asset, amount, "call", 1)
            return status, id
        except Exception as e:
            print(f"Error buying: {e}")
            return False, None

    def sell(self, asset, amount):
        if self.iq_option is None:
            print("Not connected to IQ Option API")
            return None
        try:
            status,id = self.iq_option.sell(asset, amount, "put", 1)
            return status, id
        except Exception as e:
            print(f"Error selling: {e}")
            return False, None
