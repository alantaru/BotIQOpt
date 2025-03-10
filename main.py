import os
import argparse
from dotenv import load_dotenv
from inteligencia.Inteligencia import Inteligencia
from ferramental.Ferramental import Ferramental
import pandas as pd

load_dotenv()

# Core modules
import configparser
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="IQ Option Bot")
    parser.add_argument("--mode", help="Bot mode (Download, Learning, Test, Real)")
    parser.add_argument("--asset", help="Asset to trade")
    parser.add_argument("--timeframe_type", help="Timeframe type (Minutes, Seconds, Hours)")
    parser.add_argument("--timeframe_value", type=int, help="Timeframe value")
    parser.add_argument("--candle_count", type=int, help="Number of candles to download")
    args = parser.parse_args()

    logging.info("IQ Option Bot started")
    print("Bot started")
    # Add main logic here
    config = configparser.ConfigParser()
    config.read('conf.ini')

    if args.mode:
        mode = args.mode
    else:
        mode = config['General']['operation_mode']

    inteligencia = Inteligencia(config['General']['model_filename'], config['General']['historical_data_filename'])
    assets = config['General']['assets'].split(',')
    ferramental = Ferramental(assets)

    print(f"Bot running in {mode} mode")

    def display_status(mode, asset, timeframe_type, timeframe_value, candle_count):
        print("-----------------------------------")
        print(f"Mode: {mode}")
        print(f"Asset: {asset}")
        print(f"Timeframe Type: {timeframe_type}")
        print(f"Timeframe Value: {timeframe_value}")
        print(f"Candle Count: {candle_count}")
        print("-----------------------------------")
    
    if ferramental.connect():
        logging.info("Connected to IQ Option API")
        inteligencia.load_model()
        inteligencia.set_mode(mode)
    else:
        logging.error("Failed to connect to IQ Option API")
        print("Failed to connect to IQ Option API")
        return

    if mode == "DOWNLOAD":
        asset = args.asset or config['General']['asset']
        timeframe_type = args.timeframe_type or config['General']['timeframe_type']
        timeframe_value = args.timeframe_value or int(config['General']['timeframe_value'])
        candle_count = args.candle_count or int(config['General']['candle_count'])
        display_status(mode, asset, timeframe_type, timeframe_value, candle_count)
        logging.info(f"Downloading historical data for {asset} with timeframe type {timeframe_type} and value {timeframe_value}, candle count {candle_count}")
        print(f"asset: {asset}, timeframe type: {timeframe_type}, timeframe value: {timeframe_value}, candle_count: {candle_count}")
        print("About to call download_historical_data")
        inteligencia.download_historical_data(ferramental, asset, timeframe_type, timeframe_value, candle_count)
        print("Returned from download_historical_data")

    elif mode == "LEARNING":
        asset = args.asset or config['General']['assets']
        timeframe_type = args.timeframe_type or config['General']['timeframe_type']
        timeframe_value = args.timeframe_value or int(config['General']['timeframe_value'])
        candle_count = args.candle_count or int(config['General']['candle_count'])
        display_status(mode, asset, timeframe_type, timeframe_value, candle_count)
        logging.info(f"Learning with historical data for {asset} with timeframe type {timeframe_type} and value {timeframe_value}, candle count {candle_count}")
        inteligencia.load_historical_data()
        test_size = float(config['Learning']['test_size'])
        if inteligencia.historical_data:
            inteligencia.train(inteligencia.historical_data, test_size)
        else:
            candles = ferramental.get_candles(asset, timeframe_type, timeframe_value, candle_count)
            if candles:
                inteligencia.train(candles, test_size)
                inteligencia.save_model()
            else:
                logging.warning("No candles to train on")
                print("No candles to train on")
    elif mode == "TEST" or mode == "REAL":
        timeframe_type = args.timeframe_type or config['General']['timeframe_type']
        timeframe_value = args.timeframe_value or int(config['General']['timeframe_value'])
        candle_count = args.candle_count or int(config['General']['candle_count'])
        inteligencia.set_mode(mode)
        for asset in assets:
            asset = asset.strip()
            display_status(mode, asset, timeframe_type, timeframe_value, candle_count)
            candles = ferramental.get_candles(asset, timeframe_type, timeframe_value, candle_count)
            if candles:
                if len(candles) >= 14:
                    prediction_data = pd.DataFrame(candles[-14:])
                else:
                    prediction_data = pd.DataFrame(candles)
                    logging.warning(f"Not enough candles for prediction for {asset}. Using all {len(candles)} available candles.")

                if not prediction_data.empty:
                    prediction = inteligencia.predict(prediction_data)
                    logging.info(f"Prediction for {asset}: {prediction}")
                    print(f"Prediction for {asset}: {prediction}")

                    last_candle = candles[-1]
                    amount = float(config['Trading']['amount'])

                    if prediction is not None and prediction > last_candle['close']:
                        logging.info(f"Buying {asset} with amount {amount}")
                        print(f"Buying {asset}")
                        try:
                            status, id = ferramental.buy(asset, amount)
                            if status:
                                logging.info(f"Buy status: {status}, id: {id}")
                                print(f"Buy status: {status}, id: {id}")
                            else:
                                logging.error(f"Buy failed for {asset}")
                                print(f"Buy failed for {asset}")
                        except Exception as e:
                            logging.error(f"Error during buy: {e}")
                            print(f"Error during buy: {e}")
                    elif prediction is not None:
                        logging.info(f"Selling {asset} with amount {amount}")
                        print(f"Selling {asset}")
                        try:
                            status, id = ferramental.sell(asset, amount)
                            if status:
                                logging.info(f"Sell status: {status}, id: {id}")
                                print(f"Sell status: {status}, id: {id}")
                            else:
                                logging.error(f"Sell failed for {asset}")
                                print(f"Sell failed for {asset}")
                        except Exception as e:
                            logging.error(f"Error during sell: {e}")
                            print(f"Error during sell: {e}")
                    else:
                        logging.warning(f"No prediction available for {asset}")
                else:
                    logging.warning(f"Not enough candles for prediction for {asset}") # This case should now be handled above.
            else:
                logging.warning(f"Could not retrieve candles for {asset} with timeframe type: {timeframe_type}, value: {timeframe_value} and candle count {candle_count}")
            inteligencia.analyze_performance()
            if inteligencia.should_switch_to_real():
                mode = "REAL"
                logging.info(f"Switching to {mode} mode")
                print(f"Switching to {mode} mode")
            time.sleep(1)


if __name__ == "__main__":
    main()
