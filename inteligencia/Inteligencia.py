import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import logging

class Inteligencia:
    def __init__(self, model_filename="model.joblib", historical_data_filename="historical_data.csv"):
        self.model = None
        self.mode = "Test" # Default mode
        self.historical_data = []
        self.model_filename = model_filename
        self.historical_data_filename = historical_data_filename
        #self.load_model(filename=self.model_filename)

    def set_mode(self, mode):
        print(f"Setting mode to: {mode}")
        self.mode = mode

    def download_historical_data(self, ferramental, asset, timeframe_type, timeframe_value, count):
        print(f"download_historical_data: self.mode = {self.mode}")
        print(f"self.mode inside download_historical_data: {self.mode}")
        logging.info(f"Downloading historical data for asset: {asset}, timeframe type: {timeframe_type}, timeframe value: {timeframe_value}, count: {count}")
        print("Downloading historical data...")
        self.historical_data = ferramental.get_candles(asset, timeframe_type, timeframe_value, count)
        if self.historical_data:
            self.save_historical_data(self.historical_data_filename)
            print("Historical data downloaded")
        else:
            print("Failed to download historical data")

    def engineer_features(self, df):
        """Adds technical indicators to the DataFrame."""
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df[' volatility'] = df['max'] - df['min']
        df['RSI'] = self.calculate_rsi(df['close'], window=14)
        df['MACD'] = self.calculate_macd(df['close'])
        return df.dropna() # Drop rows with NaN values after feature engineering

    def train(self, data, test_size):
        print(f"train: self.mode = {self.mode}")
        if not data:
            print("No data to train on")
            return

        # Assuming data is a list of candles, convert to DataFrame
        df = pd.DataFrame(data)
        
        # Split data *before* feature engineering
        X = df[['open', 'max', 'min', 'close', 'volume']]
        y = df['close'].shift(-1)  # Predict the next closing price
        X = X[:-1]
        y = y[:-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        # Now apply feature engineering to the split data
        X_train = self.engineer_features(X_train)
        X_test = self.engineer_features(X_test)
        
        # Drop corresponding rows from y_train and y_test to match X_train and X_test after feature engineering.
        # This is necessary because feature engineering can introduce NaNs (e.g., in rolling averages).
        y_train = y_train[X_train.index]
        y_test = y_test[X_test.index]

        self.X_test = X_test
        self.y_test = y_test
        self.model = RandomForestRegressor(n_estimators=100, random_state=0)
        self.model.fit(X_train, y_train)
        print("Model trained")
        self.tune_hyperparameters(X_train, y_train) # Pass processed data to tune_hyperparameters
        

    def predict(self, data):
        print(f"predict: self.mode = {self.mode}")
        if self.model is None:
            print("Model not trained")
            return None
        if self.mode not in ("Test", "Real"):
            print("Not in Test or Real mode")
            return None
        df = pd.DataFrame([data])
        print(f"DataFrame before feature engineering:\n{df}")
        df = self.engineer_features(df) # Apply feature engineering
        print(f"DataFrame after feature engineering:\n{df}")
        if df.empty:
            print("No data to predict on after feature engineering")
            return None
        X = df[['open', 'max', 'min', 'close', 'volume', 'SMA_5', 'SMA_10', ' volatility', 'RSI', 'MACD']]
        prediction = self.model.predict(X)
        return prediction[0]

    def save_model(self):
        print(f"save_model: self.mode = {self.mode}")
        if self.model is None:
            print("Model not trained")
            return
        joblib.dump(self.model, self.model_filename)
        print("Model saved")

    def load_model(self):
        print(f"load_model: self.mode = {self.mode}")
        try:
            self.model = joblib.load(self.model_filename)
            print("Model loaded")
        except FileNotFoundError:
            print("Model file not found")

    def save_historical_data(self, filename):
        print(f"save_historical_data: self.mode = {self.mode}")
        df = pd.DataFrame(self.historical_data)
        df.to_csv(filename)
        print("Historical data saved")

    def load_historical_data(self):
        print(f"load_historical_data: self.mode = {self.mode}")
        try:
            df = pd.read_csv(self.historical_data_filename)
            self.historical_data = df.to_dict('records')
            print("Historical data loaded")
        except FileNotFoundError:
            print("Historical data file not found")

    def analyze_performance(self):
            print("Historical data file not found")

    def analyze_performance(self):
        print(f"analyze_performance: self.mode = {self.mode}")
        # Add logic to analyze the performance of the model
        # and decide whether to switch to a different mode
        if self.mode == "TEST":
            print("Analyzing performance in Test mode...")
            # Add logic to evaluate the model's performance in Test mode
            # and decide whether to switch to Learning or Real mode
            # For now, let's just switch back to Learning mode after a while
            mse = self.calculate_accuracy(self.X_test, self.y_test)
            print("Switching to Learning mode.")
            self.set_mode("LEARNING")
        elif self.mode == "REAL":
            print("Analyzing performance in Real mode...")
            # Add logic to evaluate the model's performance in Real mode
            # and decide whether to switch to Learning mode
            print("Switching to Learning mode.")
            self.set_mode("LEARNING")
        else:
            print("Cannot analyze performance in this mode.")

    def calculate_accuracy(self, X_test, y_test):
        print(f"calculate_accuracy: self.mode = {self.mode}")
        predictions = self.model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, predictions)
        print("Mean Squared Error:", mse)
        return mse

    def should_switch_to_real(self):
        print(f"should_switch_to_real: self.mode = {self.mode}")
        # Add logic to determine if the model is performing well enough to switch to real mode
        # This could involve analyzing the model's accuracy, profitability, or other metrics
        # For now, let's just return True after a certain amount of time in Test mode
        if self.mode != "TEST":
            return False
        predictions = self.model.predict(self.X_test)
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_test, predictions)
        print("R2 Score:", r2)
        if r2 > 0.7:
            return True
        else:
            return False
    def calculate_rsi(self, data, window=14):
        print(f"calculate_rsi: self.mode = {self.mode}")
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up1 = up.rolling(window).mean()
        roll_down1 = down.abs().rolling(window).mean()
        RS = roll_up1 / roll_down1
        RSI = 100.0 - (100.0 / (1.0 + RS))
        return RSI

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        print(f"calculate_macd: self.mode = {self.mode}")
        EMA_fast = data.ewm(span=fast_period, adjust=False).mean()
        EMA_slow = data.ewm(span=slow_period, adjust=False).mean()
        MACD = EMA_fast - EMA_slow
        signal = MACD.ewm(span=signal_period, adjust=False).mean()
        return MACD - signal
    
    def tune_hyperparameters(self, X_train, y_train):
        print(f"tune_hyperparameters: self.mode = {self.mode}")
        # Add logic to tune the hyperparameters of the model
        # This could involve using GridSearchCV or RandomizedSearchCV
        # For now, let's just print a message
        print("Tuning hyperparameters...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print("Best hyperparameters:", grid_search.best_params_)
        self.save_model()
