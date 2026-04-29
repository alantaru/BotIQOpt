class MockIQOptionAPI:
    """Mock da classe IQ_Option para testes de integração."""
    
    def __init__(self, email, password):
        self.email = email
        self.password = password
        self.connected = False
        self.balance = 10000.0
        self.mode = "PRACTICE"
        self.trades = {}
        self.order_id_counter = 0

    def connect(self):
        if self.email == "error@test.com":
            return False, "Connection refused"
        self.connected = True
        return True, None

    def check_connect(self):
        return self.connected

    def get_balance(self):
        return self.balance

    def change_balance(self, mode):
        self.mode = mode
        return True

    def get_currency(self):
        return "USD"

    def buy(self, amount, asset, action, expiration):
        self.order_id_counter += 1
        order_id = self.order_id_counter
        self.trades[order_id] = {
            'amount': amount,
            'asset': asset,
            'action': action,
            'expiration': expiration,
            'status': 'open'
        }
        return True, order_id

    def check_win_v4(self, order_id):
        if order_id not in self.trades:
            return None
        # Simula ganho se o amount for par, perda se for ímpar (para testes determinísticos)
        amount = self.trades[order_id]['amount']
        if amount % 2 == 0:
            return amount * 0.8 # Profit
        else:
            return -amount # Loss

    def get_candles(self, asset, *args):
        import time
        # Detect standard iqoptionapi signature vs custom ones
        # Case 1: (asset, timeframe, count, endtime)
        # Case 2: (asset, tf_value, tf_type, count)
        
        tf_value = 60
        tf_type = "Minutes"
        count = 1
        
        if len(args) > 0:
            # Check if any argument is a string (e.g., "Minutes")
            str_args = [i for i, a in enumerate(args) if isinstance(a, str)]
            if str_args:
                idx = str_args[0]
                tf_type = args[idx]
                tf_value = args[idx-1] if idx > 0 else 1
                count = args[idx+1] if len(args) > idx+1 else 1
            else:
                # No string, assume (timeframe, count, endtime)
                tf_value = args[0]
                tf_type = "Seconds"
                count = args[1] if len(args) > 1 else 1
                
        # Simple conversion to seconds for mock consistency
        timeframe = tf_value * 60 if tf_type == "Minutes" else (tf_value * 3600 if tf_type == "Hours" else tf_value)
            
        now = int(time.time())
        candles = []
        for i in range(int(count)):
            candles.append({
                'id': i,
                'from': now - (i * timeframe),
                'to': now - (i * timeframe) + timeframe,
                'open': 1.0,
                'close': 1.1,
                'min': 0.9,
                'max': 1.2,
                'low': 0.9,
                'high': 1.2,
                'volume': 100

            })
        return candles



    def start_candles_stream(self, asset, timeframe, size):
        return True

    def get_realtime_candles(self, asset, timeframe):
        # Return a dict of candles as expected by Ferramental.py
        # Ferramental.py: df = pd.DataFrame(list(realtime_candles.values()))
        import time
        now = int(time.time())
        return {
            now: {
                'from': now,
                'open': 1.0,
                'close': 1.1,
                'min': 0.9,
                'max': 1.2,
                'low': 0.9,
                'high': 1.2,
                'volume': 100

            }
        }



    def get_optioninfo_v2(self, count):
        import time
        history = []
        # Return history based on what we have in self.trades
        for trade_id, trade in self.trades.items():
            history.append({
                'id': trade_id,
                'active': trade['asset'].upper(),
                'direction': trade['action'],
                'amount': trade['amount'],
                'profit': self.check_win_v4(trade_id),
                'win': self.check_win_v4(trade_id) > 0,
                'open_time': int(time.time()) - 60,
                'close_time': int(time.time())
            })
        return history[:count]


    def get_all_open_time(self):
        # Simula resposta de ativos abertos (Normalizado para facilitar busca)
        return {
            "binary": {"EURUSD": {"open": True}, "GBPUSD": {"open": True}},
            "turbo": {"EURUSD": {"open": True}, "GBPUSD": {"open": False}}
        }

