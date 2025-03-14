import torch
import torch.nn as nn
import torch.optim as optim
import tulipy as ti
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from transformers import TransformerEncoder, TransformerEncoderLayer
from torchvision import models, transforms
import joblib
import logging
from tqdm import tqdm
from datetime import datetime
import os

class CandleDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class HybridModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=512, num_heads=8, num_layers=6):
        super(HybridModel, self).__init__()
        
        # CNN para análise de gráficos
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, hidden_dim)
        
        # Transformer para séries temporais
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Camadas fully connected
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x_img, x_seq):
        # Processamento CNN
        cnn_features = self.cnn(x_img)
        
        # Processamento Transformer
        seq_features = self.transformer(x_seq)
        seq_features = seq_features.mean(dim=1)
        
        # Combinação de features
        combined = torch.cat((cnn_features, seq_features), dim=1)
        x = self.dropout(combined)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Inteligencia:
    def __init__(self, model_path="hybrid_model.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = HybridModel(num_features=100, num_classes=3).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.model_path = model_path
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_accuracy = 0
        self.mode = "LEARNING"
        self.visualization_dir = "training_visualizations"
        os.makedirs(self.visualization_dir, exist_ok=True)
        
    def set_mode(self, mode):
        valid_modes = ["LEARNING", "TEST", "REAL"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of {valid_modes}")
        self.mode = mode
        print(f"Mode set to: {mode}")
        
    def train(self, train_loader, val_loader, epochs=50):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Barra de progresso
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (images, sequences, labels) in enumerate(pbar):
                images = images.to(self.device)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, sequences)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Métricas
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Atualiza barra de progresso
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100.*correct/total
                })
                
                # Auto-salvamento
                if batch_idx % 100 == 0:
                    self._auto_save()
                    
            # Validação
            val_loss, val_acc = self.validate(val_loader)
            
            # Armazena métricas
            self.train_losses.append(epoch_loss/len(train_loader))
            self.val_losses.append(val_loss)
            self.train_accuracies.append(100.*correct/total)
            self.val_accuracies.append(val_acc)
            
            # Ajusta learning rate
            self.scheduler.step(val_loss)
            
            # Visualização do progresso
            self._plot_training_progress(epoch)
            
            # Early stopping
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model()
                
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, sequences, labels in val_loader:
                images = images.to(self.device)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, sequences)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100.*correct/total
        
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")
        return val_loss, val_acc
        
    def predict(self, image, sequence):
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            sequence = sequence.to(self.device)
            output = self.model(image, sequence)
            _, predicted = output.max(1)
            return predicted.item()
            
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_accuracy': self.best_accuracy
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def load_model(self):
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        self.best_accuracy = checkpoint['best_accuracy']
        print(f"Model loaded from {self.model_path}")
        
    def _auto_save(self):
        checkpoint_path = os.path.join(self.visualization_dir, f"autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        
    def _plot_training_progress(self, epoch):
        plt.figure(figsize=(12, 6))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.visualization_dir, f"training_progress_epoch_{epoch+1}.png")
        plt.savefig(plot_path)
        plt.close()
        
    def integrate_with_ferramental(self, ferramental):
        """Integração completa com o núcleo ferramental"""
        try:
            # Valida credenciais
            if not self._validate_credentials():
                raise ValueError("Credenciais inválidas")
                
            # Configura ferramental
            self.ferramental = ferramental
            
            # Conecta à API
            status, message = self.ferramental.connect()
            if not status:
                raise ConnectionError(f"Falha na conexão com a API: {message}")
                
            # Configura ativos e timeframes
            self._configure_assets()
            
            # Verifica saldo
            balance = self.ferramental.get_balance()
            if not balance:
                raise ValueError("Não foi possível obter saldo da conta")
                
            # Configura streaming de dados
            self._setup_data_streaming()
            
            print("Integração com Ferramental completa")
            return True
            
        except Exception as e:
            logging.error(f"Erro na integração com Ferramental: {str(e)}")
            self._handle_api_error(e)
            return False
            
    def _configure_assets(self):
        """Configura os ativos e timeframes para operação"""
        # Lista de ativos suportados
        self.assets = [
            'EURUSD', 'GBPUSD', 'USDJPY',  # Forex
            'BTCUSD', 'ETHUSD',             # Cripto
            'AAPL', 'GOOG',                 # Ações
            'GOLD', 'SILVER'                # Commodities
        ]
        
        # Timeframes suportados
        self.timeframes = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400
        }
        
        # Configura ativos no ferramental
        self.ferramental.configure_assets(self.assets)
        
    def _setup_data_streaming(self):
        """Configura streaming de dados em tempo real"""
        # Inicia streaming para todos os ativos e timeframes
        for asset in self.assets:
            for tf_name, tf_seconds in self.timeframes.items():
                self.ferramental.start_candles_stream(asset, tf_seconds)
                logging.info(f"Streaming iniciado para {asset} no timeframe {tf_name}")
                
    def _handle_two_factor_auth(self):
        """Tratamento de autenticação de dois fatores"""
        try:
            # Solicita código de autenticação
            code = self.ferramental.request_two_factor_code()
            
            # Envia código para API
            if not self.ferramental.submit_two_factor_code(code):
                raise ValueError("Código de autenticação inválido")
                
            print("Autenticação de dois fatores concluída com sucesso")
            return True
            
        except Exception as e:
            logging.error(f"Erro na autenticação de dois fatores: {str(e)}")
            self.ferramental.send_notification(f"Erro na autenticação: {str(e)}")
            return False
            
    def _validate_credentials(self):
        """Valida credenciais da API com verificação de segurança"""
        try:
            # Verifica se arquivo .env existe
            if not os.path.exists('.env'):
                raise FileNotFoundError("Arquivo .env não encontrado")
                
            # Carrega variáveis de ambiente
            load_dotenv()
            
            # Verifica credenciais obrigatórias
            required_creds = ['IQ_OPTION_EMAIL', 'IQ_OPTION_PASSWORD']
            for cred in required_creds:
                if not os.getenv(cred):
                    raise ValueError(f"Credencial {cred} não encontrada")
                    
            # Verifica formato do email
            email = os.getenv('IQ_OPTION_EMAIL')
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                raise ValueError("Formato de email inválido")
                
            # Verifica força da senha
            password = os.getenv('IQ_OPTION_PASSWORD')
            if len(password) < 8:
                raise ValueError("Senha deve ter pelo menos 8 caracteres")
            if not any(char.isdigit() for char in password):
                raise ValueError("Senha deve conter pelo menos um número")
            if not any(char.isupper() for char in password):
                raise ValueError("Senha deve conter pelo menos uma letra maiúscula")
            if not any(char in "!@#$%^&*" for char in password):
                raise ValueError("Senha deve conter pelo menos um caractere especial")
                
            # Verifica se credenciais já foram usadas
            if self._credentials_already_used(email, password):
                raise ValueError("Credenciais já foram utilizadas anteriormente")
                
            return True
            
        except Exception as e:
            logging.error(f"Erro na validação de credenciais: {str(e)}")
            raise
            
    def _credentials_already_used(self, email, password):
        """Verifica se as credenciais já foram utilizadas anteriormente"""
        # Implementação de cache de credenciais
        credentials_hash = hashlib.sha256(f"{email}:{password}".encode()).hexdigest()
        if credentials_hash in self.used_credentials:
            return True
        self.used_credentials.add(credentials_hash)
        return False
            
    def _handle_api_error(self, error):
        """Tratamento específico de erros da API IQ Option"""
        error_messages = {
            "invalid_credentials": "Credenciais inválidas",
            "two_factor_required": "Autenticação de dois fatores necessária",
            "connection_error": "Erro de conexão com a API",
            "rate_limit_exceeded": "Limite de requisições excedido",
            "maintenance": "API em manutenção",
            "asset_not_found": "Ativo não encontrado",
            "insufficient_balance": "Saldo insuficiente",
            "trade_timeout": "Tempo limite para trade excedido"
        }
        
        # Log detalhado do erro
        error_type = getattr(error, 'error_type', 'unknown_error')
        error_msg = error_messages.get(error_type, str(error))
        logging.error(f"API Error [{error_type}]: {error_msg}")
        
        # Notifica usuário
        self.ferramental.send_notification(f"Erro na API: {error_msg}")
        
        # Ajusta comportamento baseado no erro
        if error_type == "rate_limit_exceeded":
            time.sleep(60)  # Espera 1 minuto antes de tentar novamente
        elif error_type == "maintenance":
            self.mode = "TEST"
            print("API em manutenção - Mudando para modo TEST")
        elif error_type == "two_factor_required":
            self._handle_two_factor_auth()
        elif error_type == "asset_not_found":
            self._handle_asset_not_found(error)
        elif error_type == "insufficient_balance":
            self.mode = "TEST"
            print("Saldo insuficiente - Mudando para modo TEST")
            
    def _handle_two_factor_auth(self):
        """Tratamento de autenticação de dois fatores"""
        try:
            # Solicita código de autenticação
            code = self.ferramental.request_two_factor_code()
            
            # Envia código para API
            if not self.ferramental.submit_two_factor_code(code):
                raise ValueError("Código de autenticação inválido")
                
            print("Autenticação de dois fatores concluída com sucesso")
            return True
            
        except Exception as e:
            logging.error(f"Erro na autenticação de dois fatores: {str(e)}")
            self.ferramental.send_notification(f"Erro na autenticação: {str(e)}")
            return False
            
    def _handle_asset_not_found(self, error):
        """Tratamento de ativo não encontrado"""
        try:
            # Obtém lista de ativos disponíveis
            available_assets = self.ferramental.get_available_assets()
            
            # Verifica se há ativos similares
            target_asset = getattr(error, 'asset', '')
            similar_assets = [a for a in available_assets if target_asset in a]
            
            if similar_assets:
                # Seleciona ativo similar
                selected_asset = similar_assets[0]
                print(f"Ativo {target_asset} não encontrado. Usando {selected_asset} como alternativa")
                return selected_asset
                
            # Tenta encontrar ativo do mesmo tipo
            asset_type = self._get_asset_type(target_asset)
            if asset_type:
                type_assets = [a for a in available_assets if self._get_asset_type(a) == asset_type]
                if type_assets:
                    selected_asset = type_assets[0]
                    print(f"Ativo {target_asset} não encontrado. Usando {selected_asset} do mesmo tipo como alternativa")
                    return selected_asset
                
            raise ValueError(f"Nenhum ativo similar encontrado para {target_asset}")
            
        except Exception as e:
            logging.error(f"Erro ao tratar ativo não encontrado: {str(e)}")
            self.ferramental.send_notification(f"Erro com ativo: {str(e)}")
            return None
            
    def _get_asset_type(self, asset_name):
        """Identifica o tipo de ativo (forex, cripto, ação, etc)"""
        asset_types = {
            'forex': ['EUR', 'USD', 'GBP', 'JPY'],
            'crypto': ['BTC', 'ETH', 'XRP', 'LTC'],
            'stock': ['AAPL', 'GOOG', 'AMZN', 'TSLA'],
            'commodity': ['GOLD', 'SILVER', 'OIL', 'NATURALGAS']
        }
        
        for asset_type, prefixes in asset_types.items():
            if any(asset_name.startswith(prefix) for prefix in prefixes):
                return asset_type
                
        return None
        
    def _select_strategy(self, asset_type):
        """Seleciona estratégia baseada no tipo de ativo"""
        strategies = {
            'forex': self._forex_strategy,
            'crypto': self._crypto_strategy,
            'stock': self._stock_strategy,
            'commodity': self._commodity_strategy
        }
        
        return strategies.get(asset_type, self._default_strategy)
        
    def _forex_strategy(self, data):
        """Estratégia específica para Forex"""
        # Implementa estratégia de média móvel cruzada
        short_ma = data['close'].rolling(window=5).mean()
        long_ma = data['close'].rolling(window=20).mean()
        
        if short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 'call'
        else:
            return 'put'
            
    def _crypto_strategy(self, data):
        """Estratégia específica para Criptomoedas"""
        # Implementa estratégia baseada em RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if rsi.iloc[-1] < 30:
            return 'call'
        elif rsi.iloc[-1] > 70:
            return 'put'
        else:
            return None
            
    def _stock_strategy(self, data):
        """Estratégia específica para Ações"""
        # Implementa estratégia baseada em Bollinger Bands
        sma = data['close'].rolling(window=20).mean()
        std = data['close'].rolling(window=20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        if data['close'].iloc[-1] < lower_band.iloc[-1]:
            return 'call'
        elif data['close'].iloc[-1] > upper_band.iloc[-1]:
            return 'put'
        else:
            return None
            
    def _commodity_strategy(self, data):
        """Estratégia específica para Commodities"""
        # Implementa estratégia baseada em MACD
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        if macd.iloc[-1] > signal.iloc[-1]:
            return 'call'
        else:
            return 'put'
            
    def _default_strategy(self, data):
        """Estratégia padrão para ativos desconhecidos"""
        # Implementa estratégia simples de tendência
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return 'call'
        else:
            return 'put'
        
    def analyze_market(self):
        """Análise completa do mercado utilizando todos os recursos"""
        if self.mode == "LEARNING":
            return self._learning_analysis()
        elif self.mode == "TEST":
            return self._test_analysis()
        elif self.mode == "REAL":
            return self._real_analysis()
            
    def _learning_analysis(self):
        """Análise durante o aprendizado"""
        print("Executando análise de aprendizado...")
        
        # Coleta dados históricos
        data = self.ferramental.get_historical_data()
        if not data:
            print("Erro ao coletar dados históricos")
            return False
            
        # Pré-processamento
        processed_data = self._preprocess_data(data)
        
        # Análise estatística
        stats = self._calculate_statistics(processed_data)
        
        # Avaliação do modelo
        model_performance = self._evaluate_model(processed_data)
        
        # Atualização de parâmetros
        self._update_parameters(stats, model_performance)
        
        return True
        
    def _test_analysis(self):
        """Análise durante testes"""
        print("Executando análise de teste...")
        
        # Coleta dados em tempo real
        realtime_data = self.ferramental.get_realtime_data()
        if not realtime_data:
            print("Erro ao coletar dados em tempo real")
            return False
            
        # Pré-processamento
        processed_data = self._preprocess_data(realtime_data)
        
        # Simulação de trades
        simulation_results = self._simulate_trades(processed_data)
        
        # Avaliação de performance
        performance_metrics = self._calculate_performance_metrics(simulation_results)
        
        # Decisão sobre modo real
        if self._should_switch_to_real(performance_metrics):
            print("Performance satisfatória - Pronto para modo real")
            return True
            
        return False
        
    def _real_analysis(self):
        """Análise durante operações reais"""
        print("Executando análise real...")
        
        # Coleta dados em tempo real
        realtime_data = self.ferramental.get_realtime_data()
        if not realtime_data:
            print("Erro ao coletar dados em tempo real")
            return False
            
        # Pré-processamento
        processed_data = self._preprocess_data(realtime_data)
        
        # Execução de trades
        trade_results = self._execute_trades(processed_data)
        
        # Monitoramento de risco
        risk_metrics = self._calculate_risk_metrics(trade_results)
        
        # Ajuste de estratégia
        self._adjust_strategy(risk_metrics)
        
        return True
        
    def _preprocess_data(self, data):
        """Pré-processamento dos dados brutos"""
        # Normalização
        data = (data - data.mean()) / data.std()
        
        # Remoção de outliers
        data = data[(np.abs(data) < 3).all(axis=1)]
        
        # Feature engineering
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        return data.dropna()
        
    def _calculate_statistics(self, data):
        """Cálculo de estatísticas relevantes"""
        stats = {
            'mean_return': data['returns'].mean(),
            'volatility': data['volatility'].mean(),
            'sharpe_ratio': data['returns'].mean() / data['volatility'].mean(),
            'max_drawdown': (data['close'].cummax() - data['close']).max()
        }
        return stats
        
    def _evaluate_model(self, data):
        """Avaliação do modelo"""
        predictions = self.model.predict(data)
        accuracy = accuracy_score(data['target'], predictions)
        return {'accuracy': accuracy}
        
    def _update_parameters(self, stats, model_performance):
        """Atualização dos parâmetros do modelo"""
        if model_performance['accuracy'] < 0.7:
            self.optimizer.param_groups[0]['lr'] *= 0.9
            
    def _simulate_trades(self, data):
        """Simulação de trades com histórico detalhado"""
        simulated_trades = []
        virtual_balance = 10000  # Saldo virtual inicial
        
        try:
            # Verifica se há dados suficientes
            if len(data) < 20:
                raise ValueError("Dados insuficientes para simulação")
                
            # Obtém previsões do modelo
            predictions = self.model.predict(data)
            
            # Executa simulação de trades
            for i, pred in enumerate(predictions):
                # Calcula tamanho da posição
                position_size = self._calculate_position_size(data.iloc[i])
                
                # Simula trade
                if pred == 1:  # Compra
                    trade = {
                        'type': 'call',
                        'asset': data.iloc[i]['asset'],
                        'amount': position_size,
                        'entry_price': data.iloc[i]['close'],
                        'exit_price': data.iloc[i+1]['close'] if i+1 < len(data) else data.iloc[i]['close'],
                        'duration': 1  # 1 período
                    }
                elif pred == 2:  # Venda
                    trade = {
                        'type': 'put',
                        'asset': data.iloc[i]['asset'],
                        'amount': position_size,
                        'entry_price': data.iloc[i]['close'],
                        'exit_price': data.iloc[i+1]['close'] if i+1 < len(data) else data.iloc[i]['close'],
                        'duration': 1  # 1 período
                    }
                else:
                    continue
                    
                # Calcula resultado do trade
                if trade['type'] == 'call':
                    trade['result'] = 'win' if trade['exit_price'] > trade['entry_price'] else 'loss'
                else:
                    trade['result'] = 'win' if trade['exit_price'] < trade['entry_price'] else 'loss'
                    
                # Calcula P&L
                trade['pnl'] = position_size * (1 if trade['result'] == 'win' else -1)
                virtual_balance += trade['pnl']
                trade['balance'] = virtual_balance
                
                simulated_trades.append(trade)
                
                # Verifica limite de risco
                if self._exceeds_risk_limit(simulated_trades):
                    print("Limite de risco atingido - Parando simulação")
                    break
                    
        except Exception as e:
            logging.error(f"Erro na simulação de trades: {str(e)}")
            self._handle_trade_error(e)
            
        return simulated_trades
        
    def _calculate_performance_metrics(self, trades):
        """Cálculo detalhado de métricas de performance"""
        if not trades:
            return {}
            
        # Métricas básicas
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['result'] == 'win'])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L
        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Drawdown
        balances = [t['balance'] for t in trades]
        peak = max(balances)
        trough = min(balances)
        drawdown = (peak - trough) / peak if peak > 0 else 0
        
        # Risk metrics
        max_loss = min(t['pnl'] for t in trades)
        max_win = max(t['pnl'] for t in trades)
        risk_reward = abs(avg_pnl / max_loss) if max_loss < 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': drawdown,
            'max_loss': max_loss,
            'max_win': max_win,
            'risk_reward': risk_reward,
            'final_balance': balances[-1] if balances else 0
        }
        
    def _should_switch_to_real(self, metrics):
        """Decisão sobre mudança para modo real"""
        return metrics.get('profit', 0) > 0
        
    def _execute_trades(self, data):
        """Execução de trades reais com gerenciamento de risco"""
        trades = []
        
        try:
            # Verifica se há dados suficientes
            if len(data) < 20:
                raise ValueError("Dados insuficientes para execução de trades")

            # Obtém previsões do modelo
            predictions = self.model.predict(data)

            # Agrupa ativos por tipo e timeframe
            asset_groups = data.groupby([self._get_asset_type, 'timeframe'])

            # Executa trades para cada grupo de ativos
            for (asset_type, timeframe), group in asset_groups:
                # Seleciona estratégia específica
                strategy = self._select_strategy(asset_type)

                # Ajusta parâmetros baseado no timeframe
                self._adjust_parameters_for_timeframe(timeframe)

                # Executa trades para o grupo
                for i, row in group.iterrows():
                    # Obtém sinal da estratégia
                    signal = strategy(group.iloc[:i+1])

                    if signal == 'call':
                        trade = self.ferramental.execute_trade(
                            asset=row['asset'],
                            direction='call',
                            amount=self._calculate_position_size(row),
                            timeframe=timeframe
                        )
                        trades.append(trade)
                    elif signal == 'put':
                        trade = self.ferramental.execute_trade(
                            asset=row['asset'],
                            direction='put',
                            amount=self._calculate_position_size(row),
                            timeframe=timeframe
                        )
                        trades.append(trade)

                    # Verifica limite de risco
                    if self._exceeds_risk_limit(trades):
                        print("Limite de risco atingido - Parando execução")
                        break

                # Envia notificação de status
                self._send_trade_status_notification(asset_type, timeframe, trades)

                # Auto-ajuste de parâmetros
                self._auto_adjust_parameters(trades)
        except Exception as e:
            logging.error(f"Erro na execução de trades: {str(e)}")
            self._handle_trade_error(e)

        return trades
