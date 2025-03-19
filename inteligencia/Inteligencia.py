import torch
import torch.nn as nn
import torch.optim as optim
import tulipy as ti
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models, transforms
import joblib
import logging
from tqdm import tqdm
from datetime import datetime
import os
import json
import re
import hashlib
import time
from dotenv import load_dotenv

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

        # Transformer para séries temporais
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1  # Add dropout to the transformer layer
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear projection for input sequences
        self.input_projection = nn.Linear(10, hidden_dim)  # 10 is the number of features in the sequence

        # Camadas fully connected
        #self.fc1 = nn.Linear(hidden_dim*2, hidden_dim) # Removed cnn
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # Just transformer
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_img, x_seq):
        # Processamento Transformer
        x_seq = self.input_projection(x_seq)  # Project input to hidden_dim
        seq_features = self.transformer(x_seq)
        seq_features = seq_features.mean(dim=1)  # Average across the sequence length

        # Camadas fully connected
        #combined = torch.cat((cnn_features, seq_features), dim=1) # Removed CNN
        #x = self.dropout(combined)
        x = self.dropout(seq_features)  # Just transformer features
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Inteligencia:
    def __init__(self, model_path="hybrid_model.pth", device="cuda" if torch.cuda.is_available() else "cpu", historical_data_filename="historical_data.csv"):
        """Inicializa a inteligência do bot com configurações robustas e tratamento de erros.
        
        Args:
            model_path (str): Caminho para salvar/carregar o modelo
            device (str): Dispositivo para execução (cuda ou cpu)
            historical_data_filename (str): Nome do arquivo para dados históricos
            
        Raises:
            RuntimeError: Se ocorrer erro na inicialização do modelo
            ValueError: Se os parâmetros forem inválidos
        """
        try:
            # Validação dos parâmetros
            if not isinstance(model_path, str):
                raise ValueError("model_path deve ser uma string")
            if device not in ["cuda", "cpu"]:
                raise ValueError("device deve ser 'cuda' ou 'cpu'")
                
            # Configurações principais
            self.device = device
            self.model = HybridModel(num_features=10, num_classes=3).to(self.device)
            self.initial_lr = 1e-5
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.initial_lr)
            self.criterion = nn.CrossEntropyLoss()
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
            self.model_path = model_path
            
            # Métricas de treinamento
            self.train_losses = []
            self.val_losses = []
            self.train_accuracies = []
            self.val_accuracies = []
            self.best_accuracy = 0
            
            # Configurações de operação
            self.mode = "LEARNING"
            self.visualization_dir = "training_visualizations"
            os.makedirs(self.visualization_dir, exist_ok=True)
            self.historical_data_filename = historical_data_filename
            self.historical_data = None
            
            # Segurança e credenciais
            self.used_credentials = set()
            self.risk_multiplier = 1.0
            self.min_confidence = 0.5
            self.auto_switch_to_real = False
            
            # Configurações de salvamento automático
            self.autosave_interval = 1000  # Salvar a cada 1000 iterações
            self.last_save_time = time.time()
            
            # Configurações de logging
            self.logger = logging.getLogger('Inteligencia')
            self.logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.visualization_dir, 'inteligencia.log'))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        except Exception as e:
            raise RuntimeError(f"Erro na inicialização da Inteligência: {str(e)}")

    def set_mode(self, mode):
        valid_modes = ["LEARNING", "TEST", "REAL"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of {valid_modes}")
        self.mode = mode
        print(f"Mode set to: {mode}")

    def download_historical_data(self, ferramental, asset, timeframe_type, timeframe_value, candle_count):
        """Downloads historical data for a given asset and saves it to a CSV file."""
        try:
            candles = ferramental.get_candles(asset, timeframe_type, timeframe_value, candle_count)
            if candles:
                df = pd.DataFrame(candles)
                df.to_csv(self.historical_data_filename, index=False)
                print(f"Historical data downloaded and saved to {self.historical_data_filename}")
                self.historical_data = df
            else:
                print("No candles to download.")
        except Exception as e:
            print(f"Error downloading historical data: {e}")
            
    def load_historical_data(self):
        """Loads historical data from the CSV file."""
        try:
            self.historical_data = pd.read_csv(self.historical_data_filename)
            print(f"Historical data loaded from {self.historical_data_filename}")
        except FileNotFoundError:
            print("Historical data file not found. Please download historical data first.")
        except Exception as e:
            print(f"Error loading historical data: {e}")
        
    def _preprocess_candles(self, candles):
        """Preprocess raw candle data into model input format"""
        df = pd.DataFrame(candles)
        
        # Calculate technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['signal'] = self._calculate_macd(df['close'])
        
 # Drop NA values from indicators
        df = df.dropna()

        # Create sequences and labels
        sequences = []
        labels = []
        for i in range(14, len(df)):
            seq = df.iloc[i-14:i][['open', 'max', 'min', 'close', 'volume',
                                  'returns', 'sma_20', 'rsi', 'macd', 'signal']].values
            
            # Generate label based on indicators
            if (df['rsi'].iloc[i] < 30 and 
                df['macd'].iloc[i] > df['signal'].iloc[i] and
                df['close'].iloc[i] > df['sma_20'].iloc[i]):
                label = 1  # BUY
            elif (df['rsi'].iloc[i] > 70 and
                  df['macd'].iloc[i] < df['signal'].iloc[i] and
                  df['close'].iloc[i] < df['sma_20'].iloc[i]):
                label = 2  # SELL
            else:
                label = 0  # HOLD
            
            sequences.append(seq)
            labels.append(label)

        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        print(f"Sequences shape: {sequences.shape}, Labels shape: {labels.shape}")
        return sequences, labels

    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def train(self, candles, test_size=0.2, epochs=50, batch_size=32):
        """Train model on candle data"""
        # Preprocess data
        sequences, labels = self._preprocess_candles(candles)
        
        # Split into train/test
        split_idx = int(len(sequences) * (1 - test_size))
        train_sequences = sequences[:split_idx]
        train_labels = labels[:split_idx]
        test_sequences = sequences[split_idx:]
        test_labels = labels[split_idx:]
        
        # Create datasets
        train_dataset = CandleDataset(list(zip(train_sequences, train_labels)))
        test_dataset = CandleDataset(list(zip(test_sequences, test_labels)))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (seq_batch, label_batch) in enumerate(pbar):
                # Convert sequences to model input format
                seq_batch = torch.FloatTensor(seq_batch).to(self.device)
                label_batch = torch.LongTensor(label_batch).to(self.device)
                
                # Add dummy image data (zeros) since we're not using CNN yet
                img_batch = torch.zeros(len(seq_batch), 3, 224, 224).to(self.device)
                
                # Forward pass - use dummy image data and sequence data
                outputs = self.model(img_batch, seq_batch)
                loss = self.criterion(outputs, label_batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += label_batch.size(0)
                correct += predicted.eq(label_batch).sum().item()
                
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
            for seq_batch, label_batch in val_loader:
                seq_batch = torch.FloatTensor(seq_batch).to(self.device)
                label_batch = torch.LongTensor(label_batch).to(self.device)
                img_batch = torch.zeros(len(seq_batch), 3, 224, 224).to(self.device)

                outputs = self.model(img_batch, seq_batch)
                loss = self.criterion(outputs, label_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += label_batch.size(0)
                correct += predicted.eq(label_batch).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100.*correct/total
        
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")
        return val_loss, val_acc

    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            # Ensure sequence is a tensor on the correct device
            sequence = torch.tensor(sequence, dtype=torch.float32).to(self.device)
            # Add a batch dimension if necessary
            if sequence.ndim == 2:  # Assuming input is (sequence_length, num_features)
                sequence = sequence.unsqueeze(0)  # Add batch dimension: (1, sequence_length, num_features)
            # Create a dummy image tensor
            dummy_image = torch.zeros(sequence.shape[0], 3, 224, 224).to(self.device)
            output = self.model(dummy_image, sequence)
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
        """Carrega o modelo treinado usando PyTorch"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")
            
        try:
            # Verifica se o modelo foi salvo com a versão atual do PyTorch
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Carrega estados do modelo e otimizador
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Carrega métricas de treinamento
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies'] 
            self.val_accuracies = checkpoint['val_accuracies']
            self.best_accuracy = checkpoint['best_accuracy']
            
            # Verifica compatibilidade de versões
            if 'torch_version' in checkpoint:
                current_version = torch.__version__
                saved_version = checkpoint['torch_version']
                if current_version != saved_version:
                    print(f"Aviso: Versão do PyTorch diferente (salvo: {saved_version}, atual: {current_version})")
            
            print(f"Modelo carregado de {self.model_path}")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            raise
        
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

    def _select_strategy(self, asset_type, prediction):
        """Seleciona estratégia baseada no tipo de ativo e na previsão do modelo."""
        if prediction == 1:
            return 'call'  # BUY
        elif prediction == 2:
            return 'put'  # SELL
        else:
            return None  # HOLD

    def _forex_strategy(self, data):
        """Estratégia específica para Forex (agora serve como filtro/ajuste)."""
        # Poderia adicionar lógica para ajustar a confiança, por exemplo
        return 1.0  # Confiança padrão

    def _crypto_strategy(self, data):
        """Estratégia específica para Criptomoedas (agora serve como filtro/ajuste)."""
        return 1.0

    def _stock_strategy(self, data):
        """Estratégia específica para Ações (agora serve como filtro/ajuste)."""
        return 1.0

    def _commodity_strategy(self, data):
        """Estratégia específica para Commodities (agora serve como filtro/ajuste)."""
        return 1.0

    def _default_strategy(self, data):
        """Estratégia padrão para ativos desconhecidos."""
        return 1.0
    
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

    def _calculate_position_size(self, data_row):
        """Calculate appropriate position size based on risk parameters"""
        # Default position size
        default_size = 10.0

        # Apply risk multiplier (set by strategy adjustment)
        adjusted_size = default_size * self.risk_multiplier

        # Apply volatility adjustment if available
        if 'volatility' in data_row:
            # Reduce position size for high volatility
            if data_row['volatility'] > 0.02:  # 2% volatility threshold
                volatility_factor = 1.0 - (data_row['volatility'] - 0.02) * 10
                volatility_factor = max(0.2, volatility_factor)  # Don't reduce below 20%
                adjusted_size *= volatility_factor

        # Apply confidence adjustment if available
        if hasattr(self, 'prediction_confidence') and self.prediction_confidence is not None:
            confidence_factor = max(0.5, self.prediction_confidence)
            adjusted_size *= confidence_factor

        # Ensure position size is within reasonable limits
        min_size = 1.0
        max_size = 100.0
        adjusted_size = max(min_size, min(adjusted_size, max_size))

        return adjusted_size

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
        
    def _execute_trades(self, data):
        """Executa trades com base nas previsões do modelo.
        
        Args:
            data: DataFrame com dados processados
            
        Returns:
            Lista de resultados dos trades
        """
        trade_results = []
        
        try:
            # Verifica se estamos em modo REAL
            if self.mode != "REAL":
                logging.warning("Tentativa de executar trades reais em modo não-REAL")
                return []
                
            # Obtém previsões do modelo
            predictions = []
            for i in range(len(data)):
                # Prepara imagem e sequência
                img_data = self._prepare_image_data(data.iloc[i:i+1])
                seq_data = self._prepare_sequence_data(data.iloc[i:i+1])
                
                # Faz previsão
                prediction = self.predict(img_data, seq_data)
                predictions.append(prediction)
                
                # Calcula confiança da previsão
                self.prediction_confidence = self._calculate_prediction_confidence(data.iloc[i], prediction)
                
            # Para cada ativo, identifica a melhor oportunidade
            for asset in data['asset'].unique():
                asset_data = data[data['asset'] == asset]
                
                if len(asset_data) == 0:
                    continue
                    
                # Obtém o último registro para o ativo
                last_record = asset_data.iloc[-1]
                last_index = asset_data.index[-1]
                
                # Get the prediction for this asset
                i = data.index.get_loc(last_index)
                prediction = predictions[i] if i < len(predictions) else 0
                
                # Determina a ação com base no tipo de ativo e na previsão do modelo
                asset_type = self._get_asset_type(asset)
                action = self._select_strategy(asset_type, prediction)
                
                # Verifica condições mínimas para operar
                if action is None or self.prediction_confidence < self.min_confidence:
                    logging.info(f"Confiança insuficiente para operar {asset}")
                    continue
                    
                # Calcula tamanho da posição
                position_size = self._calculate_position_size(last_record)
                
                # Executa operação
                logging.info(f"Executando operação REAL: {asset} {action} {position_size}")
                
                # Se estiver em modo REAL, executa a operação na plataforma
                success, order_id = self.ferramental.buy_digital_spot(
                    asset, 
                    position_size, 
                    action, 
                    1  # 1 minuto
                )
                
                if success:
                    # Acompanha resultado da operação
                    check_close, win_money = self.ferramental.check_win_digital(order_id, 5)
                    
                    trade_result = {
                        'asset': asset,
                        'action': action,
                        'amount': position_size,
                        'result': 'win' if win_money > 0 else 'loss',
                        'pnl': win_money if win_money > 0 else -position_size,
                        'timestamp': datetime.now().isoformat(),
                        'order_id': order_id,
                        'confidence': self.prediction_confidence
                    }
                    
                    trade_results.append(trade_result)
                    
                    # Registra operação no histórico
                    self._record_trade(trade_result)
                    
                    logging.info(f"Resultado da operação: {trade_result['result']} | PnL: {trade_result['pnl']}")
                else:
                    logging.error(f"Falha ao executar operação em {asset}")
                
            return trade_results
            
        except Exception as e:
            logging.error(f"Erro ao executar trades: {str(e)}")
            return []
            
    def _prepare_image_data(self, data):
        """Prepara dados de imagem para o modelo."""
        # Implementação básica - em um cenário real, converteria os dados para uma imagem
        # representando o gráfico de velas ou outro formato visual
        dummy_image = torch.zeros((1, 3, 224, 224), device=self.device)
        return dummy_image
        
    def _prepare_sequence_data(self, data):
        """Prepara dados de sequência para o modelo."""
        # Implementação básica - em um cenário real, extrairia a sequência 
        # de preços/indicadores em formato adequado para o transformer
        dummy_sequence = torch.zeros((1, 16, 512), device=self.device)
        return dummy_sequence
        
    def _calculate_prediction_confidence(self, data_row, prediction):
        """Calcula a confiança da previsão baseada em indicadores técnicos e fortaleza do sinal.
        
        Args:
            data_row: Linha de dados com indicadores técnicos
            prediction: Previsão do modelo (0=hold, 1=call, 2=put)
            
        Returns:
            float: Valor de confiança entre 0.5 e 1.0
        """
        # Confidence starts at a baseline of 0.6
        confidence = 0.6
        
        # If hold prediction, lower confidence
        if prediction == 0:
            return 0.5
        
        # Adjust confidence based on available indicators
        if 'rsi' in data_row:
            rsi = data_row['rsi']
            # Strong oversold/overbought conditions increase confidence
            if (prediction == 1 and rsi < 30) or (prediction == 2 and rsi > 70):
                confidence += 0.1
            # Contradiction between RSI and prediction decreases confidence
            elif (prediction == 1 and rsi > 70) or (prediction == 2 and rsi < 30):
                confidence -= 0.1
        
        # Check for MACD confirmation if available
        if 'macd' in data_row and 'signal' in data_row:
            macd = data_row['macd']
            signal = data_row['signal']
            # If MACD confirms prediction, increase confidence
            if (prediction == 1 and macd > signal) or (prediction == 2 and macd < signal):
                confidence += 0.1
        
        # Check for trend confirmation with moving average
        if 'close' in data_row and 'sma_20' in data_row:
            price = data_row['close']
            sma = data_row['sma_20']
            # If price relative to MA confirms prediction, increase confidence
            if (prediction == 1 and price > sma) or (prediction == 2 and price < sma):
                confidence += 0.1
        
        # Check for volatility
        if 'volatility' in data_row:
            volatility = data_row['volatility']
            # High volatility decreases confidence
            if volatility > 0.02:  # 2% volatility threshold
                confidence -= 0.05 * (volatility / 0.02)
        
        # Ensure confidence stays in reasonable range
        return max(0.5, min(confidence, 0.95))
        
    def _record_trade(self, trade_result):
        """Registra a operação no histórico."""
        try:
            trade_history_file = os.path.join(self.visualization_dir, "trade_history.json")
            
            # Carrega histórico existente ou cria novo
            if os.path.exists(trade_history_file):
                with open(trade_history_file, 'r') as f:
                    trade_history = json.load(f)
            else:
                trade_history = []
                
            # Adiciona nova operação
            trade_history.append(trade_result)
            
            # Salva histórico atualizado
            with open(trade_history_file, 'w') as f:
                json.dump(trade_history, f, indent=2)
                
        except Exception as e:
            logging.error(f"Erro ao registrar operação: {str(e)}")
            
    def _calculate_risk_metrics(self, trade_results):
        """Calcula métricas de risco com base nos resultados dos trades."""
        if not trade_results:
            return {
                'consecutive_losses': 0,
                'drawdown': 0,
                'win_rate': 0
            }
            
        # Calcula perdas consecutivas
        consecutive_losses = 0
        for trade in reversed(trade_results):
            if trade['result'] == 'loss':
                consecutive_losses += 1
            else:
                break
                
        # Calcula drawdown
        pnl_values = [trade['pnl'] for trade in trade_results]
        cumulative_pnl = [sum(pnl_values[:i+1]) for i in range(len(pnl_values))]
        
        peak = max(cumulative_pnl)
        current = cumulative_pnl[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        # Calcula taxa de acerto
        wins = sum(1 for trade in trade_results if trade['result'] == 'win')
        win_rate = wins / len(trade_results)
        
        return {
            'consecutive_losses': consecutive_losses,
            'drawdown': drawdown,
            'win_rate': win_rate
        }
        
    def _adjust_strategy(self, risk_metrics):
        """Ajusta a estratégia com base nas métricas de risco."""
        # Ajusta o multiplicador de risco com base no drawdown
        if risk_metrics['drawdown'] > 0.2:
            self.risk_multiplier *= 0.8
            logging.info(f"Alto drawdown detectado - reduzindo multiplicador de risco para {self.risk_multiplier:.2f}")
            
        # Ajusta o limite de confiança com base na taxa de acerto
        if risk_metrics['win_rate'] < 0.4:
            self.min_confidence = min(0.9, self.min_confidence + 0.05)
            logging.info(f"Baixa taxa de acerto - aumentando limite de confiança para {self.min_confidence:.2f}")
        elif risk_metrics['win_rate'] > 0.6:
            self.min_confidence = max(0.5, self.min_confidence - 0.05)
            logging.info(f"Alta taxa de acerto - reduzindo limite de confiança para {self.min_confidence:.2f}")
            
        # Salva parâmetros da estratégia
        self._save_strategy_parameters()
        
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
        except Exception as e:
            logging.error(f"Erro na simulação de trades: {str(e)}")
            self._handle_trade_error(e)
            raise

    def _exceeds_risk_limit(self, trades):
        """Checks if the risk limit is exceeded."""
        if not trades:
            return False
            
        # Calculate consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 5  # Configure via settings
        
        for trade in reversed(trades):
            if trade['result'] == 'loss':
                consecutive_losses += 1
            else:
                break
                
        if consecutive_losses >= max_consecutive_losses:
            return True
            
        # Calculate drawdown
        equity_curve = []
        current_equity = 0.0
        
        for trade in trades:
            current_equity += trade['pnl']
            equity_curve.append(current_equity)
            
        peak = max(equity_curve) if equity_curve else 0
        current = equity_curve[-1] if equity_curve else 0
        
        # Calculate drawdown as percentage
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        # Return true if drawdown exceeds 20%
        return drawdown > 0.2
        
    def analyze_performance(self):
        """Analyzes the performance of the AI and returns performance metrics."""
        performance_data = {}
        
        try:
            # Get trading history from the last session
            trade_history = self._load_trade_history()
            
            if not trade_history:
                logging.warning("No trade history found for performance analysis")
                return {
                    'win_rate': 0,
                    'profit_factor': 0,
                    'expectancy': 0,
                    'status': 'insufficient_data'
                }
            
            # Calculate win rate
            wins = sum(1 for trade in trade_history if trade['result'] == 'win')
            total_trades = len(trade_history)
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(trade['pnl'] for trade in trade_history if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in trade_history if trade['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate expectancy
            expectancy = (win_rate * (gross_profit / wins if wins > 0 else 0)) - \
                        ((1 - win_rate) * (gross_loss / (total_trades - wins) if total_trades - wins > 0 else 0))
            
            # Calculate drawdown
            equity_curve = []
            current_equity = 0.0
            
            for trade in trade_history:
                current_equity += trade['pnl']
                equity_curve.append(current_equity)
                
            peak = max(equity_curve) if equity_curve else 0
            current = equity_curve[-1] if equity_curve else 0
            drawdown = (peak - current) / peak if peak > 0 else 0
            
            # Determine system status
            if total_trades < 30:
                status = 'collecting_data'
            elif win_rate >= 0.55 and profit_factor >= 1.5 and expectancy > 0:
                status = 'ready_for_real'
            elif win_rate >= 0.45 and profit_factor >= 1.2:
                status = 'optimizing'
            else:
                status = 'needs_improvement'
                
            performance_data = {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'drawdown': drawdown,
                'total_trades': total_trades,
                'status': status
            }
            
            # Store performance data for model tuning
            self._store_performance_data(performance_data)
            
        except Exception as e:
            logging.error(f"Error analyzing performance: {str(e)}")
            performance_data = {
                'win_rate': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'status': 'error',
                'error': str(e)
            }
            
        return performance_data
    
    def _load_trade_history(self):
        """Loads trade history from disk."""
        trade_history_file = os.path.join(self.visualization_dir, "trade_history.json")
        
        if not os.path.exists(trade_history_file):
            return []
            
        try:
            with open(trade_history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading trade history: {str(e)}")
            return []
    
    def _store_performance_data(self, performance_data):
        """Stores performance data for future reference."""
        performance_file = os.path.join(self.visualization_dir, "performance_history.json")
        
        try:
            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
                
            # Add timestamp to performance data
            performance_data['timestamp'] = datetime.now().isoformat()
            history.append(performance_data)
            
            with open(performance_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error storing performance data: {str(e)}")
    
    def adjust_strategy_parameters(self, params):
        """Adjusts strategy parameters based on performance metrics.
        
        Args:
            params: Dictionary of parameters to adjust
        """
        try:
            # Log the parameter adjustment
            logging.info(f"Adjusting strategy parameters: {params}")
            
            # Update risk multiplier if provided
            if 'risk_multiplier' in params:
                self.risk_multiplier = params['risk_multiplier']
                logging.info(f"Risk multiplier adjusted to {self.risk_multiplier}")
                
            # Update minimum confidence threshold if provided
            if 'min_confidence' in params:
                self.min_confidence = params['min_confidence']
                logging.info(f"Minimum confidence threshold adjusted to {self.min_confidence}")
                
            # Apply other parameter adjustments to the model
            if 'learning_rate' in params:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = params['learning_rate']
                logging.info(f"Learning rate adjusted to {params['learning_rate']}")
                
            # Store current parameters to disk
            self._save_strategy_parameters()
            
            return True
            
        except Exception as e:
            logging.error(f"Error adjusting strategy parameters: {str(e)}")
            return False
    
    def _save_strategy_parameters(self):
        """Saves current strategy parameters to disk."""
        params_file = os.path.join(self.visualization_dir, "strategy_parameters.json")
        
        try:
            # Collect all adjustable parameters
            parameters = {
                'risk_multiplier': getattr(self, 'risk_multiplier', 1.0),
                'min_confidence': getattr(self, 'min_confidence', 0.5),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'updated_at': datetime.now().isoformat()
            }
            
            with open(params_file, 'w') as f:
                json.dump(parameters, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving strategy parameters: {str(e)}")
    
    def should_switch_to_real(self):
        """Determines if the system should switch to REAL mode based on performance.
        
        Returns:
            bool: True if the system should switch to REAL mode
        """
        # Never automatically switch to REAL mode unless explicitly 
        # configured to do so by the user for safety reasons
        if not hasattr(self, 'auto_switch_to_real') or not self.auto_switch_to_real:
            return False

        try:
            performance = self.analyze_performance()

            # Check minimum requirements for switching to real mode
            if performance['status'] != 'ready_for_real':
                return False

            # Check minimum trade count
            if performance['total_trades'] < 100:
                logging.info("Not enough trades to switch to REAL mode")
                return False

            # Check sustained profitability
            if performance['win_rate'] < 0.55:
                logging.info(f"Win rate {performance['win_rate']:.2%} below threshold for REAL mode")
                return False

            if performance['profit_factor'] < 1.5:
                logging.info(f"Profit factor {performance['profit_factor']:.2f} below threshold for REAL mode")
                return False

            if performance['expectancy'] <= 0:
                logging.info(f"Expectancy {performance['expectancy']:.2f} not positive for REAL mode")
                return False

            if performance['drawdown'] > 0.15:
                logging.info(f"Drawdown {performance['drawdown']:.2%} too high for REAL mode")
                return False
            
            # Check for auto_switch_to_real flag
            if not self.auto_switch_to_real:
                logging.info("Auto-switch to REAL mode is disabled")
                return False

            # All criteria met
            logging.info("System performance meets criteria for REAL mode")
            return True

        except Exception as e:
            logging.error(f"Error evaluating switch to REAL mode: {str(e)}")
            return False
    
    def set_auto_switch(self, value: bool):
        """Enable or disable automatic switching to REAL mode."""
        self.auto_switch_to_real = value
        logging.info(f"Auto-switch to REAL mode set to: {value}")
