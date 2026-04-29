import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas_ta as ta
from utils.Logger import get_logger
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Opt-in to future behavior to avoid Downcasting warnings
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import logging
from tqdm import tqdm
from datetime import datetime
import os
import time
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import traceback


class LSTMModel(nn.Module):
    """Modelo LSTM para previsão de séries temporais financeiras."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        """Inicializa o modelo LSTM.

        Args:
            input_size: Número de features de entrada
            hidden_size: Tamanho do estado oculto
            num_layers: Número de camadas LSTM
            output_size: Número de classes de saída (3 para hold, buy, sell)
            dropout: Taxa de dropout
        """
        super(LSTMModel, self).__init__()

        # Armazena configurações para uso posterior
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # Camada LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Camada de atenção
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # Camadas fully connected
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        # Inicialização dos pesos
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializa os pesos do modelo para melhor convergência."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        """Forward pass do modelo.

        Args:
            x: Tensor de entrada de shape (batch_size, sequence_length, input_size)

        Returns:
            Tensor de saída de shape (batch_size, output_size)
        """
        # LSTM layers
        lstm_out, _ = self.lstm(x)  # shape: (batch_size, sequence_length, hidden_size)

        # Aplicar atenção
        attention_weights = self.attention(lstm_out)  # shape: (batch_size, sequence_length, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # shape: (batch_size, hidden_size)

        # Camadas fully connected
        x = F.relu(self.fc1(context_vector))
        x = self.dropout_layer(x)
        x = self.fc2(x)

        return x

# Classe HybridModel removida por não estar em uso e parecer incompleta.
# O foco atual é no LSTMModel.


class Inteligencia:

    def __init__(self, config_manager=None, error_tracker=None, model_path="lstm_model.pth",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 historical_data_filename="historical_data.csv"):
        """Inicializa a inteligência do bot com configurações robustas e tratamento de erros.

        Args:
            config_manager: Gerenciador de configurações
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
            self.model = None
            self.model_path = model_path
            self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
            self.batch_size = 32
            self.config_manager = config_manager

            # Métricas de treinamento
            self.train_losses = []
            self.val_losses = []
            self.train_accuracies = []
            self.val_accuracies = []
            self.best_accuracy = 0
            self.scaler = MinMaxScaler()
            self.is_trained = False

            # Configurações de operação
            self.mode = "LEARNING"
            self.visualization_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "training_visualizations"
            )
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
            self.logger = get_logger('Inteligencia')
            # A configuração do logger (nível, handler, formato) deve ser feita centralmente (ex: em main.py)
            self.error_tracker = error_tracker  # Armazena a instância do error_tracker
        except Exception as e:
            # Logar o erro aqui também, pois o tracker pode não estar disponível se falhar no init
            tb_str = traceback.format_exc()
            logging.critical(f"Erro CRÍTICO na inicialização da Inteligência: {str(e)}\n{tb_str}")
            # Tentar registrar no tracker se ele foi inicializado antes da falha
            if hasattr(self, 'error_tracker') and self.error_tracker:
                self.error_tracker.add_error("InteligenciaInitError", str(e), tb_str, critical=True)
            raise RuntimeError(f"Erro na inicialização da Inteligência: {str(e)}")

    def update_auto_switch_criteria(self, min_accuracy=0.7, min_precision=0.65, min_recall=0.65,
                                    min_f1_score=0.65, min_trades_count=20, min_win_rate=0.6,
                                    min_profit=0.0, auto_switch_to_real=False):
        """Atualiza os critérios para mudança automática entre modos de operação.

        Args:
            min_accuracy (float): Acurácia mínima para mudar para modo de teste
            min_precision (float): Precisão mínima para mudar para modo de teste
            min_recall (float): Recall mínimo para mudar para modo de teste
            min_f1_score (float): F1-score mínimo para mudar para modo de teste
            min_trades_count (int): Número mínimo de operações para considerar estatísticas
            min_win_rate (float): Taxa de acerto mínima para mudar para modo real
            min_profit (float): Lucro mínimo para mudar para modo real
            auto_switch_to_real (bool): Se deve mudar automaticamente para modo real
        """
        try:
            self.min_accuracy = min_accuracy
            self.min_precision = min_precision
            self.min_recall = min_recall
            self.min_f1_score = min_f1_score
            self.min_trades_count = min_trades_count
            self.min_win_rate = min_win_rate
            self.min_profit = min_profit
            self.auto_switch_to_real = auto_switch_to_real

            self.logger.info(
                f"Critérios de mudança automática atualizados: "
                f"min_accuracy={min_accuracy}, min_precision={min_precision}, "
                f"min_recall={min_recall}, min_f1_score={min_f1_score}, "
                f"min_trades_count={min_trades_count}, min_win_rate={min_win_rate}, "
                f"min_profit={min_profit}, auto_switch_to_real={auto_switch_to_real}"
            )

            return True
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao atualizar critérios de mudança automática: {str(e)}\n{tb_str}")
            if hasattr(self, 'error_tracker') and self.error_tracker:
                self.error_tracker.add_error("UpdateCriteriaError", str(e), tb_str)
            return False

    def update_auto_switch_criteria_from_config_manager(self, config_manager):
        """Atualiza os critérios de mudança automática para o modo real a partir do ConfigManager.

        Args:
            config_manager: Gerenciador de configurações
        """
        try:
            # Atualiza os critérios de mudança automática
            self.min_accuracy = config_manager.get_value('AutoSwitchCriteria', 'min_accuracy', 0.70, float)
            self.min_precision = config_manager.get_value('AutoSwitchCriteria', 'min_precision', 0.65, float)
            self.min_recall = config_manager.get_value('AutoSwitchCriteria', 'min_recall', 0.65, float)
            self.min_f1_score = config_manager.get_value('AutoSwitchCriteria', 'min_f1_score', 0.65, float)
            self.min_trades_count = config_manager.get_value('AutoSwitchCriteria', 'min_trades_count', 20, int)
            self.min_win_rate = config_manager.get_value('AutoSwitchCriteria', 'min_win_rate', 0.60, float)
            self.min_profit = config_manager.get_value('AutoSwitchCriteria', 'min_profit', 0.0, float)

            # Atualiza a flag de mudança automática
            self.auto_switch_to_real = config_manager.get_value('General', 'auto_switch_to_real', False, bool)

            self.logger.info(
                f"Critérios de mudança automática atualizados: "
                f"min_accuracy={self.min_accuracy:.2f}, "
                f"min_precision={self.min_precision:.2f}, "
                f"min_recall={self.min_recall:.2f}, "
                f"min_f1_score={self.min_f1_score:.2f}, "
                f"min_trades_count={self.min_trades_count}, "
                f"min_win_rate={self.min_win_rate:.2f}, "
                f"min_profit={self.min_profit:.2f}, "
                f"auto_switch_to_real={self.auto_switch_to_real}"
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao atualizar critérios de mudança automática: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("UpdateCriteriaError", str(e), tb_str)

    def _initialize_model(self, num_features, sequence_length):
        """Inicializa o modelo LSTM.

        Args:
            num_features: Número de features de entrada
            sequence_length: Tamanho da sequência

        """
        try:
            # Define modelo LSTM
            self.model = LSTMModel(
                input_size=num_features,
                hidden_size=128,
                num_layers=2,
                output_size=3,  # 3 classes: hold, buy, sell
                dropout=0.2
            ).to(self.device)

            self.logger.info(
                f"Modelo inicializado com {num_features} features e "
                f"sequência de tamanho {sequence_length}"
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao inicializar modelo: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("InitializeModelError", str(e), tb_str)

    def save_model(self, filename=None):
        """Salva o modelo treinado.

        Args:
            filename: Nome do arquivo para salvar o modelo

        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("Nenhum modelo para salvar")
                return False

            if filename is None:
                filename = os.path.join(self.model_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Salva o modelo e o scaler
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                'scaler': self.scaler, # Salva o objeto scaler (MinMaxScaler)
                'model_config': {
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'num_layers': self.model.num_layers,
                    'output_size': self.model.output_size,
                    'dropout': self.model.dropout
                },
                'timestamp': datetime.now().isoformat()
            }, filename)

            self.logger.info(f"Modelo salvo em {filename}")
            return True

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao salvar modelo: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("SaveModelError", str(e), tb_str)
            return False

    def load_model(self, filename=None):
        """Carrega o modelo treinado.

        Args:
            filename: Nome do arquivo para carregar o modelo

        Returns:
            bool: True se o modelo foi carregado com sucesso
        """
        try:
            if filename is None:
                # Procura o modelo mais recente
                model_files = glob.glob(os.path.join(self.model_dir, "model_*.pth"))
                if not model_files:
                    self.logger.error("Nenhum modelo encontrado para carregar")
                    return False

                filename = max(model_files, key=os.path.getctime)

            # Carrega o modelo
            checkpoint = torch.load(filename, map_location=self.device)

            # Recria o modelo com a configuração salva
            model_config = checkpoint['model_config']
            self.model = LSTMModel(
                input_size=model_config['input_size'],
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                output_size=model_config['output_size'],
                dropout=model_config['dropout']
            ).to(self.device)

            # Carrega os pesos
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Carrega o scaler
            if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                self.scaler = checkpoint['scaler']
                self.logger.info("Scaler carregado do checkpoint.")
            else:
                self.logger.warning("Scaler não encontrado no checkpoint.")
                # Tenta re-ajustar o scaler se houver dados históricos disponíveis
                if self.historical_data is not None:
                    self.logger.info("Tentando re-ajustar o scaler usando dados históricos carregados...")
                    self._normalize_data(self.historical_data, fit=True)
                else:
                    self.logger.warning("Dados históricos não disponíveis para re-ajustar o scaler.")

            # Carrega o otimizador se disponível e se self.optimizer já foi inicializado (ex: durante treinamento)
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                # Inicializa o otimizador ANTES de carregar o estado, se ainda não existir
                if not hasattr(self, 'optimizer') or self.optimizer is None:
                    # Usa o learning_rate padrão ou busca na config se disponível
                    lr = self.config_manager.get_value(
                        'Model', 'learning_rate', 0.001, float
                    ) if self.config_manager else 0.001
                    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                    self.logger.info("Otimizador Adam inicializado para carregar estado.")

                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.logger.info("Estado do otimizador carregado com sucesso.")
                except Exception as opt_load_e:
                    self.logger.error(
                        f"Erro ao carregar estado do otimizador: {opt_load_e}. "
                        f"Otimizador pode não estar no estado correto."
                    )

            self.logger.info(f"Modelo carregado de {filename}")
            return True

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao carregar modelo: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("LoadModelError", str(e), tb_str)
            return False

    def predict(self, data, confidence_threshold=0.6):
        """Faz previsões com o modelo treinado.

        Args:
            data: DataFrame ou sequência para previsão
            confidence_threshold: Limiar de confiança para considerar a previsão

        Returns:
            dict: Previsão e confiança
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("Nenhum modelo carregado para previsão")
                return None

            self.model.eval()

            # Prepara os dados
            if isinstance(data, pd.DataFrame):
                # Se os dados não tiverem indicadores técnicos, aplica o pré-processamento
                if 'adx_14' not in data.columns:
                    self.logger.info("Aplicando pré-processamento aos dados em tempo real...")
                    data_processed = self.preprocess_data(data)
                    if data_processed is None or data_processed.empty:
                        self.logger.error("Falha no pré-processamento dos dados em tempo real")
                        return None
                    data = data_processed
                
                # Normaliza os dados antes da previsão
                self.logger.info("Normalizando dados (fit=False)...")
                
                # Garante que usamos apenas as colunas numéricas e remove colunas indesejadas
                exclude_features = ['timestamp', 'date', 'label', 'asset', 'timeframe', 'datetime', 
                                   'active_id', 'ask', 'bid', 'from', 'at', 'max_at', 'id']
                feature_cols = [col for col in data.columns if col not in exclude_features]
                feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]
                
                # Se o scaler tiver nomes de colunas (feature_names_in_), usamos exatamente elas
                if hasattr(self.scaler, 'feature_names_in_'):
                    feature_cols = [col for col in self.scaler.feature_names_in_ if col in data.columns]
                    if len(feature_cols) < len(self.scaler.feature_names_in_):
                        missing = set(self.scaler.feature_names_in_) - set(feature_cols)
                        self.logger.warning(f"Colunas ausentes para normalização: {missing}")
                
                data_subset = data[feature_cols]
                data_norm = self._normalize_data(data_subset, fit=False)
                
                if data_norm is None:
                    self.logger.warning("Falha na normalização durante a previsão. Usando dados originais.")
                    data_norm = data_subset
                
                if hasattr(self, 'model') and self.model is not None:
                    expected_features = self.model.input_size
                    if data_norm.shape[1] > expected_features:
                        data_norm = data_norm.iloc[:, :expected_features]
                
                sequence = data_norm.values
                # Garante que sequence tem pelo menos 1 linha
                if len(sequence) == 0:
                    self.logger.error("Sequência vazia após processamento")
                    return None
                    
                sequence = torch.FloatTensor(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
            else:
                # Se já for um tensor ou array
                sequence = torch.FloatTensor(data).unsqueeze(0).to(self.device)

            # Faz a previsão
            with torch.no_grad():
                outputs = self.model(sequence)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)

            # Converte para numpy
            prediction = predicted.item()
            confidence_value = confidence.item()

            # Mapeia para ação
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action = action_map[prediction]

            # Log de auditoria de predição bruta
            self.logger.info(f"AUDIT PREDICTION: {action} (conf: {confidence_value:.4f})")

            # Verifica confiança
            if confidence_value < confidence_threshold:
                if action != 'HOLD':
                    self.logger.info(f"LOW CONFIDENCE: Reverting {action} to HOLD (threshold: {confidence_threshold})")
                action = 'HOLD'  # Se confiança baixa, não opera

            return {
                'prediction': prediction,
                'action': action,
                'confidence': confidence_value,
                'probabilities': probabilities.detach().cpu().numpy()[0]
            }

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao fazer previsão: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("PredictError", str(e), tb_str)
            return None

    def _validate(self, val_loader):
        """Valida o modelo no conjunto de validação.

        Args:
            val_loader: DataLoader com dados de validação

        Returns:
            tuple: (loss média, acurácia)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                # Move para o dispositivo correto
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                # Métricas
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calcula métricas médias
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, train_data, val_data=None, epochs=100, learning_rate=0.001,
              patience=10, sequence_length=20, test_size=0.2, early_stopping=True):
        """Treina o modelo com os dados fornecidos.

        Args:
            train_data: DataFrame com dados de treinamento
            val_data: DataFrame com dados de validação (opcional)
            epochs: Número de épocas de treinamento
            learning_rate: Taxa de aprendizado
            patience: Número de épocas para early stopping
            sequence_length: Tamanho da sequência para LSTM
            test_size: Proporção do conjunto de teste (se val_data não for fornecido)
            early_stopping: Se deve usar early stopping

        Returns:
            dict: Histórico de treinamento
        """
        try:
            # Verifica se há dados
            if train_data is None or len(train_data) < sequence_length:
                self.logger.error("Dados de treinamento insuficientes")
                return None

            # Prepara os dados
            if val_data is None and test_size > 0:
                # Divide em treino e validação
                train_data, val_data = self._split_data(train_data, test_size=test_size)

            # Normalização robusta (Prevenção de Data Leakage)
            self.logger.info("Aplicando normalização robusta (fit no treino, transform na validação)")
            train_data = self._normalize_data(train_data, fit=True)
            if val_data is not None:
                val_data = self._normalize_data(val_data, fit=False)

            # Pré-processa as sequências (X, y)
            X_train, y_train = self._prepare_sequences(train_data, sequence_length)
            
            if val_data is not None:
                X_val, y_val = self._prepare_sequences(val_data, sequence_length)
            else:
                split_idx = int(len(X_train) * 0.8)
                X_val, y_val = X_train[split_idx:], y_train[split_idx:]
                X_train, y_train = X_train[:split_idx], y_train[:split_idx]

            # Cria DataLoaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            # Inicializa o modelo se não existir
            if not hasattr(self, 'model') or self.model is None:
                self._initialize_model(X_train.shape[2], sequence_length)

            # Otimizador e função de perda
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()

            # Métricas de treinamento
            epochs_no_improve = 0

            # Loop de treinamento
            for epoch in range(epochs):
                # Modo de treinamento
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                # Loop pelos batches
                for batch_idx, (sequences, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                    try:
                        # Move para o dispositivo correto
                        sequences = sequences.to(self.device)
                        labels = labels.to(self.device)

                        # Zera gradientes
                        self.optimizer.zero_grad()

                        # Forward pass
                        outputs = self.model(sequences)
                        loss = self.criterion(outputs, labels)

                        # Backward pass
                        loss.backward()
                        self.optimizer.step()

                        # Métricas
                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += predicted.eq(labels).sum().item()
                    except Exception as batch_e:
                        self.logger.error(f"Erro no batch {batch_idx}: {str(batch_e)}")
                        self.logger.error(traceback.format_exc())
                        raise # Rethrow to be caught by the outer loop

                # Calcula métricas médias de treino
                avg_train_loss = train_loss / len(train_loader)
                train_accuracy = 100. * train_correct / train_total

                # Validação
                val_loss, val_accuracy = self._validate(val_loader)

                # Registra métricas
                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_accuracy)
                self.val_accuracies.append(val_accuracy)

                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
                )

                # Salva o melhor modelo
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    self.save_model(os.path.join(self.model_dir, "best_model.pth"))
                    epochs_no_improve = 0
                elif early_stopping:
                    epochs_no_improve += 1

                # Early stopping
                if early_stopping and epochs_no_improve >= patience:
                    self.logger.info(f"Early stopping após {epoch+1} épocas sem melhoria")
                    break

                # Autosave a cada N épocas
                if epoch % 5 == 0 and epoch > 0:
                    self.save_model(os.path.join(self.model_dir, f"model_epoch_{epoch}.pth"))

                # Plota métricas a cada 10 épocas
                if epoch % 10 == 0 and epoch > 0:
                    self._plot_training_metrics()

            # Salva o modelo final
            self.save_model(os.path.join(self.model_dir, "final_model.pth"))

            # Plota métricas finais
            self._plot_training_metrics()

            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'best_accuracy': self.best_accuracy
            }

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro durante o treinamento: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("TrainError", str(e), tb_str)
            return None

    def adversarial_test(self, data, sequence_length=20):
        """Realiza teste adversário (shuffled labels) para verificar aprendizado real.
        
        Args:
            data: DataFrame com dados
            sequence_length: Tamanho da sequência
            
        Returns:
            float: Acurácia no teste adversário (deve ser próxima ao acaso ~33%)
        """
        try:
            self.logger.info("Iniciando teste adversário (Shuffled Labels)...")
            
            # Cria uma cópia para não afetar os dados originais
            adv_data = data.copy()
            
            # Embaralha os rótulos
            if 'label' in adv_data.columns:
                adv_data['label'] = np.random.permutation(adv_data['label'].values)
            
            # Salva o estado atual para restaurar após o teste adversário
            original_state = None
            if hasattr(self, 'model') and self.model is not None:
                import io
                buffer = io.BytesIO()
                torch.save(self.model.state_dict(), buffer)
                original_state = buffer.getbuffer()
            
            # Treina um modelo temporário (o próprio self.model mas com pouco tempo)
            hist = self.train(adv_data, epochs=5, sequence_length=sequence_length, test_size=0.2)
            
            adv_acc = 0
            if hist:
                adv_acc = hist['val_accuracies'][-1]
            
            # Restaura o estado original
            if original_state is not None:
                import io
                buffer = io.BytesIO(original_state)
                self.model.load_state_dict(torch.load(buffer))
                self.logger.info("Estado original do modelo restaurado após teste adversário.")
                
            self.logger.info(f"Acurácia no teste adversário: {adv_acc:.2f}%")
            return adv_acc
        except Exception as e:
            self.logger.error(f"Erro no teste adversário: {str(e)}")
            return None

    def walk_forward_train(self, data, window_size=1000, step_size=200, sequence_length=20):
        """Realiza treinamento Walk-Forward para verificar consistência temporal.
        
        Args:
            data: DataFrame completo
            window_size: Tamanho da janela de treino
            step_size: Tamanho do passo de teste
            sequence_length: Tamanho da sequência
            
        Returns:
            list: Resultados de cada janela
        """
        try:
            self.logger.info("Iniciando validação Walk-Forward...")
            results = []
            
            total_len = len(data)
            for start in range(0, total_len - window_size - step_size, step_size):
                train_end = start + window_size
                test_end = train_end + step_size
                
                train_slice = data.iloc[start:train_end]
                test_slice = data.iloc[train_end:test_end]
                
                self.logger.info(f"Janela: {start} a {train_end} (Treino), {train_end} a {test_end} (Teste)")
                
                # Treina na janela
                self.train(train_slice, epochs=10, sequence_length=sequence_length)
                
                # Avalia no teste subsequente
                metrics = self.evaluate_model(test_slice, sequence_length=sequence_length)
                if metrics:
                    results.append(metrics)
                    
            return results
        except Exception as e:
            self.logger.error(f"Erro no Walk-Forward: {str(e)}")
            return None

    def _prepare_sequences(self, data, sequence_length):
        """Prepara sequências para treinamento LSTM.

        Args:
            data: DataFrame com dados
            sequence_length: Tamanho da sequência

        Returns:
            tuple: (X, y) arrays para treinamento
        """
        # Seleciona features e labels - apenas colunas numéricas
        exclude_cols = ['timestamp', 'date', 'label', 'asset', 'timeframe', 'datetime']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        # Filtra apenas colunas numéricas para evitar TypeError no PyTorch
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(data[col])]

        X = data[feature_cols].values
        y = data['label'].values if 'label' in data.columns else np.zeros(len(data))

        # Cria sequências
        X_sequences = []
        y_sequences = []

        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length])

        return np.array(X_sequences), np.array(y_sequences)

    def _split_data(self, data, test_size=0.2, random=False):
        """Divide os dados em treino e teste.

        Args:
            data: DataFrame com dados
            test_size: Proporção do conjunto de teste
            random: Se deve dividir aleatoriamente (True) ou cronologicamente (False)

        Returns:
            tuple: (train_data, test_data)
        """
        if random:
            # Divisão aleatória
            train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        else:
            # Divisão cronológica
            split_idx = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()

        return train_data, test_data

    def _plot_training_metrics(self):
        """Plota métricas de treinamento."""
        try:
            plt.figure(figsize=(12, 10))

            # Plot de loss
            plt.subplot(2, 1, 1)
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plot de acurácia
            plt.subplot(2, 1, 2)
            plt.plot(self.train_accuracies, label='Train Accuracy')
            plt.plot(self.val_accuracies, label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()

            # Salva a figura
            plt.tight_layout()
            save_path = os.path.join(
                self.visualization_dir,
                f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(save_path)
            plt.close()

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao plotar métricas: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("PlotMetricsError", str(e), tb_str)

    def evaluate_model(self, test_data, sequence_length=20):
        """Avalia o modelo em dados de teste e armazena as métricas.

        Args:
            test_data: DataFrame com dados de teste
            sequence_length: Tamanho da sequência

        Returns:
            dict: Métricas de avaliação ou None em caso de erro
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("Nenhum modelo carregado para avaliação")
                return None

            # Prepara os dados
            test_data_norm = self._normalize_data(test_data, fit=False)
            if test_data_norm is None:
                self.logger.error("Falha ao normalizar dados de teste.")
                return None
                
            X_test, y_test = self._prepare_sequences(test_data_norm, sequence_length)
            if X_test is None or y_test is None or len(X_test) == 0:
                self.logger.error("Falha ao preparar dados de teste para avaliação.")
                return None

            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

            # Avaliação
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for sequences, labels in test_loader:
                    sequences = sequences.to(self.device)
                    outputs = self.model(sequences)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # Calcula métricas
            accuracy = accuracy_score(all_labels, all_preds)
            # Use zero_division=0 para evitar warnings/erros se uma classe não for prevista
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)

            self.logger.info(
                f"Avaliação do modelo - Acurácia: {accuracy:.4f}, "
                f"Precisão: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )
            self._plot_confusion_matrix(cm)

            # Armazena as métricas para decisão de mudança de modo
            self.last_evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }

            return self.last_evaluation_metrics

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro durante a avaliação: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("EvaluateModelError", str(e), tb_str)
            # import traceback  # Removido
            # self.logger.error(traceback.format_exc())  # Log já inclui traceback
            self.last_evaluation_metrics = None  # Reseta em caso de erro
            return None

    def should_switch_to_test_mode(self) -> bool:
        """Verifica se o desempenho do modelo atende aos critérios para mudar para o modo Teste.

        Returns:
            bool: True se deve mudar para Teste, False caso contrário.
        """
        if not hasattr(self, 'last_evaluation_metrics') or self.last_evaluation_metrics is None:
            self.logger.warning("Métricas de avaliação não disponíveis para decidir mudança para modo Teste.")
            return False

        metrics = self.last_evaluation_metrics
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)

        # Verifica se todas as métricas de avaliação do modelo atingem o mínimo
        meets_criteria = (
            accuracy >= self.min_accuracy and
            precision >= self.min_precision and
            recall >= self.min_recall and
            f1 >= self.min_f1_score
        )

        if meets_criteria:
            self.logger.info(
                f"Critérios de avaliação do modelo atendidos (Acc: {accuracy:.2f}, "
                f"Prec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f}). "
                f"Recomendando mudança para modo Teste."
            )
            return True
        else:
            self.logger.info(
                f"Critérios de avaliação do modelo NÃO atendidos (Acc: {accuracy:.2f}, "
                f"Prec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f}). "
                f"Mantendo modo Aprendizado."
            )
            return False

    def should_switch_to_real_mode(self, performance_tracker) -> bool:
        """Verifica se o desempenho no modo Teste atende aos critérios para mudar para o modo Real.

        Args:
            performance_tracker: Objeto contendo métricas de desempenho das operações (win rate, profit, etc.)

        Returns:
            bool: True se deve mudar para Real, False caso contrário.
        """
        if not self.auto_switch_to_real:
            self.logger.info("Mudança automática para modo Real está desabilitada nas configurações.")
            return False

        if performance_tracker is None:
            self.logger.warning("Performance tracker não disponível para decidir mudança para modo Real.")
            return False

        # Acessa as métricas do PerformanceTracker (ajuste os nomes se necessário)
        # Supondo que performance_tracker tenha métodos/atributos como: win_rate(), total_profit(), trades_count()
        try:
            # Acessa as métricas do dicionário 'metrics' do PerformanceTracker
            win_rate = performance_tracker.metrics.get('win_rate', 0.0)
            profit = performance_tracker.metrics.get('total_profit', 0.0)
            trades_count = performance_tracker.metrics.get('total_trades', 0)
        except AttributeError as e:
            tb_str = traceback.format_exc()
            self.logger.error(
                f"Erro ao acessar métricas do PerformanceTracker: {e}. "
                f"Verifique os métodos/atributos.\n{tb_str}"
            )
            if self.error_tracker:
                self.error_tracker.add_error("PerformanceTrackerAccessError", str(e), tb_str)
            return False

        self.logger.info(
            f"Avaliando mudança para modo Real: Win Rate={win_rate:.2f} "
            f"(Min: {self.min_win_rate:.2f}), Profit={profit:.2f} "
            f"(Min: {self.min_profit:.2f}), Trades={trades_count} "
            f"(Min: {self.min_trades_count})"
        )

        # Verifica se os critérios de desempenho em teste são atendidos
        meets_criteria = (
            trades_count >= self.min_trades_count and
            win_rate >= self.min_win_rate and
            profit >= self.min_profit
        )

        if meets_criteria:
            self.logger.info("Critérios de desempenho em Teste atendidos. Recomendando mudança para modo Real.")
            return True
        else:
            self.logger.info("Critérios de desempenho em Teste NÃO atendidos. Mantendo modo Teste.")
            return False

    def should_stay_in_real_mode(self, performance_tracker) -> bool:
        """Verifica se o desempenho no modo Real justifica a permanência neste modo.

        Se o desempenho cair abaixo dos mínimos, recomenda voltar para o modo Teste.

        Args:
            performance_tracker: Objeto contendo métricas de desempenho das operações.

        Returns:
            bool: True se deve permanecer em Real, False se deve voltar para Teste.
        """
        if performance_tracker is None:
            self.logger.warning("Performance tracker não disponível para decidir permanência no modo Real.")
            return True  # Default: permanece em Real se não puder avaliar

        try:
            # Acessa as métricas do dicionário 'metrics' do PerformanceTracker
            win_rate = performance_tracker.metrics.get('win_rate', 0.0)
            profit = performance_tracker.metrics.get('total_profit', 0.0)
            trades_count = performance_tracker.metrics.get('total_trades', 0)
        except AttributeError as e:
            tb_str = traceback.format_exc()
            self.logger.error(
                f"Erro ao acessar métricas do PerformanceTracker: {e}. "
                f"Verifique os métodos/atributos.\n{tb_str}"
            )
            if self.error_tracker:
                self.error_tracker.add_error("PerformanceTrackerAccessError", str(e), tb_str)
            return True  # Default: permanece em Real se não puder avaliar

        self.logger.info(
            f"Avaliando permanência no modo Real: Win Rate={win_rate:.2f} "
            f"(Min: {self.min_win_rate:.2f}), Profit={profit:.2f} "
            f"(Min: {self.min_profit:.2f}), Trades={trades_count} "
            f"(Min: {self.min_trades_count})"
        )

        # Verifica se os critérios mínimos ainda são atendidos
        # Usamos os mesmos critérios de entrada, mas poderiam ser diferentes (ex: um limiar de 'degradação')
        meets_criteria = (
            trades_count >= self.min_trades_count and  # Garante que há dados suficientes
            win_rate >= self.min_win_rate and
            profit >= self.min_profit
        )

        if meets_criteria:
            self.logger.info("Desempenho em modo Real continua satisfatório. Permanecendo em modo Real.")
            return True
        else:
            self.logger.warning(
                "Desempenho em modo Real caiu abaixo dos critérios mínimos. "
                "Recomendando retorno para modo Teste."
            )
            return False

    def _plot_confusion_matrix(self, cm):
        """Plota matriz de confusão.

        Args:
            cm: Matriz de confusão
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            save_path = os.path.join(
                self.visualization_dir,
                f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            self.logger.error(f"Erro ao plotar matriz de confusão: {str(e)}")

    def preprocess_data(self, data, normalize=False, threshold=None):
        """Pré-processa dados históricos bruta para o formato do modelo.

        Args:
            data (pd.DataFrame): Dados brutos
            normalize (bool): Se deve normalizar os dados (fit=False)
 históricos

        Returns:
            DataFrame: Dados pré-processados
        """
        try:
            # Cria cópia para não modificar o original
            df = data.copy()

            # Verifica se há dados suficientes
            if len(df) < 100:
                self.logger.error("Dados insuficientes para pré-processamento")
                return None

            # Converte timestamp para datetime se necessário
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Adiciona coluna de data se não existir
            if 'date' not in df.columns and 'timestamp' in df.columns:
                df['date'] = df['timestamp'].dt.date

            # Preenche valores ausentes
            df = df.ffill().bfill()

            # Adiciona indicadores técnicos
            df = self._add_technical_indicators(df)

            # Adiciona padrões de candles
            df = self._add_candle_patterns(df)

            # Adiciona features derivadas
            df = self._add_derived_features(df)

            # Normaliza os dados (opcional e apenas se solicitado)
            if normalize:
                df = self._normalize_data(df, fit=False)

            # Cria rótulos para treinamento supervisionado
            if 'label' not in df.columns:
                if threshold is not None:
                    df = self.create_labels(df, threshold=threshold)
                else:
                    df = self.create_labels(df)

            # Remove linhas com NaN após criação de indicadores
            df = df.dropna()

            self.logger.info(f"Dados pré-processados: {len(df)} linhas, {len(df.columns)} colunas")
            return df

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro no pré-processamento: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("PreprocessDataError", str(e), tb_str)
            return None

    def _add_technical_indicators(self, df):
        """Adiciona indicadores técnicos ao DataFrame.

        Args:
            df: DataFrame com dados históricos

        Returns:
            DataFrame: DataFrame com indicadores adicionados
        """
        try:
            # Verifica se existem as colunas necessárias
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Colunas necessárias ausentes: {required_cols}")
                return df

            # Cria uma cópia para evitar avisos de SettingWithCopyWarning
            df = df.copy()

            # Médias Móveis
            for period in [5, 10, 20, 50, 200]:
                if len(df) >= period:
                    df[f'sma_{period}'] = ta.sma(df['close'], length=period)
                    df[f'ema_{period}'] = ta.ema(df['close'], length=period)

            # RSI
            if len(df) >= 14:
                df['rsi_14'] = ta.rsi(df['close'], length=14)

            # MACD
            if len(df) >= 26:
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
                if isinstance(macd, pd.DataFrame) and not macd.empty:
                    df['macd'] = macd['MACD_12_26_9']
                    df['macd_signal'] = macd['MACDs_12_26_9']
                    df['macd_hist'] = macd['MACDh_12_26_9']
                elif isinstance(macd, tuple) and len(macd) >= 3:
                    # Caso o retorno seja uma tupla (algumas versões do TA-Lib)
                    df['macd'] = macd[0]
                    df['macd_signal'] = macd[1]
                    df['macd_hist'] = macd[2]

            # Bollinger Bands
            if len(df) >= 20:
                bbands = ta.bbands(df['close'], length=20, std=2)
                if isinstance(bbands, pd.DataFrame) and not bbands.empty:
                    # Tenta nomes padrão do pandas_ta ou os mapeia dinamicamente
                    bb_up_col = [c for c in bbands.columns if c.startswith('BBU')][0] if any(c.startswith('BBU') for c in bbands.columns) else None
                    bb_mid_col = [c for c in bbands.columns if c.startswith('BBM')][0] if any(c.startswith('BBM') for c in bbands.columns) else None
                    bb_low_col = [c for c in bbands.columns if c.startswith('BBL')][0] if any(c.startswith('BBL') for c in bbands.columns) else None

                    if bb_up_col:
                        df['bb_upper'] = bbands[bb_up_col]
                    if bb_mid_col:
                        df['bb_middle'] = bbands[bb_mid_col]
                    if bb_low_col:
                        df['bb_lower'] = bbands[bb_low_col]
                    
                    if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
                        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                elif isinstance(bbands, tuple) and len(bbands) >= 3:
                    df['bb_upper'] = bbands[0]
                    df['bb_middle'] = bbands[1]
                    df['bb_lower'] = bbands[2]
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # ATR - Average True Range
            if len(df) >= 14:
                df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # Stochastic Oscillator
            if len(df) >= 14:
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
                if not stoch.empty:
                    df['stoch_k'] = stoch['STOCHk_14_3_3']
                    df['stoch_d'] = stoch['STOCHd_14_3_3']

            # OBV - On Balance Volume
            df['obv'] = ta.obv(df['close'], df['volume'])

            # ADX - Average Directional Index
            if len(df) >= 14:
                adx = ta.adx(df['high'], df['low'], df['close'], length=14)
                if not adx.empty:
                    df['adx_14'] = adx['ADX_14']

            # Adicionar outros indicadores avançados

            # Momentum
            if len(df) >= 10:
                df['mom_10'] = ta.mom(df['close'], length=10)

            # Rate of Change
            if len(df) >= 10:
                df['roc_10'] = ta.roc(df['close'], length=10)

            # Williams %R
            if len(df) >= 14:
                df['willr_14'] = ta.willr(df['high'], df['low'], df['close'], length=14)

            # CCI - Commodity Channel Index
            if len(df) >= 20:
                df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)

            # Ichimoku Cloud
            if len(df) >= 52:
                ichimoku_res = ta.ichimoku(df['high'], df['low'], df['close'])
                if ichimoku_res is not None:
                    # pandas_ta ichimoku retorna (df, span_df)
                    ichimoku = ichimoku_res[0] if isinstance(ichimoku_res, tuple) else ichimoku_res
                    
                    if isinstance(ichimoku, pd.DataFrame) and not ichimoku.empty:
                        # Map components with flexible column naming
                        col_map = {
                            'ITS_9': 'ichimoku_tenkan',
                            'IKS_26': 'ichimoku_kijun',
                            'ISA_9': 'ichimoku_senkou_a',
                            'ISB_26': 'ichimoku_senkou_b'
                        }
                        for old_col, new_col in col_map.items():
                            found = [c for c in ichimoku.columns if c.startswith(old_col)]
                            if found:
                                df[new_col] = ichimoku[found[0]]

            # Preenchimento de NaN
            df = df.bfill().ffill().fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Erro ao adicionar indicadores técnicos: {str(e)}")
            return df

    def _add_candle_patterns(self, df):
        """Adiciona padrões de candles ao DataFrame.

        Args:
            df: DataFrame com dados históricos

        Returns:
            DataFrame: DataFrame com padrões de candles adicionados
        """
        try:
            # Verifica se existem as colunas necessárias
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Colunas necessárias ausentes: {required_cols}")
                return df

            # Extrai arrays para cálculos
            open_prices = df['open'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Doji
            df['doji'] = np.where(np.abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1, 1, 0)

            # Hammer
            df['hammer'] = 0
            body_size = np.abs(open_prices - close)
            lower_shadow = np.minimum(open_prices, close) - low
            upper_shadow = high - np.maximum(open_prices, close)

            for i in range(len(df)):
                if body_size[i] > 0 and lower_shadow[i] >= 2 * body_size[i] and upper_shadow[i] <= 0.2 * body_size[i]:
                    df.loc[df.index[i], 'hammer'] = 1

            # Shooting Star
            df['shooting_star'] = 0
            for i in range(len(df)):
                if body_size[i] > 0 and upper_shadow[i] >= 2 * body_size[i] and lower_shadow[i] <= 0.2 * body_size[i]:
                    df.loc[df.index[i], 'shooting_star'] = 1

            # Engulfing patterns
            df['bullish_engulfing'] = 0
            df['bearish_engulfing'] = 0

            for i in range(1, len(df)):
                # Bullish engulfing
                if (close[i] > open_prices[i] and  # Current candle is bullish
                        close[i-1] < open_prices[i-1] and  # Previous candle is bearish
                        open_prices[i] < close[i-1] and  # Current open is lower than previous close
                        close[i] > open_prices[i-1]):  # Current close is higher than previous open
                    df.loc[df.index[i], 'bullish_engulfing'] = 1

                # Bearish engulfing
                if (close[i] < open_prices[i] and  # Current candle is bearish
                        close[i-1] > open_prices[i-1] and  # Previous candle is bullish
                        open_prices[i] > close[i-1] and  # Current open is higher than previous close
                        close[i] < open_prices[i-1]):  # Current close is lower than previous open
                    df.loc[df.index[i], 'bearish_engulfing'] = 1

            return df

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao adicionar padrões de candles: {str(e)}\n{tb_str}")
            if hasattr(self, 'error_tracker') and self.error_tracker:
                self.error_tracker.add_error("AddCandlePatternsError", str(e), tb_str)
            return None  # Retorna None em caso de erro

    def _add_derived_features(self, df):
        """Adiciona features derivadas ao DataFrame.

        Args:
            df: DataFrame com dados históricos

        Returns:
            DataFrame: DataFrame com features derivadas adicionadas
        """
        try:
            # Retornos
            df['return_1d'] = df['close'].pct_change(1)
            df['return_5d'] = df['close'].pct_change(5)
            df['return_10d'] = df['close'].pct_change(10)

            # Volatilidade
            df['volatility_5d'] = df['return_1d'].rolling(window=5).std()
            df['volatility_10d'] = df['return_1d'].rolling(window=10).std()
            df['volatility_20d'] = df['return_1d'].rolling(window=20).std()

            # Momentum
            df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1

            # Distância das médias móveis
            if 'sma_20' in df.columns:
                df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
            if 'sma_50' in df.columns:
                df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
            if 'sma_200' in df.columns:
                df['dist_sma_200'] = (df['close'] - df['sma_200']) / df['sma_200']

            # Cruzamentos de médias móveis
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)

            # Relação entre volume e preço
            df['volume_price_ratio'] = df['volume'] / df['close']
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma_10']

            return df

        except Exception as e:
            self.logger.error(f"Erro ao adicionar features derivadas: {str(e)}")
            return df

    def _normalize_data(self, df, fit=False):
        """Normaliza os dados numéricos usando MinMaxScaler.

        Args:
            df: DataFrame com dados históricos
            fit: Se deve ajustar o scaler aos dados (usar apenas em treino)

        Returns:
            DataFrame: DataFrame com dados normalizados ou None em caso de erro.
        """
        try:
            self.logger.info(f"Normalizando dados (fit={fit})...")
            df_normalized = df.copy()

            exclude_cols = [
                'timestamp', 'date', 'label', 'asset',
                'doji', 'hammer', 'shooting_star',
                'bullish_engulfing', 'bearish_engulfing'
            ]

            numeric_cols = df_normalized.select_dtypes(include=np.number).columns.tolist()
            norm_cols = [col for col in numeric_cols if col not in exclude_cols]

            if not norm_cols:
                self.logger.warning("Nenhuma coluna numérica encontrada para normalização.")
                return df_normalized

            if fit:
                df_normalized[norm_cols] = self.scaler.fit_transform(df_normalized[norm_cols])
            else:
                try:
                    df_normalized[norm_cols] = self.scaler.transform(df_normalized[norm_cols])
                except Exception as e:
                    if "not fitted" in str(e):
                        self.logger.warning("Scaler não ajustado. Retornando dados sem normalização.")
                    else:
                        raise e

            return df_normalized

        except Exception as e:
            self.logger.error(f"Erro na normalização: {str(e)}")
            # Em vez de retornar None, retorna o df original para evitar quebras em cadeia
            return df if 'df' in locals() else None

    def _calculate_financial_metrics(self, prices, preds, labels, payout=0.85):
        """Calcula métricas financeiras reais das previsões.
        
        Args:
            prices: Preços de fechamento
            preds: Previsões do modelo (0: Hold, 1: Buy, 2: Sell)
            labels: Rótulos reais
            payout: Payout médio da corretora (ex: 0.85 para 85%)
            
        Returns:
            dict: Métricas calculadas
        """
        try:
            trades = []
            balance = 1000.0 # Saldo inicial fictício para cálculo de drawdown
            peak = balance
            drawdown = 0
            max_drawdown = 0
            
            # 0: Hold, 1: Buy (Call), 2: Sell (Put)
            for i in range(len(preds)):
                if preds[i] == 0: # Hold
                    continue
                    
                is_win = (preds[i] == labels[i])
                
                if is_win:
                    profit = 1.0 * payout
                    balance += profit
                else:
                    profit = -1.0
                    balance += profit
                
                trades.append(profit)
                
                if balance > peak:
                    peak = balance
                
                drawdown = (peak - balance) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            if not trades:
                return {'ev': 0, 'sharpe': 0, 'mdd': 0, 'win_rate': 0}
                
            trades = np.array(trades)
            win_rate = np.mean(trades > 0)
            avg_return = np.mean(trades)
            std_return = np.std(trades) if len(trades) > 1 else 1e-6
            
            sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            ev = (win_rate * payout) - ((1 - win_rate) * 1.0)
            
            return {
                'ev': ev,
                'sharpe': sharpe,
                'mdd': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades)
            }
        except Exception as e:
            self.logger.error(f"Erro ao calcular métricas financeiras: {str(e)}")
            return None

    def create_labels(self, df, window=5, threshold=0.01):
        """Cria rótulos para treinamento supervisionado.

        Args:
            df: DataFrame com dados históricos
            window: Janela para calcular tendência futura
            threshold: Limiar para considerar movimento significativo

        Returns:
            DataFrame: DataFrame com rótulos adicionados
        """
        try:
            # Calcula retorno futuro
            df['future_return'] = df['close'].shift(-window) / df['close'] - 1

            # Cria rótulos
            df['label'] = 0  # 0: Hold
            df.loc[df['future_return'] > threshold, 'label'] = 1  # 1: Buy
            df.loc[df['future_return'] < -threshold, 'label'] = 2  # 2: Sell

            # Remove a coluna de retorno futuro
            df = df.drop('future_return', axis=1)

            # Conta as classes
            class_counts = df['label'].value_counts()
            self.logger.info(f"Distribuição de classes: {class_counts.to_dict()}")

            return df

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro na criação de rótulos: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("CreateLabelsError", str(e), tb_str)
            return None  # Retorna None em caso de erro

    def process_historical_data(self, data, asset_name, timeframe=60, save_processed=True, normalize=True, window=5, threshold=0.01):
        """Processa dados históricos para treinamento.

        Args:
            data (pd.DataFrame): DataFrame com dados históricos (OHLC)
            asset_name (str): Nome do ativo (ex: "EURUSD")
            timeframe (int): Timeframe em segundos (60, 300, etc.)
            save_processed (bool): Se deve salvar os dados processados
            normalize (bool): Se deve normalizar os dados (usar False para treino/validação separado)
            window (int): Janela para calcular tendência futura (lookahead)
            threshold (float): Limiar para considerar movimento significativo

        Returns:
            pd.DataFrame: Dados processados prontos para treinamento
        """
        try:
            self.logger.info(f"Processando dados históricos para {asset_name} (timeframe: {timeframe}s, window: {window}, threshold: {threshold})")

            if data is None or len(data) == 0:
                self.logger.error("Dados históricos vazios ou nulos")
                return None

            # Verifica se os dados têm as colunas necessárias
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                self.logger.error(f"Colunas ausentes nos dados: {missing_columns}")
                return None

            # Copia os dados para não modificar o original
            df = data.copy()

            # Converte a coluna de data para datetime se necessário
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)

            # 1. Pré-processamento
            self.logger.info("Aplicando pré-processamento aos dados")
            df = self.preprocess_data(df, threshold=threshold)

            # 2. Criação de rótulos
            self.logger.info(f"Criando rótulos para treinamento supervisionado (threshold={threshold}, window={window})")
            df = self.create_labels(df, window=window, threshold=threshold)

            # 3. Normalização
            if normalize:
                self.logger.info("Normalizando dados")
                df = self._normalize_data(df, fit=True)
            else:
                self.logger.info("Pulando normalização (será feita no treino/validação separadamente)")

            # 4. Salvamento (opcional)
            if save_processed:
                # Cria diretório para dados processados se não existir
                processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
                os.makedirs(processed_dir, exist_ok=True)

                # Nome do arquivo baseado no ativo e timeframe
                filename = f"{asset_name.lower()}_{timeframe}_processed.csv"
                filepath = os.path.join(processed_dir, filename)

                # Salva os dados processados
                df.to_csv(filepath)
                self.logger.info(f"Dados processados salvos em {filepath}")

            # Armazena os dados processados para uso posterior
            self.historical_data = df

            self.logger.info(f"Processamento concluído. Shape final: {df.shape}")
            return df

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao processar dados históricos: {str(e)}\n{tb_str}")
            if hasattr(self, 'error_tracker') and self.error_tracker:
                self.error_tracker.add_error("ProcessHistoricalDataError", str(e), tb_str)
            # import traceback  # Removido
            # self.logger.error(traceback.format_exc())  # Log já inclui traceback
            return None

    def setup_test_mode(self):
        """Configura a inteligência para o modo de teste.

        Este método prepara a inteligência para operar em modo de teste,
        carregando modelos treinados e configurando parâmetros específicos para testes.

        Returns:
            bool: True se o modo de teste foi configurado com sucesso
        """
        try:
            self.logger.info("Configurando inteligência para modo de teste")

            # Define o modo de operação
            self.operation_mode = 'test'

            if self.load_model():
                self.mode = "TEST"
                if self.config_manager:
                    self.update_auto_switch_criteria_from_config_manager(self.config_manager)
                self.logger.info("Modo Teste configurado com sucesso (modelo carregado)")
                return True
            else:
                self.logger.warning("Modelo treinado não encontrado. Criando modelo básico para testes")
                return self._create_basic_test_model()

            # Configura parâmetros de teste
            self.test_win_rate = 0.0
            self.test_trades_count = 0
            self.test_profit = 0.0

            self.logger.info("Modo de teste configurado com sucesso")
            return True

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao criar modelo básico: {str(e)}\n{tb_str}")
            if hasattr(self, 'error_tracker') and self.error_tracker:
                self.error_tracker.add_error("CreateBasicModelError", str(e), tb_str)
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao criar modelo básico: {str(e)}\n{tb_str}")
            if hasattr(self, 'error_tracker') and self.error_tracker:
                self.error_tracker.add_error("CreateBasicModelError", str(e), tb_str)
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao configurar modo de teste: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("SetupTestModeError", str(e), tb_str)
            return False

    def _create_basic_test_model(self):
        """Cria um modelo básico para testes quando não há modelo treinado disponível."""
        try:
            self.logger.info("Criando modelo básico para testes")

            # Instancia um LSTMModel com parâmetros padrão como fallback
            # Nota: O input_size=50 é um palpite; idealmente, seria inferido dos dados.
            default_input_size = 50
            self.model = LSTMModel(
                input_size=default_input_size,
                hidden_size=128,
                num_layers=2,
                output_size=3,  # 3 classes: hold, buy, sell
                dropout=0.2
            ).to(self.device)
            self.logger.warning(f"Modelo LSTM básico criado com input_size={default_input_size} (fallback).")
            self.mode = "TEST"
            return True

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao criar modelo básico: {str(e)}\n{tb_str}")
            if self.error_tracker:
                self.error_tracker.add_error("CreateBasicModelError", str(e), tb_str)

    def distribution_drift_test(self, train_data, val_data, sequence_length=20):
        """Realiza teste de validação adversária para detectar drift de distribuição.
        
        Tenta treinar um classificador para distinguir entre dados de treino e validação.
        Acurácia próxima a 50% indica que as distribuições são similares (bom).
        Acurácia alta indica leak de dados ou mudança de regime (ruim).
        
        Args:
            train_data: Dados de treinamento
            val_data: Dados de validação
            sequence_length: Tamanho da sequência
            
        Returns:
            float: Acurácia da distinção (0.5 é ideal)
        """
        try:
            self.logger.info("Iniciando teste de Distribution Drift (Adversarial Validation)...")
            
            # Prepara sequências de treino e validação
            X_train, _ = self._prepare_sequences(self._normalize_data(train_data, fit=False), sequence_length)
            X_val, _ = self._prepare_sequences(self._normalize_data(val_data, fit=False), sequence_length)
            
            # Cria labels adversários (0 para treino, 1 para validação)
            y_adv_train = np.zeros(len(X_train))
            y_adv_val = np.ones(len(X_val))
            
            X_adv = np.concatenate([X_train, X_val])
            y_adv = np.concatenate([y_adv_train, y_adv_val])
            
            # Shuffle
            idx = np.random.permutation(len(X_adv))
            X_adv, y_adv = X_adv[idx], y_adv[idx]
            
            # Divide para um treino rápido do auditor
            split = int(0.8 * len(X_adv))
            X_adv_train, X_adv_test = X_adv[:split], X_adv[split:]
            y_adv_train_final, y_adv_test_final = y_adv[:split], y_adv[split:]
            
            # Treina um mini-modelo auditor (Linear)
            # Para simplicidade e velocidade, usamos o próprio framework mas com 1 epoch
            input_dim = X_adv.shape[2]
            auditor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim * sequence_length, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ).to(self.device)
            
            optimizer = optim.Adam(auditor.parameters(), lr=0.01)
            criterion = nn.BCELoss()
            
            # Treino rápido do auditor
            auditor.train()
            batch_size = 32
            for i in range(0, len(X_adv_train), batch_size):
                end = min(i + batch_size, len(X_adv_train))
                xb = torch.FloatTensor(X_adv_train[i:end]).to(self.device)
                yb = torch.FloatTensor(y_adv_train_final[i:end]).unsqueeze(1).to(self.device)
                
                optimizer.zero_grad()
                out = auditor(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            
            # Avaliação do drift
            auditor.eval()
            with torch.no_grad():
                xt = torch.FloatTensor(X_adv_test).to(self.device)
                preds = (auditor(xt) > 0.5).cpu().numpy().flatten()
                drift_acc = accuracy_score(y_adv_test_final, preds)
                
            self.logger.info(f"Acurácia do Distribution Drift: {drift_acc:.2f}")
            return drift_acc
            
        except Exception as e:
            self.logger.error(f"Erro no teste de distribution drift: {str(e)}")
            return None

    def generate_training_report(self, train_data=None, val_data=None, sequence_length=20):
        """Gera um relatório completo da qualidade do treinamento com Auditoria Dupla.
        
        Args:
            train_data: Dados de treino para drift test
            val_data: Dados de validação
            sequence_length: Tamanho da sequência
            
        Returns:
            str: Relatório formatado em Markdown
        """
        try:
            self.logger.info("Gerando relatório de qualidade do treinamento...")
            
            # 1. Avaliação do Modelo Original
            metrics = self.last_evaluation_metrics if hasattr(self, 'last_evaluation_metrics') else None
            if metrics is None and val_data is not None:
                metrics = self.evaluate_model(val_data, sequence_length)
            
            # 2. Dual Adversarial Audit
            adv_shuffle_acc = None
            adv_drift_acc = None
            if val_data is not None:
                adv_shuffle_acc = self.adversarial_test(val_data, sequence_length)
                if train_data is not None:
                    adv_drift_acc = self.distribution_drift_test(train_data, val_data, sequence_length)
            
            # 3. Métricas Financeiras
            fin_metrics = None
            if val_data is not None and metrics:
                 # Precisamos das previsões reais para calcular EV/Sharpe
                 X_val, y_val = self._prepare_sequences(self._normalize_data(val_data, fit=False), sequence_length)
                 self.model.eval()
                 with torch.no_grad():
                     outputs = self.model(torch.FloatTensor(X_val).to(self.device))
                     _, preds = outputs.max(1)
                     preds = preds.cpu().numpy()
                 
                 # Pega preços reais de fechamento correspondentes aos labels
                 # Nota: y_val já está alinhado com o final da sequência (sequence_length:)
                 prices = val_data['close'].values[sequence_length:]
                 fin_metrics = self._calculate_financial_metrics(prices, preds, y_val)

            # Monta o report
            report = []
            report.append("# 📊 Relatório de Qualidade Absoluta (Dual Adversarial Audit)")
            report.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("\n## 1. Métricas de Performance Primária")
            
            if metrics:
                report.append(f"- **Acurácia Final**: {metrics['accuracy']*100:.2f}%")
                report.append(f"- **F1-Score (Weighted)**: {metrics['f1']:.4f}")
                report.append(f"- **Precisão**: {metrics['precision']:.4f}")
            else:
                report.append("- *Métricas não disponíveis*")

            report.append("\n## 2. Auditoria Adversária (Dual-check)")
            
            # Auditor 1: Shuffle Labels
            if adv_shuffle_acc is not None:
                real_acc = metrics['accuracy'] * 100 if metrics else 0
                is_shuffle_legit = (real_acc > adv_shuffle_acc * 1.5) and (adv_shuffle_acc < 45)
                report.append(f"### Auditor A: Label Shuffle (Noise Fitting Check)")
                report.append(f"- **Acurácia Aleatória**: {adv_shuffle_acc:.2f}%")
                report.append(f"- **Status**: {'✅ LEGÍTIMO' if is_shuffle_legit else '❌ POSSÍVEL OVERFITTING'}")
            
            # Auditor 2: Distribution Drift
            if adv_drift_acc is not None:
                is_drift_legit = adv_drift_acc < 0.70 # Ideal é 0.5, acima de 0.7 indica drift/leakage
                report.append(f"### Auditor B: Distribution Drift (Regime Consistency Check)")
                report.append(f"- **Acurácia de Distinção**: {adv_drift_acc:.2f}")
                report.append(f"- **Status**: {'✅ CONSISTENTE' if is_drift_legit else '❌ DRIFT DETECTADO'}")
            
            report.append("\n## 3. Viabilidade Financeira (Simulação)")
            if fin_metrics:
                report.append(f"- **Win Rate Simulado**: {fin_metrics['win_rate']*100:.2f}%")
                report.append(f"- **Expected Value per Trade (EV)**: {fin_metrics['ev']:.4f}")
                report.append(f"- **Sharpe Ratio (Anualizado)**: {fin_metrics['sharpe']:.4f}")
                report.append(f"- **Max Drawdown**: {fin_metrics['mdd']*100:.2f}%")
                
                is_profitable = fin_metrics['ev'] > 0
                report.append(f"- **Status de Lucratividade**: {'💰 LUCRATIVO' if is_profitable else '📉 NÃO LUCRATIVO (Ajustar Limiar ou Modelos)'}")
            
            report_str = "\n".join(report)
            
            # Salva o report em arquivo
            report_path = os.path.join(self.visualization_dir, "training_quality_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            
            self.logger.info(f"Relatório gerado com sucesso em: {report_path}")
            return report_str

        except Exception as e:
            self.logger.error(f"Erro ao gerar relatório: {str(e)}")
            return f"Erro ao gerar relatório: {str(e)}"
