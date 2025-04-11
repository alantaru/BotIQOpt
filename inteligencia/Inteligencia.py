import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas_ta as ta # Importar pandas_ta explicitamente
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

# from torchvision import models, transforms
import joblib
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import json
import re
import hashlib
import time
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import copy
import traceback # Importar traceback para logs de erro mais detalhados


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
    def __init__(self, config_manager=None, error_tracker=None, model_path="lstm_model.pth", device="cuda" if torch.cuda.is_available() else "cpu", historical_data_filename="historical_data.csv"): # Adicionado error_tracker
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
            self.error_tracker = error_tracker # Armazena a instância do error_tracker
            self.scaler = None # Adicionado para armazenar o scaler ajustado
            self.feature_cols = None # Adicionado para armazenar as colunas usadas no treino
            
            # Métricas de treinamento
            self.train_losses = []
            self.val_losses = []
            self.train_accuracies = []
            self.val_accuracies = []
            self.best_accuracy = 0
            
            # Configurações de operação
            self.mode = "LEARNING"
            self.visualization_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_visualizations")
            os.makedirs(self.visualization_dir, exist_ok=True)
            self.historical_data_filename = historical_data_filename
            self.historical_data = None
            
            # Segurança e credenciais
            self.used_credentials = set()
            self.risk_multiplier = 1.0
            self.min_confidence = 0.75 # Aumentado o limiar de confiança
            
            # Inicializa os critérios com valores padrão ou do config, mas permite atualização
            self.min_accuracy = config_manager.get_value('AutoSwitch', 'min_accuracy', 0.75, float) # Atualizado para 0.75
            self.min_precision = config_manager.get_value('AutoSwitch', 'min_precision', 0.75, float) # Atualizado para 0.75
            self.min_recall = config_manager.get_value('AutoSwitch', 'min_recall', 0.75, float) # Atualizado para 0.75
            self.min_f1_score = config_manager.get_value('AutoSwitch', 'min_f1_score', 0.75, float) # Atualizado para 0.75
            self.min_trades_count = config_manager.get_value('AutoSwitch', 'min_trades_count', 20, int)
            self.min_win_rate = config_manager.get_value('AutoSwitch', 'min_win_rate', 0.75, float) # Atualizado para 0.75
            self.min_profit = config_manager.get_value('AutoSwitch', 'min_profit', 0.0, float)
            self.auto_switch_to_real = config_manager.get_value('AutoSwitch', 'auto_switch_to_real', False, bool)

            # Configurações de salvamento automático
            self.autosave_interval = 1000  # Salvar a cada 1000 iterações
            self.last_save_time = time.time()
            
            # Configurações de logging
            self.logger = logging.getLogger('Inteligencia')
            # A configuração do logger (nível, handler, formato) deve ser feita centralmente (ex: em main.py)

        except Exception as e:
            # Logar o erro aqui também, pois o tracker pode não estar disponível se falhar no init
            tb_str = traceback.format_exc()
            logging.critical(f"Erro CRÍTICO na inicialização da Inteligência: {str(e)}\n{tb_str}")
            # Tentar registrar no tracker se ele foi inicializado antes da falha
            if hasattr(self, 'error_tracker') and self.error_tracker:
                 self.error_tracker.add_error("InteligenciaInitError", str(e), tb_str, critical=True)
            raise RuntimeError(f"Erro na inicialização da Inteligência: {str(e)}")
            
    def update_auto_switch_criteria(self, min_accuracy, min_precision, min_recall, min_f1_score, min_trades_count, min_win_rate, min_profit, auto_switch_to_real):
        """Atualiza os critérios de avaliação usados pela instância."""
        self.min_accuracy = min_accuracy
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_f1_score = min_f1_score
        self.min_trades_count = min_trades_count
        self.min_win_rate = min_win_rate
        self.min_profit = min_profit
        self.auto_switch_to_real = auto_switch_to_real
        self.logger.info("Critérios de avaliação atualizados na instância da Inteligencia.") # Traduzido
        self.logger.info(f"Acurácia >= {self.min_accuracy:.3f}, Precisão >= {self.min_precision:.3f}, Recall >= {self.min_recall:.3f}, F1 >= {self.min_f1_score:.3f}") # Traduzido

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
            
            self.logger.info(f"Modelo inicializado com {num_features} features e sequência de tamanho {sequence_length}")
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao inicializar modelo: {str(e)}\n{tb_str}")
            if self.error_tracker: self.error_tracker.add_error("InitializeModelError", str(e), tb_str)
            
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
                # Salva como best_model.pth por padrão, ou com timestamp se especificado
                filename = os.path.join(self.model_dir, "best_model.pth") 
                
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Salva o modelo
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
                'model_config': {
                    'input_size': self.model.input_size,
                'scaler': self.scaler, # Salva o scaler
                'feature_cols': self.feature_cols, # Salva as colunas usadas
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
            if self.error_tracker: self.error_tracker.add_error("SaveModelError", str(e), tb_str)
            return False
            
    def load_model(self, filename=None):
        """Carrega o modelo treinado.
        
        Args:
            filename: Nome do arquivo para carregar o modelo (default: best_model.pth)
            
        Returns:
            bool: True se o modelo foi carregado com sucesso
        """
        try:
            if filename is None:
                filename = os.path.join(self.model_dir, "best_model.pth") # Tenta carregar o melhor modelo por padrão

            if not os.path.exists(filename):
                 # Se best_model não existe, tenta o mais recente
                 model_files = glob.glob(os.path.join(self.model_dir, "model_*.pth"))
                 if not model_files:
                     self.logger.error("Nenhum modelo encontrado para carregar (nem best_model.pth, nem outros).")
                     return False
                 filename = max(model_files, key=os.path.getctime)
                 self.logger.info(f"best_model.pth não encontrado, carregando o mais recente: {filename}")

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
            
            # Carrega o otimizador se disponível e se self.optimizer já foi inicializado (ex: durante treinamento)
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                # Inicializa o otimizador ANTES de carregar o estado, se ainda não existir
                if not hasattr(self, 'optimizer') or self.optimizer is None:
                     # Usa o learning_rate padrão ou busca na config se disponível
                     lr = self.config_manager.get_value('Model', 'learning_rate', 0.001, float) if self.config_manager else 0.001
                     self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                     self.logger.info("Otimizador Adam inicializado para carregar estado.")
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.logger.info("Estado do otimizador carregado com sucesso.")
                except Exception as opt_load_e:
                    self.logger.error(f"Erro ao carregar estado do otimizador: {opt_load_e}. Otimizador pode não estar no estado correto.")
            
            # Carrega o scaler e as colunas se existirem no checkpoint
            self.scaler = checkpoint.get('scaler')
            self.feature_cols = checkpoint.get('feature_cols')
            if self.scaler and self.feature_cols:
                 self.logger.info(f"Scaler e {len(self.feature_cols)} colunas de features carregados do checkpoint.")
            else:
                 self.logger.warning("Scaler ou colunas de features não encontrados no checkpoint. A normalização pode falhar.")
                 
            return True
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao carregar modelo: {str(e)}\n{tb_str}")
            if self.error_tracker: self.error_tracker.add_error("LoadModelError", str(e), tb_str)
            return False
            
    def predict(self, data, confidence_threshold=0.75): # Limiar de confiança padrão aumentado
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
                # --- Normalização ---
                # Aplica o scaler ajustado aos dados de entrada
                if self.scaler is None or self.feature_cols is None:
                    self.logger.error("Scaler ou feature_cols não definidos. Não é possível normalizar dados para previsão.")
                    return None
                
                missing_cols_pred = [col for col in self.feature_cols if col not in data.columns]
                if missing_cols_pred:
                    self.logger.error(f"Colunas esperadas pelo scaler não encontradas nos dados de previsão: {missing_cols_pred}")
                    return None
                
                data_to_scale = data[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                data[self.feature_cols] = self.scaler.transform(data_to_scale)
                self.logger.info("Scaler aplicado aos dados de previsão.")
                # --- Fim Normalização ---

                # Prepara a sequência APÓS a normalização
                # Usa self.feature_cols para garantir consistência
                sequence = data[self.feature_cols].values # .fillna(0) não é mais necessário aqui se tratado antes
                sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            elif isinstance(data, (np.ndarray, torch.Tensor)):
                 # Se já for um tensor ou array
                 sequence = torch.FloatTensor(data).unsqueeze(0).to(self.device)
            else:
                 self.logger.error(f"Tipo de dado não suportado para previsão: {type(data)}")
                 return None

            # Verifica shape da sequência
            if sequence.shape[1] == 0 or sequence.shape[2] == 0:
                 self.logger.error(f"Sequência de entrada para previsão está vazia ou inválida. Shape: {sequence.shape}")
                 return None
            if sequence.shape[2] != self.model.input_size:
                 self.logger.error(f"Incompatibilidade no número de features da entrada ({sequence.shape[2]}) vs modelo ({self.model.input_size}).")
                 return None

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
            action = action_map.get(prediction, 'HOLD') # Default para HOLD se prediction for inesperado
            
            # Verifica confiança (garante que threshold é float)
            if confidence_value < float(confidence_threshold):
                action = 'HOLD'  # Se confiança baixa, não opera
                
            return {
                'prediction': prediction,
                'action': action,
                'confidence': confidence_value,
                'probabilities': probabilities.cpu().numpy()[0]
            }
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao fazer previsão: {str(e)}\n{tb_str}")
            if self.error_tracker: self.error_tracker.add_error("PredictError", str(e), tb_str)
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
        
        if val_loader is None or len(val_loader) == 0: # Adicionado check para val_loader None
             self.logger.warning("DataLoader de validação está vazio ou não foi criado.")
             return float('inf'), 0.0 # Retorna loss infinita e acurácia zero

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
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf') # Evita divisão por zero
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

    def train(self, train_data, val_data=None, epochs=100, learning_rate=0.001, patience=500, sequence_length=20, test_size=0.2, early_stopping=True):
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
                
            # --- Normalização ---
            # Define as colunas a serem normalizadas (excluindo timestamp, label, etc.)
            exclude_cols = ['timestamp', 'date', 'label', 'asset'] # Ajuste conforme necessário
            numeric_cols = train_data.select_dtypes(include=np.number).columns.tolist()
            self.feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not self.feature_cols:
                self.logger.error("Nenhuma coluna de feature encontrada para normalização no treino.")
                return None

            # Ajusta o scaler APENAS nos dados de treino
            self.scaler = MinMaxScaler()
            train_data_to_scale = train_data[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            train_data[self.feature_cols] = self.scaler.fit_transform(train_data_to_scale)
            self.logger.info(f"Scaler ajustado e aplicado aos dados de treino ({len(self.feature_cols)} colunas).")

            # Aplica o scaler ajustado aos dados de validação (se existirem)
            if val_data is not None and not val_data.empty:
                # Garante que as colunas existem e estão na ordem correta
                missing_cols_val = [col for col in self.feature_cols if col not in val_data.columns]
                if missing_cols_val:
                    self.logger.error(f"Colunas esperadas pelo scaler não encontradas nos dados de validação: {missing_cols_val}")
                    return None
                val_data_to_scale = val_data[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                val_data[self.feature_cols] = self.scaler.transform(val_data_to_scale)
                self.logger.info("Scaler aplicado aos dados de validação.")
            # --- Fim Normalização ---

            # Prepara as sequências APÓS a normalização
            X_train, y_train = self._prepare_sequences(train_data, sequence_length)
            if X_train is None or y_train is None or len(X_train) == 0:
                 self.logger.error("Falha ao preparar sequências de treinamento após normalização.")
                 return None

            X_val, y_val = None, None # Inicializa
            if val_data is not None and not val_data.empty:
                X_val, y_val = self._prepare_sequences(val_data, sequence_length)
                if X_val is None or y_val is None or len(X_val) == 0:
                     self.logger.warning("Falha ao preparar sequências de validação após normalização. Validação será pulada.")
                     X_val, y_val = None, None # Garante que não será usado
            else:
                 # Se não tiver validação explícita, usa 20% do treino (já normalizado)
                 split_idx = int(len(X_train) * 0.8)
                 if split_idx > 0 and len(X_train) - split_idx > 0: # Garante que ambos os conjuntos não sejam vazios
                     X_val, y_val = X_train[split_idx:], y_train[split_idx:]
                     X_train, y_train = X_train[:split_idx], y_train[:split_idx]
                 else:
                     self.logger.warning("Não foi possível criar conjunto de validação a partir do treino. Validação será pulada.")
                     X_val, y_val = None, None

            # Cria DataLoaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            val_loader = None
            if X_val is not None and y_val is not None:
                 val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
                 val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            
            # Inicializa o modelo se não existir
            if not hasattr(self, 'model') or self.model is None:
                self._initialize_model(X_train.shape[2], sequence_length)
                
            # Otimizador e função de perda
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()
            
            # Métricas de treinamento
            best_val_loss = float('inf')
            epochs_no_improve = 0
            
            # Loop de treinamento
            for epoch in range(epochs):
                # Modo de treinamento
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                # Loop pelos batches
                # Adicionado file=sys.stderr para tqdm
                for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stderr):
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
                    
                # Calcula métricas médias de treino
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
                train_accuracy = 100. * train_correct / train_total if train_total > 0 else 0.0
                
                # Validação (se houver val_loader)
                val_loss, val_accuracy = float('inf'), 0.0
                if val_loader:
                    val_loss, val_accuracy = self._validate(val_loader)
                
                # Registra métricas
                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_accuracy)
                self.val_accuracies.append(val_accuracy)
                
                # Log
                log_msg = f"Época {epoch+1}/{epochs} - Perda Treino: {avg_train_loss:.4f}, Acurácia Treino: {train_accuracy:.2f}%" # Traduzido
                if val_loader:
                     log_msg += f" - Perda Validação: {val_loss:.4f}, Acurácia Validação: {val_accuracy:.2f}%" # Traduzido
                self.logger.info(log_msg)
                
                # Salva o melhor modelo baseado na acurácia de validação (ou treino se não houver validação)
                current_best_metric = val_accuracy if val_loader else train_accuracy
                if current_best_metric > self.best_accuracy:
                    self.best_accuracy = current_best_metric
                    self.save_model(os.path.join(self.model_dir, "best_model.pth")) # Salva como best_model
                    epochs_no_improve = 0
                    self.logger.info(f"Novo melhor modelo salvo na época {epoch+1} com Acurácia {'Val' if val_loader else 'Treino'}: {self.best_accuracy:.2f}%") # Traduzido
                elif early_stopping:
                    epochs_no_improve += 1
                    
                # Early stopping
                if early_stopping and epochs_no_improve >= patience:
                    self.logger.info(f"Parada antecipada após {epoch+1} épocas sem melhoria na acurácia de validação.") # Traduzido
                    break
                    
                # Autosave a cada N épocas (salva com nome da época)
                if epoch % 5 == 0 and epoch > 0:
                    self.save_model(os.path.join(self.model_dir, f"model_epoch_{epoch+1}.pth"))
                    
                # Plota métricas a cada 10 épocas
                if epoch % 10 == 0 and epoch > 0:
                    self._plot_training_metrics()
                    
            # Salva o modelo final (última época)
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
            self.logger.error(f"Erro durante o treinamento: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("TrainError", str(e), tb_str)
            return None
            
    def _prepare_sequences(self, data, sequence_length):
        """Prepara sequências para treinamento LSTM.
        
        Args:
            data: DataFrame com dados
            sequence_length: Tamanho da sequência
            
        Returns:
            tuple: (X, y) arrays para treinamento
        """
        # Seleciona features (apenas numéricas) e labels
        # Exclui colunas não numéricas e o próprio label
        exclude_cols = ['timestamp', 'date', 'label', 'asset'] # Adiciona 'asset' e outras não numéricas se houver
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if not feature_cols:
            self.logger.error("Nenhuma coluna de feature numérica encontrada após a filtragem.")
            return None, None # Retorna None se não houver features

        # Garante que X seja float e lida com possíveis NaNs restantes (embora preprocess deve ter tratado)
        X = data[feature_cols].fillna(0).values.astype(np.float32)
        y = data['label'].values if 'label' in data.columns else np.zeros(len(data))
        
        # Cria sequências
        X_sequences = []
        y_sequences = []
        
        if len(X) <= sequence_length:
             self.logger.error(f"Dados insuficientes ({len(X)} linhas) para criar sequências de tamanho {sequence_length}.")
             return None, None

        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length])
            
        # Garante que os arrays retornados sejam do tipo correto
        return np.array(X_sequences).astype(np.float32), np.array(y_sequences).astype(np.int64)
        
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
            plt.plot(self.train_losses, label='Perda Treino') # Traduzido
            plt.plot(self.val_losses, label='Perda Validação') # Traduzido
            plt.title('Perda de Treinamento e Validação') # Traduzido
            plt.xlabel('Época') # Traduzido
            plt.ylabel('Perda') # Traduzido
            plt.legend()
            
            # Plot de acurácia
            plt.subplot(2, 1, 2)
            plt.plot(self.train_accuracies, label='Acurácia Treino') # Traduzido
            plt.plot(self.val_accuracies, label='Acurácia Validação') # Traduzido
            plt.title('Acurácia de Treinamento e Validação') # Traduzido
            plt.xlabel('Época') # Traduzido
            plt.ylabel('Acurácia (%)') # Traduzido
            plt.legend()
            
            # Salva a figura
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualization_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            plt.close()
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao plotar métricas: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("PlotMetricsError", str(e), tb_str)
            
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
                
            # --- Normalização ---
            # Aplica o scaler ajustado aos dados de teste
            if self.scaler is None or self.feature_cols is None:
                self.logger.error("Scaler ou feature_cols não definidos. Não é possível normalizar dados de teste.")
                return None
            
            missing_cols_test = [col for col in self.feature_cols if col not in test_data.columns]
            if missing_cols_test:
                self.logger.error(f"Colunas esperadas pelo scaler não encontradas nos dados de teste: {missing_cols_test}")
                return None
            
            test_data_to_scale = test_data[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            test_data[self.feature_cols] = self.scaler.transform(test_data_to_scale)
            self.logger.info("Scaler aplicado aos dados de teste.")
            # --- Fim Normalização ---

            # Prepara as sequências APÓS a normalização
            X_test, y_test = self._prepare_sequences(test_data, sequence_length)
            if X_test is None or y_test is None or len(X_test) == 0:
                self.logger.error("Falha ao preparar dados de teste para avaliação após normalização.")
                return None
                
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            
            # Avaliação
            self.model.eval()
            all_preds = []
            all_labels = []
            
            if len(test_loader) == 0:
                 self.logger.warning("DataLoader de teste está vazio. Não é possível avaliar.")
                 return None

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
            
            self.logger.info(f"Avaliação do modelo - Acurácia: {accuracy:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}") # Traduzido
            self._plot_confusion_matrix(cm)
            
            # Armazena as métricas para decisão de mudança de modo
            self.last_evaluation_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1, # Mantido como 'f1' para consistência com o código anterior, embora a variável seja f1_score
                'confusion_matrix': cm.tolist() # Convertendo para lista para serialização JSON
            }
            
            return self.last_evaluation_metrics
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro durante a avaliação: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("EvaluateModelError", str(e), tb_str)
            self.last_evaluation_metrics = None # Reseta em caso de erro
            return None

    def should_switch_to_test_mode(self):
        """Verifica se o desempenho do modelo atende aos critérios (possivelmente relaxados) para passar para o modo de teste."""
        # Usa os critérios armazenados na instância (que podem ter sido relaxados)
        min_accuracy = self.min_accuracy
        min_precision = self.min_precision
        min_recall = self.min_recall
        min_f1_score = self.min_f1_score
        
        if not hasattr(self, 'last_evaluation_metrics') or self.last_evaluation_metrics is None:
            self.logger.warning("Métricas de avaliação não disponíveis para decidir mudança para modo Teste.") # Traduzido
            return False

        metrics = self.last_evaluation_metrics
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0) # Usa a chave 'f1' que foi salva

        # Verifica se todas as métricas de avaliação do modelo atingem o mínimo
        meets_criteria = (
            accuracy >= self.min_accuracy and
            precision >= self.min_precision and
            recall >= self.min_recall and
            f1 >= self.min_f1_score
        )
        self.logger.info(f"Verificando critérios para modo teste: Acurácia ({accuracy:.4f} >= {self.min_accuracy:.3f}), Precisão ({precision:.4f} >= {self.min_precision:.3f}), Recall ({recall:.4f} >= {self.min_recall:.3f}), F1 ({f1:.4f} >= {self.min_f1_score:.3f}) -> {meets_criteria}") # Traduzido
        return meets_criteria

    def should_switch_to_real_mode(self, performance_tracker) -> bool:
        """Verifica se o desempenho no modo Teste atende aos critérios para mudar para o modo Real.
        
        Args:
            performance_tracker: Objeto contendo métricas de desempenho das operações.
            
        Returns:
            bool: True se deve mudar para Real, False caso contrário.
        """
        if not self.auto_switch_to_real:
            self.logger.info("Mudança automática para modo Real está desabilitada nas configurações.")
            return False

        if performance_tracker is None:
            self.logger.warning("Performance tracker não disponível para decidir mudança para modo Real.")
            return False # Alterado para False, pois não há dados para justificar a mudança

        # Acessa as métricas do dicionário 'metrics' do PerformanceTracker
        try:
            # Acessa as métricas do dicionário 'metrics' do PerformanceTracker
            win_rate = performance_tracker.metrics.get('win_rate', 0.0)
            profit = performance_tracker.metrics.get('total_profit', 0.0)
            trades_count = performance_tracker.metrics.get('total_trades', 0)
        except AttributeError as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao acessar métricas do PerformanceTracker: {e}. Verifique os métodos/atributos.\n{tb_str}")
            if self.error_tracker: self.error_tracker.add_error("PerformanceTrackerAccessError", str(e), tb_str)
            return False # Não pode mudar para Real se não conseguir ler as métricas

        self.logger.info(f"Avaliando mudança para modo Real: Taxa de Acerto={win_rate:.2f} (Min: {self.min_win_rate:.2f}), Lucro={profit:.2f} (Min: {self.min_profit:.2f}), Operações={trades_count} (Min: {self.min_trades_count})") # Traduzido

        # Verifica se os critérios mínimos ainda são atendidos
        meets_criteria = (
            trades_count >= self.min_trades_count and
            win_rate >= self.min_win_rate and
            profit >= self.min_profit
        )

        if meets_criteria:
            self.logger.info("Critérios de desempenho em Teste atendidos. Recomendando mudança para modo Real.") # Traduzido
            return True
        else:
            self.logger.info("Critérios de desempenho em Teste NÃO atendidos. Mantendo modo Teste.") # Traduzido
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
            return True # Default: permanece em Real se não puder avaliar

        try:
            # Acessa as métricas do dicionário 'metrics' do PerformanceTracker
            win_rate = performance_tracker.metrics.get('win_rate', 0.0)
            profit = performance_tracker.metrics.get('total_profit', 0.0)
            trades_count = performance_tracker.metrics.get('total_trades', 0)
        except AttributeError as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao acessar métricas do PerformanceTracker: {e}. Verifique os métodos/atributos.\n{tb_str}")
            if self.error_tracker: self.error_tracker.add_error("PerformanceTrackerAccessError", str(e), tb_str)
            return True # Default: permanece em Real se não puder avaliar

        self.logger.info(f"Avaliando permanência no modo Real: Taxa de Acerto={win_rate:.2f} (Min: {self.min_win_rate:.2f}), Lucro={profit:.2f} (Min: {self.min_profit:.2f}), Operações={trades_count} (Min: {self.min_trades_count})") # Traduzido

        # Verifica se os critérios mínimos ainda são atendidos
        # Usamos os mesmos critérios de entrada, mas poderiam ser diferentes (ex: um limiar de 'degradação')
        meets_criteria = (
            trades_count >= self.min_trades_count and # Garante que há dados suficientes
            win_rate >= self.min_win_rate and
            profit >= self.min_profit
        )

        if meets_criteria:
            self.logger.info("Desempenho em modo Real continua satisfatório. Permanecendo em modo Real.") # Traduzido
            return True
        else:
            self.logger.warning("Desempenho em modo Real caiu abaixo dos critérios mínimos. Recomendando retorno para modo Teste.") # Traduzido
            return False


    def _plot_confusion_matrix(self, cm):
        """Plota matriz de confusão.
        
        Args:
            cm: Matriz de confusão
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matriz de Confusão') # Traduzido
            plt.ylabel('Rótulo Verdadeiro') # Traduzido
            plt.xlabel('Rótulo Previsto') # Traduzido
            plt.savefig(os.path.join(self.visualization_dir, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            plt.close()
        except Exception as e:
            self.logger.error(f"Erro ao plotar matriz de confusão: {str(e)}") # Traduzido
            
    # A função preprocess_data foi removida para unificar a lógica em process_historical_data.
    # As chamadas a preprocess_data devem ser substituídas por process_historical_data(..., save_processed=False).
            
    def _add_technical_indicators(self, df):
        """Adiciona indicadores técnicos ao DataFrame."""
        try:
            df = df.copy() # Garante que estamos trabalhando em uma cópia
            
            # Verifica se existem as colunas necessárias
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Colunas necessárias ausentes para indicadores: {required_cols}") # Traduzido
                return None # Retorna None se faltar colunas essenciais

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
                # Verifica se o retorno é DataFrame ou tupla e extrai os dados
                if isinstance(macd, pd.DataFrame) and not macd.empty:
                    df['macd'] = macd.get('MACD_12_26_9') # Usar .get para segurança
                    df['macd_signal'] = macd.get('MACDs_12_26_9')
                    df['macd_hist'] = macd.get('MACDh_12_26_9')
                elif isinstance(macd, tuple) and len(macd) >= 3: # Verifica se é tupla
                    df['macd'] = macd[0]
                    df['macd_signal'] = macd[1]
                    df['macd_hist'] = macd[2]
                else:
                    self.logger.warning("Retorno inesperado ou vazio da função MACD.")
            
            # Bollinger Bands
            if len(df) >= 20:
                bbands = ta.bbands(df['close'], length=20, std=2)
                # Verifica se o retorno é DataFrame ou tupla
                if isinstance(bbands, pd.DataFrame) and not bbands.empty:
                    df['bb_upper'] = bbands.get('BBU_20_2.0')
                    df['bb_middle'] = bbands.get('BBM_20_2.0')
                    df['bb_lower'] = bbands.get('BBL_20_2.0')
                    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                         df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan) 
                elif isinstance(bbands, tuple) and len(bbands) >= 3: # Verifica se é tupla
                    df['bb_upper'] = bbands[0]
                    df['bb_middle'] = bbands[1]
                    df['bb_lower'] = bbands[2]
                    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
                else:
                    self.logger.warning("Retorno inesperado ou vazio da função BBANDS.")
            
            # ATR
            if len(df) >= 14:
                df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Stochastic Oscillator
            if len(df) >= 14:
                stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
                # Verifica se o retorno é DataFrame
                if isinstance(stoch, pd.DataFrame) and not stoch.empty:
                    df['stoch_k'] = stoch.get('STOCHk_14_3_3')
                    df['stoch_d'] = stoch.get('STOCHd_14_3_3')
                # --- CORREÇÃO DE INDENTAÇÃO ---
                else: 
                     self.logger.warning("Retorno inesperado ou vazio da função STOCH.")
            
            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # ADX
            if len(df) >= 14:
                adx = ta.adx(df['high'], df['low'], df['close'], length=14)
                if isinstance(adx, pd.DataFrame) and not adx.empty:
                    df['adx_14'] = adx.get('ADX_14')
                else:
                     self.logger.warning("Retorno inesperado ou vazio da função ADX.")
            
            # Momentum
            if len(df) >= 10:
                df['mom_10'] = ta.mom(df['close'], length=10)
            
            # Rate of Change
            if len(df) >= 10:
                df['roc_10'] = ta.roc(df['close'], length=10)
            
            # Williams %R
            if len(df) >= 14:
                df['willr_14'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            
            # CCI
            if len(df) >= 20:
                df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # Ichimoku Cloud
            ichimoku_result = None # Inicializa a variável
            self.logger.debug(f"[DEBUG Ichimoku] Verificando tamanho do DF antes do cálculo: {len(df)} linhas")
            if len(df) >= 52: 
                try:
                    self.logger.debug(f"Calculando Ichimoku. Shape do DF de entrada: {df.shape}")
                    if not df.index.is_monotonic_increasing:
                        df = df.sort_index()
                    ichimoku_result = ta.ichimoku(df['high'], df['low'], df['close'])
                    # Log para inspecionar o resultado IMEDIATAMENTE após a chamada
                    self.logger.debug(f"Resultado bruto de ta.ichimoku: Tipo={type(ichimoku_result)}, Comprimento={len(ichimoku_result) if hasattr(ichimoku_result, '__len__') else 'N/A'}")
                    if isinstance(ichimoku_result, pd.DataFrame) and not ichimoku_result.empty and len(ichimoku_result.columns) >= 4:
                        df['ichimoku_a'] = ichimoku_result.iloc[:, 0]
                        df['ichimoku_b'] = ichimoku_result.iloc[:, 1]
                        df['ichimoku_tenkan'] = ichimoku_result.iloc[:, 2]
                        df['ichimoku_kijun'] = ichimoku_result.iloc[:, 3]
                        self.logger.debug("Ichimoku calculado com sucesso (pandas_ta).")
                    elif hasattr(ichimoku_result, '__iter__') and not isinstance(ichimoku_result, (pd.DataFrame, str, bytes)): # Verifica se é iterável (como tupla) mas não DataFrame/str/bytes
                         self.logger.debug(f"Ichimoku retornou tupla com {len(ichimoku_result)} elementos.")
                         # Loga o tipo do primeiro elemento para inspeção
                         if len(ichimoku_result) > 0:
                              self.logger.debug(f"Tipo do primeiro elemento da tupla Ichimoku: {type(ichimoku_result[0])}")

                         if len(ichimoku_result) >= 4: # Agora verifica o comprimento
                              try:
                                   # Atribui assumindo a ordem padrão: SPAN A, SPAN B, TENKAN, KIJUN
                                   df['ichimoku_a'] = ichimoku_result[0]
                                   df['ichimoku_b'] = ichimoku_result[1]
                                   df['ichimoku_tenkan'] = ichimoku_result[2]
                                   df['ichimoku_kijun'] = ichimoku_result[3]
                                   # Opcionalmente, atribui Chikou Span se presente (geralmente 5º elemento)
                                   # if len(ichimoku_result) >= 5:
                                   #     df['ichimoku_chikou'] = ichimoku_result[4]
                                   self.logger.debug("Ichimoku atribuído com sucesso a partir da tupla.")
                              except Exception as e_assign:
                                   self.logger.error(f"Erro ao atribuir valores Ichimoku da tupla: {e_assign}")
                                   # Define como NaN se a atribuição falhar
                                   df['ichimoku_a'] = np.nan
                                   df['ichimoku_b'] = np.nan
                                   df['ichimoku_tenkan'] = np.nan
                                   df['ichimoku_kijun'] = np.nan
                         else:
                              # Este caso significa que é uma tupla, mas muito curta
                              self.logger.warning(f"Tupla Ichimoku retornada tem menos de 4 elementos ({len(ichimoku_result)}). Não foi possível atribuir.")
                              df['ichimoku_a'] = np.nan
                              df['ichimoku_b'] = np.nan
                              df['ichimoku_tenkan'] = np.nan
                              df['ichimoku_kijun'] = np.nan
                    else:
                        self.logger.warning(f"Retorno inesperado ou vazio da função ta.ichimoku. Tipo: {type(ichimoku_result)}. Conteúdo: {str(ichimoku_result)[:200]}...")
                        df['ichimoku_a'] = np.nan
                        df['ichimoku_b'] = np.nan
                        df['ichimoku_tenkan'] = np.nan
                        df['ichimoku_kijun'] = np.nan
                except Exception as e_ichi:
                    tb_str = traceback.format_exc()
                    self.logger.error(f"Erro ao calcular Ichimoku Cloud: {e_ichi}\n{tb_str}")
                    if self.error_tracker: self.error_tracker.add_error("IchimokuError", str(e_ichi), tb_str)
                    df['ichimoku_a'] = np.nan
                    df['ichimoku_b'] = np.nan
                    df['ichimoku_tenkan'] = np.nan
                    df['ichimoku_kijun'] = np.nan
            else:
                self.logger.warning(f"Dados insuficientes ({len(df)} linhas) para calcular Ichimoku Cloud (necessário >= 52).")
                df['ichimoku_a'] = np.nan
                df['ichimoku_b'] = np.nan
                df['ichimoku_tenkan'] = np.nan
                df['ichimoku_kijun'] = np.nan
                    
            # Preenchimento de NaN (após todos os indicadores)
            # Log para verificar NaNs antes do preenchimento final
            nan_counts_before_fill = df.isnull().sum()
            nans_present = nan_counts_before_fill[nan_counts_before_fill > 0]
            if not nans_present.empty:
                self.logger.warning(f"NaNs presentes ANTES do preenchimento final em _add_technical_indicators:\n{nans_present.to_string()}")
            else:
                self.logger.debug("Nenhum NaN encontrado antes do preenchimento final em _add_technical_indicators.")

            df = df.bfill().ffill().fillna(0) 
            
            return df
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao adicionar indicadores técnicos: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("AddTechnicalIndicatorsError", str(e), tb_str)
            return None # Retorna None em caso de erro
            
    def _add_candle_patterns(self, df):
        """Adiciona padrões de candles ao DataFrame."""
        try:
            # Verifica se existem as colunas necessárias
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Colunas necessárias ausentes para padrões de candle: {required_cols}") # Traduzido
                return None # Retorna None se faltar colunas essenciais
                
            df = df.copy()
            
            # Extrai arrays para cálculos
            open_prices = df['open'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Doji
            # Evita divisão por zero se high == low
            range_hl = df['high'] - df['low']
            df['doji'] = np.where( (range_hl > 0) & (np.abs(df['open'] - df['close']) / range_hl < 0.1), 1, 0)
            
            # Hammer
            df['hammer'] = 0
            body_size = np.abs(open_prices - close)
            lower_shadow = np.minimum(open_prices, close) - low
            upper_shadow = high - np.maximum(open_prices, close)
            
            # Vetorização para performance
            is_hammer = (body_size > 0) & (lower_shadow >= 2 * body_size) & (upper_shadow <= 0.2 * body_size)
            df.loc[is_hammer, 'hammer'] = 1
                    
            # Shooting Star
            df['shooting_star'] = 0
            is_shooting_star = (body_size > 0) & (upper_shadow >= 2 * body_size) & (lower_shadow <= 0.2 * body_size)
            df.loc[is_shooting_star, 'shooting_star'] = 1
                    
            # Engulfing patterns (requer shift, mais difícil de vetorizar completamente)
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
            self.logger.error(f"Erro ao adicionar padrões de candles: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("AddCandlePatternsError", str(e), tb_str)
            return None # Retorna None em caso de erro
            
    def _add_derived_features(self, df):
        """Adiciona features derivadas ao DataFrame."""
        try:
            df = df.copy()
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
            
            # Distância das médias móveis (com verificação e tratamento de divisão por zero)
            if 'sma_20' in df.columns:
                df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20'].replace(0, np.nan)
            if 'sma_50' in df.columns:
                df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50'].replace(0, np.nan)
            if 'sma_200' in df.columns:
                df['dist_sma_200'] = (df['close'] - df['sma_200']) / df['sma_200'].replace(0, np.nan)
                
            # Cruzamentos de médias móveis
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
                
            # Relação entre volume e preço
            df['volume_price_ratio'] = df['volume'] / df['close'].replace(0, np.nan)
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma_10'].replace(0, np.nan)
            
            # Preenchimento de NaN (após todos os indicadores)
            # Log para verificar NaNs antes do preenchimento final
            nan_counts_before_fill = df.isnull().sum()
            nans_present = nan_counts_before_fill[nan_counts_before_fill > 0]
            if not nans_present.empty:
                self.logger.warning(f"NaNs presentes ANTES do preenchimento final em _add_derived_features:\n{nans_present.to_string()}")
            else:
                self.logger.debug("Nenhum NaN encontrado antes do preenchimento final em _add_derived_features.")

            df = df.bfill().ffill().fillna(0) 
            
            return df
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao adicionar features derivadas: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("AddDerivedFeaturesError", str(e), tb_str)
            return None # Retorna None em caso de erro
            
    def create_labels(self, df, n_periods=5, threshold=0.001):
        """Cria rótulos (0: HOLD, 1: BUY, 2: SELL) com base na variação futura do preço.

        Args:
            df (pd.DataFrame): DataFrame com dados OHLCV e índice de timestamp.
            n_periods (int): Número de períodos à frente para calcular a variação.
            threshold (float): Limiar percentual para definir BUY/SELL (ex: 0.001 para 0.1%).

        Returns:
            pd.DataFrame: DataFrame com a coluna 'label' adicionada ou None em erro.
        """
        try:
            self.logger.info(f"Criando rótulos com horizonte de {n_periods} períodos e limiar de {threshold*100:.3f}%")
            df = df.copy()

            if 'close' not in df.columns:
                self.logger.error("Coluna 'close' não encontrada para criar rótulos.")
                return None

            # Calcula o preço futuro
            df['future_close'] = df['close'].shift(-n_periods)

            # Calcula a variação percentual futura
            df['future_pct_change'] = (df['future_close'] - df['close']) / df['close']

            # Define as condições para BUY e SELL
            buy_condition = df['future_pct_change'] > threshold
            sell_condition = df['future_pct_change'] < -threshold

            # Atribui os rótulos: 1 para BUY, 2 para SELL, 0 para HOLD
            df['label'] = 0 # Default HOLD
            df.loc[buy_condition, 'label'] = 1
            df.loc[sell_condition, 'label'] = 2

            # Remove colunas auxiliares e NaNs introduzidos pelo shift
            # Os NaNs serão tratados posteriormente em process_historical_data
            df = df.drop(columns=['future_close', 'future_pct_change'])

            self.logger.info("Rótulos criados com sucesso.")
            return df

        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao criar rótulos: {str(e)}\n{tb_str}")
            if self.error_tracker: self.error_tracker.add_error("CreateLabelsError", str(e), tb_str)
            return None


    def process_historical_data(self, data, asset_name, timeframe=60, save_processed=True):
        """Processa dados históricos para treinamento ou previsão.
        
        Args:
            data (pd.DataFrame): DataFrame com dados históricos (OHLCV)
            asset_name (str): Nome do ativo (ex: "EURUSD")
            timeframe (int): Timeframe em segundos (60, 300, etc.)
            save_processed (bool): Se deve salvar os dados processados
            
        Returns:
            pd.DataFrame: Dados pré-processados ou None em caso de erro.
        """
        try:
            self.logger.info(f"Iniciando pré-processamento para {asset_name}. Shape entrada: {data.shape}") # Traduzido
            df = data.copy()
            
            self.logger.debug(f"[DEBUG process_historical_data] Shape inicial: {df.shape}, Colunas: {df.columns.tolist()}")
            # Verifica colunas essenciais
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Colunas necessárias ausentes para pré-processamento: {required_cols}") # Traduzido
                return None

            # Garante que o índice seja datetime (importante para TA-Lib/pandas_ta)
            self.logger.debug("[DEBUG process_historical_data] Verificando e tratando timestamp...")
            if 'timestamp' in df.columns:
                 try:
                     self.logger.debug("[DEBUG process_historical_data] Tentando converter 'timestamp' (segundos)...")
                     # Tenta converter de segundos primeiro, depois milissegundos
                     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                     self.logger.debug(f"[DEBUG process_historical_data] Após to_datetime (s): NaNs={df['timestamp'].isnull().sum()}")
                     if df['timestamp'].isnull().all(): # Se falhou com segundos, tenta ms
                          self.logger.debug("[DEBUG process_historical_data] Falhou com segundos, tentando converter 'timestamp' (ms)...")
                          df['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', errors='coerce') # Usa 'data' original
                          self.logger.debug(f"[DEBUG process_historical_data] Após to_datetime (ms): NaNs={df['timestamp'].isnull().sum()}")
                     
                     self.logger.debug("[DEBUG process_historical_data] Tentando set_index e sort_index...")
                     df = df.set_index('timestamp').sort_index()
                     self.logger.debug("[DEBUG process_historical_data] set_index e sort_index concluídos.")
                 except Exception as e_time:
                     self.logger.error(f"Erro ao converter ou definir índice 'timestamp': {e_time}")
                     self.logger.debug(traceback.format_exc()) # Adiciona traceback para depuração
                     return None
                 self.logger.debug(f"[DEBUG process_historical_data] Índice após conversão de timestamp: {df.index}")
            elif isinstance(df.index, pd.DatetimeIndex):
                 self.logger.debug("Usando índice DatetimeIndex existente.")
                 if not df.index.is_monotonic_increasing:
                      self.logger.debug("[DEBUG process_historical_data] Índice não monotônico, ordenando...")
                      df = df.sort_index() # Garante ordenação
                      self.logger.debug("[DEBUG process_historical_data] sort_index concluído.")
            else:
                 self.logger.error("Coluna 'timestamp' ou índice datetime 'timestamp' não encontrado ou inválido.")
                 self.logger.debug(f"[DEBUG process_historical_data] Tipo do índice atual: {type(df.index)}")
                 # self.logger.debug(f"[DEBUG process_historical_data] Shape após sort: {df.shape}, Colunas: {df.columns.tolist()}") # Log movido ou removido se redundante
                 self.logger.error("[DEBUG process_historical_data] Falha ao definir índice datetime.") # Log de erro mantido
                 return None
            
            # Adiciona indicadores técnicos
            self.logger.debug(f"[DEBUG process_historical_data] Shape ANTES de chamar _add_technical_indicators: {df.shape}")
            df = self._add_technical_indicators(df)
            if df is None:
                 self.logger.error("Falha ao adicionar indicadores técnicos.")
                 # O log de debug aqui não faz sentido se df é None
                 return None
            else: # Adiciona o log no else
                 self.logger.debug(f"[DEBUG process_historical_data] Shape após indicadores: {df.shape}, NaNs: {df.isnull().sum().sum()}")

            # Adiciona padrões de candles
            self.logger.debug(f"[DEBUG process_historical_data] Shape antes de padrões candle: {df.shape}, NaNs: {df.isnull().sum().sum()}")
            df = self._add_candle_patterns(df)
            if df is None:
                 self.logger.error("Falha ao adicionar padrões de candle.")
                 # O log de debug aqui não faz sentido se df é None
                 return None
            else: # Adiciona o log no else
                 self.logger.debug(f"[DEBUG process_historical_data] Shape após padrões candle: {df.shape}, NaNs: {df.isnull().sum().sum()}")
            
            # Adiciona features derivadas
            self.logger.debug(f"[DEBUG process_historical_data] Shape antes de features derivadas: {df.shape}, NaNs: {df.isnull().sum().sum()}")
            df = self._add_derived_features(df)
            if df is None:
                 self.logger.error("Falha ao adicionar features derivadas.")
                 # O log de debug aqui não faz sentido se df é None
                 return None
            else: # Adiciona o log no else
                 self.logger.debug(f"[DEBUG process_historical_data] Shape após features derivadas: {df.shape}, NaNs: {df.isnull().sum().sum()}")
            
            # Normaliza os dados (APENAS se não for para previsão em tempo real, ou usar scaler salvo)
            # TODO: Implementar salvamento/carregamento do scaler ou normalizar após divisão treino/teste/validação.
            self.logger.debug(f"[DEBUG process_historical_data] Shape antes de normalizar: {df.shape}, NaNs: {df.isnull().sum().sum()}")
            # A normalização foi movida para as funções train/evaluate/predict
            # df = self._normalize_data(df) # REMOVIDO
            # if df is None:
            #      self.logger.error("Falha ao normalizar dados.")
            #      # O log de debug aqui não faz sentido se df é None
            #      return None
            # else: # Adiciona o log no else
            #      self.logger.debug(f"[DEBUG process_historical_data] Shape após normalizar: {df.shape}, NaNs: {df.isnull().sum().sum()}")

            # Cria rótulos para treinamento supervisionado (APENAS se 'label' não existir e estiver no modo learning)
            if 'label' not in df.columns and self.mode == "LEARNING": # Adicionado check de modo
                self.logger.debug(f"[DEBUG process_historical_data] Shape antes de criar labels: {df.shape}, NaNs: {df.isnull().sum().sum()}")
                df = self.create_labels(df)
                if df is None:
                     self.logger.error("Falha ao criar rótulos.")
                     # O log de debug aqui não faz sentido se df é None
                     return None
                else: # Adiciona o log no else
                     self.logger.debug(f"[DEBUG process_historical_data] Shape após criar labels: {df.shape}, NaNs: {df.isnull().sum().sum()}")

            # Remove linhas com NaN após todas as adições
            # A linha abaixo estava com indentação incorreta
            initial_rows = len(df)
            self.logger.debug(f"[DEBUG process_historical_data] Shape antes de dropna: {df.shape}, NaNs: {df.isnull().sum().sum()}")
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                 self.logger.warning(f"{dropped_rows} linhas removidas devido a NaNs após pré-processamento.") # Traduzido
            # Log movido para fora do if, para sempre mostrar o shape após dropna
            self.logger.debug(f"[DEBUG process_historical_data] Shape após dropna: {df.shape}")
            
            if df.empty:
                 self.logger.error("DataFrame ficou vazio após pré-processamento e remoção de NaNs.")
                 return None

            # 4. Salvamento (opcional, geralmente só no modo download/learning inicial)
            if save_processed and self.mode != 'test' and self.mode != 'real': # Evita salvar em teste/real
                # Cria diretório para dados processados se não existir
                processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
                os.makedirs(processed_dir, exist_ok=True)
                
                # Nome do arquivo baseado no ativo e timeframe
                filename = f"{asset_name.lower()}_{timeframe}_processed.csv"
                filepath = os.path.join(processed_dir, filename)
                
                # Salva os dados processados
                df.to_csv(filepath) # Salva com índice (timestamp)
                self.logger.info(f"Dados processados salvos em {filepath}") # Traduzido
            
            self.logger.info(f"Pré-processamento concluído. Shape final: {df.shape}") # Traduzido
            self.logger.debug(f"[DEBUG process_historical_data] Shape final antes de retornar: {df.shape}")
            return df.reset_index() # Retorna com timestamp como coluna
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro geral no pré-processamento: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("PreprocessDataError", str(e), tb_str)
            return None
            
    def setup_test_mode(self):
        """Configura a inteligência para o modo de teste."""
        try:
            self.logger.info("Configurando inteligência para modo de teste") # Traduzido
            
            # Define o modo de operação
            self.mode = 'test'
            
            # Carrega o modelo treinado (best_model.pth por padrão)
            if self.load_model(): # load_model agora tenta best_model.pth por padrão
                self.logger.info("Modelo treinado carregado com sucesso para o modo de teste.") # Traduzido
            else:
                self.logger.warning("Modelo treinado não encontrado ou falha ao carregar. Criando modelo básico para testes.") # Traduzido
                self._create_basic_test_model() # Cria um modelo básico como fallback
            
            # Configura parâmetros de teste (resetar contadores)
            self.test_win_rate = 0.0
            self.test_trades_count = 0
            self.test_profit = 0.0
            
            self.logger.info("Modo de teste configurado com sucesso") # Traduzido
            return True
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao configurar modo de teste: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("SetupTestModeError", str(e), tb_str)
            return False
    
    def _create_basic_test_model(self):
        """Cria um modelo básico para testes quando não há modelo treinado disponível."""
        try:
            self.logger.info("Criando modelo básico para testes") # Traduzido
            
            # Instancia um LSTMModel com parâmetros padrão como fallback
            # Tenta inferir input_size de dados processados, se disponíveis
            default_input_size = 50 # Valor fallback
            processed_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "*_processed.csv"))
            if processed_files:
                 try:
                      # Carrega um arquivo processado para inferir o número de features
                      temp_df = pd.read_csv(processed_files[0])
                      exclude_cols = ['timestamp', 'date', 'label', 'asset'] 
                      numeric_cols = temp_df.select_dtypes(include=np.number).columns.tolist()
                      feature_cols = [col for col in numeric_cols if col not in exclude_cols]
                      if feature_cols:
                           default_input_size = len(feature_cols)
                           self.logger.info(f"Input size inferido dos dados processados: {default_input_size}")
                 except Exception as e_infer:
                      self.logger.warning(f"Não foi possível inferir input_size dos dados processados: {e_infer}. Usando fallback {default_input_size}.")

            self.model = LSTMModel(
                input_size=default_input_size,
                hidden_size=128,
                num_layers=2,
                output_size=3,  # 3 classes: hold, buy, sell
                dropout=0.2
            ).to(self.device)
            self.logger.warning(f"Modelo LSTM básico criado com input_size={default_input_size} (fallback).") # Traduzido
            
            self.logger.info("Modelo básico criado com sucesso") # Traduzido
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao criar modelo básico: {str(e)}\n{tb_str}") # Traduzido
            if self.error_tracker: self.error_tracker.add_error("CreateBasicModelError", str(e), tb_str)
