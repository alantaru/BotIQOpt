import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas_ta as ta
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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
    def __init__(self, config_manager=None, model_path="hybrid_model.pth", device="cuda" if torch.cuda.is_available() else "cpu", historical_data_filename="historical_data.csv"):
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
            
            # Configurações de operação
            self.mode = "LEARNING"
            self.visualization_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_visualizations")
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
            
            self.logger.info(f"Critérios de mudança automática atualizados: "
                           f"min_accuracy={self.min_accuracy:.2f}, "
                           f"min_precision={self.min_precision:.2f}, "
                           f"min_recall={self.min_recall:.2f}, "
                           f"min_f1_score={self.min_f1_score:.2f}, "
                           f"min_trades_count={self.min_trades_count}, "
                           f"min_win_rate={self.min_win_rate:.2f}, "
                           f"min_profit={self.min_profit:.2f}, "
                           f"auto_switch_to_real={self.auto_switch_to_real}")
                           
        except Exception as e:
            self.logger.error(f"Erro ao atualizar critérios de mudança automática: {str(e)}")
            
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
            self.logger.error(f"Erro ao inicializar modelo: {str(e)}")
            
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
            
            # Salva o modelo
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
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
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
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
            
            # Carrega o otimizador se disponível
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                self.optimizer = optim.Adam(self.model.parameters())
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            self.logger.info(f"Modelo carregado de {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
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
                # Se for DataFrame, converte para tensor
                feature_cols = [col for col in data.columns if col not in ['timestamp', 'date', 'label']]
                sequence = data[feature_cols].values
                sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
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
            
            # Verifica confiança
            if confidence_value < confidence_threshold:
                action = 'HOLD'  # Se confiança baixa, não opera
                
            return {
                'prediction': prediction,
                'action': action,
                'confidence': confidence_value,
                'probabilities': probabilities.cpu().numpy()[0]
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao fazer previsão: {str(e)}")
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

    def train(self, train_data, val_data=None, epochs=100, learning_rate=0.001, patience=10, sequence_length=20, test_size=0.2, early_stopping=True):
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
                
            # Pré-processa os dados
            X_train, y_train = self._prepare_sequences(train_data, sequence_length)
            
            if val_data is not None:
                X_val, y_val = self._prepare_sequences(val_data, sequence_length)
            else:
                # Se não tiver validação, usa 20% do treino
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
                for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
                avg_train_loss = train_loss / len(train_loader)
                train_accuracy = 100. * train_correct / train_total
                
                # Validação
                val_loss, val_accuracy = self._validate(val_loader)
                
                # Registra métricas
                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_accuracy)
                self.val_accuracies.append(val_accuracy)
                
                # Log
                self.logger.info(f"Epoch {epoch+1}/{epochs} - "
                                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
                                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                
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
            self.logger.error(f"Erro durante o treinamento: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def _prepare_sequences(self, data, sequence_length):
        """Prepara sequências para treinamento LSTM.
        
        Args:
            data: DataFrame com dados
            sequence_length: Tamanho da sequência
            
        Returns:
            tuple: (X, y) arrays para treinamento
        """
        # Seleciona features e labels
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'date', 'label']]
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
            plt.savefig(os.path.join(self.visualization_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar métricas: {str(e)}")
            
    def evaluate(self, test_data, sequence_length=20):
        """Avalia o modelo em dados de teste.
        
        Args:
            test_data: DataFrame com dados de teste
            sequence_length: Tamanho da sequência
            
        Returns:
            dict: Métricas de avaliação
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("Nenhum modelo carregado para avaliação")
                return None
                
            # Prepara os dados
            X_test, y_test = self._prepare_sequences(test_data, sequence_length)
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            
            # Avaliação
            self.model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for sequences, labels in test_loader:
                    # Move para o dispositivo correto
                    sequences = sequences.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(sequences)
                    _, predicted = outputs.max(1)
                    
                    # Coleta previsões
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    
            # Converte para arrays
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Calcula métricas
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            # Matriz de confusão
            cm = confusion_matrix(all_labels, all_preds)
            
            # Log
            self.logger.info(f"Avaliação do modelo - Acurácia: {accuracy:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Plota matriz de confusão
            self._plot_confusion_matrix(cm)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            self.logger.error(f"Erro durante a avaliação: {str(e)}")
            return None

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
            plt.savefig(os.path.join(self.visualization_dir, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
            plt.close()
        except Exception as e:
            self.logger.error(f"Erro ao plotar matriz de confusão: {str(e)}")
            
    def preprocess_data(self, data):
        """Pré-processa dados históricos para treinamento.
        
        Args:
            data: DataFrame com dados históricos
            
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
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Adiciona indicadores técnicos
            df = self._add_technical_indicators(df)
            
            # Adiciona padrões de candles
            df = self._add_candle_patterns(df)
            
            # Adiciona features derivadas
            df = self._add_derived_features(df)
            
            # Normaliza os dados
            df = self._normalize_data(df)
            
            # Cria rótulos para treinamento supervisionado
            if 'label' not in df.columns:
                df = self.create_labels(df)
                
            # Remove linhas com NaN após criação de indicadores
            df = df.dropna()
            
            self.logger.info(f"Dados pré-processados: {len(df)} linhas, {len(df.columns)} colunas")
            return df
            
        except Exception as e:
            self.logger.error(f"Erro no pré-processamento: {str(e)}")
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
                if not macd.empty:
                    df['macd'] = macd['MACD_12_26_9']
                    df['macd_signal'] = macd['MACDs_12_26_9']
                    df['macd_hist'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            if len(df) >= 20:
                bbands = ta.bbands(df['close'], length=20, std=2)
                if not bbands.empty:
                    df['bb_upper'] = bbands['BBU_20_2.0']
                    df['bb_middle'] = bbands['BBM_20_2.0']
                    df['bb_lower'] = bbands['BBL_20_2.0']
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
                ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
                if not ichimoku.empty:
                    # Apenas adicione os componentes principais para evitar muitas colunas
                    df['ichimoku_tenkan'] = ichimoku['ITS_9'] 
                    df['ichimoku_kijun'] = ichimoku['IKS_26']
                    df['ichimoku_senkou_a'] = ichimoku['ISA_9_26']
                    df['ichimoku_senkou_b'] = ichimoku['ISB_52_26']
                    
            # Preenchimento de NaN
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
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
            self.logger.error(f"Erro ao adicionar padrões de candles: {str(e)}")
            return df
            
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
            
    def _normalize_data(self, df):
        """Normaliza os dados para treinamento.
        
        Args:
            df: DataFrame com dados históricos
            
        Returns:
            DataFrame: DataFrame com dados normalizados
        """
        try:
            # Colunas a não normalizar
            exclude_cols = ['timestamp', 'date', 'label', 'doji', 'hammer', 'shooting_star', 
                           'bullish_engulfing', 'bearish_engulfing']
            
            # Seleciona colunas para normalizar
            norm_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Normaliza cada coluna individualmente
            for col in norm_cols:
                # Ignora colunas não numéricas
                if not np.issubdtype(df[col].dtype, np.number):
                    continue
                    
                # Normalização Min-Max
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Evita divisão por zero
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    
            return df
            
        except Exception as e:
            self.logger.error(f"Erro na normalização: {str(e)}")
            return df
            
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
            self.logger.error(f"Erro na criação de rótulos: {str(e)}")
            return df
            
    def download_historical_data(self, api, asset, timeframe, start_date, end_date=None):
        """Baixa dados históricos da IQ Option.
        
        Args:
            api: API da IQ Option
            asset: Ativo para baixar dados
            timeframe: Timeframe (60, 300, 900, etc.)
            start_date: Data inicial (datetime ou string)
            end_date: Data final (datetime ou string)
            
        Returns:
            DataFrame: Dados históricos
        """
        try:
            # Converte datas para datetime se necessário
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now()
            elif isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                
            # Ajusta para o final do dia
            end_date = end_date.replace(hour=23, minute=59, second=59)
            
            self.logger.info(f"Baixando dados históricos para {asset} de {start_date} até {end_date}")
            
            # Baixa dados em chunks para evitar limitações da API
            current_date = start_date
            all_candles = []
            
            while current_date < end_date:
                # Define o próximo chunk (máximo 1 mês)
                next_date = min(current_date + timedelta(days=30), end_date)
                
                # Baixa dados
                candles = api.get_candles(asset, timeframe, current_date.timestamp(), next_date.timestamp())
                
                if candles:
                    all_candles.extend(candles)
                    self.logger.info(f"Baixados {len(candles)} candles de {current_date.date()} a {next_date.date()}")
                else:
                    self.logger.warning(f"Nenhum dado encontrado para {asset} de {current_date.date()} a {next_date.date()}")
                    
                # Avança para o próximo chunk
                current_date = next_date
                
                # Espera para evitar rate limiting
                time.sleep(1)
                
            # Converte para DataFrame
            if not all_candles:
                self.logger.error(f"Nenhum dado histórico encontrado para {asset}")
                return None
                
            df = pd.DataFrame(all_candles)
            
            # Renomeia colunas
            df = df.rename(columns={
                'from': 'timestamp',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'volume': 'volume'
            })
            
            # Converte timestamp para datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ordena por timestamp
            df = df.sort_values('timestamp')
            
            # Salva dados
            os.makedirs(os.path.dirname(os.path.abspath(self.historical_data_filename)), exist_ok=True)
            df.to_csv(self.historical_data_filename, index=False)
            
            self.logger.info(f"Dados históricos salvos em {self.historical_data_filename}: {len(df)} registros")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao baixar dados históricos: {str(e)}")
            return None