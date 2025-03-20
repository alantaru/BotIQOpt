import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tulipy as ti
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models, transforms
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
from dotenv import load_dotenv
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
            self.model = None
            self.model_path = model_path
            self.model_dir = os.path.join(os.path.dirname(model_path), "models")
            self.batch_size = 32
            
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

    def evaluate_model(self, test_loader):
        """Avalia o modelo no conjunto de teste com métricas específicas para trading.
        
        Args:
            test_loader: DataLoader com dados de teste
            
        Returns:
            dict: Métricas de avaliação
        """
        self.model.eval()
        
        # Inicializa métricas
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                # Move para o dispositivo correto
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                probabilities = F.softmax(outputs, dim=1)
                
                # Armazena resultados
                _, predicted = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
        # Converte para arrays numpy
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calcula métricas básicas
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Matriz de confusão
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Métricas específicas para trading
        trading_metrics = self._calculate_trading_metrics(all_labels, all_predictions, all_probabilities)
        
        # Visualizações
        self._plot_confusion_matrix(conf_matrix)
        self._plot_roc_curve(all_labels, all_probabilities)
        
        # Combina todas as métricas
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            **trading_metrics
        }
        
        # Log das métricas
        self.logger.info(f"Avaliação do modelo - Acurácia: {accuracy:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        self.logger.info(f"Métricas de trading - Profit Factor: {trading_metrics['profit_factor']:.2f}, Win Rate: {trading_metrics['win_rate']:.2f}")
        
        return metrics

    def _calculate_trading_metrics(self, true_labels, predictions, probabilities):
        """Calcula métricas específicas para trading.
        
        Args:
            true_labels: Rótulos verdadeiros
            predictions: Previsões do modelo
            probabilities: Probabilidades das previsões
            
        Returns:
            dict: Métricas de trading
        """
        # Inicializa métricas
        metrics = {}
        
        # Simula resultados de trading
        # Assumindo: 0 = Hold, 1 = Buy, 2 = Sell
        
        # Filtra apenas sinais de compra e venda (ignora hold)
        trade_indices = np.where((predictions == 1) | (predictions == 2))[0]
        
        if len(trade_indices) == 0:
            self.logger.warning("Nenhum sinal de trade gerado para avaliação")
            return {
                'profit_factor': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trade_count': 0
            }
            
        # Extrai apenas os trades
        trade_predictions = predictions[trade_indices]
        trade_true_labels = true_labels[trade_indices]
        trade_probabilities = probabilities[trade_indices]
        
        # Calcula acertos (win) e erros (loss)
        wins = np.sum((trade_predictions == 1) & (trade_true_labels == 1) | 
                      (trade_predictions == 2) & (trade_true_labels == 2))
        losses = len(trade_indices) - wins
        
        # Win rate
        win_rate = wins / len(trade_indices) if len(trade_indices) > 0 else 0
        
        # Simula retornos (simplificado)
        # Assumindo retorno fixo para cada trade correto/incorreto
        win_return = 0.1  # 10% de retorno por trade vencedor
        loss_return = -0.05  # 5% de perda por trade perdedor
        
        # Calcula retornos
        returns = []
        for i, idx in enumerate(trade_indices):
            if ((trade_predictions[i] == 1) & (trade_true_labels[i] == 1)) | \
               ((trade_predictions[i] == 2) & (trade_true_labels[i] == 2)):
                # Trade vencedor
                returns.append(win_return)
            else:
                # Trade perdedor
                returns.append(loss_return)
                
        returns = np.array(returns)
        
        # Profit factor (soma dos ganhos / soma das perdas)
        total_wins = np.sum(returns[returns > 0]) if any(returns > 0) else 0
        total_losses = abs(np.sum(returns[returns < 0])) if any(returns < 0) else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Média de ganhos e perdas
        avg_win = np.mean(returns[returns > 0]) if any(returns > 0) else 0
        avg_loss = np.mean(returns[returns < 0]) if any(returns < 0) else 0
        
        # Drawdown
        cumulative_returns = np.cumsum(returns)
        max_drawdown = 0
        peak = cumulative_returns[0]
        
        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            drawdown = peak - ret
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        # Sharpe ratio (simplificado)
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Armazena métricas
        metrics['profit_factor'] = profit_factor
        metrics['win_rate'] = win_rate
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        metrics['max_drawdown'] = max_drawdown
        metrics['sharpe_ratio'] = sharpe_ratio
        metrics['trade_count'] = len(trade_indices)
        
        return metrics

    def _plot_confusion_matrix(self, conf_matrix):
        """Plota a matriz de confusão.
        
        Args:
            conf_matrix: Matriz de confusão
        """
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Hold', 'Buy', 'Sell'],
                        yticklabels=['Hold', 'Buy', 'Sell'])
            plt.xlabel('Previsão')
            plt.ylabel('Real')
            plt.title('Matriz de Confusão')
            
            # Salva o gráfico
            os.makedirs(self.visualization_dir, exist_ok=True)
            plt.savefig(os.path.join(self.visualization_dir, 'confusion_matrix.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar matriz de confusão: {str(e)}")

    def _plot_roc_curve(self, true_labels, probabilities):
        """Plota a curva ROC.
        
        Args:
            true_labels: Rótulos verdadeiros
            probabilities: Probabilidades das previsões
        """
        try:
            plt.figure(figsize=(8, 6))
            
            # Para cada classe (hold, buy, sell)
            for i, class_name in enumerate(['Hold', 'Buy', 'Sell']):
                # Binariza os rótulos (um vs resto)
                binary_labels = (true_labels == i).astype(int)
                
                # Calcula a curva ROC
                fpr, tpr, _ = roc_curve(binary_labels, probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                
                # Plota a curva
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
                
            # Adiciona a linha diagonal
            plt.plot([0, 1], [0, 1], 'k--')
            
            plt.xlabel('Taxa de Falsos Positivos')
            plt.ylabel('Taxa de Verdadeiros Positivos')
            plt.title('Curva ROC')
            plt.legend(loc='lower right')
            
            # Salva o gráfico
            os.makedirs(self.visualization_dir, exist_ok=True)
            plt.savefig(os.path.join(self.visualization_dir, 'roc_curve.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar curva ROC: {str(e)}")

    def _plot_training_progress(self, train_losses, val_losses, train_accs, val_accs, epoch):
        """Plota o progresso do treinamento.
        
        Args:
            train_losses: Lista de losses de treinamento
            val_losses: Lista de losses de validação
            train_accs: Lista de acurácias de treinamento
            val_accs: Lista de acurácias de validação
            epoch: Época atual
        """
        try:
            plt.figure(figsize=(12, 5))
            
            # Plot de loss
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Treino')
            plt.plot(val_losses, label='Validação')
            plt.title('Loss')
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot de acurácia
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Treino')
            plt.plot(val_accs, label='Validação')
            plt.title('Acurácia')
            plt.xlabel('Época')
            plt.ylabel('Acurácia (%)')
            plt.legend()
            
            # Salva o gráfico
            os.makedirs(self.visualization_dir, exist_ok=True)
            plt.savefig(os.path.join(self.visualization_dir, f'training_progress_epoch_{epoch+1}.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Erro ao plotar progresso do treinamento: {str(e)}")

    def run_training_pipeline(self, data, sequence_length=60, epochs=50, learning_rate=0.001, val_size=0.2, batch_size=32, early_stopping_patience=10):
        """Executa o pipeline completo de treinamento, validação e avaliação.
        
        Args:
            data: DataFrame com os dados de treinamento
            sequence_length: Tamanho da sequência
            epochs: Número de épocas de treinamento
            learning_rate: Taxa de aprendizado
            val_size: Proporção dos dados para validação
            batch_size: Tamanho do batch
            early_stopping_patience: Número de épocas para parar o treinamento se não houver melhora
            
        Returns:
            dict: Métricas de desempenho do modelo
        """
        try:
            self.logger.info("Iniciando pipeline de treinamento")
            
            # 1. Preparar os dados
            X_train, X_val, y_train, y_val = self._prepare_data(data, sequence_length, val_size)
            
            # 2. Inicializar o modelo
            num_features = X_train.shape[2]
            self._initialize_model(num_features, sequence_length)
            
            # 3. Configurar otimizador e função de perda
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()
            
            # 4. Criar dataloaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # 5. Treinar o modelo
            best_val_loss = float('inf')
            early_stopping_counter = 0
            
            for epoch in range(epochs):
                # Treinar
                train_loss, train_acc = self._train_epoch(train_loader)
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                
                # Validar
                val_loss, val_acc = self._validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # Logar progresso
                self.logger.info(f"Época {epoch+1}/{epochs} - Treino: perda={train_loss:.4f}, acc={train_acc:.4f} | Val: perda={val_loss:.4f}, acc={val_acc:.4f}")
                
                # Verificar se é o melhor modelo
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(self.model_dir, "best_model.pth"))
                    early_stopping_counter = 0
                    self.logger.info(f"Novo melhor modelo salvo com perda de validação: {val_loss:.4f}")
                else:
                    early_stopping_counter += 1
                    
                # Early stopping
                if early_stopping_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping após {epoch+1} épocas sem melhora")
                    break
                    
                # A cada 5 épocas, plotar o progresso
                if (epoch + 1) % 5 == 0:
                    self.plot_training_progress()
                    
            # 6. Carregar o melhor modelo
            self.load_model(os.path.join(self.model_dir, "best_model.pth"))
            
            # 7. Avaliar no conjunto de validação
            self.model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    
                    val_predictions.extend(predicted.cpu().numpy())
                    val_targets.extend(targets.numpy())
            
            # 8. Calcular métricas finais
            accuracy = accuracy_score(val_targets, val_predictions)
            precision = precision_score(val_targets, val_predictions, average='weighted')
            recall = recall_score(val_targets, val_predictions, average='weighted')
            f1 = f1_score(val_targets, val_predictions, average='weighted')
            
            # 9. Plotar matriz de confusão
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(val_targets, val_predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matriz de Confusão')
            plt.xlabel('Predito')
            plt.ylabel('Real')
            plt.savefig(os.path.join(self.visualization_dir, 'confusion_matrix.png'))
            
            # 10. Plotar curva ROC para cada classe (one-vs-rest)
            plt.figure(figsize=(10, 8))
            
            # Converter para one-hot encoding para ROC
            y_val_onehot = np.zeros((len(val_targets), self.model.output_size))
            for i, val in enumerate(val_targets):
                y_val_onehot[i, val] = 1
                
            # Obter probabilidades para cada classe
            probs = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    probs.append(F.softmax(outputs, dim=1).cpu().numpy())
            
            probs = np.vstack(probs)
            
            # Plotar ROC para cada classe
            for i in range(self.model.output_size):
                fpr, tpr, _ = roc_curve(y_val_onehot[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Classe {i} (AUC = {roc_auc:.2f})')
                
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taxa de Falsos Positivos')
            plt.ylabel('Taxa de Verdadeiros Positivos')
            plt.title('Curva ROC')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.visualization_dir, 'roc_curve.png'))
            
            # 11. Retornar métricas
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'val_loss': best_val_loss,
                'training_epochs': epoch + 1
            }
            
            self.logger.info(f"Pipeline de treinamento concluído com sucesso. Métricas: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Erro no pipeline de treinamento: {str(e)}")
            raise
            
    def _prepare_data(self, data, sequence_length, val_size=0.2):
        """Prepara os dados para treinamento.
        
        Args:
            data: DataFrame com os dados
            sequence_length: Tamanho da sequência
            val_size: Proporção dos dados para validação
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        try:
            self.logger.info("Preparando dados para treinamento")
            
            # Verificar se há uma coluna de rótulos
            if 'label' not in data.columns:
                raise ValueError("Dados não contêm coluna 'label'")
                
            # Remover colunas não numéricas
            feature_cols = [col for col in data.columns if col not in ['timestamp', 'date', 'label']]
            
            # Criar sequências
            sequences = []
            labels = []
            
            for i in range(len(data) - sequence_length):
                seq = data[feature_cols].iloc[i:i+sequence_length].values
                label = data['label'].iloc[i+sequence_length]
                sequences.append(seq)
                labels.append(label)
                
            # Converter para arrays numpy
            X = np.array(sequences)
            y = np.array(labels)
            
            # Dividir em treino e validação
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, shuffle=False)
            
            # Converter para tensores PyTorch
            X_train = torch.FloatTensor(X_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_train = torch.LongTensor(y_train).to(self.device)
            y_val = torch.LongTensor(y_val).to(self.device)
            
            self.logger.info(f"Dados preparados: X_train={X_train.shape}, X_val={X_val.shape}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar dados: {str(e)}")
            raise

    def get_historical_data(self, ferramental_instance):
        """Obtém dados históricos para treinamento utilizando a instância do Ferramental.
        
        Args:
            ferramental_instance: Instância da classe Ferramental
            
        Returns:
            DataFrame com dados históricos ou None em caso de erro
        """
        try:
            # Utiliza o método get_historical_data da classe Ferramental
            self.logger.info("Obtendo dados históricos através do Ferramental")
            historical_data = ferramental_instance.get_historical_data()
            
            if historical_data is None or historical_data.empty:
                self.logger.error("Não foi possível obter dados históricos")
                return None
                
            self.logger.info(f"Obtidos {len(historical_data)} registros de dados históricos")
            
            # Salva os dados históricos em um arquivo para uso futuro
            os.makedirs(os.path.dirname(self.historical_data_filename), exist_ok=True)
            historical_data.to_csv(self.historical_data_filename, index=False)
            self.logger.info(f"Dados históricos salvos em {self.historical_data_filename}")
            
            self.historical_data = historical_data
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Erro ao obter dados históricos: {str(e)}")
            return None
    
    def preprocess_data(self, data=None):
        """Pré-processa dados históricos para treinamento.
        
        Args:
            data: DataFrame com dados históricos (opcional, usa self.historical_data se None)
            
        Returns:
            DataFrame pré-processado ou None em caso de erro
        """
        try:
            if data is None:
                if self.historical_data is None:
                    self.logger.error("Nenhum dado histórico disponível para pré-processamento")
                    return None
                data = self.historical_data.copy()
                
            self.logger.info(f"Iniciando pré-processamento de {len(data)} registros")
            
            # Verifica se as colunas necessárias estão presentes
            required_columns = ['open', 'close', 'high', 'low', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Colunas obrigatórias ausentes: {missing_columns}")
                return None
                
            # Preenchimento de valores ausentes
            for col in data.columns:
                if data[col].isna().any():
                    if col in ['open', 'close', 'high', 'low']:
                        # Para dados de preço, preenchemos com o valor anterior
                        data[col] = data[col].fillna(method='ffill')
                    elif col == 'volume':
                        # Para volume, preenchemos com 0
                        data[col] = data[col].fillna(0)
                    else:
                        # Para outras colunas, preenchemos com a média
                        data[col] = data[col].fillna(data[col].mean())
            
            # Adiciona indicadores técnicos
            data = self._add_technical_indicators(data)
            
            # Adiciona padrões de candles
            data = self._add_candle_patterns(data)
            
            # Adiciona features derivadas
            data = self._add_derived_features(data)
            
            # Normaliza os dados
            data = self._normalize_data(data)
            
            self.logger.info(f"Pré-processamento concluído. Resultado: {len(data)} registros com {len(data.columns)} features")
            return data
            
        except Exception as e:
            self.logger.error(f"Erro no pré-processamento de dados: {str(e)}")
            return None
    
    def _add_technical_indicators(self, data):
        """Adiciona indicadores técnicos ao DataFrame.
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            DataFrame com indicadores técnicos adicionados
        """
        try:
            df = data.copy()
            
            # Garante que os dados estão ordenados por data
            if 'from' in df.columns:
                df = df.sort_values('from')
            
            # Médias Móveis
            window_sizes = [7, 14, 21, 50, 200]
            for window in window_sizes:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
            # Bandas de Bollinger (20 períodos)
            window = 20
            df['sma_20'] = df['close'].rolling(window=window).mean()
            df['bollinger_upper'] = df['sma_20'] + 2 * df['close'].rolling(window=window).std()
            df['bollinger_lower'] = df['sma_20'] - 2 * df['close'].rolling(window=window).std()
            
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Stochastic Oscillator
            n = 14
            df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(window=n).min()) / 
                                  (df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()
            
            # Preenche valores NaN resultantes das operações de janela
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(method='bfill')
                    df[col] = df[col].fillna(method='ffill')
                    df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar indicadores técnicos: {str(e)}")
            return data
    
    def _add_candle_patterns(self, data):
        """Adiciona padrões de candles ao DataFrame.
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            DataFrame com padrões de candles adicionados
        """
        try:
            df = data.copy()
            
            # Tamanho do corpo da vela
            df['body_size'] = abs(df['close'] - df['open'])
            
            # Sombra superior
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            
            # Sombra inferior
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            # Doji (corpo muito pequeno)
            avg_body_size = df['body_size'].mean()
            df['is_doji'] = (df['body_size'] <= 0.1 * avg_body_size).astype(int)
            
            # Martelo (sombra inferior longa, corpo pequeno no topo, pouca ou nenhuma sombra superior)
            df['is_hammer'] = (
                (df['lower_shadow'] >= 2 * df['body_size']) & 
                (df['upper_shadow'] <= 0.1 * df['body_size']) &
                (df['close'] > df['open'])
            ).astype(int)
            
            # Estrela Cadente (sombra superior longa, corpo pequeno na base, pouca ou nenhuma sombra inferior)
            df['is_shooting_star'] = (
                (df['upper_shadow'] >= 2 * df['body_size']) & 
                (df['lower_shadow'] <= 0.1 * df['body_size']) &
                (df['close'] < df['open'])
            ).astype(int)
            
            # Engolfo de Alta (vela de alta que "engole" vela anterior de baixa)
            df['is_bullish_engulfing'] = (
                (df['close'] > df['open']) &
                (df['open'] < df['close'].shift()) &
                (df['close'] > df['open'].shift()) &
                (df['close'].shift() < df['open'].shift())
            ).astype(int)
            
            # Engolfo de Baixa (vela de baixa que "engole" vela anterior de alta)
            df['is_bearish_engulfing'] = (
                (df['close'] < df['open']) &
                (df['open'] > df['close'].shift()) &
                (df['close'] < df['open'].shift()) &
                (df['close'].shift() > df['open'].shift())
            ).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar padrões de candles: {str(e)}")
            return data
    
    def _add_derived_features(self, data):
        """Adiciona features derivadas ao DataFrame.
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            DataFrame com features derivadas adicionadas
        """
        try:
            df = data.copy()
            
            # Retornos
            df['return_1'] = df['close'].pct_change(1)
            df['return_5'] = df['close'].pct_change(5)
            df['return_10'] = df['close'].pct_change(10)
            
            # Volatilidade
            df['volatility_5'] = df['return_1'].rolling(window=5).std()
            df['volatility_10'] = df['return_1'].rolling(window=10).std()
            df['volatility_20'] = df['return_1'].rolling(window=20).std()
            
            # Momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Volume relativo
            df['volume_ratio_5'] = df['volume'] / df['volume'].rolling(window=5).mean()
            df['volume_ratio_10'] = df['volume'] / df['volume'].rolling(window=10).mean()
            
            # Preço relativo às médias móveis
            df['price_sma_ratio_7'] = df['close'] / df['sma_7']
            df['price_sma_ratio_21'] = df['close'] / df['sma_21']
            df['price_sma_ratio_50'] = df['close'] / df['sma_50']
            
            # Distância das Bandas de Bollinger
            df['bb_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
            
            # Preenche valores NaN resultantes das operações de janela
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(method='bfill')
                    df[col] = df[col].fillna(method='ffill')
                    df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao adicionar features derivadas: {str(e)}")
            return data
    
    def _normalize_data(self, data):
        """Normaliza os dados para treinamento.
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            DataFrame com dados normalizados
        """
        try:
            df = data.copy()
            
            # Colunas a serem normalizadas (excluindo colunas categóricas e binárias)
            exclude_cols = ['asset', 'timeframe', 'from', 'is_doji', 'is_hammer', 
                           'is_shooting_star', 'is_bullish_engulfing', 'is_bearish_engulfing']
            
            # Identifica colunas numéricas para normalização
            numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
            
            # Normalização Min-Max para cada coluna numérica
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Evita divisão por zero
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    # Se min e max são iguais, definimos como 0.5 (valor médio normalizado)
                    df[col] = 0.5
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao normalizar dados: {str(e)}")
            return data
    
    def create_labels(self, data, lookahead=5, threshold=0.005):
        """Cria rótulos para treinamento supervisionado.
        
        Args:
            data: DataFrame com dados históricos
            lookahead: Número de períodos à frente para prever
            threshold: Limiar de variação percentual para considerar como sinal
            
        Returns:
            DataFrame com rótulos adicionados
        """
        try:
            df = data.copy()
            
            # Calcula o retorno futuro
            df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
            
            # Cria rótulos baseados no retorno futuro
            conditions = [
                (df['future_return'] > threshold),  # Compra
                (df['future_return'] < -threshold),  # Venda
                (True)  # Hold (condição padrão)
            ]
            choices = [1, 2, 0]  # 1=Buy, 2=Sell, 0=Hold
            
            df['label'] = np.select(conditions, choices, default=0)
            
            # Remove as linhas finais que não têm rótulos válidos
            df = df.iloc[:-lookahead]
            
            self.logger.info(f"Rótulos criados: {df['label'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao criar rótulos: {str(e)}")
            return data
    
    def split_data(self, data, test_size=0.2, val_size=0.2, random_split=False, sequence_length=60):
        """Divide os dados em conjuntos de treinamento, validação e teste.
        
        Args:
            data: DataFrame com dados pré-processados
            test_size: Proporção dos dados para teste
            val_size: Proporção dos dados para validação
            random_split: Se True, divide aleatoriamente, senão divide cronologicamente
            sequence_length: Tamanho da sequência para o modelo
            
        Returns:
            dict: Dicionário com conjuntos de dados divididos
        """
        try:
            if data is None or len(data) < 100:
                self.logger.error("Dados insuficientes para divisão")
                return None
                
            # Remove colunas não numéricas
            features = data.select_dtypes(include=['float64', 'int64']).copy()
            
            if len(features.columns) <= sequence_length + 1:  # +1 para a coluna alvo
                self.logger.info(f"Número de features ({len(features.columns)-1}) já é menor que o solicitado ({sequence_length})")
                return None
                
            # Separa features e rótulos
            X = features.drop('label', axis=1)
            y = features['label']
            
            if random_split:
                # Divisão aleatória
                X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42)
            else:
                # Divisão cronológica
                test_idx = int(len(X) * (1 - test_size))
                val_idx = int(test_idx * (1 - val_size))
                
                X_train, y_train = X.iloc[:val_idx], y.iloc[:val_idx]
                X_val, y_val = X.iloc[val_idx:test_idx], y.iloc[val_idx:test_idx]
                X_test, y_test = X.iloc[test_idx:], y.iloc[test_idx:]
            
            # Prepara sequências para o modelo LSTM
            X_train_seq = self._create_sequences(X_train, sequence_length)
            X_val_seq = self._create_sequences(X_val, sequence_length)
            X_test_seq = self._create_sequences(X_test, sequence_length)
            
            # Ajusta os rótulos para corresponder às sequências
            y_train_seq = y_train.iloc[sequence_length-1:].values
            y_val_seq = y_val.iloc[sequence_length-1:].values
            y_test_seq = y_test.iloc[sequence_length-1:].values
            
            # Converte para tensores PyTorch
            X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
            y_train_tensor = torch.LongTensor(y_train_seq).to(self.device)
            
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_seq).to(self.device)
            
            X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
            y_test_tensor = torch.LongTensor(y_test_seq).to(self.device)
            
            # Cria datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            # Cria dataloaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
            
            self.logger.info(f"Divisão de dados concluída: {len(train_dataset)} treinamento, {len(val_dataset)} validação, {len(test_dataset)} teste")
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feature_names': list(X.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao dividir dados: {str(e)}")
            return None
    
    def _create_sequences(self, data, sequence_length):
        """Cria sequências para o modelo LSTM.
        
        Args:
            data: DataFrame com features
            sequence_length: Tamanho da sequência
            
        Returns:
            numpy.ndarray: Array com sequências
        """
        sequences = []
        data_array = data.values
        
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data_array[i:i+sequence_length])
            
        return np.array(sequences)

    def update_historical_data(self, ferramental_instance, force_update=False):
        """Atualiza dados históricos periodicamente.
        
        Args:
            ferramental_instance: Instância da classe Ferramental
            force_update: Se True, força a atualização mesmo que o cache seja recente
            
        Returns:
            DataFrame com dados históricos atualizados ou None em caso de erro
        """
        try:
            # Verifica se já temos dados em cache e se são recentes
            cache_file = self.historical_data_filename
            cache_is_recent = False
            
            if os.path.exists(cache_file) and not force_update:
                # Verifica a idade do arquivo de cache
                file_time = os.path.getmtime(cache_file)
                current_time = time.time()
                hours_old = (current_time - file_time) / 3600
                
                # Cache é considerado recente se tiver menos de 24 horas
                cache_is_recent = hours_old < 24
                
                if cache_is_recent:
                    self.logger.info(f"Usando cache de dados históricos (atualizado há {hours_old:.1f} horas)")
                    try:
                        self.historical_data = pd.read_csv(cache_file)
                        return self.historical_data
                    except Exception as e:
                        self.logger.error(f"Erro ao ler cache: {str(e)}")
                        # Se falhar na leitura do cache, continuamos para baixar novos dados
            
            # Se não temos cache recente ou a leitura falhou, baixamos novos dados
            return self.get_historical_data(ferramental_instance)
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar dados históricos: {str(e)}")
            return None

    def select_important_features(self, data, target='label', n_features=20):
        """Seleciona as features mais importantes para o treinamento.
        
        Args:
            data: DataFrame com dados pré-processados
            target: Nome da coluna alvo
            n_features: Número de features a serem selecionadas
            
        Returns:
            DataFrame com as features mais importantes
        """
        try:
            if target not in data.columns:
                self.logger.error(f"Coluna alvo '{target}' não encontrada nos dados")
                return data
                
            # Remove colunas não numéricas
            numeric_data = data.select_dtypes(include=['float64', 'int64']).copy()
            
            if len(numeric_data.columns) <= n_features + 1:  # +1 para a coluna alvo
                self.logger.info(f"Número de features ({len(numeric_data.columns)-1}) já é menor que o solicitado ({n_features})")
                return data
                
            # Separa features e target
            X = numeric_data.drop(target, axis=1)
            y = numeric_data[target]
            
            # Importa RandomForestClassifier apenas quando necessário
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import SelectFromModel
            
            # Treina um modelo de floresta aleatória para seleção de features
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Seleciona as features mais importantes
            selector = SelectFromModel(model, max_features=n_features, threshold=-np.inf)
            selector.fit(X, y)
            
            # Obtém os nomes das features selecionadas
            selected_features = X.columns[selector.get_support()]
            
            # Adiciona a coluna alvo às features selecionadas
            selected_features = list(selected_features) + [target]
            
            # Seleciona apenas as colunas importantes no DataFrame original
            # Mantém também colunas não numéricas que podem ser importantes (como 'asset', 'from', etc.)
            non_numeric_cols = [col for col in data.columns if col not in numeric_data.columns]
            final_columns = list(selected_features) + non_numeric_cols
            
            self.logger.info(f"Selecionadas {len(selected_features)-1} features importantes: {', '.join(selected_features[:-1])}")
            
            return data[final_columns]
            
        except Exception as e:
            self.logger.error(f"Erro na seleção de features: {str(e)}")
            return data
            
    def engineer_advanced_features(self, data):
        """Cria features avançadas para melhorar o desempenho do modelo.
        
        Args:
            data: DataFrame com dados pré-processados
            
        Returns:
            DataFrame com features avançadas adicionadas
        """
        try:
            df = data.copy()
            
            # 1. Features de interação entre indicadores
            if 'rsi_14' in df.columns and 'macd' in df.columns:
                # Interação RSI e MACD (detecta divergências)
                df['rsi_macd_divergence'] = ((df['rsi_14'] > 70) & (df['macd'] < 0)).astype(int) - \
                                           ((df['rsi_14'] < 30) & (df['macd'] > 0)).astype(int)
            
            # 2. Features de cruzamento de médias móveis
            for fast, slow in [(7, 21), (21, 50), (50, 200)]:
                if f'sma_{fast}' in df.columns and f'sma_{slow}' in df.columns:
                    # Cruzamento para cima (golden cross)
                    df[f'golden_cross_{fast}_{slow}'] = ((df[f'sma_{fast}'] > df[f'sma_{slow}']) & 
                                                       (df[f'sma_{fast}'].shift(1) <= df[f'sma_{slow}'].shift(1))).astype(int)
                    # Cruzamento para baixo (death cross)
                    df[f'death_cross_{fast}_{slow}'] = ((df[f'sma_{fast}'] < df[f'sma_{slow}']) & 
                                                      (df[f'sma_{fast}'].shift(1) >= df[f'sma_{slow}'].shift(1))).astype(int)
            
            # 3. Features de tendência
            if 'close' in df.columns:
                # Tendência de curto prazo (5 períodos)
                df['trend_5'] = (df['close'] > df['close'].shift(5)).astype(int)
                # Tendência de médio prazo (20 períodos)
                df['trend_20'] = (df['close'] > df['close'].shift(20)).astype(int)
                # Força da tendência (razão entre tendências)
                df['trend_strength'] = df['trend_5'].rolling(window=10).mean()
            
            # 4. Features de volatilidade relativa
            if 'atr_14' in df.columns and 'close' in df.columns:
                # Volatilidade relativa ao preço
                df['relative_volatility'] = df['atr_14'] / df['close']
            
            # 5. Features de momentum acumulado
            if 'momentum_5' in df.columns and 'momentum_20' in df.columns:
                # Aceleração do momentum
                df['momentum_acceleration'] = df['momentum_5'] - df['momentum_20']
            
            # 6. Features de suporte e resistência
            if 'high' in df.columns and 'low' in df.columns:
                # Níveis de suporte (mínimos recentes)
                df['support_level'] = df['low'].rolling(window=20).min()
                # Níveis de resistência (máximos recentes)
                df['resistance_level'] = df['high'].rolling(window=20).max()
                # Distância percentual do suporte
                df['support_distance'] = (df['close'] - df['support_level']) / df['close']
                # Distância percentual da resistência
                df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
            
            # 7. Features de volume e preço
            if 'volume' in df.columns and 'close' in df.columns:
                # Volume ponderado pelo preço
                df['volume_price_ratio'] = df['volume'] / df['close']
                # Acumulação/Distribuição
                if 'high' in df.columns and 'low' in df.columns:
                    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close']).abs())
                    money_flow_volume = money_flow_multiplier * df['volume']
                    df['acc_dist_line'] = money_flow_volume.cumsum()
            
            # Preenche valores NaN resultantes das operações
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(method='bfill')
                    df[col] = df[col].fillna(method='ffill')
                    df[col] = df[col].fillna(0)
            
            self.logger.info(f"Adicionadas {len(df.columns) - len(data.columns)} features avançadas")
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao criar features avançadas: {str(e)}")
            return data

    def balance_classes(self, data, target='label', method='smote'):
        """Balanceia as classes para evitar viés no treinamento.
        
        Args:
            data: DataFrame com dados pré-processados
            target: Nome da coluna alvo
            method: Método de balanceamento ('smote', 'adasyn', 'undersampling', 'oversampling')
            
        Returns:
            DataFrame com classes balanceadas
        """
        try:
            if target not in data.columns:
                self.logger.error(f"Coluna alvo '{target}' não encontrada nos dados")
                return data
                
            # Verifica a distribuição atual das classes
            class_distribution = data[target].value_counts()
            self.logger.info(f"Distribuição original das classes: {class_distribution.to_dict()}")
            
            # Se as classes já estão razoavelmente balanceadas (diferença < 20%), retorna os dados originais
            min_class = class_distribution.min()
            max_class = class_distribution.max()
            if min_class / max_class > 0.8:
                self.logger.info("Classes já estão razoavelmente balanceadas")
                return data
                
            # Separa features numéricas e target
            X = data.drop(target, axis=1).select_dtypes(include=['float64', 'int64'])
            y = data[target]
            
            # Guarda colunas não numéricas para reincorporar depois
            non_numeric_cols = [col for col in data.columns if col not in X.columns and col != target]
            non_numeric_data = data[non_numeric_cols] if non_numeric_cols else None
            
            # Aplica o método de balanceamento escolhido
            if method == 'smote':
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=42)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                self.logger.info("Aplicado balanceamento SMOTE")
                
            elif method == 'adasyn':
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(random_state=42)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                self.logger.info("Aplicado balanceamento ADASYN")
                
            elif method == 'undersampling':
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                self.logger.info("Aplicado undersampling aleatório")
                
            elif method == 'oversampling':
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=42)
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                self.logger.info("Aplicado oversampling aleatório")
                
            else:
                self.logger.error(f"Método de balanceamento '{method}' não reconhecido")
                return data
                
            # Cria novo DataFrame com os dados balanceados
            balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_data[target] = y_resampled
            
            # Verifica a nova distribuição das classes
            new_distribution = balanced_data[target].value_counts()
            self.logger.info(f"Nova distribuição das classes: {new_distribution.to_dict()}")
            
            # Se havia colunas não numéricas, tenta reincorporá-las
            if non_numeric_data is not None:
                # Como o balanceamento pode ter alterado o número de amostras,
                # precisamos tratar isso de forma especial
                if len(balanced_data) == len(data):
                    # Se o número de amostras é o mesmo, podemos simplesmente adicionar as colunas
                    for col in non_numeric_cols:
                        balanced_data[col] = data[col].values
                else:
                    # Se o número de amostras mudou, não podemos reincorporar diretamente
                    # Neste caso, avisamos que as colunas não numéricas foram perdidas
                    self.logger.warning(f"Colunas não numéricas {non_numeric_cols} foram perdidas no balanceamento")
            
            return balanced_data
            
        except Exception as e:
            self.logger.error(f"Erro no balanceamento de classes: {str(e)}")
            return data
            
    def create_efficient_dataloaders(self, train_data, val_data, test_data, target='label', 
                                    batch_size=None, num_workers=4, pin_memory=True):
        """Cria DataLoaders eficientes para treinamento.
        
        Args:
            train_data: DataFrame com dados de treinamento
            val_data: DataFrame com dados de validação
            test_data: DataFrame com dados de teste
            target: Nome da coluna alvo
            batch_size: Tamanho do batch (se None, usa self.batch_size)
            num_workers: Número de workers para carregamento paralelo
            pin_memory: Se True, usa pin_memory para transferência mais rápida para GPU
            
        Returns:
            dict: Dicionário com DataLoaders
        """
        try:
            if batch_size is None:
                batch_size = self.batch_size
                
            # Verifica se os dados estão presentes
            if train_data is None or val_data is None or test_data is None:
                self.logger.error("Dados de treinamento, validação ou teste ausentes")
                return None
                
            # Verifica se a coluna alvo está presente
            if target not in train_data.columns or target not in val_data.columns or target not in test_data.columns:
                self.logger.error(f"Coluna alvo '{target}' não encontrada em todos os conjuntos de dados")
                return None
                
            # Separa features e target
            X_train = train_data.drop(target, axis=1).select_dtypes(include=['float64', 'int64'])
            y_train = train_data[target]
            
            X_val = val_data.drop(target, axis=1).select_dtypes(include=['float64', 'int64'])
            y_val = val_data[target]
            
            X_test = test_data.drop(target, axis=1).select_dtypes(include=['float64', 'int64'])
            y_test = test_data[target]
            
            # Converte para tensores PyTorch
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.LongTensor(y_train.values)
            
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.LongTensor(y_val.values)
            
            X_test_tensor = torch.FloatTensor(X_test.values)
            y_test_tensor = torch.LongTensor(y_test.values)
            
            # Cria datasets
            from torch.utils.data import TensorDataset
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            # Cria dataloaders otimizados
            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                pin_memory=pin_memory and self.device == 'cuda'
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=pin_memory and self.device == 'cuda'
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=pin_memory and self.device == 'cuda'
            )
            
            self.logger.info(f"DataLoaders criados: {len(train_dataset)} amostras de treinamento, "
                            f"{len(val_dataset)} de validação, {len(test_dataset)} de teste")
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'feature_names': list(X_train.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao criar DataLoaders: {str(e)}")
            return None

    def train_with_auto_evaluation(self, train_loader, val_loader, epochs=50, 
                                  early_stopping_patience=10, learning_rate=0.001):
        """Treina o modelo com avaliação automática e early stopping.
        
        Args:
            train_loader: DataLoader com dados de treinamento
            val_loader: DataLoader com dados de validação
            epochs: Número máximo de épocas
            early_stopping_patience: Número de épocas para parar o treinamento se não houver melhora
            learning_rate: Taxa de aprendizado inicial
            
        Returns:
            dict: Histórico de treinamento e métricas finais
        """
        try:
            if self.model is None:
                self.logger.error("Modelo não inicializado")
                return None
                
            # Configurações de treinamento
            self.model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
            
            # Métricas de treinamento
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            
            # Early stopping
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            # Loop de treinamento
            for epoch in range(epochs):
                # Treinamento
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Métricas
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
                
                # Validação
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Métricas
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()
                
                # Calcula médias
                avg_train_loss = train_loss / len(train_loader.dataset)
                avg_val_loss = val_loss / len(val_loader.dataset)
                train_accuracy = train_correct / train_total
                val_accuracy = val_correct / val_total
                
                # Atualiza scheduler
                scheduler.step(avg_val_loss)
                
                # Registra métricas
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                
                # Log
                self.logger.info(f"Epoch {epoch+1}/{epochs}: "
                               f"Train Loss: {avg_train_loss:.4f}, "
                               f"Val Loss: {avg_val_loss:.4f}, "
                               f"Train Acc: {train_accuracy:.4f}, "
                               f"Val Acc: {val_accuracy:.4f}")
                
                # Salva o melhor modelo
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    self.logger.info(f"Novo melhor modelo encontrado! Val Loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    self.logger.info(f"Sem melhora por {patience_counter} épocas")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping após {epoch+1} épocas")
                    break
            
            # Restaura o melhor modelo
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                self.logger.info("Restaurado o melhor modelo")
            
            # Retorna histórico de treinamento
            history = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1
            }
            
            return history
            
        except Exception as e:
            self.logger.error(f"Erro no treinamento: {str(e)}")
            return None

    def auto_tune_hyperparameters(self, train_data, val_data, param_grid=None, n_trials=20):
        """Ajusta automaticamente hiperparâmetros usando Optuna.
        
        Args:
            train_data: DataFrame com dados de treinamento
            val_data: DataFrame com dados de validação
            param_grid: Dicionário com grades de hiperparâmetros (opcional)
            n_trials: Número de tentativas para otimização
            
        Returns:
            dict: Melhores hiperparâmetros encontrados
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
            
            # Define grade de hiperparâmetros padrão se não fornecida
            if param_grid is None:
                param_grid = {
                    'learning_rate': (1e-4, 1e-2),
                    'batch_size': [16, 32, 64, 128],
                    'hidden_size': [64, 128, 256],
                    'num_layers': [1, 2, 3],
                    'dropout': (0.1, 0.5)
                }
            
            # Função objetivo para otimização
            def objective(trial):
                # Amostra hiperparâmetros
                lr = trial.suggest_float('learning_rate', *param_grid['learning_rate'], log=True)
                batch_size = trial.suggest_categorical('batch_size', param_grid['batch_size'])
                hidden_size = trial.suggest_categorical('hidden_size', param_grid['hidden_size'])
                num_layers = trial.suggest_categorical('num_layers', param_grid['num_layers'])
                dropout = trial.suggest_float('dropout', *param_grid['dropout'])
                
                # Prepara dataloaders
                dataloaders = self.create_efficient_dataloaders(
                    train_data, val_data, val_data,  # Usamos val_data como test_data temporariamente
                    batch_size=batch_size
                )
                
                if dataloaders is None:
                    return float('inf')
                
                # Inicializa modelo com hiperparâmetros
                num_features = len(dataloaders['feature_names'])
                model = self.initialize_model(
                    input_size=num_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=3,
                    dropout=dropout
                )
                
                # Salva o modelo original
                original_model = self.model
                self.model = model
                
                # Treina por algumas épocas para avaliar
                history = self.train_with_auto_evaluation(
                    dataloaders['train_loader'],
                    dataloaders['val_loader'],
                    epochs=10,
                    early_stopping_patience=5,
                    learning_rate=lr
                )
                
                # Restaura o modelo original
                self.model = original_model
                
                if history is None:
                    return float('inf')
                
                # Retorna a melhor perda de validação como métrica a ser minimizada
                return history['best_val_loss']
            
            # Cria estudo Optuna
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42)
            )
            
            # Executa otimização
            self.logger.info(f"Iniciando otimização de hiperparâmetros com {n_trials} tentativas")
            study.optimize(objective, n_trials=n_trials)
            
            # Obtém melhores hiperparâmetros
            best_params = study.best_params
            best_value = study.best_value
            
            self.logger.info(f"Melhores hiperparâmetros encontrados: {best_params}")
            self.logger.info(f"Melhor valor de validação: {best_value:.4f}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Erro na otimização de hiperparâmetros: {str(e)}")
            return None

    def complete_training_pipeline(self, ferramental_instance):
        """Pipeline completo de treinamento com todas as etapas otimizadas.
        
        Args:
            ferramental_instance: Instância da classe Ferramental
            
        Returns:
            dict: Resultados do treinamento
        """
        try:
            # 1. Atualiza dados históricos
            self.logger.info("1. Atualizando dados históricos")
            historical_data = self.update_historical_data(ferramental_instance)
            
            if historical_data is None:
                self.logger.error("Falha ao obter dados históricos")
                return None
                
            # 2. Pré-processamento de dados
            self.logger.info("2. Pré-processando dados")
            processed_data = self.preprocess_data(historical_data)
            
            if processed_data is None:
                self.logger.error("Falha no pré-processamento de dados")
                return None
                
            # 3. Engenharia de features avançada
            self.logger.info("3. Criando features avançadas")
            featured_data = self.engineer_advanced_features(processed_data)
            
            # 4. Criação de rótulos
            self.logger.info("4. Criando rótulos para treinamento supervisionado")
            labeled_data = self.create_labels(featured_data, lookahead=5, threshold=0.005)
            
            # 5. Seleção de features importantes
            self.logger.info("5. Selecionando features importantes")
            selected_data = self.select_important_features(labeled_data, n_features=30)
            
            # 6. Divisão de dados
            self.logger.info("6. Dividindo dados em conjuntos de treinamento, validação e teste")
            data_splits = self.split_data(selected_data, test_size=0.2, val_size=0.2, 
                                         chronological=True, shuffle=False)
            
            if data_splits is None:
                self.logger.error("Falha na divisão dos dados")
                return None
                
            train_data = data_splits['train']
            val_data = data_splits['val']
            test_data = data_splits['test']
            
            # 7. Balanceamento de classes
            self.logger.info("7. Balanceando classes")
            balanced_train_data = self.balance_classes(train_data, method='smote')
            
            # 8. Otimização de hiperparâmetros
            self.logger.info("8. Otimizando hiperparâmetros")
            best_params = self.auto_tune_hyperparameters(balanced_train_data, val_data, n_trials=10)
            
            if best_params is None:
                self.logger.warning("Falha na otimização de hiperparâmetros. Usando valores padrão")
                best_params = {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.2
                }
            
            # 9. Criação de dataloaders eficientes
            self.logger.info("9. Criando dataloaders eficientes")
            dataloaders = self.create_efficient_dataloaders(
                balanced_train_data, val_data, test_data,
                batch_size=best_params['batch_size']
            )
            
            if dataloaders is None:
                self.logger.error("Falha na criação de dataloaders")
                return None
            
            # 10. Inicialização do modelo com melhores hiperparâmetros
            self.logger.info("10. Inicializando modelo com melhores hiperparâmetros")
            num_features = len(dataloaders['feature_names'])
            self.model = self.initialize_model(
                input_size=num_features,
                hidden_size=best_params['hidden_size'],
                num_layers=best_params['num_layers'],
                output_size=3,
                dropout=best_params['dropout']
            )
            
            # 11. Treinamento com auto avaliação
            self.logger.info("11. Treinando modelo com auto avaliação")
            history = self.train_with_auto_evaluation(
                dataloaders['train_loader'],
                dataloaders['val_loader'],
                epochs=50,
                early_stopping_patience=10,
                learning_rate=best_params['learning_rate']
            )
            
            if history is None:
                self.logger.error("Falha no treinamento do modelo")
                return None
            
            # 12. Avaliação no conjunto de teste
            self.logger.info("12. Avaliando modelo no conjunto de teste")
            test_metrics = self.evaluate_model(dataloaders['test_loader'])
            
            # 13. Salvamento do modelo e métricas
            self.logger.info("13. Salvando modelo e métricas")
            model_path = os.path.join(self.model_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            torch.save(self.model.state_dict(), model_path)
            
            # Salva métricas em JSON
            metrics_path = os.path.join(self.model_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            metrics = {
                'training_history': {
                    'train_losses': [float(x) for x in history['train_losses']],
                    'val_losses': [float(x) for x in history['val_losses']],
                    'train_accuracies': [float(x) for x in history['train_accuracies']],
                    'val_accuracies': [float(x) for x in history['val_accuracies']]
                },
                'test_metrics': test_metrics,
                'hyperparameters': best_params,
                'feature_names': dataloaders['feature_names'],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self.logger.info(f"Métricas salvas em {metrics_path}")
            
            # Resultado final
            result = {
                'model_path': model_path,
                'metrics_path': metrics_path,
                'test_metrics': test_metrics,
                'training_history': history,
                'hyperparameters': best_params
            }
            
            self.logger.info("Pipeline de treinamento concluído com sucesso!")
            return result
            
        except Exception as e:
            self.logger.error(f"Erro no pipeline de treinamento: {str(e)}")
            return None
