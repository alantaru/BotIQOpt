import os
import sys
import json
import numpy as np
import pandas as pd
import logging
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teste_treinamento_ia.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TesteTreinamentoIA')

# Dataset personalizado para PyTorch
class FinancialDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Modelo de rede neural simples
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class TreinamentoIASimulado:
    """Classe para testes de treinamento da IA"""
    
    def __init__(self):
        self.logger = logger
        self.model = None
        self.best_params = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        
    def _gerar_dados_simulados(self, n_samples=1000, n_features=10, n_classes=3):
        """Gera dados simulados para testes de treinamento"""
        logger.info(f"Gerando {n_samples} registros de dados simulados com {n_features} features")
        
        # Gera features aleatórias
        X = np.random.randn(n_samples, n_features)
        
        # Gera classes com distribuição desigual
        y_probs = np.array([0.6, 0.3, 0.1])  # Probabilidades para classes 0, 1, 2
        y = np.random.choice(n_classes, size=n_samples, p=y_probs)
        
        # Cria correlação entre features e classes
        for i in range(n_classes):
            mask = (y == i)
            X[mask, 0] += i * 2  # Primeira feature tem correlação com a classe
            X[mask, 1] -= i * 1.5  # Segunda feature tem correlação negativa
        
        # Adiciona ruído
        X += np.random.randn(n_samples, n_features) * 0.5
        
        # Converte para DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        X_df['label'] = y
        
        logger.info(f"Gerados {len(X_df)} registros com {len(X_df.columns)} features")
        logger.info(f"Distribuição de classes: {pd.Series(y).value_counts().to_dict()}")
        
        return X_df
    
    def prepare_data_loaders(self, data=None, batch_size=32, test_size=0.2, val_size=0.25):
        """Prepara DataLoaders para treinamento com PyTorch
        
        Args:
            data: DataFrame com dados de treinamento
            batch_size: Tamanho do batch para treinamento
            test_size: Proporção dos dados para teste
            val_size: Proporção dos dados de treinamento para validação
            
        Returns:
            dict: Dicionário com DataLoaders para treinamento, validação e teste
        """
        try:
            if data is None:
                data = self._gerar_dados_simulados()
            
            logger.info(f"Preparando DataLoaders (batch_size={batch_size})")
            
            # Separa features e target
            X = data.drop('label', axis=1).values
            y = data['label'].values
            
            # Divide em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            
            # Divide treino em treino e validação
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, stratify=y_train, random_state=42
            )
            
            # Normaliza os dados
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
            
            # Cria datasets
            train_dataset = FinancialDataset(X_train, y_train)
            val_dataset = FinancialDataset(X_val, y_val)
            test_dataset = FinancialDataset(X_test, y_test)
            
            # Cria dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            logger.info(f"DataLoaders preparados: {len(train_dataset)} treino, {len(val_dataset)} validação, {len(test_dataset)} teste")
            
            return {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader,
                'input_size': X_train.shape[1],
                'num_classes': len(np.unique(y))
            }
            
        except Exception as e:
            logger.error(f"Erro ao preparar DataLoaders: {str(e)}")
            return None
    
    def train_model(self, loaders=None, params=None, epochs=10, early_stopping_patience=3):
        """Treina um modelo de rede neural
        
        Args:
            loaders: Dicionário com DataLoaders
            params: Dicionário com hiperparâmetros
            epochs: Número máximo de épocas para treinamento
            early_stopping_patience: Número de épocas sem melhoria para parar o treinamento
            
        Returns:
            Modelo treinado e histórico de treinamento
        """
        try:
            if loaders is None:
                loaders = self.prepare_data_loaders()
            
            if params is None:
                params = {
                    'hidden_size': 64,
                    'learning_rate': 0.001,
                    'weight_decay': 1e-5
                }
            
            input_size = loaders['input_size']
            num_classes = loaders['num_classes']
            
            logger.info(f"Treinando modelo com parâmetros: {params}")
            
            # Cria o modelo
            model = SimpleNN(input_size, params['hidden_size'], num_classes)
            
            # Define otimizador e função de perda
            optimizer = optim.Adam(
                model.parameters(), 
                lr=params['learning_rate'], 
                weight_decay=params['weight_decay']
            )
            criterion = nn.CrossEntropyLoss()
            
            # Histórico de treinamento
            history = {
                'train_loss': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            # Early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # Loop de treinamento
            for epoch in range(epochs):
                # Modo de treinamento
                model.train()
                train_loss = 0.0
                
                for inputs, targets in loaders['train']:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(loaders['train'])
                
                # Modo de avaliação
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in loaders['val']:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                
                val_loss /= len(loaders['val'])
                val_accuracy = correct / total
                
                # Atualiza histórico
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                logger.info(f"Época {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping na época {epoch+1}")
                        break
            
            # Carrega o melhor modelo
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            self.model = model
            return model, history
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {str(e)}")
            return None, None
    
    def evaluate_model(self, model=None, loader=None):
        """Avalia o modelo em um conjunto de dados
        
        Args:
            model: Modelo treinado
            loader: DataLoader com dados de teste
            
        Returns:
            dict: Dicionário com métricas de avaliação
        """
        try:
            if model is None:
                model = self.model
                
            if model is None:
                logger.error("Nenhum modelo disponível para avaliação")
                return None
                
            if loader is None:
                logger.info("Nenhum loader fornecido, gerando dados e preparando loader")
                loaders = self.prepare_data_loaders()
                loader = loaders['test']
            
            logger.info("Avaliando modelo")
            
            # Modo de avaliação
            model.eval()
            
            # Métricas
            correct = 0
            total = 0
            all_targets = []
            all_predictions = []
            
            with torch.no_grad():
                for inputs, targets in loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            
            # Calcula métricas
            accuracy = correct / total
            
            # Calcula matriz de confusão
            classes = np.unique(all_targets)
            confusion = np.zeros((len(classes), len(classes)), dtype=int)
            
            for t, p in zip(all_targets, all_predictions):
                confusion[t][p] += 1
            
            # Calcula precisão, recall e F1 por classe
            precision = {}
            recall = {}
            f1 = {}
            
            for i in range(len(classes)):
                true_positive = confusion[i][i]
                false_positive = sum(confusion[:, i]) - true_positive
                false_negative = sum(confusion[i, :]) - true_positive
                
                precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
                recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
            
            # Calcula médias
            macro_precision = sum(precision.values()) / len(precision)
            macro_recall = sum(recall.values()) / len(recall)
            macro_f1 = sum(f1.values()) / len(f1)
            
            # Cria dicionário de métricas
            metrics = {
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'confusion_matrix': confusion.tolist(),
                'class_precision': precision,
                'class_recall': recall,
                'class_f1': f1
            }
            
            logger.info(f"Avaliação concluída - Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
            
            self.metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao avaliar modelo: {str(e)}")
            return None
    
    def optimize_hyperparameters(self, n_trials=10):
        """Otimiza hiperparâmetros usando Optuna
        
        Args:
            n_trials: Número de tentativas para otimização
            
        Returns:
            dict: Melhores hiperparâmetros encontrados
        """
        try:
            logger.info(f"Iniciando otimização de hiperparâmetros com {n_trials} tentativas")
            
            # Prepara dados
            loaders = self.prepare_data_loaders()
            
            # Define função objetivo para Optuna
            def objective(trial):
                # Sugere hiperparâmetros
                params = {
                    'hidden_size': trial.suggest_int('hidden_size', 32, 256),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
                }
                
                # Treina modelo com os hiperparâmetros sugeridos
                model, history = self.train_model(loaders, params, epochs=5)
                
                # Avalia modelo
                metrics = self.evaluate_model(model, loaders['val'])
                
                # Retorna métrica a ser otimizada
                return metrics['macro_f1']
            
            # Cria estudo Optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Obtém melhores hiperparâmetros
            best_params = study.best_params
            best_value = study.best_value
            
            logger.info(f"Otimização concluída - Melhores parâmetros: {best_params}, Melhor F1: {best_value:.4f}")
            
            self.best_params = best_params
            return best_params
            
        except Exception as e:
            logger.error(f"Erro na otimização de hiperparâmetros: {str(e)}")
            return None

def teste_dataloaders():
    """Testa a preparação de DataLoaders para treinamento"""
    logger.info("Iniciando teste de DataLoaders")
    
    # Inicializa a classe simulada
    treinamento = TreinamentoIASimulado()
    
    # Gera dados simulados
    dados = treinamento._gerar_dados_simulados(n_samples=500)
    
    # Prepara DataLoaders
    loaders = treinamento.prepare_data_loaders(dados, batch_size=32)
    
    if loaders is None:
        logger.error("Falha ao preparar DataLoaders")
        return False
    
    # Verifica se todos os loaders foram criados
    for key in ['train', 'val', 'test']:
        if key not in loaders:
            logger.error(f"Loader '{key}' não encontrado")
            return False
    
    # Verifica tamanho dos batches
    for batch_x, batch_y in loaders['train']:
        logger.info(f"Batch de treino: {batch_x.shape}, {batch_y.shape}")
        break
    
    logger.info("DataLoaders preparados com sucesso")
    return True

def teste_treinamento_modelo():
    """Testa o treinamento de um modelo de rede neural"""
    logger.info("Iniciando teste de treinamento de modelo")
    
    # Inicializa a classe simulada
    treinamento = TreinamentoIASimulado()
    
    # Gera dados simulados
    dados = treinamento._gerar_dados_simulados(n_samples=500)
    
    # Prepara DataLoaders
    loaders = treinamento.prepare_data_loaders(dados, batch_size=32)
    
    if loaders is None:
        logger.error("Falha ao preparar DataLoaders")
        return False
    
    # Define hiperparâmetros
    params = {
        'hidden_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }
    
    # Treina o modelo
    modelo, historico = treinamento.train_model(loaders, params, epochs=5)
    
    if modelo is None or historico is None:
        logger.error("Falha ao treinar modelo")
        return False
    
    # Verifica se o histórico foi registrado
    for key in ['train_loss', 'val_loss', 'val_accuracy']:
        if key not in historico:
            logger.error(f"Métrica '{key}' não encontrada no histórico")
            return False
    
    logger.info(f"Histórico de treinamento: {len(historico['train_loss'])} épocas")
    logger.info(f"Acurácia final: {historico['val_accuracy'][-1]:.4f}")
    
    # Avalia o modelo
    metricas = treinamento.evaluate_model(modelo, loaders['test'])
    
    if metricas is None:
        logger.error("Falha ao avaliar modelo")
        return False
    
    logger.info(f"Acurácia no teste: {metricas['accuracy']:.4f}")
    
    logger.info("Treinamento de modelo concluído com sucesso")
    return True

def teste_early_stopping():
    """Testa o early stopping durante o treinamento"""
    logger.info("Iniciando teste de early stopping")
    
    # Inicializa a classe simulada
    treinamento = TreinamentoIASimulado()
    
    # Gera dados simulados
    dados = treinamento._gerar_dados_simulados(n_samples=500)
    
    # Prepara DataLoaders
    loaders = treinamento.prepare_data_loaders(dados, batch_size=32)
    
    if loaders is None:
        logger.error("Falha ao preparar DataLoaders")
        return False
    
    # Define hiperparâmetros
    params = {
        'hidden_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5
    }
    
    # Treina o modelo com early stopping
    modelo, historico = treinamento.train_model(loaders, params, epochs=20, early_stopping_patience=2)
    
    if modelo is None or historico is None:
        logger.error("Falha ao treinar modelo")
        return False
    
    # Verifica se o early stopping funcionou
    if len(historico['train_loss']) == 20:
        logger.warning("Early stopping não foi acionado")
    else:
        logger.info(f"Early stopping acionado após {len(historico['train_loss'])} épocas")
    
    logger.info("Teste de early stopping concluído com sucesso")
    return True

def teste_otimizacao_hiperparametros():
    """Testa a otimização de hiperparâmetros usando Optuna"""
    logger.info("Iniciando teste de otimização de hiperparâmetros")
    
    # Inicializa a classe simulada
    treinamento = TreinamentoIASimulado()
    
    # Gera dados simulados (amostra pequena para teste rápido)
    dados = treinamento._gerar_dados_simulados(n_samples=300)
    
    # Prepara DataLoaders
    loaders = treinamento.prepare_data_loaders(dados, batch_size=32)
    treinamento.loaders = loaders
    
    if loaders is None:
        logger.error("Falha ao preparar DataLoaders")
        return False
    
    # Otimiza hiperparâmetros (com poucas tentativas para teste)
    melhores_params = treinamento.optimize_hyperparameters(n_trials=3)
    
    if melhores_params is None:
        logger.error("Falha na otimização de hiperparâmetros")
        return False
    
    logger.info(f"Melhores hiperparâmetros: {melhores_params}")
    
    # Treina modelo com os melhores hiperparâmetros
    modelo, historico = treinamento.train_model(loaders, melhores_params, epochs=5)
    
    if modelo is None:
        logger.error("Falha ao treinar modelo com os melhores hiperparâmetros")
        return False
    
    # Avalia o modelo
    metricas = treinamento.evaluate_model(modelo, loaders['test'])
    
    if metricas is None:
        logger.error("Falha ao avaliar modelo")
        return False
    
    logger.info(f"Acurácia com hiperparâmetros otimizados: {metricas['accuracy']:.4f}")
    
    logger.info("Otimização de hiperparâmetros concluída com sucesso")
    return True

def teste_pipeline_treinamento_completo():
    """Testa o pipeline completo de treinamento da IA"""
    logger.info("Iniciando teste do pipeline completo de treinamento")
    
    # Inicializa a classe simulada
    treinamento = TreinamentoIASimulado()
    
    # Gera dados simulados
    logger.info("Gerando dados simulados")
    dados = treinamento._gerar_dados_simulados(n_samples=500)
    
    # Prepara DataLoaders
    logger.info("Preparando DataLoaders")
    loaders = treinamento.prepare_data_loaders(dados, batch_size=32)
    
    if loaders is None:
        logger.error("Falha ao preparar DataLoaders")
        return False
    
    # Otimiza hiperparâmetros (com poucas tentativas para teste)
    logger.info("Otimizando hiperparâmetros")
    melhores_params = treinamento.optimize_hyperparameters(n_trials=3)
    
    if melhores_params is None:
        logger.error("Falha na otimização de hiperparâmetros")
        return False
    
    # Treina modelo com os melhores hiperparâmetros
    logger.info("Treinando modelo com os melhores hiperparâmetros")
    modelo, historico = treinamento.train_model(loaders, melhores_params, epochs=10)
    
    if modelo is None:
        logger.error("Falha ao treinar modelo")
        return False
    
    # Avalia o modelo
    logger.info("Avaliando modelo")
    metricas = treinamento.evaluate_model(modelo, loaders['test'])
    
    if metricas is None:
        logger.error("Falha ao avaliar modelo")
        return False
    
    # Salva métricas
    logger.info("Salvando métricas")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pipeline_treinamento_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'best_params': melhores_params,
            'metrics': metricas,
            'training_history': {
                'train_loss': [float(x) for x in historico['train_loss']],
                'val_loss': [float(x) for x in historico['val_loss']],
                'val_accuracy': [float(x) for x in historico['val_accuracy']]
            }
        }, f, indent=4)
    
    logger.info(f"Pipeline completo executado com sucesso. Resultados salvos em {filename}")
    logger.info(f"Acurácia final: {metricas['accuracy']:.4f}, Macro F1: {metricas['macro_f1']:.4f}")
    
    return True

if __name__ == "__main__":
    logger.info("==== INICIANDO TESTES DE TREINAMENTO DA IA ====")
    
    testes = [
        ("DataLoaders", teste_dataloaders),
        ("Treinamento de Modelo", teste_treinamento_modelo),
        ("Early Stopping", teste_early_stopping),
        ("Otimização de Hiperparâmetros", teste_otimizacao_hiperparametros),
        ("Pipeline Completo de Treinamento", teste_pipeline_treinamento_completo)
    ]
    
    resultados = {}
    
    for nome, teste in testes:
        logger.info(f"\n==== TESTE: {nome} ====")
        try:
            resultado = teste()
            resultados[nome] = "SUCESSO" if resultado else "FALHA"
        except Exception as e:
            logger.error(f"Erro durante o teste {nome}: {str(e)}")
            resultados[nome] = "ERRO"
    
    logger.info("\n==== RESUMO DOS TESTES ====")
    for nome, resultado in resultados.items():
        logger.info(f"{nome}: {resultado}")
