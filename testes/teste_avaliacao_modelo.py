import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teste_avaliacao_modelo.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TesteAvaliacaoModelo')

class AvaliacaoModeloSimulada:
    """Classe para testes de avaliação de modelos"""
    
    def __init__(self):
        self.logger = logger
        self.model = None
        self.metrics = {}
        
    def _gerar_dados_simulados(self, n_samples=1000, n_features=10, n_classes=3):
        """Gera dados simulados para testes de avaliação de modelos"""
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
    
    def treinar_modelo_simulado(self, data=None):
        """Treina um modelo simulado para avaliação
        
        Args:
            data: DataFrame com dados de treinamento (opcional)
            
        Returns:
            Modelo treinado
        """
        try:
            if data is None:
                data = self._gerar_dados_simulados()
            
            logger.info("Treinando modelo simulado")
            
            # Separa features e target
            X = data.drop('label', axis=1)
            y = data['label']
            
            # Normaliza dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Treina um modelo RandomForest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            self.model = model
            logger.info("Modelo treinado com sucesso")
            
            return model
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {str(e)}")
            return None
    
    def evaluate_model(self, model=None, test_data=None):
        """Avalia o desempenho do modelo
        
        Args:
            model: Modelo treinado (opcional, usa self.model se None)
            test_data: DataFrame com dados de teste (opcional)
            
        Returns:
            dict: Dicionário com métricas de avaliação
        """
        try:
            if model is None:
                if self.model is None:
                    logger.info("Nenhum modelo disponível, treinando um novo")
                    self.treinar_modelo_simulado()
                model = self.model
            
            if test_data is None:
                logger.info("Gerando dados de teste")
                test_data = self._gerar_dados_simulados(n_samples=300)
            
            logger.info("Avaliando modelo")
            
            # Separa features e target
            X_test = test_data.drop('label', axis=1)
            y_test = test_data['label']
            
            # Normaliza dados
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)
            
            # Faz previsões
            y_pred = model.predict(X_test_scaled)
            
            # Calcula métricas
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
                'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Calcula métricas por classe
            for i in range(len(np.unique(y_test))):
                metrics[f'precision_class_{i}'] = precision_score(y_test, y_pred, labels=[i], average=None)[0]
                metrics[f'recall_class_{i}'] = recall_score(y_test, y_pred, labels=[i], average=None)[0]
                metrics[f'f1_class_{i}'] = f1_score(y_test, y_pred, labels=[i], average=None)[0]
            
            # Calcula matriz de confusão
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Adiciona matriz de confusão ao dicionário de métricas
            metrics['confusion_matrix'] = conf_matrix.tolist()
            
            self.metrics = metrics
            logger.info(f"Métricas calculadas: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao avaliar modelo: {str(e)}")
            return None
    
    def save_model_metrics(self, metrics=None, filename=None):
        """Salva as métricas de avaliação do modelo em formato JSON
        
        Args:
            metrics: Dicionário com métricas (opcional, usa self.metrics se None)
            filename: Nome do arquivo para salvar as métricas (opcional)
            
        Returns:
            bool: True se as métricas foram salvas com sucesso, False caso contrário
        """
        try:
            if metrics is None:
                if not self.metrics:
                    logger.error("Nenhuma métrica disponível para salvar")
                    return False
                metrics = self.metrics
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_metrics_{timestamp}.json"
            
            logger.info(f"Salvando métricas em {filename}")
            
            # Converte numpy arrays para listas
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                else:
                    metrics_json[key] = value
            
            # Salva as métricas em formato JSON
            with open(filename, 'w') as f:
                json.dump(metrics_json, f, indent=4)
            
            logger.info(f"Métricas salvas com sucesso em {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {str(e)}")
            return False

def teste_avaliacao_modelo():
    """Testa a avaliação de modelos"""
    logger.info("Iniciando teste de avaliação de modelos")
    
    # Inicializa a classe simulada
    avaliacao = AvaliacaoModeloSimulada()
    
    # Gera dados de treinamento
    dados_treino = avaliacao._gerar_dados_simulados(n_samples=800)
    
    # Treina o modelo
    modelo = avaliacao.treinar_modelo_simulado(dados_treino)
    
    if modelo is None:
        logger.error("Falha ao treinar modelo")
        return False
    
    # Gera dados de teste
    dados_teste = avaliacao._gerar_dados_simulados(n_samples=200)
    
    # Avalia o modelo
    metricas = avaliacao.evaluate_model(modelo, dados_teste)
    
    if metricas is None:
        logger.error("Falha ao avaliar modelo")
        return False
    
    # Verifica se as métricas foram calculadas corretamente
    metricas_esperadas = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                          'precision_weighted', 'recall_weighted', 'f1_weighted',
                          'confusion_matrix']
    
    for metrica in metricas_esperadas:
        if metrica not in metricas:
            logger.error(f"Métrica esperada não encontrada: {metrica}")
            return False
    
    logger.info("Avaliação de modelo concluída com sucesso")
    return True

def teste_salvar_metricas():
    """Testa o salvamento de métricas de avaliação"""
    logger.info("Iniciando teste de salvamento de métricas")
    
    # Inicializa a classe simulada
    avaliacao = AvaliacaoModeloSimulada()
    
    # Treina e avalia o modelo
    avaliacao.treinar_modelo_simulado()
    metricas = avaliacao.evaluate_model()
    
    if metricas is None:
        logger.error("Falha ao avaliar modelo")
        return False
    
    # Salva as métricas
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"teste_metricas_{timestamp}.json"
    resultado = avaliacao.save_model_metrics(metricas, filename)
    
    if not resultado:
        logger.error("Falha ao salvar métricas")
        return False
    
    # Verifica se o arquivo foi criado
    if not os.path.exists(filename):
        logger.error(f"Arquivo de métricas não encontrado: {filename}")
        return False
    
    # Lê o arquivo para verificar o conteúdo
    try:
        with open(filename, 'r') as f:
            metricas_salvas = json.load(f)
        
        # Verifica se as métricas foram salvas corretamente
        for key in metricas:
            if key not in metricas_salvas:
                logger.error(f"Métrica não encontrada no arquivo: {key}")
                return False
        
        logger.info("Métricas salvas e verificadas com sucesso")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao verificar métricas salvas: {str(e)}")
        return False

def teste_avaliacao_por_classe():
    """Testa a avaliação de modelos por classe"""
    logger.info("Iniciando teste de avaliação por classe")
    
    # Inicializa a classe simulada
    avaliacao = AvaliacaoModeloSimulada()
    
    # Gera dados com distribuição desigual
    dados = avaliacao._gerar_dados_simulados(n_samples=1000, n_classes=3)
    
    # Treina o modelo
    modelo = avaliacao.treinar_modelo_simulado(dados)
    
    if modelo is None:
        logger.error("Falha ao treinar modelo")
        return False
    
    # Avalia o modelo
    metricas = avaliacao.evaluate_model(modelo, dados)
    
    if metricas is None:
        logger.error("Falha ao avaliar modelo")
        return False
    
    # Verifica se as métricas por classe foram calculadas
    for i in range(3):  # 3 classes
        for metrica in ['precision', 'recall', 'f1']:
            chave = f'{metrica}_class_{i}'
            if chave not in metricas:
                logger.error(f"Métrica por classe não encontrada: {chave}")
                return False
    
    logger.info("Avaliação por classe concluída com sucesso")
    return True

def teste_pipeline_avaliacao_completo():
    """Testa o pipeline completo de avaliação de modelos"""
    logger.info("Iniciando teste do pipeline completo de avaliação")
    
    # Inicializa a classe simulada
    avaliacao = AvaliacaoModeloSimulada()
    
    # Gera dados
    logger.info("Gerando dados simulados")
    dados = avaliacao._gerar_dados_simulados(n_samples=1000)
    
    # Divide em treino e teste (80/20)
    indices = np.random.permutation(len(dados))
    n_treino = int(len(dados) * 0.8)
    dados_treino = dados.iloc[indices[:n_treino]]
    dados_teste = dados.iloc[indices[n_treino:]]
    
    logger.info(f"Dados divididos: {len(dados_treino)} treino, {len(dados_teste)} teste")
    
    # Treina o modelo
    logger.info("Treinando modelo")
    modelo = avaliacao.treinar_modelo_simulado(dados_treino)
    
    if modelo is None:
        logger.error("Falha ao treinar modelo")
        return False
    
    # Avalia o modelo
    logger.info("Avaliando modelo")
    metricas = avaliacao.evaluate_model(modelo, dados_teste)
    
    if metricas is None:
        logger.error("Falha ao avaliar modelo")
        return False
    
    # Salva as métricas
    logger.info("Salvando métricas")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pipeline_metricas_{timestamp}.json"
    resultado = avaliacao.save_model_metrics(metricas, filename)
    
    if not resultado:
        logger.error("Falha ao salvar métricas")
        return False
    
    logger.info(f"Pipeline completo executado com sucesso. Métricas salvas em {filename}")
    
    # Exibe algumas métricas importantes
    logger.info(f"Acurácia: {metricas['accuracy']:.4f}")
    logger.info(f"F1 (macro): {metricas['f1_macro']:.4f}")
    logger.info(f"F1 (weighted): {metricas['f1_weighted']:.4f}")
    
    return True

if __name__ == "__main__":
    logger.info("==== INICIANDO TESTES DE AVALIAÇÃO DE MODELOS ====")
    
    testes = [
        ("Avaliação de Modelo", teste_avaliacao_modelo),
        ("Salvamento de Métricas", teste_salvar_metricas),
        ("Avaliação por Classe", teste_avaliacao_por_classe),
        ("Pipeline Completo de Avaliação", teste_pipeline_avaliacao_completo)
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
