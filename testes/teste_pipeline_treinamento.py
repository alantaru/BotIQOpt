import os
import sys
import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teste_pipeline_treinamento.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TestePipelineTreinamento')

# Simulação da classe Inteligencia para testes de pipeline de treinamento
class InteligenciaSimulada:
    """Versão simplificada da classe Inteligencia para testes de pipeline de treinamento"""
    
    def __init__(self):
        self.logger = logger
        self.historical_data = None
        self.processed_data = None
        
    def _gerar_dados_simulados(self, n_samples=1000):
        """Gera dados simulados para testes"""
        logger.info(f"Gerando {n_samples} registros de dados simulados")
        
        # Gera timestamps
        timestamps = np.arange(n_samples) * 60  # 1 minuto entre cada registro
        
        # Gera preços simulados
        base_price = 1.1
        random_walk = np.cumsum(np.random.normal(0, 0.001, n_samples))
        
        # Cria DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': base_price + random_walk,
            'close': base_price + random_walk + np.random.normal(0, 0.0005, n_samples),
            'high': base_price + random_walk + np.abs(np.random.normal(0, 0.001, n_samples)),
            'low': base_price + random_walk - np.abs(np.random.normal(0, 0.001, n_samples)),
            'volume': np.random.randint(100, 1000, n_samples)
        })
        
        # Garante que high >= open, close e low <= open, close
        for i in range(len(df)):
            df.loc[i, 'high'] = max(df.loc[i, 'high'], df.loc[i, 'open'], df.loc[i, 'close'])
            df.loc[i, 'low'] = min(df.loc[i, 'low'], df.loc[i, 'open'], df.loc[i, 'close'])
        
        # Adiciona indicadores técnicos básicos
        df['sma_5'] = df['close'].rolling(window=5).mean().fillna(0)
        df['sma_20'] = df['close'].rolling(window=20).mean().fillna(0)
        df['rsi_14'] = (df['close'] - df['close'].shift(1)).rolling(window=14).apply(
            lambda x: 100 - (100 / (1 + (sum(y for y in x if y > 0) / -sum(y for y in x if y < 0))))
        ).fillna(0)
        
        # Adiciona mais alguns indicadores
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['close'].ewm(span=26, adjust=False).mean()
        df['atr'] = (df['high'] - df['low']).rolling(window=14).mean().fillna(0)
        df['bollinger_upper'] = df['sma_20'] + 2 * df['close'].rolling(window=20).std().fillna(0)
        df['bollinger_lower'] = df['sma_20'] - 2 * df['close'].rolling(window=20).std().fillna(0)
        
        # Cria rótulos simulados (para classificação)
        future_returns = df['close'].shift(-5) / df['close'] - 1
        df['label'] = 1  # Hold (classe padrão)
        df.loc[future_returns > 0.005, 'label'] = 2  # Buy
        df.loc[future_returns < -0.005, 'label'] = 0  # Sell
        
        # Preenche valores NaN
        df = df.fillna(0)
        
        self.historical_data = df
        logger.info(f"Gerados {len(df)} registros com {len(df.columns)} features")
        
        return df
    
    def select_important_features(self, data=None, target='label', n_features=5):
        """Seleciona as features mais importantes para o treinamento usando RandomForest.
        
        Args:
            data: DataFrame com dados pré-processados
            target: Nome da coluna alvo
            n_features: Número de features a serem selecionadas
            
        Returns:
            DataFrame com as features mais importantes
        """
        try:
            if data is None:
                if self.historical_data is None:
                    self.historical_data = self._gerar_dados_simulados()
                data = self.historical_data.copy()
            
            logger.info(f"Selecionando as {n_features} features mais importantes")
            
            # Remove colunas não numéricas e a coluna alvo
            X = data.select_dtypes(include=['number'])
            if target in X.columns:
                y = X[target].copy()
                X = X.drop(columns=[target])
            else:
                logger.warning(f"Coluna alvo {target} não encontrada, usando rótulos simulados")
                y = np.random.randint(0, 3, len(X))  # Rótulos simulados (0, 1, 2)
            
            # Treina um RandomForest para selecionar features
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Obtém importância das features
            feature_importances = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Seleciona as n_features mais importantes
            top_features = feature_importances.head(n_features)['feature'].tolist()
            
            logger.info(f"Features mais importantes: {top_features}")
            
            # Retorna DataFrame apenas com as features selecionadas e o target
            selected_columns = top_features + [target] if target in data.columns else top_features
            selected_data = data[selected_columns].copy()
            
            return selected_data, top_features
            
        except Exception as e:
            logger.error(f"Erro ao selecionar features importantes: {str(e)}")
            return data, []
    
    def engineer_advanced_features(self, data=None):
        """Cria features avançadas para melhorar o desempenho do modelo.
        
        Args:
            data: DataFrame com dados pré-processados
            
        Returns:
            DataFrame com features avançadas adicionadas
        """
        try:
            if data is None:
                if self.historical_data is None:
                    self.historical_data = self._gerar_dados_simulados()
                data = self.historical_data.copy()
            
            logger.info("Criando features avançadas")
            
            # Cria DataFrame para as novas features
            df = data.copy()
            
            # 1. Interações entre indicadores
            if 'sma_5' in df.columns and 'sma_20' in df.columns:
                # Cruzamento de médias móveis
                df['sma_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
                # Distância entre médias móveis
                df['sma_distance'] = (df['sma_5'] - df['sma_20']) / df['close']
            
            # 2. Features de tendência
            if 'close' in df.columns:
                # Direção da tendência (últimos 5 períodos)
                df['trend_5'] = (df['close'] > df['close'].shift(5)).astype(int)
                # Força da tendência
                df['trend_strength'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
            
            # 3. Features de volatilidade
            if 'high' in df.columns and 'low' in df.columns:
                # Volatilidade relativa
                df['volatility'] = (df['high'] - df['low']) / df['close']
                # Volatilidade média (14 períodos)
                df['volatility_14'] = df['volatility'].rolling(window=14).mean().fillna(0)
            
            # 4. Features de momentum
            if 'close' in df.columns:
                # Rate of Change (ROC)
                df['roc_5'] = (df['close'] / df['close'].shift(5) - 1)
                df['roc_10'] = (df['close'] / df['close'].shift(10) - 1)
                
                # Aceleração do preço
                df['price_acceleration'] = df['roc_5'] - df['roc_5'].shift(5)
            
            # 5. Features baseadas em Bollinger Bands
            if 'bollinger_upper' in df.columns and 'bollinger_lower' in df.columns:
                # Posição relativa nas Bandas de Bollinger
                df['bb_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
                # Squeeze das Bandas de Bollinger
                df['bb_squeeze'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['sma_20']
            
            # Preenche valores NaN
            df = df.fillna(0)
            
            logger.info(f"Criadas {len(df.columns) - len(data.columns)} novas features avançadas")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao criar features avançadas: {str(e)}")
            return data
    
    def balance_classes(self, data=None, target='label', method='oversampling'):
        """Balanceia as classes para evitar viés no treinamento.
        
        Args:
            data: DataFrame com dados pré-processados
            target: Nome da coluna alvo
            method: Método de balanceamento ('oversampling', 'undersampling')
            
        Returns:
            DataFrame com classes balanceadas
        """
        try:
            if data is None:
                if self.historical_data is None:
                    self.historical_data = self._gerar_dados_simulados()
                data = self.historical_data.copy()
            
            if target not in data.columns:
                logger.error(f"Coluna alvo {target} não encontrada")
                return data
            
            logger.info(f"Balanceando classes usando método: {method}")
            
            # Obtém contagem de cada classe
            class_counts = data[target].value_counts()
            logger.info(f"Distribuição original das classes: {class_counts.to_dict()}")
            
            # Separa os dados por classe
            class_dfs = {}
            for class_value in class_counts.index:
                class_dfs[class_value] = data[data[target] == class_value]
            
            balanced_data = None
            
            if method == 'oversampling':
                # Oversampling: aumenta as classes minoritárias
                max_class_size = class_counts.max()
                
                balanced_samples = []
                for class_value, class_df in class_dfs.items():
                    if len(class_df) < max_class_size:
                        # Calcula quantas amostras adicionais são necessárias
                        n_samples = max_class_size - len(class_df)
                        # Amostragem com reposição
                        additional_samples = class_df.sample(n=n_samples, replace=True)
                        balanced_samples.append(pd.concat([class_df, additional_samples]))
                    else:
                        balanced_samples.append(class_df)
                
                balanced_data = pd.concat(balanced_samples)
                
            elif method == 'undersampling':
                # Undersampling: reduz as classes majoritárias
                min_class_size = class_counts.min()
                
                balanced_samples = []
                for class_value, class_df in class_dfs.items():
                    if len(class_df) > min_class_size:
                        # Amostragem sem reposição
                        balanced_samples.append(class_df.sample(n=min_class_size, replace=False))
                    else:
                        balanced_samples.append(class_df)
                
                balanced_data = pd.concat(balanced_samples)
            
            else:
                logger.error(f"Método de balanceamento desconhecido: {method}")
                return data
            
            # Embaralha os dados
            balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
            
            # Verifica a nova distribuição
            new_class_counts = balanced_data[target].value_counts()
            logger.info(f"Nova distribuição das classes: {new_class_counts.to_dict()}")
            
            return balanced_data
            
        except Exception as e:
            logger.error(f"Erro ao balancear classes: {str(e)}")
            return data
    
    def split_data(self, data=None, test_size=0.2, val_size=0.2, random_split=False):
        """Divide os dados em conjuntos de treinamento, validação e teste.
        
        Args:
            data: DataFrame com dados pré-processados
            test_size: Proporção dos dados para teste
            val_size: Proporção dos dados para validação
            random_split: Se True, divide aleatoriamente, senão divide cronologicamente
            
        Returns:
            dict: Dicionário com conjuntos de dados divididos
        """
        try:
            if data is None:
                if self.historical_data is None:
                    self.historical_data = self._gerar_dados_simulados()
                data = self.historical_data.copy()
            
            logger.info(f"Dividindo dados (test_size={test_size}, val_size={val_size}, random_split={random_split})")
            
            if random_split:
                # Divisão aleatória
                indices = np.random.permutation(len(data))
                test_count = int(len(data) * test_size)
                val_count = int(len(data) * val_size)
                train_count = len(data) - test_count - val_count
                
                train_indices = indices[:train_count]
                val_indices = indices[train_count:train_count + val_count]
                test_indices = indices[train_count + val_count:]
                
                train_data = data.iloc[train_indices].copy()
                val_data = data.iloc[val_indices].copy()
                test_data = data.iloc[test_indices].copy()
                
            else:
                # Divisão cronológica
                data = data.sort_values('timestamp') if 'timestamp' in data.columns else data
                
                test_count = int(len(data) * test_size)
                val_count = int(len(data) * val_size)
                train_count = len(data) - test_count - val_count
                
                train_data = data.iloc[:train_count].copy()
                val_data = data.iloc[train_count:train_count + val_count].copy()
                test_data = data.iloc[train_count + val_count:].copy()
            
            logger.info(f"Divisão concluída: {len(train_data)} treino, {len(val_data)} validação, {len(test_data)} teste")
            
            return {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
        except Exception as e:
            logger.error(f"Erro ao dividir dados: {str(e)}")
            return None

def teste_selecao_features():
    """Testa a seleção de features importantes"""
    logger.info("Iniciando teste de seleção de features importantes")
    
    # Inicializa a classe simulada
    inteligencia = InteligenciaSimulada()
    
    # Gera dados simulados
    dados = inteligencia._gerar_dados_simulados(n_samples=500)
    
    # Seleciona features importantes
    dados_selecionados, features_importantes = inteligencia.select_important_features(dados, n_features=5)
    
    if dados_selecionados is None or dados_selecionados.empty:
        logger.error("Falha ao selecionar features importantes")
        return False
    
    logger.info(f"Features selecionadas: {features_importantes}")
    
    # Verifica se o número correto de features foi selecionado
    if len(features_importantes) != 5:
        logger.error(f"Número incorreto de features selecionadas: {len(features_importantes)}")
        return False
    
    logger.info("Seleção de features concluída com sucesso")
    return True

def teste_engenharia_features():
    """Testa a engenharia de features avançadas"""
    logger.info("Iniciando teste de engenharia de features avançadas")
    
    # Inicializa a classe simulada
    inteligencia = InteligenciaSimulada()
    
    # Gera dados simulados
    dados = inteligencia._gerar_dados_simulados(n_samples=500)
    
    # Cria features avançadas
    dados_avancados = inteligencia.engineer_advanced_features(dados)
    
    if dados_avancados is None or dados_avancados.empty:
        logger.error("Falha ao criar features avançadas")
        return False
    
    # Verifica se novas features foram adicionadas
    novas_features = set(dados_avancados.columns) - set(dados.columns)
    logger.info(f"Novas features criadas: {novas_features}")
    
    if len(novas_features) == 0:
        logger.error("Nenhuma nova feature foi criada")
        return False
    
    # Verifica algumas features específicas
    features_esperadas = ['sma_cross', 'trend_5', 'volatility']
    for feature in features_esperadas:
        if feature not in dados_avancados.columns:
            logger.error(f"Feature esperada não encontrada: {feature}")
            return False
    
    logger.info("Engenharia de features concluída com sucesso")
    return True

def teste_balanceamento_classes():
    """Testa o balanceamento de classes"""
    logger.info("Iniciando teste de balanceamento de classes")
    
    # Inicializa a classe simulada
    inteligencia = InteligenciaSimulada()
    
    # Gera dados simulados
    dados = inteligencia._gerar_dados_simulados(n_samples=500)
    
    # Verifica distribuição original
    distribuicao_original = dados['label'].value_counts()
    logger.info(f"Distribuição original: {distribuicao_original.to_dict()}")
    
    # Testa oversampling
    dados_oversampling = inteligencia.balance_classes(dados, method='oversampling')
    
    if dados_oversampling is None or dados_oversampling.empty:
        logger.error("Falha ao realizar oversampling")
        return False
    
    distribuicao_oversampling = dados_oversampling['label'].value_counts()
    logger.info(f"Distribuição após oversampling: {distribuicao_oversampling.to_dict()}")
    
    # Verifica se as classes estão balanceadas
    if len(set(distribuicao_oversampling.values)) > 1:
        logger.warning("Classes não estão perfeitamente balanceadas após oversampling")
    
    # Testa undersampling
    dados_undersampling = inteligencia.balance_classes(dados, method='undersampling')
    
    if dados_undersampling is None or dados_undersampling.empty:
        logger.error("Falha ao realizar undersampling")
        return False
    
    distribuicao_undersampling = dados_undersampling['label'].value_counts()
    logger.info(f"Distribuição após undersampling: {distribuicao_undersampling.to_dict()}")
    
    # Verifica se as classes estão balanceadas
    if len(set(distribuicao_undersampling.values)) > 1:
        logger.warning("Classes não estão perfeitamente balanceadas após undersampling")
    
    logger.info("Balanceamento de classes concluído com sucesso")
    return True

def teste_divisao_dados():
    """Testa a divisão de dados em conjuntos de treinamento, validação e teste"""
    logger.info("Iniciando teste de divisão de dados")
    
    # Inicializa a classe simulada
    inteligencia = InteligenciaSimulada()
    
    # Gera dados simulados
    dados = inteligencia._gerar_dados_simulados(n_samples=1000)
    
    # Testa divisão aleatória
    logger.info("Testando divisão aleatória")
    conjuntos_aleatorios = inteligencia.split_data(dados, test_size=0.2, val_size=0.2, random_split=True)
    
    if conjuntos_aleatorios is None:
        logger.error("Falha ao dividir dados aleatoriamente")
        return False
    
    # Verifica tamanhos dos conjuntos
    tamanho_treino = len(conjuntos_aleatorios['train'])
    tamanho_val = len(conjuntos_aleatorios['val'])
    tamanho_teste = len(conjuntos_aleatorios['test'])
    
    logger.info(f"Divisão aleatória: {tamanho_treino} treino, {tamanho_val} validação, {tamanho_teste} teste")
    
    # Testa divisão cronológica
    logger.info("Testando divisão cronológica")
    conjuntos_cronologicos = inteligencia.split_data(dados, test_size=0.2, val_size=0.2, random_split=False)
    
    if conjuntos_cronologicos is None:
        logger.error("Falha ao dividir dados cronologicamente")
        return False
    
    # Verifica tamanhos dos conjuntos
    tamanho_treino = len(conjuntos_cronologicos['train'])
    tamanho_val = len(conjuntos_cronologicos['val'])
    tamanho_teste = len(conjuntos_cronologicos['test'])
    
    logger.info(f"Divisão cronológica: {tamanho_treino} treino, {tamanho_val} validação, {tamanho_teste} teste")
    
    # Verifica proporções
    total = len(dados)
    proporcao_treino = tamanho_treino / total
    proporcao_val = tamanho_val / total
    proporcao_teste = tamanho_teste / total
    
    logger.info(f"Proporções: {proporcao_treino:.2f} treino, {proporcao_val:.2f} validação, {proporcao_teste:.2f} teste")
    
    # Verifica se as proporções estão próximas do esperado
    if abs(proporcao_treino - 0.6) > 0.05 or abs(proporcao_val - 0.2) > 0.05 or abs(proporcao_teste - 0.2) > 0.05:
        logger.warning("Proporções dos conjuntos estão fora do esperado")
    
    logger.info("Divisão de dados concluída com sucesso")
    return True

def teste_pipeline_completo():
    """Testa o pipeline completo de processamento de dados para treinamento"""
    logger.info("Iniciando teste do pipeline completo de processamento de dados")
    
    # Inicializa a classe simulada
    inteligencia = InteligenciaSimulada()
    
    # Gera dados simulados
    logger.info("Gerando dados simulados")
    dados = inteligencia._gerar_dados_simulados(n_samples=1000)
    
    # Engenharia de features
    logger.info("Aplicando engenharia de features")
    dados_avancados = inteligencia.engineer_advanced_features(dados)
    
    # Seleção de features
    logger.info("Selecionando features importantes")
    dados_selecionados, features_importantes = inteligencia.select_important_features(dados_avancados, n_features=10)
    
    # Balanceamento de classes
    logger.info("Balanceando classes")
    dados_balanceados = inteligencia.balance_classes(dados_selecionados, method='oversampling')
    
    # Divisão de dados
    logger.info("Dividindo dados")
    conjuntos = inteligencia.split_data(dados_balanceados, random_split=False)
    
    if conjuntos is None:
        logger.error("Falha no pipeline de processamento de dados")
        return False
    
    # Verifica resultados
    logger.info(f"Pipeline completo executado com sucesso")
    logger.info(f"Features selecionadas: {features_importantes}")
    logger.info(f"Conjunto de treino: {len(conjuntos['train'])} registros")
    logger.info(f"Conjunto de validação: {len(conjuntos['val'])} registros")
    logger.info(f"Conjunto de teste: {len(conjuntos['test'])} registros")
    
    # Salva os dados processados para análise
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conjuntos['train'].to_csv(f"dados_treino_{timestamp}.csv", index=False)
    logger.info(f"Dados de treino salvos em dados_treino_{timestamp}.csv")
    
    return True

if __name__ == "__main__":
    logger.info("==== INICIANDO TESTES DE PIPELINE DE TREINAMENTO ====")
    
    testes = [
        ("Seleção de Features", teste_selecao_features),
        ("Engenharia de Features", teste_engenharia_features),
        ("Balanceamento de Classes", teste_balanceamento_classes),
        ("Divisão de Dados", teste_divisao_dados),
        ("Pipeline Completo", teste_pipeline_completo)
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
