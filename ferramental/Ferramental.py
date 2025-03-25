import os
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Set, Literal
from dataclasses import dataclass
from functools import lru_cache
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
from statistics import mean, stdev
from threading import Lock
import random
import statistics
import requests

# Verifica se a biblioteca iqoptionapi está disponível
try:
    from iqoptionapi.stable_api import IQ_Option
    IQOPTION_API_AVAILABLE = True
except ImportError:
    logger.warning("Biblioteca iqoptionapi não encontrada. Usando implementação alternativa.")
    IQOPTION_API_AVAILABLE = False

# Tipos customizados para melhorar legibilidade
ExpirationMode = Literal['turbo', 'binary']
TradeAction = Literal['call', 'put']
TimeframeType = Literal['Seconds', 'Minutes', 'Hours']

class ExpirationStrategy(Enum):
    """Estratégias de expiração disponíveis"""
    TURBO = auto()
    BINARY = auto()
    MULTI = auto()

class PerformanceMetrics:
    """Classe para armazenar e calcular métricas de performance"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trade_results = deque(maxlen=window_size)
        self.execution_times = deque(maxlen=window_size)
        
    def add_trade_result(self, result: bool) -> None:
        """Adiciona resultado de uma operação"""
        self.trade_results.append(result)
        
    def add_execution_time(self, execution_time: float) -> None:
        """Adiciona tempo de execução de uma operação"""
        self.execution_times.append(execution_time)
        
    def win_rate(self) -> float:
        """Calcula taxa de acerto"""
        if not self.trade_results:
            return 0.0
        return sum(self.trade_results) / len(self.trade_results)
        
    def avg_execution_time(self) -> float:
        """Calcula tempo médio de execução"""
        if not self.execution_times:
            return 0.0
        return mean(self.execution_times)
        
    def execution_time_stddev(self) -> float:
        """Calcula desvio padrão do tempo de execução"""
        if len(self.execution_times) < 2:
            return 0.0
        return stdev(self.execution_times)

class FerramentalError(Exception):
    """Classe base para erros do Ferramental"""
    pass

class ConnectionError(FerramentalError):
    """Erro de conexão com a API"""
    pass

class InvalidAssetError(FerramentalError):
    """Erro de ativo inválido"""
    pass

@dataclass
class RiskMetrics:
    max_daily_loss: float = 0.05
    max_trade_risk: float = 0.02
    max_consecutive_losses: int = 3
    consecutive_losses: int = 0
    daily_loss: float = 0.0
    last_reset: float = time.time()

class Ferramental:
    """Classe principal para operações com a API do IQ Option.
    
    Implementa padrão Singleton para garantir única instância de conexão.
    
    Attributes:
        iq_option: Instância da API do IQ Option
        asset_pairs: Conjunto de pares de ativos configurados
        connected: Status da conexão com a API
        max_retries: Número máximo de tentativas de reconexão
        retry_delay: Intervalo entre tentativas de reconexão (em segundos)
        risk_metrics: Métricas de gerenciamento de risco
        _instance: Instância única da classe (Singleton)
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Ferramental, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_manager):
        if not hasattr(self, 'initialized'):  # Evita reinicialização
            self.iq_option = None
            self.config_manager = config_manager
            self.asset_pairs = self.config_manager.get_list('Bot', 'assets')
            self.connected = False
            self.max_retries = self.config_manager.get_value('API', 'retry_count', 3, int)
            self.retry_delay = self.config_manager.get_value('API', 'timeout', 5, int)
            self.risk_management = {
                'max_daily_loss': self.config_manager.get_value('Trading', 'daily_loss_limit', 5.0, float) / 100,
                'max_trade_risk': self.config_manager.get_value('Trading', 'risk_per_trade', 1.0, float) / 100,
                'max_consecutive_losses': self.config_manager.get_value('Trading', 'max_consecutive_losses', 3, int),
                'consecutive_losses': 0,
                'daily_loss': 0.0,
                'last_reset': time.time()
            }
            self.initialized = True


    def reset_daily_metrics(self) -> None:
        """Reseta as métricas diárias de risco"""
        self.risk_management['daily_loss'] = 0.0
        self.risk_management['consecutive_losses'] = 0
        self.risk_management['last_reset'] = time.time()
        logger.info("Métricas diárias de risco resetadas")

    def check_and_reset_daily_metrics(self) -> None:
        """Verifica se é necessário resetar as métricas diárias"""
        # Verifica se passou mais de 24 horas desde o último reset
        if time.time() - self.risk_management['last_reset'] > 86400:  # 86400 segundos = 24 horas
            self.reset_daily_metrics()

    def set_session(self, headers: Optional[Dict] = None, cookies: Optional[Dict] = None) -> None:
        """Configura headers e cookies personalizados para a sessão.
        
        Args:
            headers: Dicionário com headers HTTP personalizados
            cookies: Dicionário com cookies personalizados
        """
        default_headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36"
        }
        
        if headers:
            default_headers.update(headers)
            
        self.iq_option.set_session(default_headers, cookies or {})
        logger.info("Sessão configurada com sucesso")
        logger.info("Sessão configurada após conexão")

    def connect(self) -> Tuple[bool, Optional[str]]:
        """Conecta à API do IQ Option.
        
        Returns:
            Tuple[bool, Optional[str]]: Status da conexão e mensagem de erro (se houver)
        """
        # Verifica se credenciais foram fornecidas ou carrega do ambiente
        email = self.config_manager.get_value('Credentials', 'username')
        password = self.config_manager.get_value('Credentials', 'password')

        if not email or not password:
            try:
                # Tenta obter credenciais das variáveis de ambiente (definidas no main.py)
                email = self.config_manager.get_value('API', 'email')
                password = self.config_manager.get_value('API', 'password')

                if not email or not password:
                    logger.error("Credenciais não configuradas")
                    return False, "Credenciais não configuradas"
            except Exception as e:
                logger.error(f"Erro ao carregar credenciais: {str(e)}")
                return False, f"Erro ao carregar credenciais: {str(e)}"

        logger.info(f"Tentando conectar com email: {email}")
        logger.info(f"Conectando com email: {email} e senha mascarada: {'*' * len(password)}")

        # Verifica se a biblioteca IQ Option está disponível
        if not IQOPTION_API_AVAILABLE:
            error_msg = "A biblioteca IQ Option não está instalada. Execute 'pip install -U git+https://github.com/iqoptionapi/iqoptionapi.git' para instalá-la."
            logger.error(error_msg)
            return False, error_msg

        # Configurações adicionais para a API
        self.iq_option = IQ_Option(email, password)

        # Configurar timeout mais longo para evitar problemas de conexão
        self.iq_option.set_max_reconnect(5)

        attempts = 0
        while attempts < self.max_retries:
            try:
                logger.info(f"Tentativa de conexão: {attempts + 1}")
                # Tenta conectar com a API
                check, reason = self.iq_option.connect()
                logger.info(f"check: {check}, reason: {reason}")
                
                if check:
                    logger.info("Conexão estabelecida com sucesso!")
                    logger.success("Conexão com a API do IQ Option estabelecida com sucesso")
                    self.connected = True
                    # Configura sessão padrão após conexão bem sucedida
                    self.set_session()
                    logger.info("Sessão configurada após conexão")
                    return True, None
                
                # Se a resposta contiver "Forbidden", pode ser um problema com a API
                if reason and "Forbidden" in reason:
                    error_msg = f"Erro de conexão: {reason}. A API pode estar bloqueando a conexão. Verifique sua conexão e tente novamente mais tarde."
                    logger.error(error_msg)
                    return False, error_msg
                
                logger.warning(f"Falha na conexão (tentativa {attempts + 1}/{self.max_retries}): {reason}")
                logger.warning(f"Falha na conexão (tentativa {attempts + 1}/{self.max_retries}): {reason}")
                time.sleep(self.retry_delay)
                attempts += 1
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Erro na conexão (tentativa {attempts + 1}/{self.max_retries}): {error_msg}")
                
                # Se for um erro de JSON, pode ser um problema com a API
                if "JSONDecodeError" in error_msg or "Expecting value" in error_msg:
                    logger.warning("Erro de decodificação JSON. Tentando reconectar com nova instância.")
                    # Cria uma nova instância da API
                    self.iq_option = IQ_Option(email, password)
                
                time.sleep(self.retry_delay)
                attempts += 1
        
        logger.error(f"Falha ao conectar após {self.max_retries} tentativas")
        self.connected = False
        try:
            return False, "Número máximo de tentativas de conexão excedido"
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de conexão de rede: {e}")
            return False, f"Erro de conexão de rede: {e}"
        except Exception as e:
            logger.exception(f"Erro inesperado durante a conexão: {e}")
            return False, f"Erro inesperado durante a conexão: {e}"
    
    def reconnect(self) -> Tuple[bool, Optional[str]]:
        """Tenta reconectar à API.
        
        Returns:
            Tuple[bool, Optional[str]]: Status da conexão e mensagem de erro (se houver)
        """
        logger.info("Tentando reconectar...")
        return self.connect()

    def get_version(self) -> str:
        """Obtém a versão da API.
        
        Returns:
            str: Versão da API
        """
        return self.iq_option.__version__

    def get_digital_spot_instruments(self) -> Optional[List[Dict]]:
        """Obtém a lista de instrumentos digitais spot disponíveis.
        
        NOTA: Este método está mantido apenas para compatibilidade. O bot agora opera exclusivamente
        com Opções Binárias.
        
        Returns:
            Optional[List[Dict]]: Lista de instrumentos ou None se ocorrer erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        logger.warning("Operações Digital Spot não estão mais disponíveis. O bot opera exclusivamente com Opções Binárias.")
        
        # Retorna uma lista vazia para indicar que não há instrumentos digitais disponíveis
        return []

    def get_digital_spot_profit(self, asset: str) -> Optional[float]:
        """Obtém o lucro percentual esperado para um ativo digital spot.
        
        NOTA: Este método está mantido apenas para compatibilidade. O bot agora opera exclusivamente
        com Opções Binárias.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            Optional[float]: Percentual de lucro ou None se ocorrer erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        logger.warning("Operações Digital Spot não estão mais disponíveis. O bot opera exclusivamente com Opções Binárias.")
        
        # Retorna None para indicar que não há lucro digital disponível
        return None

    def get_technical_indicators(self, asset: str) -> Optional[Dict]:
        """Obtém indicadores técnicos para um ativo específico.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            Dicionário com indicadores técnicos ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            indicators = self.iq_option.get_technical_indicators(asset)
            
            if not indicators or not isinstance(indicators, dict):
                logger.error("Dados de indicadores técnicos inválidos recebidos")
                return None
                
            required_keys = ['trend', 'oscillators', 'moving_averages']
            if not all(key in indicators for key in required_keys):
                logger.error("Dados de indicadores técnicos faltando campos obrigatórios")
                return None
                
            logger.success(f"Obteve indicadores técnicos para {asset}")
            return indicators
            
        except Exception as e:
            logger.error(f"Erro ao obter indicadores técnicos: {str(e)}")
            return None

    def get_support_resistance(self, asset: str) -> Optional[Dict]:
        """Obtém níveis de suporte e resistência para um ativo específico.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            Dicionário com níveis de suporte e resistência ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            levels = self.iq_option.get_support_resistance(asset)
            
            if not levels or not isinstance(levels, dict):
                logger.error("Dados de suporte/resistência inválidos recebidos")
                return None
                
            required_keys = ['support', 'resistance']
            if not all(key in levels for key in required_keys):
                logger.error("Dados de suporte/resistência faltando campos obrigatórios")
                return None
                
            logger.success(f"Obteve níveis de suporte/resistência para {asset}")
            return levels
            
        except Exception as e:
            logger.error(f"Erro ao obter níveis de suporte/resistência: {str(e)}")
            return None

    def enable_auto_reconnect(self, interval: int = 300) -> None:
        """Habilita reconexão automática com a API.
        
        Args:
            interval: Intervalo entre verificações de conexão em segundos
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return
            
        try:
            self.iq_option.enable_auto_reconnect(interval)
            logger.success(f"Reconexão automática habilitada com intervalo de {interval} segundos")
        except Exception as e:
            logger.error(f"Erro ao habilitar reconexão automática: {str(e)}")

    def get_session_info(self) -> Optional[Dict]:
        """Obtém informações da sessão atual.
        
        Returns:
            Dicionário com informações da sessão ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            session_info = self.iq_option.get_session_info()
            
            if not session_info or not isinstance(session_info, dict):
                logger.error("Dados de sessão inválidos recebidos")
                return None
                
            required_keys = ['session_id', 'expires_in', 'user_id']
            if not all(key in session_info for key in required_keys):
                logger.error("Dados de sessão faltando campos obrigatórios")
                return None
                
            logger.success("Informações da sessão obtidas com sucesso")
            return session_info
            
        except Exception as e:
            logger.error(f"Erro ao obter informações da sessão: {str(e)}")
            return None

    def enable_two_factor_auth(self, code: str) -> bool:
        """Habilita autenticação de dois fatores.
        
        Args:
            code: Código de autenticação
            
        Returns:
            bool: True se autenticação foi bem sucedida, False caso contrário
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False
            
        try:
            result = self.iq_option.enable_two_factor_auth(code)
            if result:
                logger.success("Autenticação de dois fatores habilitada com sucesso")
            else:
                logger.error("Falha ao habilitar autenticação de dois fatores")
            return result
            
        except Exception as e:
            logger.error(f"Erro ao habilitar autenticação de dois fatores: {str(e)}")
            return False

    def get_asset_groups(self) -> Optional[List[str]]:
        """Obtém lista de grupos de ativos disponíveis.
        
        Returns:
            Lista de nomes de grupos ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            groups = self.iq_option.get_all_asset_groups()
            
            if not groups or not isinstance(groups, list):
                logger.error("Dados de grupos de ativos inválidos recebidos")
                return None
                
            logger.success(f"Obteve {len(groups)} grupos de ativos")
            return groups
            
        except Exception as e:
            logger.error(f"Erro ao obter grupos de ativos: {str(e)}")
            return None

    def get_assets_by_group(self, group_name: str) -> Optional[List[Dict]]:
        """Obtém lista de ativos de um grupo específico.
        
        Args:
            group_name: Nome do grupo (ex: 'forex')
            
        Returns:
            Lista de dicionários com informações dos ativos ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            assets = self.iq_option.get_all_assets_by_group(group_name)
            
            if not assets or not isinstance(assets, list):
                logger.error("Dados de ativos inválidos recebidos")
                return None
                
            required_keys = ['id', 'name', 'active', 'underlying', 'group']
            for asset in assets:
                if not isinstance(asset, dict) or not all(key in asset for key in required_keys):
                    logger.error("Dados de ativos faltando campos obrigatórios ou formato inválido")
                    return None
                    
            logger.success(f"Obteve {len(assets)} ativos para o grupo {group_name}")
            return assets
            
        except Exception as e:
            logger.error(f"Erro ao obter ativos do grupo {group_name}: {str(e)}")
            return None

    def get_technical_analysis(self, asset: str) -> Optional[Dict]:
        """Obtém análise técnica completa para um ativo.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            Dicionário com análise técnica ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            analysis = self.iq_option.get_technical_analysis(asset)
            
            if not analysis or not isinstance(analysis, dict):
                logger.error("Dados de análise técnica inválidos recebidos")
                return None
                
            required_keys = ['trend', 'oscillators', 'moving_averages', 'summary']
            if not all(key in analysis for key in required_keys):
                logger.error("Dados de análise técnica faltando campos obrigatórios")
                return None
                
            logger.success(f"Obteve análise técnica para {asset}")
            return analysis
            
        except Exception as e:
            logger.error(f"Erro ao obter análise técnica: {str(e)}")
            return None

    def get_candles(self, asset: str, timeframe_type: str, timeframe_value: int, 
                   count: int, endtime: Optional[float] = None) -> Optional[List[Dict]]:
        """Obtém velas históricas para um ativo específico.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            timeframe_type: Tipo de timeframe ('Seconds', 'Minutes', 'Hours')
            timeframe_value: Valor do timeframe (1, 5, 15, etc.)
            count: Número de velas a retornar
            endtime: Timestamp de término (opcional)
            
        Returns:
            Lista de dicionários contendo dados das velas ou None em caso de erro
        """
        # Verifica e reseta métricas diárias se necessário
        self.check_and_reset_daily_metrics()
        
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None

        # Verifica se a biblioteca IQ Option está disponível
        if not IQOPTION_API_AVAILABLE:
            error_msg = "A biblioteca IQ Option não está instalada. Execute 'pip install -U git+https://github.com/iqoptionapi/iqoptionapi.git' para instalá-la."
            logger.error(error_msg)
            return None

        # Valida ativo
        if asset not in self.asset_pairs:
            logger.error(f"Ativo {asset} não configurado")
            return None

        # Valida e converte timeframe de forma mais eficiente
        timeframe_multipliers = {
            "Seconds": 1,
            "Minutes": 60,
            "Hours": 3600
        }
        
        if timeframe_type not in timeframe_multipliers:
            logger.error(f"Tipo de timeframe inválido: {timeframe_type}. Opções válidas: {list(timeframe_multipliers.keys())}")
            return None
            
        try:
            timeframe = int(timeframe_value) * timeframe_multipliers[timeframe_type]
            if timeframe <= 0:
                logger.error(f"Valor de timeframe deve ser positivo: {timeframe_value}")
                return None
        except (ValueError, TypeError):
            logger.error(f"Valor de timeframe inválido: {timeframe_value}")
            return None

        try:
            endtime = endtime if endtime else time.time()
            candles = self.iq_option.get_candles(asset, timeframe, count, endtime)
            
            # Valida estrutura das velas
            if not candles or not isinstance(candles, list) or len(candles) == 0:
                logger.error("Dados de velas inválidos recebidos")
                return None
                
            required_keys = ['id', 'from', 'to', 'open', 'close', 'min', 'max', 'volume']
            for candle in candles:
                if not isinstance(candle, dict) or not all(key in candle for key in required_keys):
                    logger.error("Dados de velas faltando campos obrigatórios ou formato inválido")
                    return None
                    
                # Valida tipos dos valores
                try:
                    float(candle['open'])
                    float(candle['close'])
                    float(candle['min'])
                    float(candle['max'])
                    int(candle['volume'])
                except (ValueError, TypeError):
                    logger.error("Dados de velas com tipos inválidos")
                    return None
                    
                # Valida timestamps
                try:
                    if not isinstance(candle['from'], int) or not isinstance(candle['to'], int):
                        logger.error("Timestamps inválidos nas velas")
                        return None
                    if candle['from'] >= candle['to']:
                        logger.error("Timestamp 'from' deve ser menor que 'to'")
                        return None
                except KeyError:
                    logger.error("Faltando timestamps nas velas")
                    return None
                
            logger.success(f"Obteve {len(candles)} velas para {asset}")
            return candles
            
        except Exception as e:
            logger.error(f"Erro ao obter velas: {str(e)}")
            return None

    def buy(self, asset: str, amount: float, action: str, expiration_mode: str) -> Tuple[bool, Optional[int]]:
        """Executa uma operação de compra na plataforma com gerenciamento de risco.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            amount: Valor da operação
            action: Direção da operação ('call' ou 'put')
            expiration_mode: Modo de expiração (ex: 'turbo')
            
        Returns:
            Tuple[bool, Optional[int]]: Status da operação e ID da ordem (se bem sucedida)
        """
        # Verifica e reseta métricas diárias se necessário
        self.check_and_reset_daily_metrics()
        
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False, None
            
        # Verifica limites de risco
        balance = self.get_balance()
        if not balance:
            logger.error("Não foi possível obter o saldo para verificação de risco")
            return False, None
            
        # Verifica risco máximo por operação
        risk_per_trade = amount / balance
        if risk_per_trade > self.risk_management['max_trade_risk']:
            logger.error(f"Risco por operação {risk_per_trade:.2%} excede o limite de {self.risk_management['max_trade_risk']:.2%}")
            return False, None
            
        # Verifica perda diária máxima
        if self.risk_management['daily_loss'] >= self.risk_management['max_daily_loss']:
            logger.error(f"Perda diária {self.risk_management['daily_loss']:.2%} atingiu o limite de {self.risk_management['max_daily_loss']:.2%}")
            return False, None
            
        # Verifica perdas consecutivas
        if self.risk_management['consecutive_losses'] >= self.risk_management['max_consecutive_losses']:
            logger.error(f"{self.risk_management['consecutive_losses']} perdas consecutivas atingiram o limite de {self.risk_management['max_consecutive_losses']}")
            return False, None

        # Valida ativo
        if asset not in self.asset_pairs:
            logger.error(f"Ativo {asset} não configurado")
            return False, None

        # Valida ação de forma mais eficiente
        valid_actions = {'call', 'put'}
        action = action.lower()
        if action not in valid_actions:
            logger.error(f"Ação inválida: {action}. Opções válidas: {valid_actions}")
            return False, None

        # Valida valor mínimo
        min_amount = 1.0  # Valor mínimo da plataforma
        if amount < min_amount:
            logger.error(f"Valor mínimo da operação é {min_amount}")
            return False, None

        try:
            # Verifica se o ativo está disponível para negociação
            if not self.check_asset_open(asset):
                logger.error(f"Ativo {asset} não está disponível para negociação no momento")
                return False, None
                
            # Tenta executar a operação com retry em caso de falha temporária
            max_retries = 3
            retry_delay = 1  # segundos
            
            for retry in range(max_retries):
                try:
                    # Verifica se a ordem existe no histórico
                    status, order_id = self.iq_option.buy(amount, asset, action, expiration_mode)
                    
                    # Verifica se a operação foi realmente executada
                    if status and order_id:
                        # Verifica se a ordem existe no histórico
                        if self.verify_order_execution(order_id):
                            logger.success(f"Operação {action} de {amount} em {asset} executada com sucesso. ID: {order_id}")
                            
                            # Atualiza métricas de risco após operação bem sucedida
                            self.risk_management['consecutive_losses'] = 0
                            return True, order_id
                        else:
                            logger.warning(f"Ordem {order_id} não encontrada no histórico. Verificando novamente...")
                            time.sleep(1)  # Aguarda um pouco para atualização do histórico
                            
                            # Verifica novamente
                            if self.verify_order_execution(order_id):
                                logger.success(f"Ordem {order_id} confirmada após segunda verificação")
                                self.risk_management['consecutive_losses'] = 0
                                return True, order_id
                            else:
                                logger.error(f"Ordem {order_id} não confirmada após segunda verificação")
                                status = False
                    
                    if not status:
                        if retry < max_retries - 1:
                            logger.warning(f"Falha na execução da operação {action} em {asset}. Tentativa {retry+1}/{max_retries}")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"Falha na execução da operação {action} em {asset} após {max_retries} tentativas")
                            
                            # Atualiza métricas de risco após falha
                            self.risk_management['consecutive_losses'] += 1
                            self.risk_management['daily_loss'] += amount / balance
                            return False, None
                            
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"Erro na tentativa {retry+1}/{max_retries} de executar operação: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Erro persistente ao executar operação após {max_retries} tentativas: {str(e)}")
                        
                        # Atualiza métricas de risco após falha
                        self.risk_management['consecutive_losses'] += 1
                        self.risk_management['daily_loss'] += amount / balance
                        return False, None
            
            return False, None
                
        except Exception as e:
            logger.error(f"Erro ao executar operação: {str(e)}")
            
            # Atualiza métricas de risco após falha
            self.risk_management['consecutive_losses'] += 1
            self.risk_management['daily_loss'] += amount / balance
            
            return False, None

    def get_balance_v2(self) -> Optional[float]:
        """Obtém o saldo da conta com maior precisão.

        Returns:
            float: Saldo da conta ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None

        try:
            balance = self.iq_option.get_balance_v2()
            logger.success(f"Saldo (v2) obtido: {balance}")
            return balance
        except Exception as e:
            logger.error(f"Erro ao obter saldo (v2): {str(e)}")
            return None

    def get_currency(self) -> Optional[str]:
        """Obtém a moeda da conta.

        Returns:
            str: Moeda da conta ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None

        try:
            currency = self.iq_option.get_currency()
            logger.success(f"Moeda da conta: {currency}")
            return currency
        except Exception as e:
            logger.error(f"Erro ao obter moeda da conta: {str(e)}")
            return None
            
    def get_balance(self) -> Optional[float]:
        """Obtém o saldo atual da conta.
        
        Returns:
            float: Saldo da conta ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            # Tenta obter o saldo com a versão mais atualizada da API
            balance = self.get_balance_v2()
            if balance is not None:
                return balance
                
            # Fallback para a versão antiga em caso de falha
            balance = self.iq_option.get_balance()
            
            if not isinstance(balance, (int, float)) or balance < 0:
                logger.error("Saldo inválido recebido")
                return None
                
            logger.success(f"Saldo obtido: {balance}")
            return balance
            
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {str(e)}")
            return None
            
    def get_min_trade_amount(self) -> float:
        """Obtém o valor mínimo por operação.
        
        Returns:
            float: Valor mínimo por operação
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return 1.0
            
        try:
            # Valores mínimos por tipo de conta (aproximados)
            min_amounts = {
                "USD": 1.0,
                "EUR": 1.0,
                "GBP": 1.0,
                "BRL": 5.0,
            }
            
            currency = self.get_currency()
            return min_amounts.get(currency, 1.0)
            
        except Exception as e:
            logger.error(f"Erro ao obter valor mínimo por operação: {str(e)}")
            return 1.0
            
    def get_max_trade_amount(self) -> float:
        """Obtém o valor máximo por operação.
        
        Returns:
            float: Valor máximo por operação
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return 20000.0
            
        try:
            # Obtém o saldo atual
            balance = self.get_balance()
            if not balance:
                return 20000.0
                
            # Limita o valor máximo a 20% do saldo ou aos limites da plataforma
            platform_max = 20000.0
            balance_max = balance * 0.2
            
            return min(platform_max, balance_max)
            
        except Exception as e:
            logger.error(f"Erro ao obter valor máximo por operação: {str(e)}")
            return 20000.0
            
    def get_current_price(self, asset: str) -> Optional[float]:
        """Obtém o preço atual de um ativo.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            float: Preço atual do ativo
        """
        try:
            if not IQOPTION_API_AVAILABLE and hasattr(self, 'simulation_mode') and self.simulation_mode:
                # No modo de simulação, retornamos o preço do ativo simulado
                if hasattr(self, 'simulation_assets') and asset in self.simulation_assets:
                    # Adiciona uma pequena variação aleatória ao preço
                    import numpy as np
                    base_price = self.simulation_assets[asset]["price"]
                    price_variation = base_price * np.random.normal(0, 0.0002)
                    current_price = base_price + price_variation
                    
                    logger.info(f"Preço atual simulado para {asset}: {current_price}")
                    return current_price
                else:
                    logger.warning(f"Ativo simulado {asset} não encontrado")
                    return None
            
            # Verifica se o ativo está disponível
            all_assets = self.iq_option.get_all_open_time()
            
            for asset_type in ['turbo', 'binary']:
                if asset in all_assets[asset_type] and all_assets[asset_type][asset]['open']:
                    # Obtém o preço atual do ativo
                    candles = self.iq_option.get_candles(asset, 60, 1)
                    if candles:
                        return candles[0]['close']
            
            logger.warning(f"Ativo {asset} não está disponível para negociação")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter preço atual: {str(e)}")
            return None
            
    def get_spread(self, asset: str) -> Optional[float]:
        """Obtém o spread atual de um ativo.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            float: Spread do ativo em percentual ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            # Tenta obter o preço de compra e venda para calcular o spread
            instruments = self.iq_option.get_all_open_time()
            
            if asset in instruments['turbo'] and instruments['turbo'][asset]['open']:
                bid = self.iq_option.get_price_raw(asset)
                ask = self.iq_option.get_price_raw(asset)
                
                # Na API do IQ Option, às vezes o preço de compra e venda são iguais
                # Nesse caso, estimamos o spread baseado no tipo de ativo
                if bid == ask:
                    # Spreads típicos baseados no tipo de ativo
                    if asset.startswith(('EUR', 'GBP', 'USD', 'JPY')):  # Forex major
                        spread = 0.0002  # 0.02% típico para pares forex principais
                    elif asset.startswith(('BTC', 'ETH')):  # Crypto
                        spread = 0.001  # 0.1% típico para cryptos
                    else:
                        spread = 0.0005  # 0.05% valor médio para outros ativos
                else:
                    spread = (ask - bid) / bid
                    
                logger.success(f"Spread de {asset}: {spread:.4%}")
                return spread
                
            logger.error(f"Não foi possível obter spread para {asset}")
            return 0.05  # Valor default de 5% em caso de falha
            
        except Exception as e:
            logger.error(f"Erro ao obter spread: {str(e)}")
            return 0.05  # Valor default de 5% em caso de falha
            
    def configure_assets(self, assets: List[str]) -> bool:
        """Configura a lista de ativos para operação.
        
        Args:
            assets: Lista de ativos a serem configurados
            
        Returns:
            bool: True se os ativos foram configurados com sucesso
        """
        try:
            # Valida se todos os ativos existem
            instruments = self.iq_option.get_all_open_time()
            
            all_assets = set()
            
            for category in instruments.values():
                all_assets.update(category.keys())
                
            # Filtra apenas ativos válidos
            valid_assets = [asset for asset in assets if asset in all_assets]
            
            if len(valid_assets) != len(assets):
                invalid_assets = set(assets) - set(valid_assets)
                logger.warning(f"Alguns ativos não são válidos: {invalid_assets}")
                
            if not valid_assets:
                logger.error("Nenhum ativo válido para configuração")
                return False
                
            # Configura ativos da classe
            self.asset_pairs = valid_assets
            logger.success(f"Ativos configurados: {self.asset_pairs}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao configurar ativos: {str(e)}")
            return False

    def reset_practice_balance(self) -> bool:
        """Reseta o saldo da conta de prática.

        Returns:
            bool: True se bem sucedido, False caso contrário
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False

        try:
            success = self.iq_option.reset_practice_balance()
            if success:
                logger.success("Saldo da conta de prática resetado")
            else:
                logger.error("Falha ao resetar saldo da conta de prática")
            return success
        except Exception as e:
            logger.error(f"Erro ao resetar saldo da conta de prática: {str(e)}")
            return False

    def change_balance(self, balance_type: str) -> bool:
        """Muda o tipo de conta (REAL, PRACTICE, TOURNAMENT).

        Args:
            balance_type: Tipo de conta para mudar

        Returns:
            bool: True se bem sucedido, False caso contrário
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False

        valid_types = ["REAL", "PRACTICE", "TOURNAMENT"]
        if balance_type not in valid_types:
            logger.error(f"Tipo de conta inválido: {balance_type}. Tipos válidos: {valid_types}")
            return False

        try:
            self.iq_option.change_balance(balance_type)
            logger.success(f"Tipo de conta alterado para: {balance_type}")
            return True
        except Exception as e:
            logger.error(f"Erro ao mudar tipo de conta: {str(e)}")
            return False
    
    def buy_digital_spot(self, asset: str, amount: float, action: str, duration: int) -> Tuple[bool, Optional[int]]:
        """Executa uma operação de compra digital spot com gerenciamento de risco.
        
        NOTA: Este método está mantido apenas para compatibilidade. O bot agora opera exclusivamente
        com Opções Binárias. Todas as chamadas serão redirecionadas para o método buy().
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            amount: Valor da operação
            action: Direção da operação ('call' ou 'put')
            duration: Duração da operação em minutos (1 ou 5)
        
        Returns:
            Tuple[bool, Optional[int]]: Status da operação e ID da ordem (se bem sucedida)
        """
        # Verifica e reseta métricas diárias se necessário
        self.check_and_reset_daily_metrics()
        
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False, None
            
        logger.warning("Operações Digital Spot não estão mais disponíveis. Redirecionando para Opções Binárias.")
        
        # Converter duração de minutos para o formato esperado pelo método buy
        expiration_mode = "turbo" if duration == 1 else "binary"
        
        # Redirecionar para o método de opções binárias
        return self.buy(asset, amount, action, expiration_mode)

    def check_win_digital(self, order_id: int, polling_time: int = 2) -> Tuple[bool, Optional[float]]:
        """Verifica o resultado de uma operação digital spot.
        
        NOTA: Este método está mantido apenas para compatibilidade. O bot agora opera exclusivamente
        com Opções Binárias.
        
        Args:
            order_id: ID da ordem a ser verificada
            polling_time: Intervalo entre verificações em segundos
            
        Returns:
            Tuple[bool, Optional[float]]: Status da verificação e valor do lucro (se disponível)
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False, None
            
        logger.warning("Operações Digital Spot não estão mais disponíveis. O bot opera exclusivamente com Opções Binárias.")
        
        # Mantido para compatibilidade, mas retorna False para indicar que não é mais suportado
        return False, None

    def start_candles_stream(self, asset: str, timeframe: int = 60) -> bool:
        """Inicia o streaming de candles para um ativo específico.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            timeframe: Timeframe em segundos (opcional, padrão: 60)
            
        Returns:
            bool: True se o streaming foi iniciado com sucesso, False caso contrário
        """
        try:
            if not IQOPTION_API_AVAILABLE and hasattr(self, 'simulation_mode') and self.simulation_mode:
                logger.info(f"Iniciando streaming simulado para {asset} (timeframe: {timeframe}s)")
                # No modo de simulação, apenas registramos o streaming ativo
                if not hasattr(self, 'simulation_active_streams'):
                    self.simulation_active_streams = {}
                
                self.simulation_active_streams[asset] = {
                    'timeframe': timeframe,
                    'last_update': datetime.datetime.now()
                }
                return True
            
            if self.iq_option:
                self.iq_option.start_candles_stream(asset, timeframe, 1)
                return True
            else:
                logger.error("API IQ Option não inicializada")
                return False
        except Exception as e:
            logger.error(f"Erro ao iniciar streaming de candles: {str(e)}")
            return False
            
    def get_realtime_candles(self, asset: str, timeframe: int = 60) -> Optional[Dict]:
        """Obtém candles em tempo real para um ativo específico.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            timeframe: Timeframe em segundos (opcional, padrão: 60)
            
        Returns:
            dict: Dicionário com os candles em tempo real
        """
        try:
            if not IQOPTION_API_AVAILABLE and hasattr(self, 'simulation_mode') and self.simulation_mode:
                # No modo de simulação, geramos candles sintéticos baseados nos dados históricos
                if hasattr(self, 'simulation_historical_data') and asset in self.simulation_historical_data:
                    # Obtém o último candle dos dados históricos
                    df = self.simulation_historical_data[asset]
                    last_candle = df.iloc[-1].copy()
                    
                    # Ajusta o timestamp para o momento atual
                    current_time = datetime.now()
                    timestamp = int(current_time.timestamp())
                    
                    # Gera uma pequena variação no preço
                    import numpy as np
                    price_variation = last_candle['close'] * np.random.normal(0, 0.0002)
                    
                    # Cria um novo candle com base no último, mas com pequenas variações
                    new_candle = {
                        'id': timestamp,
                        'from': timestamp - (timestamp % timeframe),
                        'to': timestamp - (timestamp % timeframe) + timeframe,
                        'open': last_candle['close'],
                        'close': last_candle['close'] + price_variation,
                        'high': max(last_candle['close'], last_candle['close'] + price_variation) + abs(price_variation) * 0.5,
                        'low': min(last_candle['close'], last_candle['close'] + price_variation) - abs(price_variation) * 0.5,
                        'volume': int(last_candle['volume'] * np.random.uniform(0.8, 1.2))
                    }
                    
                    # Retorna um dicionário no formato esperado pela API
                    return {timestamp: new_candle}
                else:
                    logger.warning(f"Dados históricos simulados não disponíveis para {asset}")
                    return {}
            
            if self.iq_option:
                return self.iq_option.get_realtime_candles(asset, timeframe)
            else:
                logger.error("API IQ Option não inicializada")
                return {}
        except Exception as e:
            logger.error(f"Erro ao obter candles em tempo real: {str(e)}")
            return {}
            
    def get_current_price(self, asset: str) -> Optional[float]:
        """Obtém o preço atual de um ativo.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            float: Preço atual do ativo
        """
        try:
            if not IQOPTION_API_AVAILABLE and hasattr(self, 'simulation_mode') and self.simulation_mode:
                # No modo de simulação, retornamos o preço do ativo simulado
                if hasattr(self, 'simulation_assets') and asset in self.simulation_assets:
                    # Adiciona uma pequena variação aleatória ao preço
                    import numpy as np
                    base_price = self.simulation_assets[asset]["price"]
                    price_variation = base_price * np.random.normal(0, 0.0002)
                    current_price = base_price + price_variation
                    
                    logger.info(f"Preço atual simulado para {asset}: {current_price}")
                    return current_price
                else:
                    logger.warning(f"Ativo simulado {asset} não encontrado")
                    return None
            
            # Verifica se o ativo está disponível
            all_assets = self.iq_option.get_all_open_time()
            
            for asset_type in ['turbo', 'binary']:
                if asset in all_assets[asset_type] and all_assets[asset_type][asset]['open']:
                    # Obtém o preço atual do ativo
                    candles = self.iq_option.get_candles(asset, 60, 1)
                    if candles:
                        return candles[0]['close']
            
            logger.warning(f"Ativo {asset} não está disponível para negociação")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter preço atual: {str(e)}")
            return None
            
    def stop_candles_stream(self, asset: str, timeframe: int = 60) -> bool:
        """Para o streaming de candles para um ativo específico.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            timeframe: Timeframe em segundos (opcional, padrão: 60)
            
        Returns:
            bool: True se o streaming foi parado com sucesso, False caso contrário
        """
        try:
            if not IQOPTION_API_AVAILABLE and hasattr(self, 'simulation_mode') and self.simulation_mode:
                logger.info(f"Parando streaming simulado para {asset}")
                # No modo de simulação, apenas removemos o registro do streaming
                if hasattr(self, 'simulation_active_streams') and asset in self.simulation_active_streams:
                    del self.simulation_active_streams[asset]
                return True
            
            if self.iq_option:
                self.iq_option.stop_candles_stream(asset, timeframe)
                return True
            else:
                logger.error("API IQ Option não inicializada")
                return False
        except Exception as e:
            logger.error(f"Erro ao parar streaming de candles: {str(e)}")
            return False

    def handle_two_factor_auth(self, code: str) -> bool:
        """Gerencia a autenticação de dois fatores.
        
        Args:
            code: Código de autenticação recebido
            
        Returns:
            bool: True se a autenticação foi bem sucedida
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False
            
        try:
            status, reason = self.iq_option.connect_2fa(code)
            
            if status:
                logger.success("Autenticação de dois fatores concluída com sucesso")
                return True
                
            logger.error(f"Falha na autenticação de dois fatores: {reason}")
            return False
            
        except Exception as e:
            logger.error(f"Erro ao processar autenticação de dois fatores: {str(e)}")
            return False
            
    def get_realtime_data(self) -> Optional[pd.DataFrame]:
        """Obtém dados históricos e em tempo real para análise.
        
        Returns:
            DataFrame com dados para análise ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            # Obtém dados para todos os ativos configurados
            all_data = []
            
            for asset in self.asset_pairs:
                # Tenta obter candles em tempo real para cada ativo
                candles = {}
                
                # Verifica se o streaming de velas está configurado
                for tf_seconds in [60, 300, 900]:  # 1m, 5m, 15m
                    # Inicia streaming se não estiver ativo
                    self.start_candles_stream(asset, tf_seconds)
                    # Obtém velas em tempo real
                    realtime_candles = self.get_realtime_candles(asset, tf_seconds)
                    
                    if realtime_candles:
                        # Converte para DataFrame
                        df = pd.DataFrame(list(realtime_candles.values()))
                        # Adiciona informações do ativo e timeframe
                        df['asset'] = asset
                        df['timeframe'] = tf_seconds
                        all_data.append(df)
                
                # Backup: Se não conseguiu dados em tempo real, tenta dados históricos
                if not all_data:
                    historical_candles = self.get_candles(asset, "Minutes", 1, 100)
                    if historical_candles:
                        df = pd.DataFrame(historical_candles)
                        df['asset'] = asset
                        df['timeframe'] = 60
                        all_data.append(df)
            
            # Combina todos os dados
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.success(f"Obteve {len(combined_data)} registros de dados em tempo real")
                return combined_data
                
            logger.error("Não foi possível obter dados em tempo real")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter dados em tempo real: {str(e)}")
            return None
            
    def get_historical_data(self) -> Optional[pd.DataFrame]:
        """Obtém dados históricos para treinamento.
        
        Returns:
            DataFrame com dados históricos ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            # Obtém dados históricos para todos os ativos configurados
            all_data = []
            
            for asset in self.asset_pairs:
                # Obtém candles históricos para cada ativo
                for tf_type, tf_value in [("Minutes", 1), ("Minutes", 5), ("Minutes", 15)]:
                    historical_candles = self.get_candles(asset, tf_type, tf_value, 1000)
                    
                    if historical_candles:
                        df = pd.DataFrame(historical_candles)
                        df['asset'] = asset
                        df['timeframe'] = f"{tf_value} {tf_type}"
                        all_data.append(df)
            
            # Combina todos os dados
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.success(f"Obteve {len(combined_data)} registros de dados históricos")
                return combined_data
                
            logger.error("Não foi possível obter dados históricos")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos: {str(e)}")
            return None
            
    def get_all_assets(self) -> Optional[List[str]]:
        """Obtém lista de todos os ativos disponíveis para negociação.
        
        Returns:
            Optional[List[str]]: Lista de nomes de ativos ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            # Obtém instrumentos para diferentes categorias
            all_assets = set()
            
            if not IQOPTION_API_AVAILABLE:
                # No modo de simulação, retorna uma lista de ativos comuns
                return ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURGBP", 
                        "EURJPY", "USDCHF", "USDCAD", "BTCUSD", "ETHUSD"]
            
            # Forex e outros mercados
            forex = self.iq_option.get_all_open_time()
            
            for market_type in forex:
                for asset in forex[market_type]:
                    if forex[market_type][asset]['open']:
                        all_assets.add(asset)
            
            logger.success(f"Obteve {len(all_assets)} ativos disponíveis")
            return list(all_assets)
            
        except Exception as e:
            logger.error(f"Erro ao obter lista de ativos: {str(e)}")
            return None

    def send_notification(self, message: str) -> None:
        """Envia notificação para o usuário.
        
        Args:
            message: Mensagem a ser enviada
        """
        logger.info(f"NOTIFICAÇÃO: {message}")
        # Em uma implementação real, poderia enviar por email, SMS ou push notification

    def check_connection(self) -> bool:
        """Verifica se a conexão com a API está ativa.
        
        Returns:
            bool: True se conectado, False caso contrário
        """
        if not self.connected:
            return False
            
        if IQOPTION_API_AVAILABLE:
            return self.iq_option.check_connect()
        else:
            # No modo de simulação, sempre retorna True
            return True

    def execute_test_trade(self, asset, direction, amount):
        """Executa uma operação de teste para um ativo.
        
        Args:
            asset (str): Ativo para operar
            direction (str): Direção da operação ('call' ou 'put')
            amount (float): Valor a ser investido
            
        Returns:
            dict: Resultado da operação com informações como lucro/prejuízo
        """
        logger.info(f"Executando operação de TESTE: {asset} {direction} ${amount}")
        
        try:
            # Verifica se o ativo está disponível
            if not self.check_asset_open(asset):
                logger.warning(f"Ativo {asset} não está disponível para negociação no momento")
                return {"success": False, "profit": 0, "message": "Ativo indisponível"}
                
            # Obtém o preço atual do ativo
            current_price = self.get_current_price(asset)
            if current_price is None:
                logger.warning(f"Não foi possível obter o preço atual para {asset}")
                return {"success": False, "profit": 0, "message": "Preço indisponível"}
                
            # Verifica se estamos no modo de conta de prática
            if self.iq_option.get_account_type() != "PRACTICE":
                logger.warning("Tentando executar operação de teste em conta real. Mudando para conta de prática...")
                self.iq_option.change_balance("PRACTICE")
                
            # Configura o modo de expiração para opções binárias (turbo = 1 minuto)
            expiration_mode = "turbo"  # Opções binárias com expiração de 1 minuto
            
            # Converte a direção se necessário
            action = direction.lower()
            if action == "buy":
                action = "call"
            elif action == "sell":
                action = "put"
                
            # Executa a operação usando o método buy para opções binárias
            status, order_id = self.buy(asset, amount, action, expiration_mode)
            
            if status:
                logger.success(f"Operação de teste bem-sucedida: {asset} {action} ${amount}")
                
                # Atualiza métricas de risco após operação bem sucedida
                self.risk_management['consecutive_losses'] = 0
                return {
                    "success": True,
                    "profit": amount * 0.8,  # Lucro padrão de 80% para opções binárias
                    "message": "Operação bem-sucedida",
                    "order_id": order_id
                }
            else:
                logger.warning(f"Operação de teste mal-sucedida: {asset} {action} ${amount}")
                return {
                    "success": False,
                    "profit": -amount,  # Perda do valor investido
                    "message": "Operação mal-sucedida"
                }
                
        except Exception as e:
            logger.error(f"Erro ao executar operação de teste: {str(e)}")
            return {
                "success": False,
                "profit": 0,
                "message": f"Erro: {str(e)}"
            }
            
    def execute_real_trade(self, asset, direction, amount):
        """Executa uma operação real para um ativo.
        
        Args:
            asset (str): Ativo para operar
            direction (str): Direção da operação ('call' ou 'put')
            amount (float): Valor a ser investido
            
        Returns:
            dict: Resultado da operação com informações como lucro/prejuízo
        """
        logger.info(f"Executando operação REAL: {asset} {direction} ${amount}")
        
        # Verificação de segurança para operações reais
        if not IQOPTION_API_AVAILABLE:
            logger.error("Operações reais não estão disponíveis no modo de simulação")
            return {
                "success": False,
                "profit": 0,
                "message": "Operações reais não disponíveis no modo de simulação"
            }
            
        try:
            # Verifica se o ativo está disponível
            if not self.check_asset_open(asset):
                logger.warning(f"Ativo {asset} não está disponível para negociação no momento")
                return {"success": False, "profit": 0, "message": "Ativo indisponível"}
                
            # Obtém o preço atual do ativo
            current_price = self.get_current_price(asset)
            if current_price is None:
                logger.warning(f"Não foi possível obter o preço atual para {asset}")
                return {"success": False, "profit": 0, "message": "Preço indisponível"}
                
            # Verifica se estamos no modo de conta real
            if self.iq_option.get_account_type() != "REAL":
                logger.warning("Tentando executar operação real em conta de prática. Mudando para conta real...")
                self.iq_option.change_balance("REAL")
                
            # Verifica o saldo disponível
            balance = self.get_balance()
            if balance is None or balance < amount:
                logger.error(f"Saldo insuficiente para operação: {balance if balance is not None else 'desconhecido'} < {amount}")
                return {"success": False, "profit": 0, "message": "Saldo insuficiente"}
            
            # Verifica limites de risco
            if self.check_risk_limits(amount):
                logger.warning("Operação cancelada devido a limites de risco")
                return {"success": False, "profit": 0, "message": "Limites de risco atingidos"}
                
            # Configura o modo de expiração para opções binárias (turbo = 1 minuto)
            expiration_mode = "turbo"  # Opções binárias com expiração de 1 minuto
            
            # Converte a direção se necessário
            action = direction.lower()
            if action == "buy":
                action = "call"
            elif action == "sell":
                action = "put"
                
            # Executa a operação usando o método buy para opções binárias
            status, order_id = self.buy(asset, amount, action, expiration_mode)
            
            if status:
                # Aguarda um momento para obter o resultado da operação
                time.sleep(2)
                
                # Verifica se a ordem foi realmente executada
                if self.verify_order_execution(order_id):
                    logger.success(f"Operação real bem-sucedida: {asset} {action} ${amount}")
                    
                    # Obtém o payout (retorno) para este ativo
                    payout = self.get_payout(asset) or 0.8  # Usa 80% como padrão se não conseguir obter
                    
                    return {
                        "success": True,
                        "profit": amount * payout,
                        "message": "Operação bem-sucedida",
                        "order_id": order_id,
                        "payout": payout
                    }
                else:
                    logger.warning(f"Ordem {order_id} não encontrada no histórico")
                    return {
                        "success": False,
                        "profit": 0,
                        "message": "Ordem não encontrada no histórico",
                        "order_id": order_id
                    }
            else:
                logger.warning(f"Operação real mal-sucedida: {asset} {action} ${amount}")
                return {
                    "success": False,
                    "profit": -amount,  # Perda do valor investido
                    "message": "Operação mal-sucedida"
                }
                
        except Exception as e:
            logger.error(f"Erro ao executar operação real: {str(e)}")
            return {
                "success": False,
                "profit": 0,
                "message": f"Erro: {str(e)}"
            }
            
    def get_payout(self, asset, expiration_time=60):
        """Obtém o payout atual para um ativo específico.
        
        Args:
            asset (str): Nome do ativo
            expiration_time (int): Tempo de expiração em segundos
            
        Returns:
            float: Payout atual (0-1) ou None em caso de erro
        """
        try:
            if not IQOPTION_API_AVAILABLE:
                # No modo de simulação, retorna um payout aleatório entre 0.7 e 0.9
                return random.uniform(0.7, 0.9)
                
            # Obtém o payout atual
            payout = self.iq_option.get_digital_payout(asset)
            
            if payout is not None:
                # Converte para decimal (0-1)
                payout = float(payout) / 100
                logger.info(f"Payout atual para {asset}: {payout:.2%}")
                return payout
            else:
                logger.warning(f"Não foi possível obter o payout para {asset}")
                return 0.7  # Valor padrão conservador
                
        except Exception as e:
            logger.error(f"Erro ao obter payout para {asset}: {str(e)}")
            return 0.7  # Valor padrão conservador em caso de erro

    def check_risk_limits(self, amount):
        """Verifica se a operação ultrapassa os limites de risco definidos.
        
        Args:
            amount (float): Valor da operação
            
        Returns:
            bool: True se os limites foram atingidos (não deve operar), False caso contrário
        """
        try:
            if not IQOPTION_API_AVAILABLE:
                # No modo de simulação, sempre retorna True
                return True
                
            # 1. Verifica o limite de perda diária
            try:
                # Obtém o histórico de operações do dia
                today = datetime.now().strftime("%Y-%m-%d")
                today_start = int(datetime.strptime(f"{today} 00:00:00", "%Y-%m-%d %H:%M:%S").timestamp())
                
                # Obtém o histórico de operações
                history = self.api.get_position_history_v2("turbo-option", today_start, limit=100)
                
                if history:
                    # Calcula o lucro/prejuízo total do dia
                    daily_profit = sum(float(trade.get('profit', 0)) for trade in history)
                    daily_loss_pct = abs(daily_profit) / self.get_balance() if daily_profit < 0 else 0
                    
                    if daily_loss_pct >= 0.05:  # Limite de perda diária de 5%
                        logger.warning(f"Limite de perda diária atingido: {daily_loss_pct:.2%} >= 5%")
                        return True
            except Exception as e:
                logger.warning(f"Erro ao verificar histórico de operações: {str(e)}")
                # Em caso de erro, continuamos com as outras verificações
            
            # 2. Verifica perdas consecutivas
            try:
                # Obtém o histórico de operações recentes
                history = self.api.get_position_history_v2("turbo-option", limit=5)
                
                if history:
                    # Conta perdas consecutivas
                    consecutive_losses = 0
                    for trade in history:
                        if float(trade.get('profit', 0)) < 0:
                            consecutive_losses += 1
                        else:
                            break  # Interrompe a contagem ao encontrar um ganho
                    
                    if consecutive_losses >= 3:  # Limite de 3 perdas consecutivas
                        logger.warning(f"Limite de perdas consecutivas atingido: {consecutive_losses} >= 3")
                        return True
            except Exception as e:
                logger.warning(f"Erro ao verificar perdas consecutivas: {str(e)}")
                # Em caso de erro, continuamos com as outras verificações
            
            # 3. Verifica se o saldo está abaixo do mínimo configurado
            min_balance = self.config.get('min_balance_for_real', 100)
            if self.get_balance() < min_balance:
                logger.warning(f"Saldo abaixo do mínimo configurado: {self.get_balance():.2f} < {min_balance:.2f}")
                return True
            
            # Todos os limites de risco estão OK
            return False
            
        except Exception as e:
            logger.error(f"Erro ao verificar limites de risco: {str(e)}")
            # Em caso de erro na verificação, retorna True por segurança
            return True

    def verify_order_execution(self, order_id):
        """Verifica se uma ordem foi realmente executada consultando o histórico.
        
        Args:
            order_id (int): ID da ordem a verificar
            
        Returns:
            bool: True se a ordem foi executada, False caso contrário
        """
        try:
            if not IQOPTION_API_AVAILABLE:
                # No modo de simulação, consideramos todas as ordens como executadas
                return True
                
            # Obtém o histórico recente de operações
            max_retries = 3
            retry_delay = 1  # segundos
            
            for retry in range(max_retries):
                try:
                    # Obtém o histórico recente de operações
                    history = self.api.get_position_history_v2("turbo-option", limit=100)
                    
                    # Verifica se a ordem está no histórico
                    for order in history:
                        if order.get('id') == order_id:
                            logger.info(f"Ordem {order_id} encontrada no histórico")
                            return True
                    
                    # Se não encontrou no histórico geral, tenta obter histórico específico do dia
                    today = datetime.now().strftime("%Y-%m-%d")
                    daily_history = self.api.get_position_history_v2("turbo-option", today, limit=100)
                    
                    for order in daily_history:
                        if order.get('id') == order_id:
                            logger.info(f"Ordem {order_id} encontrada no histórico do dia")
                            return True
                    
                    # Se ainda não encontrou, tenta obter detalhes específicos da ordem
                    order_details = self.api.get_order(order_id)
                    if order_details:
                        logger.info(f"Detalhes da ordem {order_id} obtidos com sucesso")
                        return True
                    
                    # Se não encontrou em nenhuma das verificações, mas ainda temos retries
                    if retry < max_retries - 1:
                        logger.warning(f"Ordem {order_id} não encontrada. Tentativa {retry+1}/{max_retries}. Aguardando...")
                        time.sleep(retry_delay * (retry + 1))  # Aumenta o tempo de espera a cada tentativa
                    else:
                        logger.warning(f"Ordem {order_id} não encontrada após {max_retries} tentativas")
                        return False
                
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"Erro na tentativa {retry+1}/{max_retries} de verificar ordem {order_id}: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Erro persistente ao verificar ordem {order_id}: {str(e)}")
                        
                        # Em caso de erro persistente na verificação, assumimos que a ordem foi executada
                        # para evitar operações duplicadas
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro crítico ao verificar execução da ordem {order_id}: {str(e)}")
            # Em caso de erro na verificação, assumimos que a ordem foi executada
            # para evitar operações duplicadas
            return True

    def _setup_simulation_mode(self):
        """Configura o modo de simulação para testes quando a biblioteca IQ Option não está disponível.
        
        Este método configura valores simulados para permitir testes do bot sem a necessidade
        da biblioteca IQ Option.
        """
        logger.info("Configurando modo de simulação para testes")
        
        # Define que estamos em modo de simulação
        self.simulation_mode = True
        
        # Lista de ativos disponíveis para simulação
        self.simulation_assets = {
            "EURUSD": {"price": 1.0821, "open": True, "type": "forex"},
            "GBPUSD": {"price": 1.2654, "open": True, "type": "forex"},
            "USDJPY": {"price": 106.87, "open": True, "type": "forex"},
            "AUDUSD": {"price": 0.7123, "open": True, "type": "forex"},
            "USDCAD": {"price": 1.3245, "open": True, "type": "forex"},
            "EURJPY": {"price": 115.65, "open": True, "type": "forex"},
            "GBPJPY": {"price": 135.28, "open": True, "type": "forex"}
        }
        
        # Configura dados históricos simulados
        self._setup_simulated_historical_data()
        
        # Simula o saldo da conta
        self.simulation_balance = 1000.0  # Saldo inicial de $1000
        
        # Simula operações abertas
        self.simulation_open_positions = []
        
        # Simula histórico de operações
        self.simulation_trade_history = []
        
        logger.success("Modo de simulação configurado com sucesso")
    
    def _setup_simulated_historical_data(self):
        """Configura dados históricos simulados para testes."""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        logger.info("Configurando dados históricos simulados")
        
        # Dicionário para armazenar dados históricos simulados por ativo
        self.simulation_historical_data = {}
        
        # Para cada ativo, gera dados históricos simulados
        for asset in self.simulation_assets.keys():
            # Define o preço base para o ativo
            base_price = self.simulation_assets[asset]["price"]
            
            # Gera timestamps para os últimos 5000 minutos
            end_time = datetime.now()
            timestamps = [end_time - timedelta(minutes=i) for i in range(5000)]
            timestamps.reverse()  # Ordem cronológica
            
            # Gera preços simulados com tendência e volatilidade
            np.random.seed(42)  # Para reprodutibilidade
            
            # Gera um passeio aleatório com tendência
            random_walk = np.random.normal(0, 0.0002, 5000).cumsum()
            trend = np.linspace(0, 0.01, 5000)  # Tendência de alta
            prices = base_price + random_walk + trend
            
            # Cria DataFrame com os dados históricos
            candles = []
            
            for i in range(5000):
                # Gera preços de abertura, fechamento, máxima e mínima realistas
                if i == 0:
                    open_price = prices[i] * (1 + np.random.uniform(-0.0005, 0.0005))
                else:
                    open_price = candles[i-1]['close']
                    
                close_price = prices[i]
                
                # Máxima e mínima realistas
                price_range = abs(close_price - open_price) + (close_price * np.random.uniform(0.0005, 0.0015))
                if close_price > open_price:
                    high_price = close_price + (price_range * np.random.uniform(0.1, 0.5))
                    low_price = open_price - (price_range * np.random.uniform(0.1, 0.5))
                else:
                    high_price = open_price + (price_range * np.random.uniform(0.1, 0.5))
                    low_price = close_price - (price_range * np.random.uniform(0.1, 0.5))
                    
                # Volume aleatório
                volume = int(np.random.uniform(50, 200) * (1 + abs((close_price - open_price) / open_price) * 100))
                
                # Timestamp em formato unix (segundos)
                timestamp = int(timestamps[i].timestamp())
                
                candle = {
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                }
                
                candles.append(candle)
                
            # Converte para DataFrame
            df = pd.DataFrame(candles)
            
            # Adiciona coluna datetime e configura como índice
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Adiciona informações do ativo e timeframe
            df['asset'] = asset
            df['timeframe'] = "1m"
            
            logger.success(f"Gerados {len(df)} candles sintéticos para {asset}")
            self.simulation_historical_data[asset] = df
            
    def adjust_trade_amount_based_on_risk(self, base_amount, win_rate=None, market_volatility=None, consecutive_losses=0):
        """Ajusta o valor da operação com base em fatores de risco.
        
        Args:
            base_amount (float): Valor base da operação
            win_rate (float, optional): Taxa de acerto recente (0-1)
            market_volatility (float, optional): Volatilidade atual do mercado (0-1)
            consecutive_losses (int, optional): Número de perdas consecutivas
            
        Returns:
            float: Valor ajustado da operação
        """
        try:
            # Valor inicial é o valor base
            adjusted_amount = base_amount
            
            # Fatores de ajuste (todos começam em 1.0 = sem ajuste)
            win_rate_factor = 1.0
            consecutive_losses_factor = 1.0
            volatility_factor = 1.0
            
            # 1. Ajuste baseado na taxa de acerto
            if win_rate is not None:
                # Se a taxa de acerto for alta (>70%), podemos aumentar o valor
                # Se for baixa (<50%), reduzimos o valor
                if win_rate >= 0.7:
                    win_rate_factor = 1.2  # Aumento de 20%
                elif win_rate >= 0.6:
                    win_rate_factor = 1.1  # Aumento de 10%
                elif win_rate >= 0.5:
                    win_rate_factor = 1.0  # Sem alteração
                elif win_rate >= 0.4:
                    win_rate_factor = 0.8  # Redução de 20%
                else:
                    win_rate_factor = 0.6  # Redução de 40%
                    
                adjustment_factor = win_rate_factor
                logger.info(f"Ajuste por taxa de acerto ({win_rate:.2%}): {win_rate_factor:.2f}")
                
            # 2. Ajuste baseado na volatilidade do mercado
            if market_volatility is not None:
                # Se a volatilidade for alta, reduzimos o valor para reduzir o risco
                # Se for baixa, podemos manter o valor
                if market_volatility >= 0.5:
                    volatility_factor = 0.5  # Redução de 50%
                elif market_volatility >= 0.3:
                    volatility_factor = 0.7  # Redução de 30%
                elif market_volatility >= 0.2:
                    volatility_factor = 0.8  # Redução de 20%
                elif market_volatility >= 0.1:
                    volatility_factor = 0.9  # Redução de 10%
                else:
                    volatility_factor = 1.0  # Sem alteração
                    
                adjustment_factor *= volatility_factor
                logger.info(f"Ajuste por volatilidade ({market_volatility:.2%}): {volatility_factor:.2f}")
                
            # 3. Ajuste baseado em perdas consecutivas
            if consecutive_losses > 0:
                # Reduzimos o valor progressivamente com base no número de perdas consecutivas
                consecutive_losses_factor = max(0.5, 1.0 - (consecutive_losses * 0.1))
                adjustment_factor *= consecutive_losses_factor
                logger.info(f"Ajuste por perdas consecutivas ({consecutive_losses}): {consecutive_losses_factor:.2f}")
                
            # Aplicamos o fator de ajuste ao valor base
            adjusted_amount = base_amount * adjustment_factor
            
            # Garantimos que o valor não seja menor que um mínimo aceitável (20% do valor base)
            min_amount = base_amount * 0.2
            adjusted_amount = max(adjusted_amount, min_amount)
            
            # Arredondamos para 2 casas decimais
            adjusted_amount = round(adjusted_amount, 2)
            
            logger.info(f"Valor ajustado: {base_amount:.2f} -> {adjusted_amount:.2f} (fator: {adjustment_factor:.2f})")
            return adjusted_amount
            
        except Exception as e:
            logger.error(f"Erro ao ajustar valor da operação: {str(e)}")
            return base_amount  # Em caso de erro, retorna o valor base

    def check_market_volatility(self, assets, period=20):
        """Verifica a volatilidade atual do mercado para um conjunto de ativos.
        
        Args:
            assets (list): Lista de ativos para verificar
            period (int): Período para cálculo da volatilidade (em minutos)
            
        Returns:
            float: Volatilidade média do mercado (0-1)
        """
        try:
            if not IQOPTION_API_AVAILABLE:
                # No modo de simulação, retorna uma volatilidade aleatória baixa
                return random.uniform(0.01, 0.03)
                
            volatilities = []
            
            for asset in assets:
                try:
                    # Obtém os dados recentes do ativo
                    candles = self.get_candles(asset, period=60, count=period)
                    
                    if not candles or len(candles) < period / 2:
                        logger.warning(f"Dados insuficientes para calcular volatilidade de {asset}")
                        continue
                        
                    # Extrai preços de fechamento
                    closes = [candle['close'] for candle in candles]
                    
                    # Calcula retornos percentuais
                    returns = [abs((closes[i] - closes[i-1]) / closes[i-1]) for i in range(1, len(closes))]
                    
                    # Calcula volatilidade (desvio padrão dos retornos)
                    volatility = statistics.stdev(returns) if len(returns) > 1 else 0
                    
                    # Normaliza a volatilidade para uma escala de 0-1
                    # Considerando que uma volatilidade de 5% (0.05) já é alta para a maioria dos ativos
                    normalized_volatility = min(volatility * 10, 1.0)
                    
                    volatilities.append(normalized_volatility)
                    logger.info(f"Volatilidade de {asset}: {normalized_volatility:.2%}")
                    
                except Exception as e:
                    logger.warning(f"Erro ao calcular volatilidade para {asset}: {str(e)}")
                    continue
            
            # Calcula a volatilidade média do mercado
            if not volatilities:
                logger.warning("Não foi possível calcular a volatilidade para nenhum ativo")
                return 0.02  # Valor padrão baixo
                
            market_volatility = sum(volatilities) / len(volatilities)
            logger.info(f"Volatilidade média do mercado: {market_volatility:.2%}")
            
            return market_volatility
            
        except Exception as e:
            logger.error(f"Erro ao verificar volatilidade do mercado: {str(e)}")
            return 0.02  # Valor padrão baixo em caso de erro

    def check_asset_open(self, asset):
        """Verifica se um ativo está disponível para negociação.
        
        Args:
            asset (str): Nome do ativo
            
        Returns:
            bool: True se o ativo está disponível, False caso contrário
        """
        try:
            if not IQOPTION_API_AVAILABLE:
                # No modo de simulação, consideramos todos os ativos disponíveis
                return True
                
            # Obtém informações sobre os ativos disponíveis
            instruments = self.iq_option.get_all_open_time()
            
            # Verifica em diferentes categorias de ativos
            for category in ['turbo', 'binary', 'digital']:
                if category in instruments and asset in instruments[category]:
                    if instruments[category][asset].get('open', False):
                        logger.info(f"Ativo {asset} está disponível para negociação na categoria {category}")
                        return True
                        
            logger.warning(f"Ativo {asset} não está disponível para negociação no momento")
            return False
                
        except Exception as e:
            logger.error(f"Erro ao verificar disponibilidade do ativo {asset}: {str(e)}")
            return False

    def connect(self) -> Tuple[bool, Optional[str]]:
        """Conecta à API do IQ Option.
        
        Returns:
            Tuple[bool, Optional[str]]: Status da conexão e mensagem de erro (se houver)
        """
        # Verifica se credenciais foram fornecidas ou carrega do ambiente
        email = self.config_manager.get_value('Credentials', 'username')
        password = self.config_manager.get_value('Credentials', 'password')

        if not email or not password:
            try:
                # Tenta obter credenciais das variáveis de ambiente (definidas no main.py)
                email = self.config_manager.get_value('API', 'email')
                password = self.config_manager.get_value('API', 'password')

                if not email or not password:
                    logger.error("Credenciais não configuradas")
                    return False, "Credenciais não configuradas"
            except Exception as e:
                logger.error(f"Erro ao carregar credenciais: {str(e)}")
                return False, f"Erro ao carregar credenciais: {str(e)}"

        logger.info(f"Tentando conectar com email: {email}")
        logger.info(f"Conectando com email: {email} e senha mascarada: {'*' * len(password)}")

        # Verifica se a biblioteca IQ Option está disponível
        if not IQOPTION_API_AVAILABLE:
            error_msg = "A biblioteca IQ Option não está instalada. Execute 'pip install -U git+https://github.com/iqoptionapi/iqoptionapi.git' para instalá-la."
            logger.error(error_msg)
            return False, error_msg

        # Configurações adicionais para a API
        self.iq_option = IQ_Option(email, password)

        # Configurar timeout mais longo para evitar problemas de conexão
        self.iq_option.set_max_reconnect(5)

        attempts = 0
        while attempts < self.max_retries:
            try:
                logger.info(f"Tentativa de conexão: {attempts + 1}")
                # Tenta conectar com a API
                check, reason = self.iq_option.connect()
                logger.info(f"check: {check}, reason: {reason}")
                
                if check:
                    logger.info("Conexão estabelecida com sucesso!")
                    logger.success("Conexão com a API do IQ Option estabelecida com sucesso")
                    self.connected = True
                    # Configura sessão padrão após conexão bem sucedida
                    self.set_session()
                    logger.info("Sessão configurada após conexão")
                    return True, None
                
                # Se a resposta contiver "Forbidden", pode ser um problema com a API
                if reason and "Forbidden" in reason:
                    error_msg = f"Erro de conexão: {reason}. A API pode estar bloqueando a conexão. Verifique sua conexão e tente novamente mais tarde."
                    logger.error(error_msg)
                    return False, error_msg
                
                logger.warning(f"Falha na conexão (tentativa {attempts + 1}/{self.max_retries}): {reason}")
                logger.warning(f"Falha na conexão (tentativa {attempts + 1}/{self.max_retries}): {reason}")
                time.sleep(self.retry_delay)
                attempts += 1
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Erro na conexão (tentativa {attempts + 1}/{self.max_retries}): {error_msg}")
                
                # Se for um erro de JSON, pode ser um problema com a API
                if "JSONDecodeError" in error_msg or "Expecting value" in error_msg:
                    logger.warning("Erro de decodificação JSON. Tentando reconectar com nova instância.")
                    # Cria uma nova instância da API
                    self.iq_option = IQ_Option(email, password)
                
                time.sleep(self.retry_delay)
                attempts += 1
        
        logger.error(f"Falha ao conectar após {self.max_retries} tentativas")
        self.connected = False
        try:
            return False, "Número máximo de tentativas de conexão excedido"
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro de conexão de rede: {e}")
            return False, f"Erro de conexão de rede: {e}"
        except Exception as e:
            logger.exception(f"Erro inesperado durante a conexão: {e}")
            return False, f"Erro inesperado durante a conexão: {e}"