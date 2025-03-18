import os
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Set, Literal
from dataclasses import dataclass
from functools import lru_cache
from loguru import logger
from iqoptionapi.stable_api import IQ_Option
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
from statistics import mean, stdev
from threading import Lock

# Tipos customizados para melhorar legibilidade
ExpirationMode = Literal['turbo', 'binary', 'digital']
TradeAction = Literal['call', 'put']
TimeframeType = Literal['Seconds', 'Minutes', 'Hours']

class ExpirationStrategy(Enum):
    """Estratégias de expiração disponíveis"""
    TURBO = auto()
    BINARY = auto()
    DIGITAL = auto()
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
    
    def __init__(self, asset_pairs: List[str] = [], max_retries: int = 3, retry_delay: int = 5):
        if not hasattr(self, 'initialized'):  # Evita reinicialização
            self.iq_option = None
            self.asset_pairs = asset_pairs
            self.connected = False
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            self.risk_management = {
                'max_daily_loss': 0.05,  # 5% do saldo
                'max_trade_risk': 0.02,  # 2% por operação
                'max_consecutive_losses': 3,
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

    def connect(self) -> Tuple[bool, Optional[str]]:
        """Conecta à API do IQ Option com reconexão automática.
        
        Returns:
            Tuple[bool, Optional[str]]: Status da conexão e mensagem de erro (se houver)
        """
        # Verifica e reseta métricas diárias se necessário
        self.check_and_reset_daily_metrics()
        
        # Verifica credenciais
        try:
            email = os.environ["IQ_OPTION_EMAIL"]
            password = os.environ["IQ_OPTION_PASSWORD"]
        except KeyError:
            logger.error("Variáveis de ambiente IQ_OPTION_EMAIL e IQ_OPTION_PASSWORD devem ser configuradas")
            return False, "Credenciais ausentes"
            
        attempts = 0
        while attempts < self.max_retries:
            try:
                self.iq_option = IQ_Option(email, password)
                check, reason = self.iq_option.connect()
                
                if check:
                    logger.success("Conexão com a API do IQ Option estabelecida com sucesso")
                    self.connected = True
                    # Configura sessão padrão após conexão bem sucedida
                    self.set_session()
                    return True, None
                
                logger.warning(f"Falha na conexão (tentativa {attempts + 1}/{self.max_retries}): {reason}")
                time.sleep(self.retry_delay)
                attempts += 1
                
            except Exception as e:
                logger.error(f"Erro na conexão (tentativa {attempts + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay)
                attempts += 1
        
        logger.error(f"Falha ao conectar após {self.max_retries} tentativas")
        self.connected = False
        return False, "Max retries exceeded"
            
        attempts = 0
        while attempts < self.max_retries:
            try:
                self.iq_option = IQ_Option(email, password)
                check, reason = self.iq_option.connect()
                
                if check:
                    logger.success("Conexão com a API do IQ Option estabelecida com sucesso")
                    self.connected = True
                    return True, None
                
                logger.warning(f"Falha na conexão (tentativa {attempts + 1}/{self.max_retries}): {reason}")
                time.sleep(self.retry_delay)
                attempts += 1
                
            except Exception as e:
                logger.error(f"Erro na conexão (tentativa {attempts + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay)
                attempts += 1
        
        logger.error(f"Falha ao conectar após {self.max_retries} tentativas")
        self.connected = False
        return False, "Max retries exceeded"

    def check_connection(self) -> bool:
        """Verifica se a conexão com a API está ativa.
        
        Returns:
            bool: True se conectado, False caso contrário
        """
        if not self.connected:
            return False
            
        return self.iq_option.check_connect()

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
        """Obtém lista de instrumentos disponíveis para operações digitais spot.
        
        Returns:
            Lista de dicionários com informações dos instrumentos ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            instruments = self.iq_option.get_digital_spot_instruments()
            
            if not instruments or not isinstance(instruments, list):
                logger.error("Dados de instrumentos inválidos recebidos")
                return None
                
            required_keys = ['id', 'name', 'active', 'underlying', 'group']
            for instrument in instruments:
                if not isinstance(instrument, dict) or not all(key in instrument for key in required_keys):
                    logger.error("Dados de instrumentos faltando campos obrigatórios ou formato inválido")
                    return None
                    
            logger.success(f"Obteve {len(instruments)} instrumentos digitais spot")
            return instruments
            
        except Exception as e:
            logger.error(f"Erro ao obter instrumentos digitais spot: {str(e)}")
            return None

    def get_digital_spot_profit(self, asset: str) -> Optional[float]:
        """Obtém o lucro percentual esperado para um ativo digital spot.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            float: Lucro percentual esperado ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            profit = self.iq_option.get_digital_spot_profit(asset)
            
            if not isinstance(profit, (int, float)):
                logger.error("Dados de lucro inválidos recebidos")
                return None
                
            logger.success(f"Obteve lucro de {profit:.2%} para {asset}")
            return profit
            
        except Exception as e:
            logger.error(f"Erro ao obter lucro digital spot: {str(e)}")
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
            status, order_id = self.iq_option.buy(amount, asset, action, expiration_mode)
            
            if status:
                logger.success(f"Operação {action} de {amount} em {asset} executada com sucesso. ID: {order_id}")
                
                # Atualiza métricas de risco após operação bem sucedida
                self.risk_management['consecutive_losses'] = 0
                return True, order_id
                
            logger.error(f"Falha na execução da operação {action} em {asset}")
            
            # Atualiza métricas de risco após falha
            self.risk_management['consecutive_losses'] += 1
            self.risk_management['daily_loss'] += amount / balance
            
            return False, None
            
        except Exception as e:
            logger.error(f"Erro ao executar operação: {str(e)}")
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
            float: Preço atual do ativo ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            # Tenta obter candles em tempo real e usar o preço do último candle
            self.start_candles_stream(asset, 60, 10)
            candles = self.get_realtime_candles(asset, 60)
            
            if candles and len(candles) > 0:
                # Obtém o último candle disponível
                last_candle = list(candles.values())[-1]
                price = last_candle.get('close', None)
                
                if price is not None:
                    logger.success(f"Preço atual de {asset}: {price}")
                    self.stop_candles_stream(asset, 60)
                    return price
            
            # Backup se não conseguir obter pelos candles em tempo real
            instruments = self.iq_option.get_all_open_time()
            if asset in instruments['turbo'] and instruments['turbo'][asset]['open']:
                price = self.iq_option.get_price_raw(asset)
                logger.success(f"Preço atual de {asset}: {price}")
                return price
                
            logger.error(f"Não foi possível obter preço atual para {asset}")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter preço atual: {str(e)}")
            self.stop_candles_stream(asset, 60)
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
            status, order_id = self.iq_option.buy_digital_spot(asset, amount, action, duration)
            
            if status:
                logger.success(f"Operação digital spot {action} de {amount} em {asset} executada com sucesso. ID: {order_id}")
                
                # Atualiza métricas de risco após operação bem sucedida
                self.risk_management['consecutive_losses'] = 0
                return True, order_id
                
            logger.error(f"Falha na execução da operação digital spot {action} em {asset}")
            
            # Atualiza métricas de risco após falha
            self.risk_management['consecutive_losses'] += 1
            self.risk_management['daily_loss'] += amount / balance
            
            return False, None
            
        except Exception as e:
            logger.error(f"Erro ao executar operação digital spot: {str(e)}")
            return False, None

    def check_win_digital(self, order_id: int, polling_time: int = 2) -> Tuple[bool, Optional[float]]:
        """Verifica o resultado de uma operação digital spot.
        
        Args:
            order_id: ID da ordem a ser verificada
            polling_time: Intervalo entre verificações em segundos
            
        Returns:
            Tuple[bool, Optional[float]]: Status da verificação e valor do lucro (se disponível)
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False, None
            
        try:
            check_close, win_money = self.iq_option.check_win_digital_v2(order_id, polling_time)
            
            if check_close:
                if win_money is not None:
                    logger.success(f"Resultado da operação {order_id} verificado. Lucro: {win_money}")
                else:
                    logger.warning(f"Operação {order_id} finalizada sem lucro")
                return True, win_money
                
            logger.info(f"Operação {order_id} ainda em andamento")
            return False, None
            
        except Exception as e:
            logger.error(f"Erro ao verificar resultado da operação: {str(e)}")
            return False, None

    def start_candles_stream(self, asset: str, timeframe: int, max_buffersize: int = 10) -> bool:
        """Inicia o streaming de candles em tempo real.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            timeframe: Duração do candle em segundos
            max_buffersize: Tamanho máximo do buffer de candles
            
        Returns:
            bool: True se o streaming foi iniciado com sucesso
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False
            
        try:
            self.iq_option.start_candles_stream(asset, timeframe, max_buffersize)
            logger.success(f"Streaming de candles iniciado para {asset} com timeframe {timeframe}s")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar streaming de candles: {str(e)}")
            return False

    def get_realtime_candles(self, asset: str, timeframe: int) -> Optional[Dict]:
        """Obtém candles em tempo real.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            timeframe: Duração do candle em segundos
            
        Returns:
            Dicionário com candles em tempo real ou None em caso de erro
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            candles = self.iq_option.get_realtime_candles(asset, timeframe)
            
            if not candles or not isinstance(candles, dict):
                logger.error("Dados de candles inválidos recebidos")
                return None
                
            logger.success(f"Obteve candles em tempo real para {asset}")
            return candles
            
        except Exception as e:
            logger.error(f"Erro ao obter candles em tempo real: {str(e)}")
            return None

    def stop_candles_stream(self, asset: str, timeframe: int) -> bool:
        """Para o streaming de candles em tempo real.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            timeframe: Duração do candle em segundos
            
        Returns:
            bool: True se o streaming foi parado com sucesso
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return False
            
        try:
            self.iq_option.stop_candles_stream(asset, timeframe)
            logger.success(f"Streaming de candles parado para {asset}")
            return True
            
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
                        df['timeframe'] = tf_value * 60  # Converte para segundos
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
            
    def get_available_assets(self) -> List[str]:
        """Obtém lista de ativos disponíveis na plataforma.
        
        Returns:
            Lista de ativos disponíveis
        """
        if not self.connected:
            logger.error("Não conectado à API do IQ Option")
            return []
            
        try:
            # Obtém instrumentos para diferentes categorias
            all_assets = set()
            
            # Forex
            forex = self.iq_option.get_all_open_time()
            
            for market_type in forex:
                for asset in forex[market_type]:
                    if forex[market_type][asset]['open']:
                        all_assets.add(asset)
            
            logger.success(f"Obteve {len(all_assets)} ativos disponíveis")
            return list(all_assets)
            
        except Exception as e:
            logger.error(f"Erro ao obter ativos disponíveis: {str(e)}")
            return []
            
    def send_notification(self, message: str) -> None:
        """Envia notificação para o usuário.
        
        Args:
            message: Mensagem a ser enviada
        """
        logger.info(f"NOTIFICAÇÃO: {message}")
        # Em uma implementação real, poderia enviar por email, SMS ou push notification
