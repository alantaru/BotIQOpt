import os
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Set, Literal
from dataclasses import dataclass
from functools import lru_cache
import logging
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
    # Usar logging diretamente aqui, pois self.logger ainda não foi inicializado
    logging.warning("Biblioteca iqoptionapi não encontrada. Usando implementação alternativa.")
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
    
    def __init__(self, config_manager, error_tracker): # Adicionado error_tracker
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
            self.logger = logging.getLogger('Ferramental')
            self.error_tracker = error_tracker # Armazena a instância do error_tracker
            self.initialized = True


    def reset_daily_metrics(self) -> None:
        """Reseta as métricas diárias de risco"""
        self.risk_management['daily_loss'] = 0.0
        self.risk_management['consecutive_losses'] = 0
        self.risk_management['last_reset'] = time.time()
        self.logger.info("Métricas diárias de risco resetadas")

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
        self.logger.info("Sessão configurada com sucesso")
        self.logger.info("Sessão configurada após conexão")

    def connect_2fa(self, code: str) -> Tuple[bool, Optional[str]]:
        """Completa a autenticação de dois fatores.
        
        Args:
            code: Código SMS recebido
            
        Returns:
            Tuple[bool, Optional[str]]: Status da conexão e mensagem de erro (se houver)
        """
        try:
            check, reason = self.iq_option.connect_2fa(code)
            if check:
                self.connected = True
                self.logger.info("Autenticação 2FA concluída com sucesso")
                return True, None
            else:
                self.logger.error(f"Erro na autenticação 2FA: {reason}")
                return False, str(reason)
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao processar 2FA: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("Connect2FAError", str(e), tb_str)
            return False, str(e)

    def get_balance(self) -> float:
        """Obtém o saldo da conta usando o método mais preciso (v2).
        
        Returns:
            float: Saldo atual da conta
        """
        try:
            return self.iq_option.get_balance_v2()
        except:
            self.logger.warning("Erro ao usar get_balance_v2, tentando get_balance")
            return self.iq_option.get_balance()

    def reconnect(self) -> Tuple[bool, Optional[str]]:
        """Tenta reconectar à API.
        
        Returns:
            Tuple[bool, Optional[str]]: Status da conexão e mensagem de erro (se houver)
        """
        self.logger.info("Tentando reconectar...")
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
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        self.logger.warning("Operações Digital Spot não estão mais disponíveis. O bot opera exclusivamente com Opções Binárias.")
        
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
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        self.logger.warning("Operações Digital Spot não estão mais disponíveis. O bot opera exclusivamente com Opções Binárias.")
        
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
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            indicators = self.iq_option.get_technical_indicators(asset)
            
            if not indicators or not isinstance(indicators, dict):
                self.logger.error("Dados de indicadores técnicos inválidos recebidos")
                return None
                
            required_keys = ['trend', 'oscillators', 'moving_averages']
            if not all(key in indicators for key in required_keys):
                self.logger.error("Dados de indicadores técnicos faltando campos obrigatórios")
                return None
                
            self.logger.info(f"Obteve indicadores técnicos para {asset}")
            return indicators
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter indicadores técnicos: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetIndicatorsError", str(e), tb_str)
            return None

    def get_support_resistance(self, asset: str) -> Optional[Dict]:
        """Obtém níveis de suporte e resistência para um ativo específico.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            Dicionário com níveis de suporte e resistência ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            levels = self.iq_option.get_support_resistance(asset)
            
            if not levels or not isinstance(levels, dict):
                self.logger.error("Dados de suporte/resistência inválidos recebidos")
                return None
                
            required_keys = ['support', 'resistance']
            if not all(key in levels for key in required_keys):
                self.logger.error("Dados de suporte/resistência faltando campos obrigatórios")
                return None
                
            self.logger.info(f"Obteve níveis de suporte/resistência para {asset}")
            return levels
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter níveis de suporte/resistência: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetSupportResistanceError", str(e), tb_str)
            return None

    def enable_auto_reconnect(self, interval: int = 300) -> None:
        """Habilita reconexão automática com a API.
        
        Args:
            interval: Intervalo entre verificações de conexão em segundos
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return
            
        try:
            self.iq_option.enable_auto_reconnect(interval)
            self.logger.info(f"Reconexão automática habilitada com intervalo de {interval} segundos")
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao habilitar reconexão automática: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("EnableAutoReconnectError", str(e), tb_str)

    def get_session_info(self) -> Optional[Dict]:
        """Obtém informações da sessão atual.
        
        Returns:
            Dicionário com informações da sessão ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            session_info = self.iq_option.get_session_info()
            
            if not session_info or not isinstance(session_info, dict):
                self.logger.error("Dados de sessão inválidos recebidos")
                return None
                
            required_keys = ['session_id', 'expires_in', 'user_id']
            if not all(key in session_info for key in required_keys):
                self.logger.error("Dados de sessão faltando campos obrigatórios")
                return None
                
            self.logger.info("Informações da sessão obtidas com sucesso")
            return session_info
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter informações da sessão: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetSessionInfoError", str(e), tb_str)
            return None

    def enable_two_factor_auth(self, code: str) -> bool:
        """Habilita autenticação de dois fatores.
        
        Args:
            code: Código de autenticação
            
        Returns:
            bool: True se autenticação foi bem sucedida, False caso contrário
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return False
            
        try:
            result = self.iq_option.enable_two_factor_auth(code)
            if result:
                self.logger.info("Autenticação de dois fatores habilitada com sucesso")
            else:
                self.logger.error("Falha ao habilitar autenticação de dois fatores")
            return result
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao habilitar autenticação de dois fatores: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("Enable2FAError", str(e), tb_str)
            return False

    def get_asset_groups(self) -> Optional[List[str]]:
        """Obtém lista de grupos de ativos disponíveis.
        
        Returns:
            Lista de nomes de grupos ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            groups = self.iq_option.get_all_asset_groups()
            
            if not groups or not isinstance(groups, list):
                self.logger.error("Dados de grupos de ativos inválidos recebidos")
                return None
                
            self.logger.info(f"Obteve {len(groups)} grupos de ativos")
            return groups
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter grupos de ativos: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetAssetGroupsError", str(e), tb_str)
            return None

    def get_assets_by_group(self, group_name: str) -> Optional[List[Dict]]:
        """Obtém lista de ativos de um grupo específico.
        
        Args:
            group_name: Nome do grupo (ex: 'forex')
            
        Returns:
            Lista de dicionários com informações dos ativos ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            assets = self.iq_option.get_all_assets_by_group(group_name)
            
            if not assets or not isinstance(assets, list):
                self.logger.error("Dados de ativos inválidos recebidos")
                return None
                
            required_keys = ['id', 'name', 'active', 'underlying', 'group']
            for asset in assets:
                if not isinstance(asset, dict) or not all(key in asset for key in required_keys):
                    self.logger.error("Dados de ativos faltando campos obrigatórios ou formato inválido")
                    return None
                    
            self.logger.info(f"Obteve {len(assets)} ativos para o grupo {group_name}")
            return assets
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter ativos do grupo {group_name}: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetAssetsByGroupError", str(e), tb_str)
            return None

    def get_technical_analysis(self, asset: str) -> Optional[Dict]:
        """Obtém análise técnica completa para um ativo.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            Dicionário com análise técnica ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            analysis = self.iq_option.get_technical_analysis(asset)
            
            if not analysis or not isinstance(analysis, dict):
                self.logger.error("Dados de análise técnica inválidos recebidos")
                return None
                
            required_keys = ['trend', 'oscillators', 'moving_averages', 'summary']
            if not all(key in analysis for key in required_keys):
                self.logger.error("Dados de análise técnica faltando campos obrigatórios")
                return None
                
            self.logger.info(f"Obteve análise técnica para {asset}")
            return analysis
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter análise técnica: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetTechnicalAnalysisError", str(e), tb_str)
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
            self.logger.error("Não conectado à API do IQ Option")
            return None

        # Verifica se a biblioteca IQ Option está disponível
        if not IQOPTION_API_AVAILABLE:
            error_msg = "A biblioteca IQ Option não está instalada. Execute 'pip install -U git+https://github.com/iqoptionapi/iqoptionapi.git' para instalá-la."
            self.logger.error(error_msg)
            return None

        # Valida ativo
        if asset not in self.asset_pairs:
            self.logger.error(f"Ativo {asset} não configurado")
            return None

        # Valida e converte timeframe de forma mais eficiente
        timeframe_multipliers = {
            "Seconds": 1,
            "Minutes": 60,
            "Hours": 3600
        }
        
        if timeframe_type not in timeframe_multipliers:
            self.logger.error(f"Tipo de timeframe inválido: {timeframe_type}. Opções válidas: {list(timeframe_multipliers.keys())}")
            return None
            
        try:
            timeframe = int(timeframe_value) * timeframe_multipliers[timeframe_type]
            if timeframe <= 0:
                self.logger.error(f"Valor de timeframe deve ser positivo: {timeframe_value}")
                return None
        except (ValueError, TypeError):
            self.logger.error(f"Valor de timeframe inválido: {timeframe_value}")
            return None

        try:
            endtime = endtime if endtime else time.time()
            candles = self.iq_option.get_candles(asset, timeframe, count, endtime)
            
            # Valida estrutura das velas
            if not candles or not isinstance(candles, list) or len(candles) == 0:
                self.logger.error("Dados de velas inválidos recebidos")
                return None
                
            required_keys = ['id', 'from', 'to', 'open', 'close', 'min', 'max', 'volume']
            for candle in candles:
                if not isinstance(candle, dict) or not all(key in candle for key in required_keys):
                    self.logger.error("Dados de velas faltando campos obrigatórios ou formato inválido")
                    return None
                    
                # Valida tipos dos valores
                try:
                    float(candle['open'])
                    float(candle['close'])
                    float(candle['min'])
                    float(candle['max'])
                    int(candle['volume'])
                except (ValueError, TypeError):
                    self.logger.error("Dados de velas com tipos inválidos")
                    return None
                    
                # Valida timestamps
                try:
                    if not isinstance(candle['from'], int) or not isinstance(candle['to'], int):
                        self.logger.error("Timestamps inválidos nas velas")
                        return None
                    if candle['from'] >= candle['to']:
                        self.logger.error("Timestamp 'from' deve ser menor que 'to'")
                        return None
                except KeyError:
                    self.logger.error("Faltando timestamps nas velas")
                    return None
                
            self.logger.info(f"Obteve {len(candles)} velas para {asset}")
            return candles
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter velas: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetCandlesError", str(e), tb_str)
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
            self.logger.error("Não conectado à API do IQ Option")
            return False, None
            
        # Verifica limites de risco
        balance = self.get_balance()
        if not balance:
            self.logger.error("Não foi possível obter o saldo para verificação de risco")
            return False, None
            
        # Verifica risco máximo por operação
        risk_per_trade = amount / balance
        if risk_per_trade > self.risk_management['max_trade_risk']:
            self.logger.error(f"Risco por operação {risk_per_trade:.2%} excede o limite de {self.risk_management['max_trade_risk']:.2%}")
            return False, None
            
        # Verifica perda diária máxima
        if self.risk_management['daily_loss'] >= self.risk_management['max_daily_loss']:
            self.logger.error(f"Perda diária {self.risk_management['daily_loss']:.2%} atingiu o limite de {self.risk_management['max_daily_loss']:.2%}")
            return False, None
            
        # Verifica perdas consecutivas
        if self.risk_management['consecutive_losses'] >= self.risk_management['max_consecutive_losses']:
            self.logger.error(f"{self.risk_management['consecutive_losses']} perdas consecutivas atingiram o limite de {self.risk_management['max_consecutive_losses']}")
            return False, None

        # Valida ativo
        if asset not in self.asset_pairs:
            self.logger.error(f"Ativo {asset} não configurado")
            return False, None

        # Valida ação de forma mais eficiente
        valid_actions = {'call', 'put'}
        action = action.lower()
        if action not in valid_actions:
            self.logger.error(f"Ação inválida: {action}. Opções válidas: {valid_actions}")
            return False, None

        # Valida valor mínimo
        min_amount = 1.0  # Valor mínimo da plataforma
    def verify_order_execution(self, order_id: int, max_wait_seconds: int = 5) -> bool:
        """Verifica se uma ordem foi realmente registrada na plataforma.

        Tenta obter informações da ordem por um tempo limitado.

        Args:
            order_id: ID da ordem a ser verificada.
            max_wait_seconds: Tempo máximo de espera em segundos.

        Returns:
            bool: True se a ordem foi encontrada, False caso contrário.
        """
        if not self.connected or not IQOPTION_API_AVAILABLE:
            self.logger.warning(f"Não é possível verificar a ordem {order_id} (não conectado ou API indisponível)")
            # Em simulação ou sem API, assumimos que a ordem foi criada se a chamada buy retornou ID
            return True if not IQOPTION_API_AVAILABLE else False

        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            try:
                # Tenta obter informações da aposta/ordem
                is_successful, bet_info = self.iq_option.get_betinfo(order_id)

                if is_successful and bet_info:
                    self.logger.info(f"Ordem {order_id} encontrada no histórico.")
                    return True
                elif bet_info is False: # API retorna False se a ordem não existe ou não fechou ainda
                    self.logger.debug(f"Aguardando informações da ordem {order_id}...")
                else:
                    # Pode ter retornado um dict vazio ou outra coisa
                    self.logger.warning(f"Resposta inesperada de get_betinfo para ordem {order_id}: {bet_info}")

            except Exception as e:
                self.logger.error(f"Erro ao verificar ordem {order_id} com get_betinfo: {str(e)}")
                # Consideramos que não foi possível verificar
                return False
            
            time.sleep(0.5)  # Pequena pausa antes de tentar novamente

        self.logger.error(f"Não foi possível confirmar a execução da ordem {order_id} após {max_wait_seconds} segundos.")
        return False


        if amount < min_amount:
            self.logger.error(f"Valor mínimo da operação é {min_amount}")
            return False, None

        try:
            # Verifica se o ativo está disponível para negociação
            if not self.check_asset_open(asset):
                self.logger.error(f"Ativo {asset} não está disponível para negociação no momento")
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
                            self.logger.info(f"Operação {action} de {amount} em {asset} executada com sucesso. ID: {order_id}")
                            
                            # Atualiza métricas de risco após operação bem sucedida
                            self.risk_management['consecutive_losses'] = 0
                            return True, order_id
                        else:
                            self.logger.warning(f"Ordem {order_id} não encontrada no histórico. Verificando novamente...")
                            time.sleep(1)  # Aguarda um pouco para atualização do histórico
                            
                            # Verifica novamente
                            if self.verify_order_execution(order_id):
                                self.logger.info(f"Ordem {order_id} confirmada após segunda verificação")
                                self.risk_management['consecutive_losses'] = 0
                                return True, order_id
                            else:
                                self.logger.error(f"Ordem {order_id} não confirmada após segunda verificação")
                                status = False
                    
                    if not status:
                        if retry < max_retries - 1:
                            self.logger.warning(f"Falha na execução da operação {action} em {asset}. Tentativa {retry+1}/{max_retries}")
                            time.sleep(retry_delay)
                        else:
                            self.logger.error(f"Falha na execução da operação {action} em {asset} após {max_retries} tentativas")
                            
                            # Falha na execução após retries
                            return False, None
                            
                except Exception as e:
                    if retry < max_retries - 1:
                        self.logger.warning(f"Erro na tentativa {retry+1}/{max_retries} de executar operação: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        tb_str = traceback.format_exc()
                        self.logger.error(f"Erro persistente ao executar operação após {max_retries} tentativas: {str(e)}\n{tb_str}")
                        self.error_tracker.add_error("BuyRetryError", str(e), tb_str)
                        # Falha na execução após retries (exceção)
                        return False, None
            
            return False, None
                
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao executar operação: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("BuyExecutionError", str(e), tb_str)
            # Falha na execução (exceção externa)
            return False, None

    def get_balance_v2(self) -> Optional[float]:
        """Obtém o saldo da conta com maior precisão.

        Returns:
            float: Saldo da conta ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return None

        try:
            balance = self.iq_option.get_balance_v2()
            self.logger.info(f"Saldo (v2) obtido: {balance}")
            return balance
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter saldo (v2): {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetBalanceV2Error", str(e), tb_str)
            return None

    def get_currency(self) -> Optional[str]:
        """Obtém a moeda da conta.

        Returns:
            str: Moeda da conta ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return None

        try:
            currency = self.iq_option.get_currency()
            self.logger.info(f"Moeda da conta: {currency}")
            return currency
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter moeda da conta: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetCurrencyError", str(e), tb_str)
            return None
            
    def get_balance(self) -> float:
        """Obtém o saldo da conta usando o método mais preciso (v2).
        
        Returns:
            float: Saldo atual da conta
        """
        try:
            return self.iq_option.get_balance_v2()
        except:
            self.logger.warning("Erro ao usar get_balance_v2, tentando get_balance")
            return self.iq_option.get_balance()
            
    def get_min_trade_amount(self) -> float:
        """Obtém o valor mínimo por operação.
        
        Returns:
            float: Valor mínimo por operação
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
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
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter valor mínimo por operação: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetMinTradeAmountError", str(e), tb_str)
            return 1.0
            
    def get_max_trade_amount(self) -> float:
        """Obtém o valor máximo por operação.
        
        Returns:
            float: Valor máximo por operação
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
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
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter valor máximo por operação: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetMaxTradeAmountError", str(e), tb_str)
            return 20000.0
            
    def get_spread(self, asset: str) -> Optional[float]:
        """Obtém o spread atual de um ativo.
        
        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            
        Returns:
            float: Spread do ativo em percentual ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
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
                    
                self.logger.info(f"Spread de {asset}: {spread:.4%}")
                return spread
                
            self.logger.error(f"Não foi possível obter spread para {asset}")
            return 0.05  # Valor default de 5% em caso de falha
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter spread: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetSpreadError", str(e), tb_str)
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
                self.logger.warning(f"Alguns ativos não são válidos: {invalid_assets}")
                
            if not valid_assets:
                self.logger.error("Nenhum ativo válido para configuração")
                return False
                
            # Configura ativos da classe
            self.asset_pairs = valid_assets
            self.logger.info(f"Ativos configurados: {self.asset_pairs}")
            return True
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao configurar ativos: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("ConfigureAssetsError", str(e), tb_str)
            return False

    def reset_practice_balance(self) -> bool:
        """Reseta o saldo da conta de prática.

        Returns:
            bool: True se bem sucedido, False caso contrário
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return False

        try:
            success = self.iq_option.reset_practice_balance()
            if success:
                self.logger.info("Saldo da conta de prática resetado")
            else:
                self.logger.error("Falha ao resetar saldo da conta de prática")
            return success
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao resetar saldo da conta de prática: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("ResetPracticeBalanceError", str(e), tb_str)
            return False

    def change_balance(self, balance_type: str) -> bool:
        """Muda o tipo de conta (REAL, PRACTICE, TOURNAMENT).

        Args:
            balance_type: Tipo de conta para mudar

        Returns:
            bool: True se bem sucedido, False caso contrário
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return False

        valid_types = ["REAL", "PRACTICE", "TOURNAMENT"]
        if balance_type not in valid_types:
            self.logger.error(f"Tipo de conta inválido: {balance_type}. Tipos válidos: {valid_types}")
            return False

        try:
            self.iq_option.change_balance(balance_type)
            self.logger.info(f"Tipo de conta alterado para: {balance_type}")
            return True
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao mudar tipo de conta: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("ChangeBalanceError", str(e), tb_str)
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
            self.logger.error("Não conectado à API do IQ Option")
            return False, None
            
        self.logger.warning("Operações Digital Spot não estão mais disponíveis. Redirecionando para Opções Binárias.")
        
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
            self.logger.error("Não conectado à API do IQ Option")
            return False, None
            
        self.logger.warning("Operações Digital Spot não estão mais disponíveis. O bot opera exclusivamente com Opções Binárias.")
        
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
                self.logger.info(f"Iniciando streaming simulado para {asset} (timeframe: {timeframe}s)")
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
                self.logger.error("API IQ Option não inicializada")
                return False
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao iniciar streaming de candles: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("StartCandlesStreamError", str(e), tb_str)
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
                    self.logger.warning(f"Dados históricos simulados não disponíveis para {asset}")
                    return {}
            
            if self.iq_option:
                try:
                    # Tenta obter os candles em tempo real diretamente do objeto iq_option
                    return self.iq_option.get_realtime_candles(asset, timeframe)
                except AttributeError:
                    # Se o método get_realtime_candles não existir, tenta uma abordagem alternativa
                    self.logger.warning("Método get_realtime_candles não disponível, tentando alternativa")
                    
                    # Tenta obter os candles mais recentes como alternativa
                    candles = self.get_candles(asset, "Seconds", timeframe, 1)
                    if candles:
                        timestamp = int(datetime.now().timestamp())
                        candle = candles[0]
                        return {timestamp: candle}
                    return {}
            else:
                self.logger.error("API IQ Option não inicializada")
                return {}
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter candles em tempo real: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetRealtimeCandlesError", str(e), tb_str)
            return {}
            
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
                self.logger.info(f"Parando streaming simulado para {asset}")
                # No modo de simulação, apenas removemos o registro do streaming
                if hasattr(self, 'simulation_active_streams') and asset in self.simulation_active_streams:
                    del self.simulation_active_streams[asset]
                return True
            
            if self.iq_option:
                self.iq_option.stop_candles_stream(asset, timeframe)
                return True
            else:
                self.logger.error("API IQ Option não inicializada")
                return False
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao parar streaming de candles: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("StopCandlesStreamError", str(e), tb_str)
            return False

    def handle_two_factor_auth(self, code: str) -> bool:
        """Gerencia a autenticação de dois fatores.
        
        Args:
            code: Código de autenticação recebido
            
        Returns:
            bool: True se a autenticação foi bem sucedida
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return False
            
        try:
            status, reason = self.iq_option.connect_2fa(code)
            
            if status:
                self.logger.info("Autenticação de dois fatores concluída com sucesso")
                return True
                
            self.logger.error(f"Falha na autenticação de dois fatores: {reason}")
            return False
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao processar autenticação de dois fatores: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("Handle2FAError", str(e), tb_str)
            return False
            
    def get_realtime_data(self, asset: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Obtém dados históricos e em tempo real para análise.
        
        Args:
            asset (Optional[str]): Ativo específico para obter dados. Se None, obtém para todos os ativos configurados.
            
        Returns:
            DataFrame com dados para análise ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            # Obtém dados para todos os ativos configurados ou apenas para o ativo especificado
            all_data = []
            
            # Define a lista de ativos a serem processados
            assets_to_process = [asset] if asset else self.asset_pairs
            
            for current_asset in assets_to_process:
                # Tenta obter candles em tempo real para cada ativo
                candles = {}
                
                # Verifica se o streaming de velas está configurado
                for tf_seconds in [60, 300, 900]:  # 1m, 5m, 15m
                    # Inicia streaming se não estiver ativo
                    self.start_candles_stream(current_asset, tf_seconds)
                    # Obtém velas em tempo real
                    realtime_candles = self.get_realtime_candles(current_asset, tf_seconds)
                    
                    if realtime_candles:
                        # Converte para DataFrame
                        df = pd.DataFrame(list(realtime_candles.values()))
                        # Adiciona informações do ativo e timeframe
                        df['asset'] = current_asset
                        df['timeframe'] = tf_seconds
                        all_data.append(df)
                
                # Backup: Se não conseguiu dados em tempo real, tenta dados históricos
                if not all_data:
                    historical_candles = self.get_candles(current_asset, "Minutes", 1, 100)
                    if historical_candles:
                        df = pd.DataFrame(historical_candles)
                        df['asset'] = current_asset
                        df['timeframe'] = 60
                        all_data.append(df)
            
            # Combina todos os dados
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                self.logger.info(f"Obteve {len(combined_data)} registros de dados em tempo real")
                return combined_data
                
            self.logger.error("Não foi possível obter dados em tempo real")
            return None
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter dados em tempo real: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetRealtimeDataError", str(e), tb_str)
            return None
            
    def get_historical_data(self, asset: str = None, timeframe: int = 60, count: int = 1000) -> Optional[pd.DataFrame]:
        """Obtém dados históricos para treinamento ou análise.
        
        Args:
            asset (str): Nome do ativo (ex: "EURUSD")
            timeframe (int): Timeframe em segundos (60, 300, 900, etc.)
            count (int): Quantidade de candles a serem obtidos
            
        Returns:
            pd.DataFrame: DataFrame com dados históricos ou None em caso de erro
        """
        if not self.connected and not hasattr(self, 'simulation_mode'):
            self.logger.error("Não conectado à API do IQ Option")
            return None
            
        try:
            # Verifica se estamos em modo de simulação
            if hasattr(self, 'simulation_mode') and self.simulation_mode:
                if asset and asset.upper() in self.simulation_assets:
                    self.logger.info(f"Obtendo dados históricos simulados para {asset}")
                    return self.simulation_historical_data.get(asset.upper())
                else:
                    self.logger.error(f"Ativo {asset} não disponível em modo de simulação")
                    return None
            
            # Verifica se o ativo foi especificado
            if not asset:
                self.logger.error("Ativo não especificado")
                return None
                
            # Converte timeframe para o formato esperado pela API
            if timeframe == 60:
                tf_type, tf_value = "Minutes", 1
            elif timeframe == 300:
                tf_type, tf_value = "Minutes", 5
            elif timeframe == 900:
                tf_type, tf_value = "Minutes", 15
            elif timeframe == 3600:
                tf_type, tf_value = "Hours", 1
            else:
                tf_type, tf_value = "Minutes", 1  # Padrão: 1 minuto
                
            # Obtém candles históricos para o ativo
            self.logger.info(f"Obtendo {count} candles históricos para {asset} (timeframe: {timeframe}s)")
            
            if not IQOPTION_API_AVAILABLE:
                self.logger.warning("API IQ Option não disponível. Usando dados sintéticos.")
                # Gera dados sintéticos
                return self._generate_synthetic_data(asset, count)
            
            # Usa a API para obter dados reais
            historical_candles = self.iq_option.get_candles(asset, tf_value, tf_type, count)
            
            if not historical_candles or len(historical_candles) == 0:
                self.logger.error(f"Nenhum dado histórico encontrado para {asset}")
                return None
                
            # Converte para DataFrame
            df = pd.DataFrame(historical_candles)
            
            # Renomeia colunas para o formato padrão
            if 'from' in df.columns:
                df['timestamp'] = df['from']
                df.drop('from', axis=1, inplace=True)
                
            # Garante que temos as colunas necessárias
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Coluna {col} ausente nos dados históricos")
                    return None
                    
            # Ordena por timestamp
            if 'timestamp' in df.columns:
                df.sort_values('timestamp', inplace=True)
                
            self.logger.info(f"Obtidos {len(df)} candles históricos para {asset}")
            return df
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter dados históricos: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetHistoricalDataError", str(e), tb_str)
            import traceback # Manter import local para log detalhado
            self.logger.error(traceback.format_exc())
            return None
            
    def _generate_synthetic_data(self, asset, count=1000):
        """Gera dados sintéticos para testes quando a API não está disponível.
        
        Args:
            asset (str): Nome do ativo
            count (int): Quantidade de candles a serem gerados
            
        Returns:
            pd.DataFrame: DataFrame com dados sintéticos
        """
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        self.logger.info(f"Gerando {count} candles sintéticos para {asset}")
        
        # Define o preço base para o ativo
        if asset.upper() in self.simulation_assets:
            base_price = self.simulation_assets[asset.upper()]["price"]
        else:
            base_price = 100.0  # Valor padrão
        
        # Gera timestamps para os últimos 5000 minutos
        end_time = datetime.now()
        timestamps = [end_time - timedelta(minutes=i) for i in range(5000)]
        timestamps.reverse()  # Ordem cronológica
        
        # Gera preços simulados com tendência e volatilidade
        np.random.seed(42)  # Para reprodutibilidade
        
        # Usa os parâmetros de simulação definidos na instância
        volatility = getattr(self, 'simulation_volatility', 0.0002) # Usa valor padrão se atributo não existir
        trend_strength = getattr(self, 'simulation_trend', 0.00001) # Usa valor padrão se atributo não existir
        
        # Gera um passeio aleatório com a volatilidade configurada
        random_walk = np.random.normal(0, volatility, 5000).cumsum()
        # Gera a tendência com a força configurada
        trend = np.linspace(0, trend_strength * 5000, 5000)
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
        
        self.logger.info(f"Gerados {len(df)} candles sintéticos para {asset}")
        self.simulation_historical_data[asset] = df
        
    def get_all_assets(self) -> Optional[List[str]]:
        """Obtém lista de todos os ativos disponíveis para negociação.
        
        Returns:
            Optional[List[str]]: Lista de nomes de ativos ou None em caso de erro
        """
        if not self.connected:
            self.logger.error("Não conectado à API do IQ Option")
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
            
            self.logger.info(f"Obteve {len(all_assets)} ativos disponíveis")
            return list(all_assets)
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter lista de ativos: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetAllAssetsError", str(e), tb_str)
            return None

    def send_notification(self, message: str) -> None:
        """Envia notificação para o usuário.
        
        Args:
            message: Mensagem a ser enviada
        """
        self.logger.info(f"NOTIFICAÇÃO: {message}")
        # Em uma implementação real, poderia enviar por email, SMS ou push notification

    def check_connection(self) -> bool:
        """Verifica se a conexão com a API está ativa.
        
        Returns:
            bool: True se conectado, False caso contrário
        """
        if not self.connected:
            self.logger.warning("Não há conexão ativa com a API")
            return False
            
        try:
            if IQOPTION_API_AVAILABLE:
                is_connected = self.iq_option.check_connect()
                if is_connected:
                    self.logger.info("Conexão com a API está ativa")
                else:
                    self.logger.warning("Conexão com a API foi perdida")
                return is_connected
            else:
                # No modo de simulação, sempre retorna True
                self.logger.info("Modo de simulação: conexão simulada está ativa")
                return True
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao verificar conexão: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("CheckConnectionError", str(e), tb_str)
            return False

    def connect(self) -> Tuple[bool, Optional[str]]:
        """Conecta à API do IQ Option.
        
        Returns:
            Tuple[bool, Optional[str]]: Status da conexão e mensagem de erro (se houver)
            Se retornar (False, "2FA"), é necessário chamar connect_2fa com o código SMS
        """
        email = self.config_manager.get_value('Credentials', 'username')
        password = self.config_manager.get_value('Credentials', 'password')

        if not email or not password:
            try:
                email = self.config_manager.get_value('API', 'email')
                password = self.config_manager.get_value('API', 'password')

                if not email or not password:
                    self.logger.error("Credenciais não configuradas")
                    return False, "Credenciais não configuradas"
            except Exception as e:
                self.logger.error(f"Erro ao carregar credenciais: {str(e)}")
                return False, f"Erro ao carregar credenciais: {str(e)}"

        self.logger.info(f"Tentando conectar com email: {email}")

        if not IQOPTION_API_AVAILABLE:
            error_msg = "A biblioteca IQ Option não está instalada. Execute 'pip install -U git+https://github.com/iqoptionapi/iqoptionapi.git' para instalá-la."
            self.logger.error(error_msg)
            return False, error_msg

        try:
            self.iq_option = IQ_Option(email, password)
            check, reason = self.iq_option.connect()
            
            if check:
                self.connected = True
                self.logger.info("Conexão estabelecida com sucesso")
                return True, None
            else:
                if reason == "2FA":
                    self.logger.info("Autenticação de dois fatores necessária")
                    return False, "2FA"
                elif reason == "[Errno -2] Name or service not known":
                    self.logger.error("Erro de rede - Serviço não encontrado")
                    return False, "Erro de rede"
                elif "invalid_credentials" in str(reason):
                    self.logger.error("Credenciais inválidas")
                    return False, "Credenciais inválidas"
                else:
                    self.logger.error(f"Erro de conexão: {reason}")
                    return False, str(reason)
                
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao conectar: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("ConnectError", str(e), tb_str)
            return False, str(e)

    def reconnect(self) -> Tuple[bool, Optional[str]]:
        """Tenta reconectar à API.
        
        Returns:
            Tuple[bool, Optional[str]]: Status da conexão e mensagem de erro (se houver)
        """
        self.logger.info("Tentando reconectar...")
        return self.connect()

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
            for category in ['turbo', 'binary']:
                if category in instruments and asset in instruments[category]:
                    if instruments[category][asset].get('open', False):
                        self.logger.info(f"Ativo {asset} está disponível para negociação na categoria {category}")
                        return True
                        
            self.logger.warning(f"Ativo {asset} não está disponível para negociação no momento")
            return False
                
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao verificar disponibilidade do ativo {asset}: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("CheckAssetOpenError", str(e), tb_str)
            return False

    def _setup_simulation_mode(self):
        """Configura o modo de simulação para testes quando a biblioteca IQ Option não está disponível.
        
        Este método configura valores simulados para permitir testes do bot sem a necessidade
        da biblioteca IQ Option.
        """
        self.logger.info("Configurando modo de simulação para testes")
        
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
        self.simulation_trades = []
        
        self.logger.info("Modo de simulação configurado com sucesso")
    
    def _setup_simulated_historical_data(self):
        """Configura dados históricos simulados para testes."""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        self.logger.info("Configurando dados históricos simulados")
        
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
            
            self.logger.info(f"Gerados {len(df)} candles sintéticos para {asset}")
            self.simulation_historical_data[asset] = df
            
    def setup_simulation_mode(self, volatility=None, trend=None, win_rate=None):
        """Configura o modo de simulação para testes quando a API IQ Option não está disponível.
        
        Este método público configura o bot para operar em modo de simulação, permitindo
        testes sem a necessidade de conexão com a API real.
        
        Returns:
            bool: True se o modo de simulação foi configurado com sucesso
        """
        try:
            self.logger.info("Ativando modo de simulação para testes")
            
            # Configura o modo de simulação
            self._setup_simulation_mode()
            
            # Define que estamos conectados (em modo simulado)
            self.connected = True
            
            # Carrega configurações de simulação
            # Usa os valores passados ou busca na configuração como fallback
            self.simulation_volatility = volatility if volatility is not None else self.config_manager.get_value('Simulation', 'synthetic_volatility', 0.0002, float) # Ajustado para escala decimal
            self.simulation_trend = trend if trend is not None else self.config_manager.get_value('Simulation', 'synthetic_trend', 0.00001, float) # Ajustado para escala decimal
            self.simulation_win_rate = win_rate if win_rate is not None else self.config_manager.get_value('Simulation', 'simulated_win_rate', 60.0, float) / 100.0
            
            self.logger.info(f"Modo de simulação configurado com volatilidade: {self.simulation_volatility:.5f}, tendência: {self.simulation_trend:.5f}, taxa de acerto: {self.simulation_win_rate*100:.1f}%")
            return True
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao configurar modo de simulação: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("SetupSimulationModeError", str(e), tb_str)
            return False
            
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
                    
                    self.logger.info(f"Preço atual simulado para {asset}: {current_price}")
                    return current_price
                else:
                    self.logger.warning(f"Ativo simulado {asset} não encontrado")
                    return None
            
            # Verifica se o ativo está disponível
            all_assets = self.iq_option.get_all_open_time()
            
            for asset_type in ['turbo', 'binary']:
                if asset in all_assets[asset_type] and all_assets[asset_type][asset]['open']:
                    # Obtém o preço atual do ativo
                    candles = self.iq_option.get_candles(asset, 60, 1)
                    if candles:
                        return candles[0]['close']
            
            self.logger.warning(f"Ativo {asset} não está disponível para negociação")
            return None
            
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro ao obter preço atual: {str(e)}\n{tb_str}")
            self.error_tracker.add_error("GetCurrentPriceError", str(e), tb_str)
            return None
            
# A função execute_test_trade foi removida pois sua lógica foi integrada diretamente em main.py
# A chamada agora é feita diretamente para ferramental.buy nos modos 'test' e 'real'.
    def get_trade_results(self):
        """Obtém os resultados das operações realizadas.
        
        Returns:
            list: Lista de resultados das operações ou None em caso de erro
        """
        try:
            if not self.connected:
                self.logger.warning("Não conectado à API do IQ Option")
                return None
                
            # No modo de simulação, gera resultados simulados
            if not IQOPTION_API_AVAILABLE:
                self.logger.info("Gerando resultados simulados para operações")
                
                # Usa a taxa de acerto definida para a simulação
                win_rate = getattr(self, 'simulation_win_rate', 0.6) # Usa valor padrão se atributo não existir
                
                # Obtém histórico de operações simuladas
                simulated_trades = self.simulation_trades if hasattr(self, 'simulation_trades') else []
                
                # Se não houver operações simuladas, retorna lista vazia
                if not simulated_trades:
                    return []
                    
                # Processa os resultados
                results = []
                for trade in simulated_trades:
                    # Determina se a operação foi vencedora com base na taxa de acerto configurada
                    is_win = random.random() < win_rate
                    
                    # Calcula lucro/prejuízo
                    profit = trade['amount'] * 0.8 if is_win else -trade['amount']
                    
                    # Adiciona resultado à lista
                    results.append({
                        'id': trade['id'],
                        'asset': trade['asset'],
                        'direction': trade['direction'],
                        'amount': trade['amount'],
                        'profit': profit,
                        'is_win': is_win,
                        'timestamp': trade['timestamp'],
                        'expiration': trade['expiration']
                    })
                    
                self.logger.info(f"Gerados {len(results)} resultados simulados de operações")
                return results
            else:
                # Obtém resultados reais da API
                try:
                    # Obtém histórico de operações
                    # Obtém histórico de operações usando o método documentado v2 (buscando os últimos 100)
                    history = self.iq_option.get_optioninfo_v2(count=100)
                    
                    if not history:
                        self.logger.warning("Não foi possível obter histórico de operações")
                        return []
                        
                    # Processa os resultados
                    results = []
                    for trade in history:
                        results.append({
                            'id': trade.get('id'),
                            'asset': trade.get('active'),
                            'direction': trade.get('direction'),
                            'amount': trade.get('amount'),
                            'profit': trade.get('profit'),
                            'is_win': trade.get('win'),
                            'timestamp': trade.get('open_time'),
                            'expiration': trade.get('close_time')
                        })
                        
                    self.logger.info(f"Obtidos {len(results)} resultados de operações da API")
                    return results
                except Exception as e:
                    tb_str = traceback.format_exc()
                    self.logger.error(f"Erro ao obter resultados de operações da API: {str(e)}\n{tb_str}")
                    self.error_tracker.add_error("GetTradeResultsAPIError", str(e), tb_str)
                    return []
        except Exception as e:
            tb_str = traceback.format_exc()
            self.logger.error(f"Erro geral ao obter resultados de operações: {str(e)}\n{tb_str}") # Mensagem ligeiramente diferente para clareza
            self.error_tracker.add_error("GetTradeResultsGeneralError", str(e), tb_str)
            return [] # Retorna lista vazia em caso de erro geral