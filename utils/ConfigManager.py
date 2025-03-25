"""
Gerenciador centralizado de configurações para o BotIQOpt.
Responsável por carregar, validar e fornecer acesso às configurações do sistema.
"""

import os
import logging
import configparser
from typing import Any, Dict, List, Optional, Union, Tuple

logger = logging.getLogger('ConfigManager')

class ConfigManager:
    """
    Classe para gerenciar as configurações do sistema de forma centralizada.
    Implementa o padrão Singleton para garantir uma única instância das configurações.
    """
    _instance = None
    
    def __new__(cls, config_file=None):
        """Implementação do padrão Singleton."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_file=None):
        """
        Inicializa o gerenciador de configurações.
        
        Args:
            config_file (str, optional): Caminho para o arquivo de configuração.
                Se None, usa o arquivo padrão 'config.ini' no diretório raiz.
        """
        # Evita reinicialização se já inicializado (padrão Singleton)
        if self._initialized:
            return
            
        self._initialized = True
        self.config_file = config_file or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.ini')
        self.config = configparser.ConfigParser()
        self.load_config()
        self.load_credentials()
        
    def load_config(self) -> bool:
        """
        Carrega as configurações do arquivo.
        
        Returns:
            bool: True se as configurações foram carregadas com sucesso, False caso contrário.
        """
        try:
            if not os.path.exists(self.config_file):
                logger.error(f"Arquivo de configuração não encontrado: {self.config_file}")
                return False
                
            self.config.read(self.config_file)
            logger.info(f"Configurações carregadas com sucesso de {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar configurações: {str(e)}")
            return False
            
    def validate_config(self) -> bool:
        """
        Valida se as seções e parâmetros obrigatórios estão presentes no arquivo de configuração.
        
        Returns:
            bool: True se a configuração é válida, False caso contrário.
        """
        try:
            required_sections = ['General', 'Learning', 'Trading', 'Logging']
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"Seção obrigatória não encontrada: {section}")
                    return False
                    
            # Valida seção General
            if not self.config['General'].get('operation_mode'):
                logger.error("operation_mode é obrigatório na seção General")
                return False
                
            # Valida seção Trading
            if not self.config['Trading'].get('amount'):
                logger.error("amount é obrigatório na seção Trading")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Erro ao validar configurações: {str(e)}")
            return False
            
    def get_value(self, section: str, key: str, default: Any = None, value_type: type = None) -> Any:
        """
        Obtém um valor da configuração com conversão de tipo opcional.
        
        Args:
            section (str): Nome da seção.
            key (str): Nome da chave.
            default (Any, optional): Valor padrão se a chave não existir.
            value_type (type, optional): Tipo para conversão do valor.
            
        Returns:
            Any: Valor da configuração convertido para o tipo especificado.
        """
        try:
            if section not in self.config:
                logger.warning(f"Seção não encontrada: {section}")
                return default
                
            value = self.config[section].get(key, default)
            
            if value is None:
                return default
                
            value = str(value).lower()
            
            if key == 'auto_switch_modes':
                return value in ('true', 'yes', '1', 'sim', 'verdadeiro')
            
            if value_type is bool:
                return value in ('true', 'yes', '1', 'sim', 'verdadeiro')
            elif value_type is not None:
                return value_type(value)
                
            return value
        except Exception as e:
            logger.error(f"Erro ao obter valor de configuração {section}.{key}: {str(e)}")
            return default
            
    def get_list(self, section: str, key: str, default: List = None, item_type: type = str) -> List:
        """
        Obtém uma lista de valores da configuração.
        
        Args:
            section (str): Nome da seção.
            key (str): Nome da chave.
            default (List, optional): Lista padrão se a chave não existir.
            item_type (type, optional): Tipo para conversão dos itens da lista.
            
        Returns:
            List: Lista de valores convertidos para o tipo especificado.
        """
        if default is None:
            default = []
            
        try:
            value = self.get_value(section, key)
            
            if not value:
                return default
                
            # Divide a string por vírgulas e remove espaços em branco
            items = [item.strip() for item in value.split(',') if item.strip()]
            
            # Converte os itens para o tipo especificado
            if item_type is not str:
                return [item_type(item) for item in items]
                
            return items
        except Exception as e:
            logger.error(f"Erro ao obter lista de configuração {section}.{key}: {str(e)}")
            return default
            
    def get_auto_switch_criteria(self) -> Dict[str, Any]:
        """
        Obtém os critérios para mudança automática para o modo real.
        
        Este método carrega os critérios definidos na seção AutoSwitchCriteria do arquivo
        de configuração. Estes critérios são utilizados pela IA para determinar quando
        é seguro mudar automaticamente para o modo real de operação.
        
        Os critérios incluem:
        - min_accuracy: Acurácia mínima no conjunto de teste
        - min_precision: Precisão mínima no conjunto de teste
        - min_recall: Recall mínimo no conjunto de teste
        - min_f1_score: F1-score mínimo no conjunto de teste
        - min_trades_count: Número mínimo de operações para avaliação
        - min_win_rate: Taxa mínima de acerto nas operações
        - min_profit: Lucro mínimo acumulado nas operações
        
        Returns:
            Dict[str, Any]: Dicionário com os critérios para mudança automática.
        """
        criteria = {
            'min_accuracy': self.get_value('AutoSwitchCriteria', 'min_accuracy', 0.70, float),
            'min_precision': self.get_value('AutoSwitchCriteria', 'min_precision', 0.65, float),
            'min_recall': self.get_value('AutoSwitchCriteria', 'min_recall', 0.65, float),
            'min_f1_score': self.get_value('AutoSwitchCriteria', 'min_f1_score', 0.65, float),
            'min_trades_count': self.get_value('AutoSwitchCriteria', 'min_trades_count', 20, int),
            'min_win_rate': self.get_value('AutoSwitchCriteria', 'min_win_rate', 0.60, float),
            'min_profit': self.get_value('AutoSwitchCriteria', 'min_profit', 0.0, float)
        }
        
        logger.debug(f"Critérios para mudança automática: {criteria}")
        return criteria
        
    #def is_auto_switch_enabled(self) -> bool:
    #    """
    #    Verifica se a mudança automática para o modo real está habilitada.
    #    
    #    Returns:
    #        bool: True se a mudança automática está habilitada, False caso contrário.
    #    """
    #    return self.get_value('General', 'auto_switch_modes', False, bool)
        
    def get_operation_mode(self) -> str:
        """
        Obtém o modo de operação configurado.
        
        Returns:
            str: Modo de operação (DOWNLOAD, LEARNING, TEST, REAL).
        """
        return self.get_value('General', 'operation_mode', 'TEST').upper()
        
    def get_assets(self) -> List[str]:
        """
        Obtém a lista de ativos configurados.
        
        Returns:
            List[str]: Lista de ativos.
        """
        return self.get_list('General', 'assets')
        
    def get_timeframe(self) -> Tuple[str, int]:
        """
        Obtém o timeframe configurado.
        
        Returns:
            Tuple[str, int]: Tipo de timeframe e valor.
        """
        timeframe_type = self.get_value('General', 'timeframe_type', 'Minutes')
        timeframe_value = self.get_value('General', 'timeframe_value', 1, int)
        
        return timeframe_type, timeframe_value
        
    def get_risk_params(self) -> Dict[str, float]:
        """
        Obtém os parâmetros de gerenciamento de risco.
        
        Returns:
            Dict: Dicionário com os parâmetros de risco.
        """
        return {
            'amount': self.get_value('Trading', 'amount', 1.0, float),
            'stop_loss': self.get_value('Trading', 'stop_loss', 0.1, float),
            'take_profit': self.get_value('Trading', 'take_profit', 0.2, float),
            'max_daily_loss': self.get_value('Trading', 'max_daily_loss', 5.0, float),
            'max_consecutive_losses': self.get_value('Trading', 'max_consecutive_losses', 5, int)
        }
        
    def get_learning_params(self) -> Dict[str, Any]:
        """
        Obtém os parâmetros de aprendizado.
        
        Returns:
            Dict: Dicionário com os parâmetros de aprendizado.
        """
        return {
            'epochs': self.get_value('Learning', 'epochs', 100, int),
            'batch_size': self.get_value('Learning', 'batch_size', 32, int),
            'test_size': self.get_value('Learning', 'test_size', 0.2, float),
            'learning_rate': self.get_value('Learning', 'learning_rate', 0.001, float),
            'early_stopping_patience': self.get_value('Learning', 'early_stopping_patience', 10, int),
            'hidden_layers': self.get_value('Learning', 'hidden_layers', 3, int),
            'neurons_per_layer': self.get_value('Learning', 'neurons_per_layer', 128, int),
            'activation_function': self.get_value('Learning', 'activation_function', 'relu'),
            'dropout_rate': self.get_value('Learning', 'dropout_rate', 0.2, float)
        }
        
    def get_download_params(self) -> Dict[str, Any]:
        """
        Obtém os parâmetros para download de dados históricos.
        
        Returns:
            Dict: Dicionário com os parâmetros de download.
        """
        return {
            'start_date': self.get_value('Download', 'start_date', '2023-01-01'),
            'end_date': self.get_value('Download', 'end_date', '2024-03-19'),
            'max_candles_per_request': self.get_value('Download', 'max_candles_per_request', 1000, int)
        }
        
    def get_logging_params(self) -> Dict[str, Any]:
        """
        Obtém os parâmetros de logging.
        
        Returns:
            Dict: Dicionário com os parâmetros de logging.
        """
        return {
            'log_level': self.get_value('Logging', 'log_level', 'INFO'),
            'log_file': self.get_value('Logging', 'log_file', 'bot.log'),
            'max_log_size': self.get_value('Logging', 'max_log_size', '10MB'),
            'log_backup_count': self.get_value('Logging', 'log_backup_count', 5, int)
        }
        
    def get_api_params(self) -> Dict[str, Any]:
        """
        Obtém os parâmetros da API.
        
        Returns:
            Dict: Dicionário com os parâmetros da API.
        """
        return {
            'timeout': self.get_value('API', 'timeout', 30, int),
            'retry_count': self.get_value('API', 'retry_count', 3, int)
        }
        
    def get_config_parser(self) -> configparser.ConfigParser:
        """
        Obtém o objeto ConfigParser.
        
        Returns:
            ConfigParser: Objeto ConfigParser com as configurações carregadas.
        """
        return self.config

    def load_credentials(self) -> None:
        """
        Carrega as credenciais do arquivo de configuração.
        """
        self.username = self.get_value('Credentials', 'username')
        self.password = self.get_value('Credentials', 'password')
        
        if not self.username or not self.password:
            logger.warning("Credenciais da IQ Option não encontradas no arquivo de configuração")
        else:
            logger.info("Credenciais da IQ Option carregadas do arquivo de configuração")
