#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import colorama

# Inicializa o colorama para suporte a cores no terminal
colorama.init()

# Definição de cores para diferentes níveis de log
COLORS = {
    'DEBUG': colorama.Fore.BLUE,
    'INFO': colorama.Fore.GREEN,
    'WARNING': colorama.Fore.YELLOW,
    'ERROR': colorama.Fore.RED,
    'CRITICAL': colorama.Fore.RED + colorama.Style.BRIGHT,
    'SUCCESS': colorama.Fore.GREEN + colorama.Style.BRIGHT
}

# Adiciona nível de log personalizado para SUCCESS
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

class ColoredFormatter(logging.Formatter):
    """Formatter personalizado que adiciona cores aos logs no terminal."""
    
    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{colorama.Style.RESET_ALL}"
        return super().format(record)

class LogManager:
    """Gerenciador de logs para o Bot IQ Option."""
    
    def __init__(self, log_dir='logs', log_level=logging.INFO, max_size=10*1024*1024, backup_count=5):
        """Inicializa o gerenciador de logs.
        
        Args:
            log_dir: Diretório para armazenar os arquivos de log
            log_level: Nível de log (DEBUG, INFO, etc.)
            max_size: Tamanho máximo do arquivo de log antes de rotacionar
            backup_count: Número de arquivos de backup a manter
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.max_size = max_size
        self.backup_count = backup_count
        
        # Cria o diretório de logs se não existir
        os.makedirs(log_dir, exist_ok=True)
        
        # Configura o logger raiz
        self.setup_root_logger()
        
    def setup_root_logger(self):
        """Configura o logger raiz com handlers para arquivo e console."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remove handlers existentes para evitar duplicação
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Adiciona handler para arquivo com rotação
        log_file = os.path.join(self.log_dir, f'bot_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=self.max_size, 
            backupCount=self.backup_count
        )
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Adiciona handler para console com cores
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    def get_logger(self, name):
        """Obtém um logger configurado com o nome especificado.
        
        Args:
            name: Nome do logger
            
        Returns:
            Logger configurado
        """
        logger = logging.getLogger(name)
        
        # Adiciona método success
        def success(msg, *args, **kwargs):
            logger.log(SUCCESS_LEVEL_NUM, msg, *args, **kwargs)
        
        logger.success = success
        
        return logger

def setup_logger(name, log_level=logging.INFO):
    """Função de conveniência para configurar um logger.
    
    Args:
        name: Nome do logger
        log_level: Nível de log
        
    Returns:
        Logger configurado
    """
    # Configura o logger raiz se ainda não foi configurado
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        log_manager = LogManager(log_level=log_level)
    
    # Obtém o logger com o nome especificado
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Adiciona método success se não existir
    if not hasattr(logger, 'success'):
        def success(msg, *args, **kwargs):
            logger.log(SUCCESS_LEVEL_NUM, msg, *args, **kwargs)
        logger.success = success
    
    return logger
