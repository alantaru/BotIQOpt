#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import logging.handlers
from datetime import datetime
from .ConfigManager import ConfigManager # Importar ConfigManager

def setup_logger(name, override_level=None):
    """Configura e retorna um logger com base nas configurações do ConfigManager,
    permitindo sobrescrever o nível via argumento.
    
    Args:
        name (str): Nome do logger
        override_level (int, optional): Nível de logging para sobrescrever a configuração.
        
    Returns:
        logging.Logger: Logger configurado
    """
    # Obter instância do ConfigManager
    config_manager = ConfigManager() # Singleton
    
    # Obter parâmetros de logging da configuração
    log_params = config_manager.get_logging_params()
    # Define o nível de log: usa override_level se fornecido, senão busca na config
    if override_level is not None:
        log_level = override_level
        log_level_str = logging.getLevelName(log_level)
        print(f"Nível de log sobrescrito para {log_level_str} via argumento.") # Adiciona print para feedback imediato
    else:
        log_level_str = log_params.get('log_level', 'INFO').upper()
        # Mapear string de nível para constante de logging
        log_level = getattr(logging, log_level_str, logging.INFO)
        
    log_file_path = log_params.get('log_file', f'logs/{name}.log')
    max_bytes_str = log_params.get('max_log_size', '10MB')
    backup_count = log_params.get('log_backup_count', 5)
    log_format = config_manager.get_value('Logging', 'log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Busca formato específico

    # Mapear string de nível para constante de logging (movido para cima)
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Converter max_bytes_str para bytes (ex: 10MB -> 10 * 1024 * 1024)
    try:
        size_mb = int(max_bytes_str.lower().replace('mb', ''))
        max_bytes = size_mb * 1024 * 1024
    except ValueError:
        logger.warning(f"Formato inválido para max_log_size: {max_bytes_str}. Usando 10MB.")
        max_bytes = 10 * 1024 * 1024
    # Cria o logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Define o formato do log
    formatter = logging.Formatter(log_format)
    
    # Adiciona handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Cria diretório de logs se não existir
    log_dir = os.path.dirname(log_file_path)
    if log_dir: # Verifica se há um diretório no path
        os.makedirs(log_dir, exist_ok=True)
    
    # Adiciona handler para arquivo com rotação
    # RotatingFileHandler(filename, maxBytes=0, backupCount=0)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger '{name}' configurado: Nível={log_level_str}, Arquivo={log_file_path}, Rotação={max_bytes_str}/{backup_count} backups")
    
    return logger

def get_logger(name):
    """Obtém um logger existente ou cria um novo.
    
    Args:
        name (str): Nome do logger
        
    Returns:
        logging.Logger: Logger solicitado
    """
    logger = logging.getLogger(name)
    
    # Se o logger não tem handlers, configura-o
    if not logger.handlers:
        # Passa apenas o nome, setup_logger agora lê a config e aceita override
        return setup_logger(name)
    
    return logger
