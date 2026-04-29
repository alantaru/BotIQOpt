#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
from datetime import datetime
import logging

logger = logging.getLogger('ErrorTracker')

class ErrorTracker:
    """Classe para rastrear e gerenciar erros ocorridos durante a execução do bot."""
    
    def __init__(self, max_errors=1000):
        """Inicializa o rastreador de erros.
        
        Args:
            max_errors (int): Número máximo de erros a serem armazenados
        """
        self.errors = []
        self.max_errors = max_errors
        self.error_counts = {}  # Contagem de erros por tipo
        self.critical_errors = []  # Erros críticos que requerem atenção
        
    def add_error(self, error_type, error_message, error_traceback=None, critical=False):
        """Adiciona um erro ao rastreador.
        
        Args:
            error_type (str): Tipo do erro (ex: "API", "Connection", "Model")
            error_message (str): Mensagem de erro
            error_traceback (str, optional): Traceback do erro
            critical (bool): Se o erro é crítico e requer atenção
        """
        # Cria registro de erro
        error = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'traceback': error_traceback
        }
        
        # Adiciona à lista de erros
        self.errors.append(error)
        
        # Limita o tamanho da lista
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)
            
        # Atualiza contagem de erros
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1
        else:
            self.error_counts[error_type] = 1
            
        # Registra erro crítico
        if critical:
            self.critical_errors.append(error)
            logger.critical(f"Erro crítico: {error_type} - {error_message}")
        else:
            logger.error(f"Erro: {error_type} - {error_message}")
            
    def get_errors(self, error_type=None, limit=None, critical_only=False):
        """Obtém erros do rastreador.
        
        Args:
            error_type (str, optional): Filtra por tipo de erro
            limit (int, optional): Limita o número de erros retornados
            critical_only (bool): Retorna apenas erros críticos
            
        Returns:
            list: Lista de erros
        """
        if critical_only:
            errors = self.critical_errors
        else:
            errors = self.errors
            
        # Filtra por tipo
        if error_type:
            errors = [e for e in errors if e['type'] == error_type]
            
        # Limita o número de erros
        if limit and limit > 0:
            errors = errors[-limit:]
            
        return errors
        
    def get_error_counts(self):
        """Obtém contagem de erros por tipo.
        
        Returns:
            dict: Dicionário com contagem de erros por tipo
        """
        return self.error_counts
        
    def clear_errors(self, error_type=None):
        """Limpa erros do rastreador.
        
        Args:
            error_type (str, optional): Limpa apenas erros do tipo especificado
        """
        if error_type:
            self.errors = [e for e in self.errors if e['type'] != error_type]
            self.critical_errors = [e for e in self.critical_errors if e['type'] != error_type]
            if error_type in self.error_counts:
                del self.error_counts[error_type]
        else:
            self.errors = []
            self.critical_errors = []
            self.error_counts = {}
            
    def save_to_file(self, filename='error_log.json'):
        """Salva erros em um arquivo JSON.
        
        Args:
            filename (str): Nome do arquivo para salvar os erros
        """
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'error_counts': self.error_counts,
                'errors': self.errors,
                'critical_errors': self.critical_errors
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Erros salvos em {filename}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar erros em arquivo: {str(e)}")
            return False
            
    def load_from_file(self, filename='error_log.json'):
        """Carrega erros de um arquivo JSON.
        
        Args:
            filename (str): Nome do arquivo para carregar os erros
            
        Returns:
            bool: True se carregou com sucesso, False caso contrário
        """
        try:
            if not os.path.exists(filename):
                logger.warning(f"Arquivo {filename} não encontrado")
                return False
                
            with open(filename, 'r') as f:
                data = json.load(f)
                
            self.error_counts = data.get('error_counts', {})
            self.errors = data.get('errors', [])
            self.critical_errors = data.get('critical_errors', [])
            
            logger.info(f"Erros carregados de {filename}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar erros do arquivo: {str(e)}")
            return False
