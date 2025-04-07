#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Any, Dict

from datetime import datetime, timedelta
import logging

logger = logging.getLogger('PerformanceTracker')

class PerformanceTracker:
    """Classe para rastrear e analisar o desempenho do bot."""
    
    def __init__(self):
        """Inicializa o rastreador de desempenho."""
        self.trades = []
        self.metrics = {
            'total_trades': 0,
            'win_count': 0,
            'loss_count': 0,
            'draw_count': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'avg_trade_duration': 0.0,
            'best_asset': None,
            'worst_asset': None,
            'best_timeframe': None,
            'worst_timeframe': None,
            'last_updated': datetime.now().isoformat()
        }
        self.asset_performance = {}
        self.timeframe_performance = {}
        self.daily_performance = {}
        self.equity_curve = []
        
    def add_trade(self, trade: Dict[str, Any]):
        """Adiciona uma operação ao rastreador, com validação.
        
        Args:
            trade (dict): Informações da operação. Espera chaves como:
                'asset', 'profit', 'result' (opcional, pode ser inferido de profit),
                'timestamp' (opcional, será adicionado se ausente), etc.
        """
        # Validação básica da estrutura do trade
        required_keys = ['asset', 'profit'] # Chaves mínimas esperadas
        if not isinstance(trade, dict) or not all(key in trade for key in required_keys):
            logger.error(f"Formato inválido do trade recebido: {trade}. Pulando adição.")
            return

        # Inferir 'result' se não estiver presente
        if 'result' not in trade:
            profit = trade.get('profit', 0)
            if profit > 0:
                trade['result'] = 'win'
            elif profit < 0:
                trade['result'] = 'loss'
            else:
                trade['result'] = 'draw'
            logger.debug(f"Resultado inferido para trade: {trade.get('id', 'N/A')} como '{trade['result']}' baseado no lucro {profit}")

        # Adiciona timestamp se não existir
        if 'timestamp' not in trade:
            trade['timestamp'] = datetime.now().isoformat()
            
        # Adiciona à lista de operações
        self.trades.append(trade)
        
        # Atualiza métricas
        self._update_metrics()
        
    def update(self, trades):
        """Atualiza o rastreador com múltiplas operações.
        
        Args:
            trades (list): Lista de operações
        """
        for trade in trades:
            self.add_trade(trade)
            
    def _update_metrics(self):
        """Atualiza as métricas de desempenho."""
        if not self.trades:
            return
            
        # Métricas básicas
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['win_count'] = sum(1 for t in self.trades if t.get('result') == 'win')
        self.metrics['loss_count'] = sum(1 for t in self.trades if t.get('result') == 'loss')
        self.metrics['draw_count'] = sum(1 for t in self.trades if t.get('result') == 'draw')
        
        # Taxa de acerto
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['win_count'] / self.metrics['total_trades']
        
        # Lucro total
        self.metrics['total_profit'] = sum(t.get('profit', 0) for t in self.trades)
        
        # Fator de lucro
        total_gain = sum(t.get('profit', 0) for t in self.trades if t.get('profit', 0) > 0)
        total_loss = sum(abs(t.get('profit', 0)) for t in self.trades if t.get('profit', 0) < 0)
        if total_loss > 0:
            self.metrics['profit_factor'] = total_gain / total_loss
        
        # Curva de patrimônio
        self.equity_curve = self._calculate_equity_curve()
        
        # Drawdown máximo
        self.metrics['max_drawdown'] = self._calculate_max_drawdown()
        
        # Duração média das operações
        durations = []
        for trade in self.trades:
            if 'open_time' in trade and 'close_time' in trade:
                try:
                    open_time = datetime.fromisoformat(trade['open_time']) if isinstance(trade['open_time'], str) else trade['open_time']
                    close_time = datetime.fromisoformat(trade['close_time']) if isinstance(trade['close_time'], str) else trade['close_time']
                    duration = (close_time - open_time).total_seconds()
                    durations.append(duration)
                except (ValueError, TypeError):
                    pass
        
        if durations:
            self.metrics['avg_trade_duration'] = sum(durations) / len(durations)
        
        # Desempenho por ativo
        self.asset_performance = self._calculate_asset_performance()
        
        # Melhor e pior ativo
        if self.asset_performance:
            self.metrics['best_asset'] = max(self.asset_performance.items(), key=lambda x: x[1]['win_rate'])
            self.metrics['worst_asset'] = min(self.asset_performance.items(), key=lambda x: x[1]['win_rate'])
        
        # Desempenho por timeframe
        self.timeframe_performance = self._calculate_timeframe_performance()
        
        # Melhor e pior timeframe
        if self.timeframe_performance:
            self.metrics['best_timeframe'] = max(self.timeframe_performance.items(), key=lambda x: x[1]['win_rate'])
            self.metrics['worst_timeframe'] = min(self.timeframe_performance.items(), key=lambda x: x[1]['win_rate'])
        
        # Desempenho diário
        self.daily_performance = self._calculate_daily_performance()
        
        # Atualiza timestamp
        self.metrics['last_updated'] = datetime.now().isoformat()
        
    def _calculate_equity_curve(self):
        """Calcula a curva de patrimônio.
        
        Returns:
            list: Lista de pontos da curva de patrimônio
        """
        equity = 0
        curve = []
        
        for trade in sorted(self.trades, key=lambda x: x.get('timestamp', '')):
            equity += trade.get('profit', 0)
            curve.append({
                'timestamp': trade.get('timestamp', ''),
                'equity': equity,
                'trade_profit': trade.get('profit', 0)
            })
            
        return curve
        
    def _calculate_max_drawdown(self):
        """Calcula o drawdown máximo.
        
        Returns:
            float: Drawdown máximo em percentual
        """
        if not self.equity_curve:
            return 0.0
            
        # Extrai valores de patrimônio
        equity_values = [point['equity'] for point in self.equity_curve]
        
        # Calcula drawdown
        max_dd = 0
        peak = equity_values[0]
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_asset_performance(self):
        """Calcula o desempenho por ativo.
        
        Returns:
            dict: Dicionário com métricas por ativo
        """
        performance = {}
        
        for trade in self.trades:
            asset = trade.get('asset')
            if not asset:
                continue
                
            if asset not in performance:
                performance[asset] = {
                    'total_trades': 0,
                    'win_count': 0,
                    'loss_count': 0,
                    'draw_count': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0
                }
                
            performance[asset]['total_trades'] += 1
            
            result = trade.get('result')
            if result == 'win':
                performance[asset]['win_count'] += 1
            elif result == 'loss':
                performance[asset]['loss_count'] += 1
            elif result == 'draw':
                performance[asset]['draw_count'] += 1
                
            performance[asset]['total_profit'] += trade.get('profit', 0)
            
            # Calcula taxa de acerto
            if performance[asset]['total_trades'] > 0:
                performance[asset]['win_rate'] = performance[asset]['win_count'] / performance[asset]['total_trades']
                
        return performance
        
    def _calculate_timeframe_performance(self):
        """Calcula o desempenho por timeframe.
        
        Returns:
            dict: Dicionário com métricas por timeframe
        """
        performance = {}
        
        for trade in self.trades:
            timeframe = trade.get('timeframe')
            if not timeframe:
                continue
                
            if timeframe not in performance:
                performance[timeframe] = {
                    'total_trades': 0,
                    'win_count': 0,
                    'loss_count': 0,
                    'draw_count': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0
                }
                
            performance[timeframe]['total_trades'] += 1
            
            result = trade.get('result')
            if result == 'win':
                performance[timeframe]['win_count'] += 1
            elif result == 'loss':
                performance[timeframe]['loss_count'] += 1
            elif result == 'draw':
                performance[timeframe]['draw_count'] += 1
                
            performance[timeframe]['total_profit'] += trade.get('profit', 0)
            
            # Calcula taxa de acerto
            if performance[timeframe]['total_trades'] > 0:
                performance[timeframe]['win_rate'] = performance[timeframe]['win_count'] / performance[timeframe]['total_trades']
                
        return performance
        
    def _calculate_daily_performance(self):
        """Calcula o desempenho diário.
        
        Returns:
            dict: Dicionário com métricas por dia
        """
        performance = {}
        
        for trade in self.trades:
            # Obtém data da operação
            timestamp = trade.get('timestamp', '')
            if not timestamp:
                continue
                
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                    
                date_str = dt.strftime('%Y-%m-%d')
                
                if date_str not in performance:
                    performance[date_str] = {
                        'total_trades': 0,
                        'win_count': 0,
                        'loss_count': 0,
                        'draw_count': 0,
                        'win_rate': 0.0,
                        'total_profit': 0.0
                    }
                    
                performance[date_str]['total_trades'] += 1
                
                result = trade.get('result')
                if result == 'win':
                    performance[date_str]['win_count'] += 1
                elif result == 'loss':
                    performance[date_str]['loss_count'] += 1
                elif result == 'draw':
                    performance[date_str]['draw_count'] += 1
                    
                performance[date_str]['total_profit'] += trade.get('profit', 0)
                
                # Calcula taxa de acerto
                if performance[date_str]['total_trades'] > 0:
                    performance[date_str]['win_rate'] = performance[date_str]['win_count'] / performance[date_str]['total_trades']
                    
            except (ValueError, TypeError):
                continue
                
        return performance
        
    def get_metrics(self):
        """Obtém as métricas de desempenho.
        
        Returns:
            dict: Métricas de desempenho
        """
        return self.metrics
        
    def get_asset_performance(self, asset=None):
        """Obtém o desempenho por ativo.
        
        Args:
            asset (str, optional): Ativo específico
            
        Returns:
            dict: Desempenho por ativo
        """
        if asset:
            return self.asset_performance.get(asset)
        return self.asset_performance
        
    def get_timeframe_performance(self, timeframe=None):
        """Obtém o desempenho por timeframe.
        
        Args:
            timeframe (int, optional): Timeframe específico
            
        Returns:
            dict: Desempenho por timeframe
        """
        if timeframe:
            return self.timeframe_performance.get(timeframe)
        return self.timeframe_performance
        
    def get_daily_performance(self, date=None):
        """Obtém o desempenho diário.
        
        Args:
            date (str, optional): Data específica (formato: 'YYYY-MM-DD')
            
        Returns:
            dict: Desempenho diário
        """
        if date:
            return self.daily_performance.get(date)
        return self.daily_performance
        
    def get_equity_curve(self):
        """Obtém a curva de patrimônio.
        
        Returns:
            list: Curva de patrimônio
        """
        return self.equity_curve
        
    def save_to_file(self, filename='performance_metrics.json'):
        """Salva métricas em um arquivo JSON.
        
        Args:
            filename (str): Nome do arquivo para salvar as métricas
            
        Returns:
            bool: True se salvou com sucesso, False caso contrário
        """
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics,
                'asset_performance': self.asset_performance,
                'timeframe_performance': self.timeframe_performance,
                'daily_performance': self.daily_performance,
                'equity_curve': self.equity_curve,
                'trades_count': len(self.trades),
                # Salvar a lista completa de trades para consistência ao recarregar
                'trades': self.trades
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Métricas de desempenho salvas em {filename}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar métricas em arquivo: {str(e)}")
            return False
            
    def load_from_file(self, filename='performance_metrics.json'):
        """Carrega métricas de um arquivo JSON.
        
        Args:
            filename (str): Nome do arquivo para carregar as métricas
            
        Returns:
            bool: True se carregou com sucesso, False caso contrário
        """
        try:
            if not os.path.exists(filename):
                logger.warning(f"Arquivo {filename} não encontrado")
                return False
                
            with open(filename, 'r') as f:
                data = json.load(f)
                
            self.metrics = data.get('metrics', {})
            self.asset_performance = data.get('asset_performance', {})
            self.timeframe_performance = data.get('timeframe_performance', {})
            self.daily_performance = data.get('daily_performance', {})
            self.equity_curve = data.get('equity_curve', [])
            
            # Carrega operações se disponíveis
            if 'trades' in data:
                self.trades = data['trades']
            
            logger.info(f"Métricas de desempenho carregadas de {filename}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar métricas do arquivo: {str(e)}")
            return False
