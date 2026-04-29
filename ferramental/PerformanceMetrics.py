import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BotPerformanceMetrics:
    """
    Classe para rastrear e analisar métricas de desempenho do bot
    """
    
    def __init__(self):
        """Inicializa o rastreador de métricas de desempenho"""
        self.trades: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self.metrics_history: List[Dict[str, Any]] = []
        self.save_interval = 10  # Salva métricas a cada 10 trades
        
    def add_trade(self, asset: str, direction: str, amount: float, result: bool, 
                  confidence: Optional[float] = None, timestamp: Optional[datetime] = None):
        """
        Adiciona um trade às métricas de desempenho
        
        Args:
            asset: O ativo negociado
            direction: A direção do trade (call/put)
            amount: O valor investido
            result: O resultado do trade (True para ganho, False para perda)
            confidence: Nível de confiança da previsão (opcional)
            timestamp: Timestamp do trade (opcional, usa o tempo atual se não fornecido)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        trade_info = {
            'timestamp': timestamp,
            'asset': asset,
            'direction': direction,
            'amount': amount,
            'result': result
        }
        
        if confidence is not None:
            trade_info['confidence'] = confidence
            
        self.trades.append(trade_info)
        logger.debug(f"Trade adicionado: {trade_info}")
        
        # Salva métricas periodicamente
        if len(self.trades) % self.save_interval == 0:
            self.save_metrics()
            
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de desempenho com base nos trades registrados
        
        Returns:
            Dict contendo as métricas calculadas
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'runtime': str(datetime.now() - self.start_time),
                'last_updated': datetime.now().isoformat()
            }
            
        # Métricas básicas
        wins = sum(1 for trade in self.trades if trade['result'])
        losses = len(self.trades) - wins
        
        win_rate = wins / len(self.trades) if len(self.trades) > 0 else 0
        
        # Cálculo de lucro/perda
        total_profit = sum(trade['amount'] if trade['result'] else -trade['amount'] for trade in self.trades)
        
        # Profit factor (razão entre ganhos e perdas)
        total_wins = sum(trade['amount'] for trade in self.trades if trade['result'])
        total_losses = sum(trade['amount'] for trade in self.trades if not trade['result'])
        
        profit_factor = (
            total_wins / abs(total_losses)
            if total_losses != 0 else 
            float('inf') if total_wins > 0 else 0
        )
        
        # Métricas avançadas
        consecutive_wins = self._calculate_max_consecutive(True)
        consecutive_losses = self._calculate_max_consecutive(False)
        
        # Métricas por ativo
        assets = set(trade['asset'] for trade in self.trades)
        asset_metrics = {}
        
        for asset in assets:
            asset_trades = [trade for trade in self.trades if trade['asset'] == asset]
            asset_wins = sum(1 for trade in asset_trades if trade['result'])
            
            asset_metrics[asset] = {
                'total_trades': len(asset_trades),
                'wins': asset_wins,
                'losses': len(asset_trades) - asset_wins,
                'win_rate': asset_wins / len(asset_trades) if len(asset_trades) > 0 else 0,
                'profit': sum(trade['amount'] if trade['result'] else -trade['amount'] for trade in asset_trades)
            }
        
        # Métricas por direção
        direction_metrics = {}
        directions = set(trade['direction'] for trade in self.trades)
        
        for direction in directions:
            direction_trades = [trade for trade in self.trades if trade['direction'] == direction]
            direction_wins = sum(1 for trade in direction_trades if trade['result'])
            
            direction_metrics[direction] = {
                'total_trades': len(direction_trades),
                'wins': direction_wins,
                'losses': len(direction_trades) - direction_wins,
                'win_rate': direction_wins / len(direction_trades) if len(direction_trades) > 0 else 0,
                'profit': sum(trade['amount'] if trade['result'] else -trade['amount'] for trade in direction_trades)
            }
        
        # Métricas temporais (últimas 10 operações)
        recent_trades = self.trades[-10:] if len(self.trades) >= 10 else self.trades
        recent_wins = sum(1 for trade in recent_trades if trade['result'])
        
        recent_metrics = {
            'total_trades': len(recent_trades),
            'wins': recent_wins,
            'losses': len(recent_trades) - recent_wins,
            'win_rate': recent_wins / len(recent_trades) if len(recent_trades) > 0 else 0,
            'profit': sum(trade['amount'] if trade['result'] else -trade['amount'] for trade in recent_trades)
        }
        
        # Compilação de todas as métricas
        metrics = {
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'assets': asset_metrics,
            'directions': direction_metrics,
            'recent': recent_metrics,
            'runtime': str(datetime.now() - self.start_time),
            'last_updated': datetime.now().isoformat()
        }
        
        # Adiciona às métricas históricas
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_profit': total_profit
        })
        
        return metrics
    
    def _calculate_max_consecutive(self, win: bool) -> int:
        """
        Calcula o número máximo de vitórias ou derrotas consecutivas
        
        Args:
            win: True para calcular vitórias consecutivas, False para derrotas
            
        Returns:
            Número máximo de resultados consecutivos do tipo especificado
        """
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade['result'] == win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def save_metrics(self, filename: Optional[str] = None) -> bool:
        """
        Salva as métricas atuais em um arquivo JSON
        
        Args:
            filename: Nome do arquivo para salvar as métricas (opcional)
            
        Returns:
            True se as métricas foram salvas com sucesso, False caso contrário
        """
        try:
            metrics = self.calculate_metrics()
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bot_performance_{timestamp}.json"
                
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=4, default=str)
                
            logger.info(f"Métricas de desempenho salvas em {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {str(e)}")
            return False
    
    def get_equity_curve_data(self) -> List[float]:
        """
        Retorna os dados da curva de equity para visualização
        
        Returns:
            Lista de valores de equity ao longo do tempo
        """
        equity_curve = []
        current_equity = 0.0
        
        for trade in self.trades:
            current_equity += trade['amount'] if trade['result'] else -trade['amount']
            equity_curve.append(current_equity)
            
        return equity_curve
    
    def reset(self) -> None:
        """
        Reinicia as métricas de desempenho
        """
        # Salva as métricas atuais antes de resetar
        if self.trades:
            self.save_metrics()
            
        self.trades = []
        self.start_time = datetime.now()
        logger.info("Métricas de desempenho resetadas")
