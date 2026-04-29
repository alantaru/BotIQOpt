# Documentação: Ferramental.py

O módulo `Ferramental.py` é o coração da integração com a plataforma de negociação (IQ Option) e do sistema de simulação do BotIQOpt. Ele gerencia a conectividade, execução de ordens, obtenção de dados e o motor de simulação.

## Arquitetura

O `Ferramental` é implementado como um **Singleton**, garantindo que apenas uma instância gerencie a conexão com a API e o estado da simulação durante toda a execução do bot.

### Principais Componentes

1.  **Gerenciamento de Conexão**:
    *   `connect()`: Realiza a autenticação inicial.
    *   `handle_two_factor_auth(code)`: Gerencia o fluxo de autenticação 2FA.
    *   `reconnect()`: Tenta restabelecer a conexão perdida.
    *   `check_connection()`: Verifica o status atual da conexão.

2.  **Execução de Operações**:
    *   `buy(asset, amount, action, expiration)`: Método principal para executar ordens (Call/Put).
    *   `get_trade_results()`: Recupera o histórico de operações (reais ou simuladas).
    *   `get_balance()`: Obtém o saldo atual da conta (suporta conta real/prática e simulação).

3.  **Obtenção de Dados**:
    *   `get_candles()`: Obtém dados históricos básicos.
    *   `get_historical_data()`: Obtém dados estruturados (Pandas DataFrame) para análise.
    *   `get_realtime_data()`: Agrega dados de múltiplos ativos em tempo real.
    *   `start_candles_stream()` / `stop_candles_stream()`: Gerencia o fluxo de dados em tempo real via WebSocket.

4.  **Gerenciamento de Risco**:
    *   Integrado com a dataclass `RiskMetrics`.
    *   Verifica limites de perda diária e risco por operação antes de cada execução.
    *   Bloqueia operações se os limites de loss consecutivo forem atingidos.

## Modo de Simulação

O `Ferramental` possui um motor de simulação robusto que permite testar estratégias sem risco financeiro e sem dependência da API.

*   **Ativação**: Controlada pelo parâmetro `simulation_mode`.
*   **Geração de Dados**: Se os dados históricos não estiverem disponíveis, o método `_generate_synthetic_data` gera um passeio aleatório (Random Walk) com tendência e volatilidade configuráveis.
*   **Execução Simulada**: As ordens enviadas via `buy` no modo de simulação são registradas internamente e processadas com base em uma taxa de acerto estatística (`simulation_win_rate`).

## Dependências

*   `iqoptionapi.stable_api`: Biblioteca base para conexão com a IQ Option.
*   `pandas`: Para manipulação de séries temporais de preços.
*   `numpy`: Usado na geração de dados sintéticos.
*   `ConfigManager`: Para carregar parâmetros de conexão e limites de risco.
*   `ErrorTracker`: Para registro centralizado de falhas técnicas.

## Tratamento de Erros

Todos os métodos críticos são envolvidos em blocos `try-except` com registro detalhado de traceback via `logging` e integração com o `ErrorTracker`. Isso garante que erros na API não causem o fechamento abrupto do bot.
