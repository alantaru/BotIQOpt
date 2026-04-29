# Documentação: main.py

O arquivo `main.py` é o ponto de entrada principal e o orquestrador do BotIQOpt. Ele gerencia o ciclo de vida do bot, o loop principal de operação e a coordenação entre os módulos de ferramentas e inteligência.

## Responsabilidades

1.  **Bootstrapping**:
    *   Verificação de dependências (`check_dependencies`).
    *   Criação de diretórios (`create_directories`).
    *   Configuração do sistema de logging (`setup_logger`).
    *   Tratamento de sinais (SIGINT/SIGTERM).

2.  **Gerenciamento de Modos de Operação**:
    *   **download**: Baixa dados históricos para treinamento.
    *   **learning**: Processa dados e treina o modelo de IA.
    *   **test**: Simula operações na conta DEMO ou via motor interno de simulação.
    *   **real**: Executa operações reais na conta de saldo ao vivo.

3.  **Loop Principal (`main()`)**:
    *   Executa continuamente em um loop `while not stop_event.is_set()`.
    *   Implementa fallback automático para o modo de simulação se a conexão com a API falhar.
    *   Coordena a análise de múltiplos ativos em paralelo (iteração serial sobre a lista de ativos).

4.  **Integração de Componentes**:
    *   Instancia `Ferramental` e `Inteligencia`.
    *   Gerencia a lógica de transição automática de modos (`should_switch_to_test_mode`, `should_switch_to_real_mode`).
    *   Atualiza métricas de desempenho e gerenciamento de risco após cada operação.

## Gerenciamento de Estado

*   **`stop_event`**: Um `threading.Event` que sinaliza a todos os loops para encerrarem graciosamente.
*   **Performance & Error Tracking**: Utiliza instâncias globais de `PerformanceTracker` e `ErrorTracker` para registrar a saúde do sistema.

## Fluxo de Execução

1.  Processar argumentos via `argparse`.
2.  Carregar `config.ini` via `ConfigManager`.
3.  Verificar conexão.
4.  Entrar no loop de operação do modo selecionado.
5.  Em cada iteração:
    *   Obter dados de mercado.
    *   Solicitar previsão à `Inteligencia`.
    *   Executar compra via `Ferramental` se houver sinal.
    *   Verificar resultados e atualizar métricas.
    *   Aguardar intervalo configurado.

## Encerramento (`cleanup()`)

Garante que as métricas de performance e os logs de erro sejam salvos em arquivos JSON antes de fechar o processo, permitindo auditoria posterior.
