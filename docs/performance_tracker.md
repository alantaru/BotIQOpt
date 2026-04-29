# Documentação Técnica: PerformanceTracker.py

O `PerformanceTracker.py` é responsável pela contabilidade detalhada de todas as operações realizadas pelo bot, fornecendo métricas críticas para a tomada de decisão.

## Métricas Monitoradas

- **Win Rate**: Taxa de acerto percentual das operações.
- **Profit/Loss (P&L)**: Lucro ou prejuízo líquido total e por ativo.
- **Drawdown**: Monitoramento da queda máxima de capital a partir do pico.
- **Expectativa Matemática**: Valor médio esperado por cada trade.
- **Tempo de Execução**: Latência média entre o sinal e a confirmação da ordem.

## Funcionalidades Avançadas

- **Curva de Patrimônio**: Gera dados para visualização da evolução do saldo ao longo do tempo.
- **Performance por Ativo**: Identifica quais pares de moedas são mais lucrativos para a estratégia atual.
- **Performance Diária**: Consolida resultados por sessões de 24 horas.
- **Persistência**: Salva e carrega o histórico de operações para que o bot saiba seu desempenho acumulado mesmo após reinicializações.

## Integração

O módulo alimenta a `Inteligencia` com dados de performance real, permitindo que o sistema recomende automaticamente a troca entre os modos de operação (Simulação vs. Real) baseando-se em resultados estatisticamente válidos.
