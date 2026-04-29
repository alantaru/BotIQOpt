# Documentação: Inteligencia.py

O módulo `Inteligencia.py` é o cérebro analítico do BotIQOpt. Ele é responsável por processar dados de mercado, aplicar indicadores técnicos e tomar decisões de compra (Call) ou venda (Put) com base em lógica estatística e heurística.

## Arquitetura

Assim como o Ferramental, a `Inteligencia` é um **Singleton**, garantindo consistência na análise de ativos e indicadores durante a vida útil da aplicação.

### Principais Componentes

1.  **Motor de Decisão (`analyze_and_trade`)**:
    *   Este é o ponto de entrada principal, que recebe os dados do `Ferramental`.
    *   Ele avalia múltiplos ativos simultaneamente, ponderando sinais de compra/venda.
    *   Coordena a execução de ordens via `Ferramental` quando um sinal forte é detectado.

2.  **Cálculo de Indicadores (`calculate_indicators`)**:
    *   **RSI (Relative Strength Index)**: Utilizado para identificar condições de sobrecompra (>70) ou sobrevenda (<30).
    *   **Médias Móveis (SMA/EMA)**: Usado para detectar tendências de curto e longo prazo.
    *   **Bandas de Bollinger**: Usado para medir a volatilidade e identificar possíveis reversões de preço.

3.  **Lógica de Sinais (`get_signal`)**:
    *   Avalia a confluência de múltiplos indicadores para gerar uma recomendação (`call`, `put` ou `neutral`).
    *   A recomendação é baseada em uma pontuação acumulada (score) de vários componentes técnicos.

4.  **Processamento de Dados**:
    *   `prepare_data()`: Realiza o pré-processamento de DataFrames (limpeza, ordenação).

## Funcionamento

O fluxo típico de operação da `Inteligencia` é:
1.  Receber dados históricos/tempo real do `Ferramental`.
2.  Calcular indicadores técnicos sobre esses dados.
3.  Avaliar a tendência atual do mercado.
4.  Verificar se os indicadores sugerem um ponto de entrada.
5.  Se um sinal for gerado, disparar a ordem de compra via `Ferramental`.

## Dependências

*   `pandas` & `numpy`: Para análise quantitativa de dados.
*   `Ferramental`: Para execução de ordens no mercado.
*   `ConfigManager`: Para carregar parâmetros de estratégia (ex: limiares de RSI).
*   `PerformanceTracker`: Para registrar a eficácia dos sinais gerados.

## Customização

A estratégia de negociação pode ser ajustada via arquivo de configuração (`config.ini`), permitindo alterar:
*   Períodos de RSI e Bollinger.
*   Limiares de volatilidade aceitáveis.
*   Ativos que devem ser monitorados prioritariamente.
