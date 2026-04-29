# Visão Geral da Arquitetura do Sistema

O BotIQOpt é projetado seguindo princípios de modularidade, alta coesão e baixo acoplamento. A arquitetura é separada em camadas de responsabilidade clara.

## Visão Geral das Camadas

### 1. Camada de Aplicação (`main.py`)
- Ponto de entrada do sistema.
- Orquestra a inicialização de todos os módulos.
- Gerencia o loop principal de execução (ciclo de análise e trading).

### 2. Camada de Inteligência (`inteligencia/`)
- Contém a lógica de Deep Learning (LSTM) e Análise Técnica.
- Responsável por transformar dados brutos em decisões acionáveis.
- Implementa o mecanismo de auto-switch entre modos de operação.

### 3. Camada de Ferramental (`ferramental/`)
- Encapsula a comunicação com o IQ Option.
- Gere o Singleton da conexão e a segurança de rede.
- Implementa o Gerenciador de Risco que atua como porta de saída para todas as ordens.

### 4. Camada de Utilidades (`utils/`)
- **ConfigManager**: Centraliza todas as configurações via arquivo `.ini`.
- **ErrorTracker**: Monitora e persiste erros em tempo real.
- **PerformanceTracker**: Calcula métricas de acerto (Win Rate), lucro e estatísticas de trading.

## Fluxo de Dados (Data Flow)

1.  **Coleta de Dados**: O `Ferramental` é responsável por coletar dados históricos e em tempo real da IQ Option. Ele atua como a única fonte de dados de mercado para o sistema.
2.  **Pré-processamento e Análise**: Os dados brutos são enviados para a `Inteligencia`, que os pré-processa, adiciona indicadores técnicos e padrões de velas.
3.  **Previsão**: O modelo LSTM da `Inteligencia` utiliza os dados processados para prever a direção do mercado (compra, venda ou aguardar).
4.  **Validação de Confiança**: A `Inteligencia` avalia a confiança da previsão e, com base nas métricas de desempenho do `PerformanceTracker`, decide se deve ou não operar.
5.  **Gerenciamento de Risco**: Se a decisão for operar, a ordem é enviada para o `Ferramental`, que a submete ao `RiskManager`. O `RiskManager` verifica se a operação está dentro dos limites de risco definidos (por exemplo, perda diária, risco por operação).
6.  **Execução da Ordem**: Se a ordem for aprovada pelo `RiskManager`, o `Ferramental` a envia para a API da IQ Option.
7.  **Rastreamento de Desempenho**: O resultado da operação é capturado e enviado para o `PerformanceTracker`, que atualiza as métricas de desempenho do bot.

## Padrões de Design Utilizados

- **Singleton**: Garante uma única instância do `Ferramental` e do `ConfigManager`, evitando múltiplas conexões com a API e leituras repetidas do arquivo de configuração.
- **Circuit Breaker**: O `RiskManager` atua como um Circuit Breaker, interrompendo as operações de trading se os limites de perda forem atingidos, prevenindo perdas catastróficas.
- **Strategy**: O módulo `Inteligencia` pode ser visto como uma implementação do padrão Strategy, onde diferentes estratégias de análise (por exemplo, diferentes modelos ou conjuntos de indicadores) podem ser implementadas e trocadas.
- **Observer**: O `PerformanceTracker` e o `ErrorTracker` atuam como Observers, monitorando o estado do sistema e registrando informações relevantes sem acoplar-se diretamente aos componentes que observam.
