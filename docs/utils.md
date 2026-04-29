# Documentação: Utilitários (ConfigManager, ErrorTracker, PerformanceTracker)

O BotIQOpt conta com três módulos de utilitários fundamentais que garantem a estabilidade, configurabilidade e monitoramento do sistema.

## 1. ConfigManager.py

O `ConfigManager` é responsável pelo carregamento e gerenciamento centralizado de todas as configurações do bot, lidas a partir de um arquivo `.ini`.

*   **Singleton**: Garante consistência em todo o sistema.
*   **Acesso Tipado**: Métodos como `get_value(section, key, default, type)` permitem recuperar valores convertidos automaticamente para o tipo correto (int, float, bool).
*   **Gerenciamento de Listas**: O método `get_list()` facilita a recuperação de ativos (ex: `EURUSD,GBPUSD`) como uma lista Python.

## 2. ErrorTracker.py

O `ErrorTracker` atua como um repositório centralizado para erros técnicos e falhas de runtime detectadas pelo sistema.

*   **Identificação de Falhas**: Registra o tipo do erro, a mensagem e o traceback completo.
*   **Métricas de Erro**: Permite visualizar o número total de erros ocorridos em uma sessão.
*   **Recuperação de Erros**: Fornece métodos para obter todos os erros de um tipo específico ou de toda a sessão para análise posterior.

## 3. PerformanceTracker.py

O `PerformanceTracker` monitora a eficiência operacional do bot, focando principalmente no tempo de execução de métodos críticos e na taxa de acerto das estratégias.

*   **Monitoramento de Tempo**: Fornece ferramentas para medir o tempo gasto em operações de rede ou processamento analítico.
*   **Estatísticas de Assertividade**: (Integrado com Inteligencia) Calcula a performance histórica da estratégia atual.
*   **Relatórios**: Gera métricas sumarizadas sobre o comportamento do sistema sob carga.

## Integração

Estes três módulos trabalham em conjunto para fornecer uma base sólida para os módulos de alta complexidade (`Ferramental` e `Inteligencia`). O `ConfigManager` dita o comportamento, o `ErrorTracker` observa as falhas e o `PerformanceTracker` valida a eficiência.
