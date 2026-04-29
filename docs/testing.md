# Estratégia de Testes: Pirâmide de Automação

O BotIQOpt utiliza uma estratégia de testes baseada na **Pirâmide de Automação**, garantindo que o sistema seja resiliente, fácil de manter e que novas funcionalidades não quebrem a lógica existente.

## A Pirâmide

A pirâmide é dividida em três camadas principais:

### 1. Testes de Unidade (Base)
*   **Foco**: Testar isoladamente cada função e classe.
*   **Módulos Cobertos**: `ConfigManager`, `ErrorTracker`, `PerformanceTracker`, `Ferramental` (métricas e lógica interna).
*   **Velocidade**: Extremamente rápidos.
*   **Localização**: `tests/unit/`

### 2. Testes de Integração (Meio)
*   **Foco**: Verificar a comunicação entre módulos e o comportamento com dependências externas (mockadas).
*   **Módulos Cobertos**: `Ferramental` integrated with `IQOptionAPI` mocks, `Inteligencia` processing pipelines.
*   **Objetivo**: Garantir que os dados fluam corretamente entre o motor analítico e a execução de ordens.
*   **Localização**: `tests/integration/` (ex: `test_main_flow.py`)

### 3. Testes de Fluxo / E2E (Topo)
*   **Foco**: Simular cenários reais de ponta a ponta em ambiente controlado (Modo de Simulação).
*   **Cenários**: Conexão -> Download -> Aprendizado -> Teste -> Operação Real.
*   **Ferramental**: Utiliza o motor de simulação interna para validar a lógica de lucro/prejuízo sem expor capital.

## Cobertura de Código

Utilizamos o `pytest-cov` para monitorar a abrangência dos testes:
*   **Meta**: Manter cobertura acima de **80%** nos módulos críticos (`Ferramental.py`, `Inteligencia.py`).
*   **Relatórios**: Gerados automaticamente na pasta `tests/coverage_report`.

## Como Executar os Testes

Para rodar a suíte completa de testes com relatório de cobertura:

```powershell
.\venv_win\Scripts\pytest.exe -v --cov=. --cov-report=term-missing tests/
```

## Benefícios da Estratégia
*   **Refatoração Segura**: Alterações estruturais no `Ferramental` podem ser validadas instantaneamente pelos +60 testes existentes.
*   **Isolamento de Erros**: O `ErrorTracker` é testado unitariamente para garantir que falhas reais sejam sempre reportadas.
*   **Simulação Realista**: O motor de simulação permite testar o bot 24/7 sem depender da disponibilidade da API real.
