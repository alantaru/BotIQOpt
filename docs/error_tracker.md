# Documentação Técnica: ErrorTracker.py

O `ErrorTracker.py` provê um sistema robusto de monitoramento e auditoria de erros em tempo real para o BotIQOpt.

## Funcionalidades

- **Registro Centralizado**: Captura erros de todos os módulos (Inteligência, Ferramental, Main).
- **Persistência em JSON**: Salva a lista de erros em um arquivo para análise posterior após o fechamento do bot.
- **Deduplicação**: Agrupa erros idênticos para evitar logs repetitivos e facilitar a identificação de problemas sistêmicos.
- **Traceback Integrado**: Armazena o stack trace completo de cada exceção para depuração rápida.

## Métodos de Interface

- `add_error(error_type, message, traceback=None)`: Registra uma nova falha no sistema.
- `get_errors()`: Retorna a lista completa de erros registrados na sessão.
- `clear_errors()`: Limpa a memória de erros (útil após resoluções automáticas).
- `save_to_file(filepath)`: Exporta o histórico de erros para um arquivo JSON.

## Importância

Este módulo é vital para a melhoria contínua do bot, permitindo que o desenvolvedor identifique padrões de falhas de conexão ou erros de lógica que ocorrem apenas em condições específicas de mercado.
