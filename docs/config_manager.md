# Documentação Técnica: ConfigManager.py

O módulo `ConfigManager.py` é responsável pela centralização e gerenciamento de todas as configurações do bot, utilizando arquivos no formato `.ini`.

## Funcionalidades Principais

- **Leitura de Arquivo INI**: Carrega parâmetros de seções como `[General]`, `[API]`, `[Trading]`, `[LSTM]`, etc.
- **Tipagem Segura**: Métodos para obter valores convertidos automaticamente para `int`, `float`, `bool` ou `list`.
- **Valores Default**: Permite definir valores padrão caso uma chave esteja ausente no arquivo de configuração.
- **Escrita de Configuração**: Permite atualizar parâmetros em tempo de execução e persistir as mudanças no disco.

## Métodos Destacados

- `get_value(section, key, default=None, type=None)`: Recupera um valor específico com conversão de tipo opcional.
- `get_list(section, key)`: Converte uma string separada por vírgula em uma lista Python.
- `update_value(section, key, value)`: Modifica uma configuração na memória e salva no arquivo.

## Segurança

O `ConfigManager` valida a existência do arquivo e das seções solicitadas, prevenindo falhas catastróficas por erros de digitação ou arquivos corrompidos.
