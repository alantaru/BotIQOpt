# BotIQOpt

Um bot avançado para IQ Option com núcleo de inteligência artificial para trading automatizado.

## Recursos

- **Núcleo Ferramental**: Implementa operações com a API do IQ Option
- **Núcleo de Inteligência**: IA para análise e previsão de mercados
- **Modos de Operação**: Download, Aprendizado, Teste e Real
- **Auto-treinamento**: IA capaz de aprender e melhorar com o tempo
- **Dashboard de Performance**: Visualização de métricas de desempenho
- **Gerenciamento de Risco**: Estratégias inteligentes para proteção de capital

## Instalação

### Linux/Mac

```bash
chmod +x install.sh
./install.sh
```

### Windows

```
install.bat
```

## Configuração

1. Edite o arquivo `.env` com suas credenciais da IQ Option
2. Configure suas preferências de trading no arquivo `conf.ini`

## Modos de Uso

### Download de Dados Históricos

```bash
python main.py --mode DOWNLOAD --asset EURUSD --timeframe_type Minutes --timeframe_value 1 --candle_count 1000
```

### Aprendizado

```bash
python main.py --mode LEARNING
```

### Teste

```bash
python main.py --mode TEST
```

### Operação Real

```bash
python main.py --mode REAL
```

## Arquitetura

- **main.py**: Ponto de entrada principal
- **ferramental/**: Módulo para interação com a API do IQ Option
- **inteligencia/**: Implementação da IA e algoritmos de aprendizado

## Segurança

- **Modo de Teste**: Sempre execute em modo de teste antes de usar o modo real
- **Auto-Switch**: O sistema só muda para modo real após atingir critérios rigorosos

## Aviso Legal

Este software é para uso educacional apenas. Trading envolve risco de perda parcial ou total do capital investido. Use por sua conta e risco.
