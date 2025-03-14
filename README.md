# IQ Option Bot

## Descrição
Bot de trading automatizado para a plataforma IQ Option com núcleo de IA auto-aprendizagem. Desenvolvido para operar em diferentes ativos financeiros (Forex, Ações, Criptomoedas) com estratégias baseadas em análise técnica e machine learning.

## Requisitos do Sistema
- Python 3.9 ou superior
- 4GB de RAM (8GB recomendado)
- Conexão estável com a internet
- Conta IQ Option (demo ou real)

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/iq-option-bot.git
cd iq-option-bot
```

2. Execute o script de instalação:
```bash
chmod +x install.sh
./install.sh
```

3. Ative o ambiente virtual:
```bash
source .venv/bin/activate
```

## Configuração

1. Edite o arquivo `.env` com suas credenciais da IQ Option:
```bash
IQ_OPTION_API_KEY=sua_chave_api
IQ_OPTION_EMAIL=seu_email
IQ_OPTION_PASSWORD=sua_senha
```

2. Configure os parâmetros no `conf.ini`:
- Modo de operação (TEST/REAL)
- Ativos a serem negociados
- Parâmetros de gerenciamento de risco
- Configurações do modelo de IA

## Uso Básico

Para iniciar o bot:
```bash
python main.py
```

Modos de operação:
- **DOWNLOAD**: Baixa dados históricos
- **LEARNING**: Treina o modelo de IA
- **TEST**: Executa em modo de teste
- **REAL**: Executa operações reais

## Modelo de IA

O bot utiliza uma rede neural profunda com as seguintes características:
- 3 camadas ocultas com 128 neurônios cada
- Função de ativação ReLU
- Dropout de 20% para prevenção de overfitting
- Early stopping com paciência de 10 épocas

## Estratégias Implementadas

1. Análise Técnica:
- Médias móveis
- RSI
- MACD
- Bandas de Bollinger

2. Machine Learning:
- Previsão de tendência
- Detecção de padrões
- Gerenciamento de risco adaptativo

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contribuição

Contribuições são bem-vindas! Siga as diretrizes no arquivo [CONTRIBUTING.md](CONTRIBUTING.md).

## Suporte

Para reportar bugs ou solicitar novas funcionalidades, abra uma issue no repositório.
