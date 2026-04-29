<div align="center">

# BotIQOpt

### Bot de Trading Automatizado com Inteligência Artificial

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![pytest](https://img.shields.io/badge/pytest-tested-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**👤 [Isaac Oliveira](https://www.linkedin.com/in/isaac-oliveira-a0924441) · 📧 [isaac_oliveira@live.com](mailto:isaac_oliveira@live.com)**

</div>

---

## Sobre o Projeto

Bot de trading automatizado para a plataforma **IQ Option**, com núcleo de inteligência artificial capaz de aprender e melhorar com o tempo. Desenvolvido em Python 3.12 com arquitetura modular separando ferramental de API, lógica de inteligência e utilitários.

O sistema opera em 4 modos distintos — download de dados históricos, aprendizado, teste e operação real — permitindo validar estratégias antes de expô-las ao mercado.

---

## Funcionalidades

- **Núcleo Ferramental** — integração com a API do IQ Option (operações, dados de mercado, gestão de conta)
- **Núcleo de Inteligência** — modelo de IA para análise e previsão de movimentos de mercado
- **Auto-treinamento** — IA aprende continuamente com novos dados históricos
- **4 modos de operação** — Download, Aprendizado, Teste e Real
- **Dashboard de Performance** — métricas de desempenho em tempo real
- **Gestão de Risco** — estratégias configuráveis de proteção de capital
- **Logging estruturado** — rastreabilidade completa de todas as operações

---

## Arquitetura

```
BotIQOpt/
├── main.py                  # Ponto de entrada — orquestra os modos
├── ferramental/
│   ├── Ferramental.py       # Integração com API IQ Option
│   ├── PerformanceMetrics.py
│   └── ErrorTracker.py
├── inteligencia/
│   └── Inteligencia.py      # Núcleo de IA (análise + previsão)
├── brain/                   # Modelos treinados e dados de aprendizado
├── utils/
│   ├── ConfigManager.py
│   ├── LogManager.py
│   └── PerformanceTracker.py
├── tests/                   # Suíte de testes pytest
├── docs/                    # Documentação técnica
└── config.ini.template      # Template de configuração (sem credenciais)
```

---

## Instalação

### Pré-requisitos

- Python 3.12+
- pip

### Linux / macOS

```bash
git clone https://github.com/alantaru/BotIQOpt.git
cd BotIQOpt
chmod +x install.sh
./install.sh
```

### Windows

```bash
git clone https://github.com/alantaru/BotIQOpt.git
cd BotIQOpt
install.bat
```

---

## Configuração

1. Copie o template de configuração:

```bash
cp config.ini.template config.ini
```

2. Copie o template de variáveis de ambiente:

```bash
cp .env.example .env
```

3. Edite `.env` com suas credenciais da IQ Option:

```env
IQOPTION_EMAIL=seu_email@exemplo.com
IQOPTION_PASSWORD=sua_senha
```

4. Ajuste as preferências de trading em `config.ini`

> ⚠️ **Nunca commite `.env` ou `config.ini` com credenciais reais.**

---

## Uso

### Download de Dados Históricos

```bash
python main.py --mode DOWNLOAD --asset EURUSD --timeframe_type Minutes --timeframe_value 1 --candle_count 1000
```

### Aprendizado (treinar o modelo)

```bash
python main.py --mode LEARNING
```

### Teste (backtesting sem dinheiro real)

```bash
python main.py --mode TEST
```

### Operação Real

```bash
python main.py --mode REAL
```

---

## Testes

```bash
pytest tests/ -v
```

---

## Stack

| Tecnologia | Uso |
|---|---|
| **Python 3.12** | Linguagem principal |
| **iqoptionapi** | Integração com a plataforma IQ Option |
| **pandas / numpy** | Processamento de dados de mercado |
| **scikit-learn** | Modelos de machine learning |
| **pytest** | Suíte de testes automatizados |

---

## Aviso Legal

> Este projeto é desenvolvido para fins educacionais e de estudo de algoritmos de trading. Trading envolve risco de perda de capital. Use por sua conta e risco.

---

<div align="center">

**Isaac Oliveira** · [LinkedIn](https://www.linkedin.com/in/isaac-oliveira-a0924441) · [isaac_oliveira@live.com](mailto:isaac_oliveira@live.com)

</div>
