<div align="center">

# BotIQOpt

### Bot de Trading Automatizado com Inteligência Artificial

[![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![pytest](https://img.shields.io/badge/pytest-tested-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

**Desenvolvido por [Isaac Oliveira](https://www.linkedin.com/in/isaac-oliveira-a0924441)**
*Líder de Serviços II · Desenvolvedor Fullstack · Analista de Sistemas*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-isaac--oliveira-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/isaac-oliveira-a0924441)
[![Email](https://img.shields.io/badge/Email-isaac__oliveira@live.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:isaac_oliveira@live.com)

</div>

---

## Sobre o Projeto

Bot de trading automatizado para a plataforma **IQ Option** com núcleo de inteligência artificial capaz de aprender e melhorar continuamente. Arquitetura modular com separação clara entre ferramental de API, lógica de inteligência e utilitários.

O sistema opera em 4 modos — download de dados históricos, aprendizado, teste e operação real — permitindo validar estratégias antes de expô-las ao mercado.

---

## Funcionalidades

| Módulo | Descrição |
|---|---|
| **Núcleo Ferramental** | Integração com a API do IQ Option — operações, dados de mercado, gestão de conta |
| **Núcleo de Inteligência** | Modelo de IA para análise e previsão de movimentos de mercado |
| **Auto-treinamento** | IA aprende continuamente com novos dados históricos |
| **4 Modos de Operação** | Download, Aprendizado, Teste e Real |
| **Dashboard de Performance** | Métricas de desempenho em tempo real |
| **Gestão de Risco** | Estratégias configuráveis de proteção de capital |
| **Logging Estruturado** | Rastreabilidade completa de todas as operações |

---

## Arquitetura

```
BotIQOpt/
├── main.py                   # Orquestrador — seleciona o modo de operação
├── ferramental/
│   ├── Ferramental.py        # Integração com API IQ Option
│   ├── PerformanceMetrics.py # Métricas de desempenho
│   └── ErrorTracker.py       # Rastreamento de erros
├── inteligencia/
│   └── Inteligencia.py       # Núcleo de IA (análise + previsão)
├── brain/                    # Modelos treinados e dados de aprendizado
├── utils/
│   ├── ConfigManager.py
│   ├── LogManager.py
│   └── PerformanceTracker.py
├── tests/                    # Suíte pytest
└── config.ini.template       # Template de configuração (sem credenciais)
```

---

## Instalação

**Linux / macOS**
```bash
git clone https://github.com/alantaru/BotIQOpt.git
cd BotIQOpt
chmod +x install.sh && ./install.sh
```

**Windows**
```bash
git clone https://github.com/alantaru/BotIQOpt.git
cd BotIQOpt
install.bat
```

---

## Configuração

```bash
# 1. Copiar templates
cp config.ini.template config.ini
cp .env.example .env

# 2. Editar .env com suas credenciais
IQOPTION_EMAIL=seu_email@exemplo.com
IQOPTION_PASSWORD=sua_senha
```

> ⚠️ Nunca commite `.env` ou `config.ini` com credenciais reais.

---

## Uso

```bash
# Download de dados históricos
python main.py --mode DOWNLOAD --asset EURUSD --timeframe_type Minutes --timeframe_value 1 --candle_count 1000

# Treinar o modelo
python main.py --mode LEARNING

# Backtesting (sem dinheiro real)
python main.py --mode TEST

# Operação real
python main.py --mode REAL
```

---

## Testes

```bash
pytest tests/ -v
```

---

## Stack

`Python 3.12` `iqoptionapi` `pandas` `numpy` `scikit-learn` `pytest`

---

> **Aviso Legal:** Projeto desenvolvido para fins educacionais e estudo de algoritmos de trading. Trading envolve risco de perda de capital.

---

<div align="center">

**Isaac Oliveira** — Transformando operações reais em software de qualidade

[![LinkedIn](https://img.shields.io/badge/Conectar_no_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/isaac-oliveira-a0924441)

</div>
