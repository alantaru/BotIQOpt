#!/bin/bash

echo "=============================================="
echo "Iniciando instalação das dependências do Bot IQ Option"
echo "=============================================="

# Ativa ambiente virtual se existir
if [ -d "venv" ]; then
    echo "Ativando ambiente virtual..."
    source venv/bin/activate
else
    echo "Nenhum ambiente virtual encontrado. Recomenda-se criar um com 'python -m venv venv'"
fi

echo "Instalando dependências do requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "----------------------------------------------"
echo "Tentando instalar api-iqoption-faria via pip"
echo "----------------------------------------------"
pip install --upgrade api-iqoption-faria || echo "Falha ao instalar api-iqoption-faria, tentando fallback..."

echo "----------------------------------------------"
echo "Tentando instalar iqoptionapi oficial via GitHub"
echo "----------------------------------------------"
pip install -U git+https://github.com/iqoptionapi/iqoptionapi.git || echo "Falha ao instalar iqoptionapi oficial."

echo "----------------------------------------------"
echo "Instalação concluída. Verifique mensagens acima para erros."
echo "Recomenda-se que pelo menos uma das bibliotecas (api-iqoption-faria ou iqoptionapi) esteja instalada."
echo "=============================================="