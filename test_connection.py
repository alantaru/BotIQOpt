import os
from iqoptionapi.stable_api import IQ_Option
from dotenv import load_dotenv

load_dotenv()

email = os.getenv("IQ_OPTION_EMAIL")
password = os.getenv("IQ_OPTION_PASSWORD")

if not email or not password:
    print("Erro: Credenciais não encontradas no arquivo .env")
    exit()

iq = IQ_Option(email, password)
status, reason = iq.connect()

if status:
    print("Conexão bem-sucedida!")
else:
    print(f"Falha na conexão. Motivo: {reason}")
