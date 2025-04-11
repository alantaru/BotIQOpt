import logging
import time
from iqoptionapi.stable_api import IQ_Option
from typing import Optional, Tuple, Dict, List
import os
from dotenv import load_dotenv
import pandas as pd # Adicionado import para uso futuro

# Configuração básica de logging (pode ser ajustada externamente se necessário)
# Removido basicConfig para permitir configuração externa pelo main.py
logger = logging.getLogger(__name__)

class NovoFerramental:
    """
    Nova classe para interagir com a API da IQ Option, construída do zero.
    Foco em estabilidade e funcionalidades essenciais para Opções Binárias.
    """
    def __init__(self, email: str, password: str):
        """
        Inicializa a instância e tenta conectar à IQ Option.

        Args:
            email: Email da conta IQ Option.
            password: Senha da conta IQ Option.
        """
        if not email or not password:
            logger.critical("Email ou senha não fornecidos para NovoFerramental.")
            raise ValueError("Email e senha são obrigatórios.")

        self.email = email
        # Não armazenar a senha diretamente se possível, mas necessário para a API
        self._password = password
        self.api: Optional[IQ_Option] = None
        self.connected: bool = False
        self._connect() # Tenta conectar na inicialização

    def _connect(self) -> None:
        """
        Método interno para estabelecer a conexão inicial com a API.
        """
        # Log sem expor o email completo
        email_masked = self.email[:3] + '***' + self.email[self.email.find('@'):] if '@' in self.email else self.email[:3] + '***'
        logger.info(f"Tentando conectar à IQ Option com o email: {email_masked}")
        try:
            # Instancia a API aqui
            self.api = IQ_Option(self.email, self._password)
            # Limpa a senha da memória o quanto antes (embora a instância da API possa mantê-la)
            # self._password = None # Comentar se a API precisar da senha posteriormente

            connected, reason = self.api.connect()

            if connected:
                logger.info("Conexão inicial com a API bem-sucedida. Aguardando autenticação...")
                # Aguarda um tempo para a autenticação (pode ser ajustado ou usar callbacks/verificações mais robustas)
                time.sleep(3) # Ajuste este tempo se necessário
                if self.api.check_connect():
                    self.connected = True
                    logger.info("Autenticação na IQ Option bem-sucedida. Conectado!")
                else:
                    self.connected = False
                    logger.error("Falha na autenticação após conexão inicial. Verifique as credenciais ou status da conta.")
                    self.api = None # Limpa a instância se a autenticação falhar
            else:
                self.connected = False
                logger.error(f"Falha na conexão inicial com a API. Motivo: {reason}")
                # Tratar motivos específicos se necessário (ex: "2FA", "invalid_credentials")
                if reason == "2FA":
                    logger.warning("Autenticação de dois fatores (2FA) é necessária.")
                elif "invalid_credentials" in str(reason):
                     logger.error("Credenciais inválidas.")
                self.api = None # Limpa a instância se a conexão falhar

        except Exception as e:
            self.connected = False
            logger.exception(f"Erro inesperado durante a conexão com a IQ Option: {e}")
            self.api = None

    def check_connection(self) -> bool:
        """
        Verifica o status atual da conexão com a API.

        Returns:
            True se conectado e autenticado, False caso contrário.
        """
        if self.api and self.connected:
            # Verifica novamente usando check_connect para garantir que a sessão ainda é válida
            try:
                if self.api.check_connect():
                    # logger.debug("Verificação de conexão: SUCESSO") # Log muito frequente
                    return True
                else:
                    logger.warning("Verificação de conexão: FALHA (check_connect retornou False). Conexão pode ter sido perdida.")
                    self.connected = False
                    return False
            except Exception as e:
                logger.exception(f"Erro ao verificar conexão com check_connect: {e}")
                self.connected = False
                return False
        # logger.debug(f"Verificação de conexão: FALHA (API não instanciada ou não conectada inicialmente: api={self.api is not None}, connected={self.connected})")
        return False

    def get_profile(self) -> Optional[Dict]:
        """
        Obtém os dados do perfil do usuário conectado.

        Returns:
            Dicionário com dados do perfil (ex: nome, email, saldo, moeda)
            ou None em caso de falha ou não conexão.
        """
        if not self.check_connection():
            logger.error("Não conectado à IQ Option. Não é possível obter o perfil.")
            return None

        logger.info("Obtendo dados do perfil do usuário...")
        try:
            # A API stable_api pode ter um método síncrono ou que bloqueia.
            # Se get_profile_ansyc for realmente assíncrono, precisaria de `await` em um contexto `async`.
            # Assumindo que ele funciona de forma bloqueante aqui.
            profile = self.api.get_profile_ansyc() # Verificar documentação para método síncrono se disponível

            if profile and isinstance(profile, dict):
                logger.info("Dados do perfil obtidos com sucesso.")
                # Log seguro (sem dados muito sensíveis)
                logger.info(f"Perfil - Nome: {profile.get('name', 'N/A')}, Saldo: {profile.get('balance', 'N/A')}, Moeda: {profile.get('currency', 'N/A')}")
                return profile
            elif profile is None:
                 logger.error("Não foi possível obter os dados do perfil (API retornou None).")
                 return None
            else:
                 logger.error(f"Resposta inesperada ao obter perfil: {type(profile)} - {profile}")
                 return None
        except Exception as e:
            logger.exception(f"Erro ao obter dados do perfil da IQ Option: {e}")
            return None

    def get_balance(self) -> Optional[float]:
        """
        Obtém o saldo atual da conta.

        Returns:
            Saldo atual como float ou None em caso de erro.
        """
        if not self.check_connection():
            logger.error("Não conectado. Não é possível obter o saldo.")
            return None

        logger.info("Obtendo saldo da conta...")
        try:
            # A API stable_api geralmente tem o método get_balance()
            # get_balance_v2() pode não estar disponível nela.
            balance = self.api.get_balance()
            if balance is not None:
                logger.info(f"Saldo obtido: {balance}")
                return float(balance) # Garante que é float
            else:
                logger.error("API retornou None para o saldo.")
                return None
        except Exception as e:
            logger.exception(f"Erro ao obter saldo da IQ Option: {e}")
            return None

    def get_historical_data(self, asset: str, timeframe_type: str, timeframe_value: int, count: int, endtime: Optional[float] = None) -> Optional[List[Dict]]:
        """
        Obtém dados históricos (velas) para um ativo específico.

        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            timeframe_type: Tipo de timeframe ('Seconds', 'Minutes', 'Hours')
            timeframe_value: Valor do timeframe (1, 5, 15, etc.)
            count: Número de velas a retornar
            endtime: Timestamp de término (opcional)

        Returns:
            Lista de dicionários contendo dados das velas ou None em caso de erro
        """
        if not self.check_connection():
            logger.error("Não conectado. Não é possível obter dados históricos.")
            return None

        logger.info(f"Obtendo {count} velas históricas para {asset} ({timeframe_value} {timeframe_type})...")
        try:
            # Converte timeframe para o formato esperado pela API (em segundos)
            timeframe_multipliers = {
                "Seconds": 1,
                "Minutes": 60,
                "Hours": 3600
            }
            timeframe_seconds = int(timeframe_value) * timeframe_multipliers.get(timeframe_type, 1)
            if timeframe_seconds <= 0:
                logger.error(f"Timeframe inválido: {timeframe_value} {timeframe_type}")
                return None

            if endtime is None:
                endtime = time.time()

            candles = self.api.get_candles(asset, timeframe_seconds, count, endtime)

            if candles and isinstance(candles, list):
                logger.info(f"Obtidos {len(candles)} candles para {asset}")
                # Validação básica da estrutura do candle (opcional, mas recomendado)
                if len(candles) > 0 and isinstance(candles[0], dict) and 'close' in candles[0]:
                     return candles
                elif len(candles) == 0:
                     logger.warning(f"API retornou lista vazia de candles para {asset}.")
                     return [] # Retorna lista vazia em vez de None
                else:
                     logger.error(f"Formato inesperado dos candles recebidos para {asset}: {type(candles[0]) if len(candles)>0 else 'vazio'}")
                     return None
            elif candles is None:
                 logger.error(f"API retornou None ao obter candles para {asset}.")
                 return None
            else:
                 logger.error(f"Resposta inesperada da API ao obter candles para {asset}: {type(candles)}")
                 return None
        except Exception as e:
            logger.exception(f"Erro ao obter dados históricos da IQ Option: {e}")
            return None

    def buy(self, asset: str, amount: float, action: str, duration: int) -> Tuple[bool, Optional[int]]:
        """
        Executa uma operação de compra de OPÇÃO BINÁRIA (call/put).

        Args:
            asset: Nome do ativo (ex: 'EURUSD')
            amount: Valor da operação (deve ser maior ou igual ao mínimo permitido, ex: 1.0)
            action: Direção da operação ('call' ou 'put')
            duration: Duração da expiração em minutos (geralmente 1 a 5 para binárias)

        Returns:
            Tuple[bool, Optional[int]]: (True, order_id) se bem sucedida, (False, None) caso contrário.
        """
        if not self.check_connection():
            logger.error("Não conectado. Não é possível executar a operação 'buy'.")
            return False, None

        # Validação dos parâmetros
        action = action.lower()
        if action not in ['call', 'put']:
            logger.error(f"Ação inválida: '{action}'. Use 'call' ou 'put'.")
            return False, None

        if not isinstance(duration, int) or duration <= 0:
            logger.error(f"Duração inválida: {duration}. Deve ser um inteiro positivo (minutos).")
            return False, None

        # Valida valor mínimo (pode variar, mas $1 é comum)
        # Idealmente, buscaria esse valor da API se disponível
        min_amount = 1.0
        if amount < min_amount:
            logger.error(f"Valor da operação ({amount}) é menor que o mínimo permitido ({min_amount}).")
            return False, None

        logger.info(f"Executando operação BINÁRIA: {action.upper()} em {asset}, Valor: {amount}, Duração: {duration} min")
        try:
            # A API stable_api usa o método buy para opções binárias
            # O quarto argumento é a duração em minutos
            status, order_id = self.api.buy(amount, asset, action, duration)

            if status and order_id:
                logger.info(f"Operação {action.upper()} em {asset} executada com sucesso. Order ID: {order_id}")
                # Adicionar verificação de execução se necessário (como no Ferramental antigo)
                # if self.verify_order_execution(order_id):
                #    return True, order_id
                # else:
                #    logger.warning(f"Ordem {order_id} não confirmada imediatamente no histórico.")
                #    return False, None # Ou retornar True, order_id e tratar a confirmação depois
                return True, order_id
            elif status and not order_id:
                 logger.error(f"Falha na execução da operação {action.upper()} em {asset}. Status True, mas Order ID é None/False.")
                 return False, None
            else:
                # A API pode retornar uma mensagem no order_id em caso de falha
                error_reason = order_id if order_id else "Razão desconhecida"
                logger.error(f"Falha na execução da operação {action.upper()} em {asset}. Razão: {error_reason}")
                return False, None

        except Exception as e:
            logger.exception(f"Erro inesperado ao executar operação 'buy': {e}")
            return False, None

    def check_win(self, order_id: int) -> Tuple[Optional[str], Optional[float]]:
        """
        Verifica o resultado de uma operação de opção binária finalizada.

        Args:
            order_id: O ID da ordem retornado pelo método 'buy'.

        Returns:
            Tuple[Optional[str], Optional[float]]:
            - ('win', lucro) se a operação foi vencedora.
            - ('loss', prejuizo) se a operação foi perdedora (prejuizo será negativo).
            - ('equal', 0.0) se houve empate.
            - (None, None) se a ordem não foi encontrada, ainda não finalizou, ou ocorreu um erro.
        """
        if not self.check_connection():
            logger.error("Não conectado. Não é possível verificar o resultado da operação.")
            return None, None

        logger.info(f"Verificando resultado da ordem ID: {order_id}...")
        try:
            # A API stable_api pode usar get_optioninfo ou um método similar.
            # Vamos usar check_win que é comum em algumas versões/wrappers.
            # É importante verificar a documentação da API exata que está sendo usada.
            # Assumindo que check_win retorna (win_status, profit)
            # win_status pode ser 'win', 'loose', 'equal'

            # Tentativa 1: Usar check_win_v4 (mais recente em algumas APIs)
            try:
                # Nota: check_win_v4 pode não existir na stable_api padrão.
                # Adapte conforme a API real.
                # Se check_win_v4 não existir, um AttributeError ocorrerá e o bloco except será executado.
                profit = self.api.check_win_v4(order_id)
                # check_win_v4 geralmente retorna apenas o lucro/perda.
                # Precisamos inferir o status.
                if profit is not None:
                    profit = float(profit) # Garante que é float
                    if profit > 0:
                        status = 'win'
                    elif profit < 0:
                        status = 'loss'
                    else:
                        status = 'equal'
                    logger.info(f"Resultado da ordem {order_id} (v4): Status={status}, Lucro/Perda={profit}")
                    return status, profit
                else:
                    # Pode significar que a ordem ainda está aberta ou não encontrada
                    logger.warning(f"check_win_v4 retornou None para a ordem {order_id}. Pode estar aberta ou não existir.")
                    return None, None
            except AttributeError:
                logger.warning("Método check_win_v4 não encontrado, tentando check_win...")
                # Tentativa 2: Usar check_win (mais comum)
                # Este método pode retornar diretamente o status ('win'/'loose'/'equal') ou lucro
                # A implementação exata varia muito entre as bibliotecas.
                # Vamos assumir um retorno de lucro/perda numérico.
                result = self.api.check_win(order_id)

                if isinstance(result, (int, float)):
                    profit = float(result)
                    if profit > 0:
                        status = 'win'
                    elif profit < 0:
                        status = 'loss'
                    else:
                        status = 'equal'
                    logger.info(f"Resultado da ordem {order_id} (check_win): Status={status}, Lucro/Perda={profit}")
                    return status, profit
                elif result is None:
                     logger.warning(f"check_win retornou None para a ordem {order_id}. Pode estar aberta ou não existir.")
                     return None, None
                else:
                    # Se retornar string ('win', 'loose', 'equal') - precisa buscar o valor
                    # Esta parte exigiria outra chamada (ex: get_optioninfo) ou uma API diferente.
                    # Por simplicidade, vamos tratar como não encontrado por agora.
                    logger.warning(f"check_win retornou um tipo inesperado ({type(result)}): {result}. Não foi possível determinar o lucro.")
                    return None, None
            except Exception as e_check:
                 logger.exception(f"Erro ao verificar resultado da ordem {order_id} com check_win/v4: {e_check}")
                 return None, None

        except Exception as e:
            logger.exception(f"Erro inesperado ao verificar resultado da ordem {order_id}: {e}")
            return None, None

    # --- Métodos Futuros (a serem implementados incrementalmente) ---

    # ... outros métodos conforme necessário (get_all_assets, etc.) ...


# --- Bloco de Teste (Executar este arquivo diretamente para testar a classe) ---
if __name__ == "__main__":
    # Configura o logging para exibir mensagens INFO durante o teste
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Iniciando teste da classe NovoFerramental...")

    # Carrega variáveis de ambiente do arquivo .env (se existir)
    # Crie um arquivo .env na raiz do projeto com:
    # IQ_OPTION_EMAIL=seu_email@exemplo.com
    # IQ_OPTION_PASSWORD=sua_senha
    load_dotenv()

    EMAIL_TEST = os.getenv("IQ_OPTION_EMAIL")
    PASSWORD_TEST = os.getenv("IQ_OPTION_PASSWORD")

    if not EMAIL_TEST or not PASSWORD_TEST:
        logger.critical("Credenciais de teste (IQ_OPTION_EMAIL, IQ_OPTION_PASSWORD) não encontradas nas variáveis de ambiente ou no arquivo .env.")
        logger.critical("Crie um arquivo .env ou defina as variáveis de ambiente.")
    else:
        logger.info("Credenciais carregadas. Instanciando NovoFerramental...")
        try:
            ferramenta = NovoFerramental(EMAIL_TEST, PASSWORD_TEST)
            last_order_id = None # Para armazenar o ID da última ordem bem sucedida

            # Teste 1: Verificar conexão
            logger.info("--- Teste 1: Verificação de Conexão ---")
            if ferramenta.check_connection():
                logger.info("Resultado Teste 1: SUCESSO - Conectado.")

                # Teste 2: Obter Perfil (só executa se conectado)
                logger.info("--- Teste 2: Obter Perfil ---")
                profile_data = ferramenta.get_profile()
                if profile_data:
                    logger.info(f"Resultado Teste 2: SUCESSO - Perfil obtido.")
                    # print(f"Detalhes do Perfil: {profile_data}") # Descomente para ver detalhes
                else:
                    logger.error("Resultado Teste 2: FALHA - Não foi possível obter o perfil.")

                # Teste 3: Obter Saldo
                logger.info("--- Teste 3: Obter Saldo ---")
                balance = ferramenta.get_balance()
                if balance is not None:
                    logger.info(f"Resultado Teste 3: SUCESSO - Saldo: {balance}")
                else:
                    logger.error("Resultado Teste 3: FALHA - Não foi possível obter o saldo.")

                # Teste 4: Obter Dados Históricos
                logger.info("--- Teste 4: Obter Dados Históricos ---")
                asset_test = "EURUSD-OTC" # Tentar um ativo OTC que pode estar aberto
                try:
                    historical_data = ferramenta.get_historical_data(asset_test, "Minutes", 1, 10)
                    if historical_data is not None:
                        logger.info(f"Resultado Teste 4: SUCESSO - Obtidos {len(historical_data)} candles para {asset_test}")
                        print(f"Último candle: {historical_data[-1] if historical_data else 'N/A'}") # Descomente para ver detalhes
                    else:
                        logger.error(f"Resultado Teste 4: FALHA - Não foi possível obter dados históricos para {asset_test}.")
                except Exception as hist_e:
                     logger.exception(f"Erro durante o Teste 4 (get_historical_data): {hist_e}")
                     logger.error(f"Resultado Teste 4: FALHA - Exceção ao obter dados históricos para {asset_test}.")

                # Teste 5: Executar Operação de Compra (Exemplo: CALL em EURUSD-OTC, $1, 1 min)
                # CUIDADO: Este teste executará uma operação real na conta PRACTICE/DEMO.
                logger.info("--- Teste 5: Executar Operação 'buy' (CALL) ---")
                buy_asset = "EURUSD-OTC" # Usar o mesmo ativo OTC
                buy_amount = 1.0
                buy_action = "call"
                buy_duration = 1 # 1 minuto (típico para turbo)
                try:
                    buy_success, buy_order_id = ferramenta.buy(buy_asset, buy_amount, buy_action, buy_duration)
                    if buy_success and buy_order_id:
                        logger.info(f"Resultado Teste 5: SUCESSO - Operação {buy_action.upper()} executada. Order ID: {buy_order_id}")
                        last_order_id = buy_order_id # Armazena para o próximo teste
                    else:
                        logger.error(f"Resultado Teste 5: FALHA - Não foi possível executar a operação {buy_action.upper()}.")
                except Exception as buy_e:
                    logger.exception(f"Erro durante o Teste 5 (buy): {buy_e}")
                    logger.error(f"Resultado Teste 5: FALHA - Exceção ao executar a operação {buy_action.upper()}.")

                # Teste 6: Verificar Resultado da Operação (se a ordem foi criada)
                if last_order_id:
                    logger.info(f"--- Teste 6: Verificar Resultado da Ordem ID: {last_order_id} ---")
                    logger.info("Aguardando tempo de expiração (1 minuto + margem)...")
                    time.sleep(70) # Espera 1 minuto + 10 segundos de margem
                    try:
                        win_status, win_profit = ferramenta.check_win(last_order_id)
                        if win_status is not None:
                            logger.info(f"Resultado Teste 6: SUCESSO - Status: {win_status}, Lucro/Perda: {win_profit}")
                        else:
                            logger.warning(f"Resultado Teste 6: INDETERMINADO - Não foi possível obter o resultado final da ordem {last_order_id} (pode não ter fechado ou erro na API).")
                    except Exception as check_e:
                        logger.exception(f"Erro durante o Teste 6 (check_win): {check_e}")
                        logger.error(f"Resultado Teste 6: FALHA - Exceção ao verificar resultado da ordem {last_order_id}.")
                else:
                    logger.warning("Teste 6 pulado: Nenhuma ordem foi criada com sucesso no Teste 5.")


                # Adicionar mais testes aqui à medida que os métodos forem implementados

            else:
                logger.error("Resultado Teste 1: FALHA - Não conectado.")
                logger.warning("Verifique suas credenciais, conexão com a internet ou status da conta IQ Option.")
                logger.warning("Se o 2FA estiver ativo, a conexão inicial pode falhar aqui. Funcionalidade 2FA precisa ser implementada.")

        except ValueError as ve:
             logger.critical(f"Erro ao instanciar NovoFerramental: {ve}")
        except Exception as e:
             logger.exception(f"Erro inesperado durante o teste: {e}")

    logger.info("Teste da classe NovoFerramental finalizado.")