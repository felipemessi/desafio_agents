from typing import Dict, List
from autogen import ConversableAgent
import sys
import os
import re
import math
from dotenv import load_dotenv

load_dotenv()


def fetch_restaurant_data(restaurant_name: str) -> Dict[str, List[str]]:
    """
    Recupera avaliações de um restaurante específico do arquivo restaurantes.txt.
    
    Args:
        restaurant_name: Nome do restaurante a ser pesquisado
        
    Returns:
        Dicionário com o nome do restaurante como chave e lista de avaliações como valor
        
    Example:
        {"Santo Pão": ["Sanduíches e sopas de boa qualidade...", "Atendimento eficiente..."]}
    """
    try:
        with open("restaurantes.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        return {restaurant_name: []}
    
    reviews = []
    
    for line in lines:
        line = line.strip()
        if line.startswith(restaurant_name + "."):
            review = line.split(".", 1)[1].strip()
            reviews.append(review)
    
    return {restaurant_name: reviews}


def calculate_overall_score(
    restaurant_name: str, 
    food_scores: List[int], 
    customer_service_scores: List[int]
) -> Dict[str, float]:
    """
    Calcula a pontuação geral de um restaurante baseada nos scores de comida e atendimento.
    
    Args:
        restaurant_name: Nome do restaurante
        food_scores: Lista de scores da comida (1-5)
        customer_service_scores: Lista de scores do atendimento (1-5)
        
    Returns:
        Dicionário com nome do restaurante e pontuação final (0-10) com 3 casas decimais
        
    Formula:
        SUM(sqrt(food_scores[i]^2 * customer_service_scores[i]) * 1/(N * sqrt(125)) * 10
    """
    if not food_scores or not customer_service_scores or len(food_scores) != len(customer_service_scores):
        return {restaurant_name: 0.000}
    
    n = len(food_scores)
    total_score = 0.0
    
    for i in range(n):
        score_component = math.sqrt(food_scores[i]**2 * customer_service_scores[i])
        total_score += score_component
    
    final_score = total_score * (1 / (n * math.sqrt(125))) * 10
    
    return {restaurant_name: round(final_score, 3)}


def get_data_fetch_agent_prompt(restaurant_query: str) -> str:
    """
    Gera prompt para o agente de busca de dados.
    
    Args:
        restaurant_query: Consulta do usuário sobre o restaurante
        
    Returns:
        Prompt formatado para o agente
    """
    return f"""
    Você é um agente especializado em recuperar dados de restaurantes.
    Sua única tarefa é identificar o nome do restaurante na consulta e usar a função fetch_restaurant_data.
    
    Consulta: "{restaurant_query}"
    
    INSTRUÇÕES:
    1. Identifique exatamente o nome do restaurante (ex: "Bob's", "Paris 6", "KFC", "China in Box")
    2. Chame OBRIGATORIAMENTE a função fetch_restaurant_data com o nome correto
    3. Retorne as avaliações encontradas para o próximo agente
    
    SEMPRE use a função fetch_restaurant_data - não tente responder sem ela.
    """


def get_review_analysis_agent_prompt() -> str:
    """
    Gera prompt para o agente de análise de avaliações.
    
    Returns:
        Prompt formatado para o agente de análise
    """
    return """
    Você é um agente especializado em análise de sentimentos para avaliações de restaurantes.
    Analise as avaliações recebidas e extraia scores numéricos para comida e atendimento.
    
    ESCALA OBRIGATÓRIA (NÃO MODIFICAR):
    - 1/5: horrível, nojento, terrível
    - 2/5: ruim, desagradável, ofensivo  
    - 3/5: mediano, sem graça, irrelevante
    - 4/5: bom, agradável, satisfatório
    - 5/5: incrível, impressionante, surpreendente
    
    PROCESSO:
    1. Para cada avaliação recebida, identifique aspectos de COMIDA e ATENDIMENTO
    2. Converta adjetivos usando EXATAMENTE a escala acima
    3. Retorne no formato: "food_scores: [X,Y,Z] customer_service_scores: [A,B,C]"
    
    MAPEAMENTO DE CONTEXTO:
    - COMIDA: comida, sabor, ingredientes, pratos, qualidade, saboroso, preparado, sanduíches, hambúrgueres
    - ATENDIMENTO: atendimento, funcionários, serviço, garçons, baristas, eficiente
    
    Se uma avaliação não mencionar especificamente um aspecto, use score 3 (mediano).
    """


def get_score_agent_prompt() -> str:
    """
    Gera prompt para o agente de cálculo de pontuação.
    
    Returns:
        Prompt formatado para o agente de pontuação
    """
    return """
    Você é responsável por calcular a pontuação final do restaurante.
    Receba os scores do agente anterior e use a função calculate_overall_score.
    
    INSTRUÇÕES:
    1. Extraia food_scores e customer_service_scores da mensagem anterior
    2. SEMPRE chame a função calculate_overall_score com:
       - restaurant_name: nome do restaurante
       - food_scores: lista de integers [1,2,3,etc]  
       - customer_service_scores: lista de integers [1,2,3,etc]
    3. Retorne a pontuação com exatamente 3 casas decimais
    
    FORMATO ESPERADO:
    "A avaliação média do [RESTAURANTE] é [X.XXX]."
    
    SEMPRE use a função - não calcule manualmente.
    """


def main(user_query: str) -> None:
    """
    Função principal que coordena o sistema de agentes conversacionais.
    
    Args:
        user_query: Consulta do usuário sobre um restaurante
    """
    llm_config = {
        "config_list": [
            {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}
        ]
    }
    
    # Agente supervisor/ponto de entrada
    entrypoint_agent = ConversableAgent(
        "entrypoint_agent",
        system_message="""
        Você é o agente supervisor responsável por coordenar o processo de análise de restaurantes.
        Coordene a execução sequencial dos outros agentes e formate a resposta final para o usuário.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    
    # Agente de busca de dados
    data_fetch_agent = ConversableAgent(
        "data_fetch_agent",
        system_message=get_data_fetch_agent_prompt(user_query),
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    
    # Registrar função no entrypoint_agent para que todos possam usá-la
    entrypoint_agent.register_for_llm(
        name="fetch_restaurant_data", 
        description="Obtém as avaliações de um restaurante específico."
    )(fetch_restaurant_data)
    entrypoint_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)
    
    # Registrar também no data_fetch_agent
    data_fetch_agent.register_for_llm(
        name="fetch_restaurant_data", 
        description="Obtém as avaliações de um restaurante específico."
    )(fetch_restaurant_data)
    data_fetch_agent.register_for_execution(name="fetch_restaurant_data")(fetch_restaurant_data)
    
    # Agente de análise de avaliações
    review_analysis_agent = ConversableAgent(
        "review_analysis_agent",
        system_message=get_review_analysis_agent_prompt(),
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    
    # Agente de cálculo de pontuação
    score_agent = ConversableAgent(
        "score_agent",
        system_message=get_score_agent_prompt(),
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    
    # Registrar função no entrypoint_agent
    entrypoint_agent.register_for_llm(
        name="calculate_overall_score",
        description="Calcula a pontuação final do restaurante baseada nos scores."
    )(calculate_overall_score)
    entrypoint_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)
    
    # Registrar também no score_agent
    score_agent.register_for_llm(
        name="calculate_overall_score",
        description="Calcula a pontuação final do restaurante baseada nos scores."
    )(calculate_overall_score)
    score_agent.register_for_execution(name="calculate_overall_score")(calculate_overall_score)
    
    # Configuração do fluxo de conversação
    chat_sequence = [
        {
            "sender": entrypoint_agent,
            "recipient": data_fetch_agent,
            "message": f"Recupere as avaliações para: {user_query}",
            "max_turns": 3
        },
        {
            "sender": data_fetch_agent,
            "recipient": review_analysis_agent,
            "message": "Analise as avaliações e extraia scores de comida e atendimento usando a escala definida.",
            "max_turns": 3
        },
        {
            "sender": review_analysis_agent,
            "recipient": score_agent,
            "message": "Calcule a pontuação final do restaurante usando os scores extraídos.",
            "max_turns": 3
        }
    ]
    
    # Executa o pipeline de agentes
    result = entrypoint_agent.initiate_chats(chat_sequence)
    
    # Exibe a resposta final
    if result and len(result) > 0:
        final_chat = result[-1]
        if hasattr(final_chat, 'chat_history') and final_chat.chat_history:
            last_message = final_chat.chat_history[-1]['content']
            
            # Extrai a pontuação da resposta e formata adequadamente
            score_match = re.search(r'(\d+\.\d{3})', last_message)
            restaurant_match = re.search(r'(?:do|da) ([^?]+)\?', user_query, re.IGNORECASE)
            
            if score_match and restaurant_match:
                score = score_match.group(1)
                restaurant = restaurant_match.group(1).strip()
                print(f"A avaliação média do {restaurant} é {score}.")
            else:
                print(last_message)
        else:
            print("Não foi possível processar a consulta adequadamente.")

# NÃO modifique o código abaixo.
if __name__ == "__main__":
    assert len(sys.argv) > 1, "Certifique-se de incluir uma consulta para algum restaurante ao executar a função main."
    main(sys.argv[1])