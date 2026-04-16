import os 
from autogen import  ConversableAgent,register_function
from dotenv import load_dotenv
import json
import re
from tavily import TavilyClient
import agentops
import logging


load_dotenv()



#AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") 
#agentops.init(
#   api_key=AGENTOPS_API_KEY,
 #   default_tags=['autogen']

#)


llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY")
}

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

logging.basicConfig(filename="output.log", level=logging.INFO)

temas = ["economía"]

def buscar_en_tavily(query: str) -> str:
    """Busca información en internet usando Tavily.
    
    Args:
        query: La búsqueda a realizar
    
    Returns:
        Resultados de la búsqueda como texto
    """
    response = tavily_client.search(query=query, max_results=2,search_depth="advanced")
    resultados = ""
    for r in response["results"]:
        resultados += f"URL: {r['url']}\n" + r["content"] + "\n"
    return resultados

def recolectar_articulos(temas: list) -> list:
    articulos = []
    for tema in temas:
        query = f"{tema} en colombia hoy"
        resultado = buscar_en_tavily(query)
        articulos.append({
            "tema": tema,
            "contenido": resultado
        })
    return articulos

articulos = recolectar_articulos(temas)

for articulo in articulos:
    print(f"\n{'='*50}")
    print(f"TEMA: {articulo['tema']}")
    print(f"{'='*50}")
    print(articulo['contenido'])

extraccion_agent = ConversableAgent(
    name="Extraccion_agent",
    system_message="""

Eres un experto en análisis de artículos periodísticos colombianos.

Tu tarea es:
1. Leer el artículo que te proporcionan e identificar todas las estadísticas, cifras o indicadores mencionados.
2. Por cada estadística encontrada, identificar la fuente que el artículo cita (DANE, Banco Mundial, ONU, etc.).
3. Usar la herramienta buscar_en_tavily para buscar si esa fuente realmente publicó ese dato.
4. Cada resultado de búsqueda comienza con una línea 'URL: ...' seguida del contenido. Usa esa URL como url_articulo_origen cuando corresponda al artículo que estás analizando, y como url_fuente_original cuando corresponda a la fuente que estás verificando.
5. Si la búsqueda no retorna una URL que corresponda exactamente a la fuente citada, retorna verificado: false y url_fuente_original: null.
6. Retornar SOLO un JSON válido, sin explicaciones adicionales.

Si no encuentras estadísticas retorna: []

Formato de respuesta:
[
    {
        "tema": "tema del artículo",
        "declaracion": "frase exacta del artículo que contiene el dato",
        "dato": "el número o cifra específica",
        "fuente_citada": "fuente mencionada en el artículo",
        "url_articulo_origen": "url del artículo que contiene la estadística",
        "url_fuente_original": "url de la fuente que publicó el dato original",
        "verificado": true/false
    }
]

Cuando termines escribe TERMINATE.

""",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

executor_agent = ConversableAgent(
    name="Executor_agent",
    llm_config=False,
    is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content", "") or ""),
    default_auto_reply="No tools to execute.",
    human_input_mode="NEVER",

)

register_function(
    buscar_en_tavily,
    caller=extraccion_agent,
    executor=executor_agent,
    name="buscar_en_tavily",
    description="Busca en internet si una fuente (DANE, Banco Mundial, ONU, etc.) realmente publicó un dato o estadística específica. Recibe el nombre de la fuente y el dato como query.",
)

for articulo in articulos:
    resultado = executor_agent.initiate_chat(
        extraccion_agent,
        message=f"Analiza este artículo del tema {articulo['tema']}:\n\n{articulo['contenido']}"
    )


