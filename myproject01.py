import os 
from autogen import  ConversableAgent,register_function
from dotenv import load_dotenv
import json
import re
from tavily import TavilyClient
from autogen import GroupChat, GroupChatManager
import agentops

load_dotenv()



AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") 
agentops.init(
    api_key=AGENTOPS_API_KEY,
    default_tags=['autogen']

)


llm_config = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY")
}

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


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
        resultados += r["content"] + "\n"
    return resultados

def calcular_pert(minimo: float, modal: float, maximo: float) -> float:
    """Tool to determine the expected time of a procedure using the PERT formula.
    Args:
    minimo (float): Minimum time found on the web for this specific procedure.
    modal (float): Most common time found on the web for this specific procedure.
    maximo (float): Maximum time found on the web for this specific procedure.
      
    Returns:
        float: Expected time for the procedure calculated with PERT formula.
    """
    T_esperado = (minimo + 4 * modal + maximo) / 6
    return T_esperado

Investigacion_agent = ConversableAgent(
    name="Investigacion_agent",
    system_message="""You are an expert researcher on business formation procedures in Colombia.

You will receive a procedure name and a city.

Your goal is to:
1. Search for the minimum time this procedure takes in the given city
2. Search for the most common (modal) time this procedure takes
3. Search for the maximum time this procedure takes
4. Search for the minimum, modal and maximum cost of this procedure
5. Apply the PERT formula by calling calcular_pert with the three values 
   found. The result must be stored in the "esperado" field of the JSON.
   Do this separately for time and cost.
6. Return the results with the sources found

To find the information use buscar_en_tavily to search for:
- Minimum, modal and maximum time for the procedure in the given city
- Minimum, modal and maximum cost for the procedure in Colombia

If you cannot find a specific value, estimate it based on similar procedures
and mark it as estimated.

Return ONLY this JSON in the same message as TERMINATE:
{
  "tramite": "...",
  "ciudad": "...",
  "tiempo": {
    "minimo": ...,
    "modal": ...,
    "maximo": ...,
    "esperado": ...,
    "unidad": "días hábiles"
  },
  "costo": {
    "minimo": ...,
    "modal": ...,
    "maximo": ...,
    "esperado": ...,
    "unidad": "COP"
  },
  "fuentes": ["url1", "url2"]
}
TERMINATE""",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)




Diagnostico_agent = ConversableAgent(
            name="Diagnostico_agent",
            system_message="""You are an expert advisor in company formation in Colombia.

Your goal is to determine the most appropriate legal structure 
for the user's company. To do this:
1. Research legal structures in Colombia using buscar_en_tavily
2. Based on that research, ask the user relevant questions one 
   at a time to define their company structure
3. Once you have the answers, research the specific procedures 
   required to formally incorporate their company in Colombia

IMPORTANT — SEARCH RULES:
When using buscar_en_tavily, ALWAYS write the query in Spanish.
Example queries:
- "trámites para constituir una SAS en Colombia"
- "procedimientos registro mercantil Colombia"
- "requisitos DIAN registro empresa Colombia"
NEVER search in English.

IMPORTANT — DEFINITION OF TRAMITE:
A "tramite" is ONLY a formal procedure performed before an official 
Colombian government entity such as:
- Cámara de Comercio
- DIAN
- Alcaldía municipal
- Secretaría de Hacienda
- Bomberos
- Curaduría urbana

Internal documents like bylaws, acceptance letters, or shareholder 
documents are NOT tramites — do not include them in the list.

IMPORTANT — CITY:
You MUST ask explicitly:
"¿En qué ciudad o municipio de Colombia va a operar la empresa?"
Never infer the city from other answers.

IMPORTANT — TRAMITES LIST:
Always include these base tramites:
1. Verificar disponibilidad del nombre en RUES
2. Registro mercantil ante Cámara de Comercio
3. Inscripción en el RUT ante la DIAN
4. Apertura de cuenta bancaria empresarial
5. Autorización de facturación electrónica DIAN
6. Registro de Industria y Comercio municipal

Add ONLY if applicable:
- Afiliación seguridad social (if employees)
- Permiso de bomberos (if physical establishment)
- Certificado uso de suelo (if physical establishment)
- Permiso sanitario INVIMA (research with buscar_en_tavily based on activity)
- Licencia de funcionamiento (research with buscar_en_tavily based on activity)

Return ONLY this JSON in the same message as TERMINATE:
{
  "tipo_empresa": "...",
  "regimen_tributario": "...",
  "ciudad": "...",
  "tramites": [
    "tramite 1",
    "tramite 2",
    ...
  ]
}
TERMINATE

When you have collected all the necessary information from the user 
and researched the procedures, you MUST stop asking questions and 
immediately return the JSON followed by TERMINATE.
Do NOT continue the conversation after producing the JSON.
Do NOT ask if the user needs more help.
Do NOT say goodbye.
Just return the JSON and TERMINATE. Nothing else.""",

             llm_config=llm_config,
             code_execution_config=False,
             human_input_mode="NEVER",)

user_proxy = ConversableAgent(
    name="user_proxy",
    llm_config=False,       
    human_input_mode="ALWAYS",  
    code_execution_config=False,
    is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content", "") or ""),
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
    caller=Diagnostico_agent,
    executor=executor_agent,
    name="buscar_en_tavily",
    description="Search for information about legal procedures, permits and requirements to open a business in Colombia.",
)

register_function(
    calcular_pert,
    caller=Investigacion_agent,
    executor=executor_agent,
    name="calcular_pert",
    description="Calculate the expected time or cost of a procedure using the PERT formula. Provide minimum, modal and maximum values.",
)

register_function(
    buscar_en_tavily,
    caller=Investigacion_agent,
    executor=executor_agent,
    name="buscar_en_tavily",
    description="Search for information about time and cost of business formation procedures in Colombia.",
)


groupchat = GroupChat(
    agents=[user_proxy, Diagnostico_agent, executor_agent,],
    messages=[],
    max_round=20,
        allowed_or_disallowed_speaker_transitions={
        user_proxy: [Diagnostico_agent],
        Diagnostico_agent: [executor_agent, user_proxy],
        executor_agent: [Diagnostico_agent],
    },
    speaker_transitions_type="allowed",
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

result = user_proxy.initiate_chat(
    manager,
    message="I want to start a company in Colombia",
)


historial = result.chat_history
json_texto = None

for mensaje in historial:
    contenido = mensaje.get("content", "") or ""
    if "tipo_empresa" in contenido:
        match = re.search(r'\{.*\}', contenido, re.DOTALL)
        if match:
            json_texto = match.group()
            break

diagnostico = json.loads(json_texto)
tramites = diagnostico["tramites"]
ciudad = diagnostico["ciudad"]
tipo_empresa = diagnostico["tipo_empresa"]

print(tramites)
print(ciudad)
print(tipo_empresa)

resultados_investigacion = []

for tramite in tramites:
    result_inv = executor_agent.initiate_chat(
        Investigacion_agent,
        message=f"Investigate this procedure: {tramite} in the city of: {ciudad}",
    )
    resultados_investigacion.append(result_inv.summary)

print(resultados_investigacion)



