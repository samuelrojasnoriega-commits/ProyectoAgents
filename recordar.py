import os
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv

load_dotenv()

config_list = [{"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]

# 1. AssistantAgent
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

# 2. RetrieveUserProxyAgent
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "docs_path": [
            "C:\\Users\\panon\\Documents\\samuel_perfil.txt",
        ],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "chroma",
        "overwrite": True,
    },
    code_execution_config=False,
)

# 3. Iniciar chat
code_problem = "What has Samuel NOT built yet?"

chat_result = ragproxyagent.initiate_chat(
    assistant,
    message=ragproxyagent.message_generator,
    problem=code_problem,

)