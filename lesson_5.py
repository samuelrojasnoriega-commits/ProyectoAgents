# lesson_5.py
import datetime
import os
from dotenv import load_dotenv
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import ConversableAgent, AssistantAgent
import sys
import subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "yfinance", "-q"], check=True)




load_dotenv()

llm_config = {"model": "gpt-4-turbo"}

# ==================================================
print("=" * 50)
print("  PARTE 1: Codigo generado desde cero")
print("=" * 50)

import types
venv_context = types.SimpleNamespace(
    bin_path=r"C:\Users\panon\autogen_proyecto\venv\Scripts"
)

executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    virtual_env_context=venv_context,

)

code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply="Please continue. If everything is done, reply 'TERMINATE'.",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),

)

code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

today = datetime.datetime.now().date()
message = (
    f"Today is {today}. "
    "Create a plot showing stock gain YTD for NVDA and TSLA. "
    "Make sure the code is in markdown code block and save the figure "
    "to a file ytd_stock_gains.png."
)

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message,
)

# ==================================================
print("=" * 50)
print("  PARTE 2: Codigo con funciones predefinidas")
print("=" * 50)

def get_stock_prices(stock_symbols, start_date, end_date):
    """Get the stock prices for the given stock symbols between
    the start and end dates.

    Args:
        stock_symbols (str or list): The stock symbols to get the
        prices for.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
        pandas.DataFrame: The stock prices for the given stock
        symbols indexed by date, with one column per stock symbol.
    """
    import yfinance
    stock_data = yfinance.download(
        stock_symbols, start=start_date, end=end_date
    )
    return stock_data.get("Close")

def plot_stock_prices(stock_prices, filename):
    """Plot the stock prices for the given stock symbols.

    Args:
        stock_prices (pandas.DataFrame): The stock prices for the
        given stock symbols.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for column in stock_prices.columns:
        plt.plot(
            stock_prices.index, stock_prices[column], label=column
        )
    plt.title("Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)

executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    functions=[get_stock_prices, plot_stock_prices],
    virtual_env_context=venv_context,
)

code_writer_agent_system_message = code_writer_agent.system_message
code_writer_agent_system_message += executor.format_functions_for_prompt()

code_writer_agent = ConversableAgent(
    name="code_writer_agent",
    system_message=code_writer_agent_system_message,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply="Please continue. If everything is done, reply 'TERMINATE'.",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),

)

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=(
        f"Today is {today}. "
        "Download the stock prices YTD for NVDA and TSLA and create "
        "a plot. Make sure the code is in markdown code block and "
        "save the figure to a file stock_prices_YTD_plot.png."
    ),
)