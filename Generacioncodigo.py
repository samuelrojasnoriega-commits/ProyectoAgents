import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import autogen 
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv

load_dotenv()

#config_list = autogen.config_list_from_json(
#   "OAI_CONFIG_LIST",
#   filter_dict={"tags": ["gpt-4o"]},
#)
# When using a single openai endpoint, you can use the following:
# config_list = [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]
config_list  = [{
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY")
}]

# create an AssistantAgent named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "cache_seed": 41,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0,  # temperature for sampling
    },
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="coding"),
    },
)

# the assistant receives a message from the user_proxy, which contains the task description
chat_res = user_proxy.initiate_chat(
    assistant,
    message="""Search for the latest news about artificial intelligence. Summarize the top 5 headlines and save the results to ai_news.csv""",
    summary_method="reflection_with_llm",
)

print("Chat history:", chat_res.chat_history)
print("Summary:", chat_res.summary)
print("Cost info:", chat_res.cost)

# followup of the previous question
user_proxy.send(
    recipient=assistant,
    message="""Create a bar chart showing the number of news per source. Save the plot to ai_news.png""",
)

try:
    img = mpimg.imread("coding/ai_news.png")
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()
except FileNotFoundError:
    print("Image not found. Please check the file name and modify if necessary.")

# Path to your CSV file
file_path = "coding/ai_news.csv"
try:
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            row = line.strip().split(",")
            print(row)
except FileNotFoundError:
    print("File not found. Please check the file name and modify if necessary.")

def my_message_generator(sender, recipient, context):
    # your CSV file
    file_name = context.get("file_name")
    try:
        with open(file_name, mode="r", encoding="utf-8") as file:
            file_content = file.read()
    except FileNotFoundError:
        file_content = "No data found."
    return "Analyze the data and write a brief but engaging blog post. \n Data: \n" + file_content

# followup of the previous question
chat_res = user_proxy.initiate_chat(
    recipient=assistant,
    message=my_message_generator,
    file_name="coding/ai_news.csv",
    summary_method="reflection_with_llm",
    summary_args={"summary_prompt": "Return the blog post in Markdown format."},
)

print(chat_res.summary)
print(chat_res.cost)