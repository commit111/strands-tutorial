from strands import Agent, tool
from strands_tools import handoff_to_user
from strands.models.openai import OpenAIModel

import json
import os
import dotenv

dotenv.load_dotenv()

model = OpenAIModel(
    client_args={
        "base_url": os.getenv("LLM_URL")
    },
    # **model_config
    model_id=os.getenv("MODEL_ID"),
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

def message_buffer_handler(**kwargs):
    # When a new message is created from the assistant, print its content
    if "message" in kwargs and kwargs["message"].get("role") == "assistant":
        print(json.dumps(kwargs["message"], indent=2))

agent = Agent(tools=[handoff_to_user], callback_handler=message_buffer_handler)

print("Agent model:", agent.model.config)

# Define a custom tool to request user preferences
@tool
def request_user_preferences() -> str:
    """
    Requests user preferences for fashion advice.

    Returns:
        str: The user's response.
    """

    return handoff_to_user(
        message="Please share your fashion preferences (e.g., favorite colors, styles, articles of clothing).",
        breakout_of_loop=False
    )

def end_response():
        """
        Ends the agent's response and performs any necessary cleanup.
        """
        print("Thank you for using the fashion stylist agent. Have a great day!")
        exit()  # This will break out of any loop and stop the script

# # Request user input and continue
# response = agent.tool.handoff_to_user(
#     message="I need your approval to continue. Type 'yes' to confirm.",
#     breakout_of_loop=False
# )

message = """
You are an expert fashion stylist. Your goal is to help users find their personal style. 
Your task is to provide fashion advice and offer products based on the user's preferences.
You can use the tools available to you to assist with this.
Note that for any prompts you ask the user, make sure you actually explicitly state the question you are asking, and possible some sample answers so they know what to type. 
Keep the questions as simple as possible so the user doesn't have to type much. And don't ask more than 3 questions.
"""

agent(message)
