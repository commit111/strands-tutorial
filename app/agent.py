from strands import Agent, tool
# from strands_tools import handoff_to_user
from strands.models.openai import OpenAIModel
from flask import Flask, request, jsonify, send_from_directory

import json
import os
import time
import dotenv
from threading import Lock

dotenv.load_dotenv()

message = """
You are an expert fashion stylist. Your goal is to help users find their personal style. 
Your task is to provide fashion advice and offer products based on the user's preferences.
You can use the tools available to you to assist with this.
Note that for any prompts you ask the user, make sure you actually explicitly state the question you are asking, and possible some sample answers so they know what to type. 
Keep the questions as simple as possible so the user doesn't have to type much. And don't ask more than 3 questions.
"""

app = Flask(__name__)
# response_lock = Lock()
latest_response = {"message": "Hello! I'm your fashion stylist assistant. How can I help you with your style today?"}

model = OpenAIModel(
    client_args={
        "base_url": os.getenv("LLM_URL"),
        # "api_key": os.getenv("OPENAI_API_KEY")
    },
    model_id=os.getenv("LLM_MODEL"),
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

def parse_assistant_response(**kwargs):
    # print(json.dumps(kwargs["message"], indent=2))
    # Extract the assistant's text message
    assistant_text = kwargs["message"]["content"][0]["text"]

    # Append tool message if used
    if len(kwargs["message"]["content"]) > 1:
        tool_content = kwargs["message"]["content"][1]
        if "toolUse" in tool_content:
            tool_message = tool_content["toolUse"]["input"]["kwargs"]

            if tool_message:
                assistant_text += "\n\n" + tool_message

    print("Assistant Text: ", assistant_text)
    return assistant_text


def message_buffer_handler(**kwargs):
    # When a new message is created from the assistant, print its content
    global latest_response
    try:
        if "message" in kwargs and kwargs["message"].get("role") == "assistant":
            # Parse the assistant's response from JSON
            assistant_text = parse_assistant_response(**kwargs)

            # Send the assistant's message content back to the UI
            latest_response = {"message": assistant_text}

            # Prevent the agent from closing by not calling exit() or any termination logic here.
            # If you have any cleanup or state reset, do it here, but do not terminate the process.
            pass

    except Exception as e:
        print(f"Error in message_buffer_handler: {str(e)}")

# Define the handoff_to_user tool specification for the agent to use
TOOL_SPEC = {
    "name": "handoff_to_user",
    "description": "Hand off control from agent to user for input",
    "inputSchema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message to display to the user with context and instructions",
            },
        },
        "required": ["message"],
    }
}

@tool
def handoff_to_user(tool, **kwargs):
    """
    Custom handoff_to_user function that works with the UI website.
    This function is called when the agent needs input from the user.
    
    Args:
        tool: The tool use object from Strands
        **kwargs: Additional arguments
        
    Returns:
        The tool result containing the user's response
    """
    # Extract message from the tool use
    tool_input = tool["input"]
    message = tool_input.get("message", "Please provide input:")
    tool_use_id = tool["toolUseId"]

        # Make the message HTML-friendly for the UI
    if isinstance(message, str):
        message = message.replace('\n', '<br>')
    
    # Update latest response to prompt user for input
    global latest_response
    latest_response = {"message": message}
    
    # For demonstration purposes, we'll return a message indicating the handoff
    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": f"Waiting for the next user input. Message: {message}"}]
    }

agent = Agent(
    tools=[handoff_to_user], 
    model=model, 
    callback_handler=message_buffer_handler, 
    system_prompt=message
)

print("Agent model:", agent.model.config)


# Flask routes
@app.route('/')
def index():
    # This assumes index.html is in the same directory as this script
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        global latest_response
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Received message: {user_message}")

        agent(f"Continue the conversation with the user. The user says: {user_message}")

        # # Return the response from latest_response
        response_content = latest_response.get("message", "I'm thinking about your question...")

        return jsonify({
            "response": response_content
        })
    
    except Exception as e:
        print(f"Error in /chat endpoint: {str(e)}")
        return jsonify({"error": str(e), "response": str(e)}), 500

# Define a custom tool to request user preferences
@tool
def request_user_preferences() -> str:
    """
    Requests user preferences for fashion advice.

    Returns:
        str: The user's response.
    """
    # Create a mock tool use object with the format expected by handoff_to_user
    tool_use = {
        "toolUseId": "request_preferences_" + str(hash(str(time.time()))),
        "input": {
            "message": "Please share your fashion preferences (e.g., favorite colors, styles, articles of clothing)."
        }
    }
    return handoff_to_user(tool_use)

def end_response():
        """
        Ends the agent's response and performs any necessary cleanup.
        """
        print("Thank you for using the fashion stylist agent. Have a great day!")
        exit()  # This will break out of any loop and stop the script

# Start Flask server when this script is run directly
if __name__ == '__main__':

    print("Environment variables:")
    print(f"- LLM_URL: {os.getenv('LLM_URL')}")
    
    print("Starting Flask server on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
