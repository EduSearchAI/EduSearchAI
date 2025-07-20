import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Construct the path to the .env file in the mcp_server1 folder
# This assumes the script is run from its directory within the project structure
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'mcp_project', 'mcp_server1', '.env')

# Load environment variables from the specified .env file
load_dotenv(dotenv_path=dotenv_path)

# Configure the generative AI model with the API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please check the .env file in mcp_project/mcp_server1.")

# Initialize the ChatGoogleGenerativeAI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

def query_llm(json_input):
    """
    Queries the LLM with a question from a JSON object.

    Args:
        json_input (str or dict): A JSON string or a dictionary
                                  containing a "question" key.

    Returns:
        str: The response from the LLM, or an error message.
    """
    try:
        if isinstance(json_input, str):
            data = json.loads(json_input)
        else:
            data = json_input

        question = data.get("question")
        if not question:
            return "Error: JSON object must contain a 'question' key."

        # Invoke the model with the question
        response = llm.invoke(question)
        return response.content

    except json.JSONDecodeError:
        return "Error: Invalid JSON format."
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    # Example usage:
    json_question = '{"question": "What is the capital of France?"}'
    response = query_llm(json_question)
    print(f"Question: {json.loads(json_question)['question']}")
    print(f"LLM Response: {response}")

    # Example with a dictionary
    dict_question = {"question": "Explain the theory of relativity in simple terms."}
    response = query_llm(dict_question)
    print(f"Question: {dict_question['question']}")
    print(f"LLM Response: {response}")
