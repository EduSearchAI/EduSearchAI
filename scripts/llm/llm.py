import os
import json
from typing import Dict, Any, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Construct the path to the .env file. This assumes a specific project structure.
# The path navigates up from the current script's location to the project root
# and then into the 'mcp_project/mcp_server1' directory.
try:
    current_dir = os.path.dirname(__file__)
    dotenv_path = os.path.join(current_dir, '..', '..', '..', '..', 'mcp_project', 'mcp_server1', '.env')

    # Load environment variables from the specified .env file
    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(f".env file not found at the expected path: {dotenv_path}")
    
    load_dotenv(dotenv_path=dotenv_path)

    # Configure and initialize the generative AI model
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please check the .env file.")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

except (FileNotFoundError, ValueError) as e:
    print(f"Error initializing LLM: {e}")
    # Set llm to None so the application can handle the missing model gracefully.
    llm = None


def query_llm(json_input: Union[str, Dict[str, Any]]) -> str:
    """Queries the LLM with a question from a JSON object or dictionary.

    This function takes either a JSON string or a Python dictionary as input,
    extracts the question, sends it to the configured Generative AI model,
    and returns the model's response.

    Args:
        json_input: A JSON string or a dictionary containing a "question" key.

    Returns:
        The content of the response from the LLM, or an error message if
        the LLM is not initialized, the input is invalid, or an API error occurs.
    """
    if not llm:
        return "Error: LLM not initialized. Please check configuration and API key."

    try:
        if isinstance(json_input, str):
            data = json.loads(json_input)
        else:
            data = json_input

        question = data.get("question")
        if not question:
            return "Error: Input must contain a 'question' key."

        # Invoke the model with the question
        response = llm.invoke(question)
        return response.content

    except json.JSONDecodeError:
        return "Error: Invalid JSON format provided."
    except Exception as e:
        return f"An unexpected error occurred while querying the LLM: {e}"


def main() -> None:
    print("--- Running LLM Query Examples ---")
    
    # Example usage with a JSON string:
    json_question = '{"question": "What is the capital of France?"}'
    print(f"\nQuerying with JSON string: {json_question}")
    response = query_llm(json_question)
    print(f"LLM Response: {response}")

    # Example with a dictionary
    dict_question = {"question": "Explain the theory of relativity in simple terms."}
    print(f"\nQuerying with dictionary: {dict_question}")
    response = query_llm(dict_question)
    print(f"LLM Response: {response}")

    # Example of an invalid query
    invalid_question = {"prompt": "This should fail"}
    print(f"\nQuerying with invalid input: {invalid_question}")
    response = query_llm(invalid_question)
    print(f"LLM Response: {response}")


if __name__ == '__main__':
    main()
