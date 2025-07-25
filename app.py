import os
import pickle
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from sentence_transformers import SentenceTransformer

from semantic_context_expansion import sliding_window_search
from scripts.llm.llm import query_llm

app = Flask(__name__, template_folder='templates')

# --- Global Variables ---
model: SentenceTransformer = None
index: Dict[str, Any] = None


def load_model_and_index(index_path: str) -> None:
    """Loads the SentenceTransformer model and the search index from a file.

    This function initializes the global `model` and `index` variables.
    It loads a pre-trained SentenceTransformer model and a pickled index file
    containing lecture embeddings and metadata.

    Args:
        index_path: The file path to the pickled index file.
    """
    global model, index
    try:
        print("Loading model and index... This may take a moment.")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if 'cuda' in sys.modules else 'cpu')
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        print("✅ Model and index loaded.")
    except (FileNotFoundError, pickle.UnpicklingError, Exception) as e:
        print(f"Error loading model or index: {e}")
        sys.exit(1)


def search_engine(query: str, num_answers: int = 1) -> List[Dict[str, Any]]:
    """Finds the most relevant text segments for a given query.

    This function takes a user query, encodes it into an embedding, and then
    uses a sliding window search algorithm to find the most relevant segments
    from the indexed lecture content.

    Args:
        query: The user's search query.
        num_answers: The maximum number of answers to return.

    Returns:
        A list of dictionaries, where each dictionary represents a relevant
        search result containing segments and lecture metadata. Returns an
        empty list if no results are found.
    """
    print(f"\nSearching for: '{query}'...")
    if not model or not index:
        # This case should ideally not be hit if setup is correct.
        print("Error: Model or index not loaded.")
        return []
    
    query_embedding = model.encode(query)
    results = sliding_window_search(
        query_embedding,
        index,
        num_answers=num_answers
    )
    return results


def prepare_llm_prompt_and_contexts(query: str, results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Prepares the LLM prompt and formats the structured context data.

    This function constructs a detailed prompt for the Language Model (LLM)
    and also prepares a structured list of the context data that can be
    returned to the user.

    Args:
        query: The user's original query.
        results: A list of search results from the `search_engine`.

    Returns:
        A tuple containing:
        - The formatted prompt string for the LLM.
        - A list of dictionaries, each with a formatted answer, source,
          and start time.
    """
    context_texts = []
    structured_contexts = []

    for i, result_item in enumerate(results):
        segments = result_item.get('segments', [])
        lecture_name = result_item.get('lecture_name', 'Unknown Lecture')
        
        text = " ".join(seg.get("text", "") for seg in segments)
        start_time = segments[0].get('start') if segments else "N/A"
        
        if isinstance(start_time, (int, float)):
             start_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02}"
        else:
            start_str = "N/A"

        structured_contexts.append({
            "answer": text,
            "source": lecture_name,
            "start_time": start_str
        })

        context_texts.append(
            f"Answer Context #{i+1}:\n"
            f"- Source Lecture: {lecture_name}\n"
            f"- Start Time: {start_str}\n"
            f"- Text: {text}\n"
        )
    
    all_contexts = "\n".join(context_texts)

    question = f"""
The user asked the following question: '{query}'

Here are some possible contextual answers from lecture recordings:

{all_contexts}

Based on the context provided, which is the best answer?
Your response MUST be only the number of the best Answer Context. For example: 1
If none of the contexts are relevant, respond with 0.
"""
    return question, structured_contexts


@app.route('/')
def home() -> str:
    """Renders the main HTML page for the application.

    Returns:
        The rendered HTML content of the `index.html` template.
    """
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask() -> Response:
    """Handles the user's question from a POST request.

    This endpoint receives a JSON payload with a "question", runs the
    search, queries the LLM to select the best result, and returns the
    final answer as a JSON response.

    Returns:
        A Flask Response object containing the JSON-formatted answer.
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Invalid request: 'question' is required."}), 400

    query = data['question']
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Question cannot be empty."}), 400

    results = search_engine(query, num_answers=3)
    if not results:
        return jsonify({"answer": "No relevant results found in the lecture recordings.", "source": "N/A", "start_time": "N/A"})

    llm_prompt, structured_contexts = prepare_llm_prompt_and_contexts(query, results)

    print("\nAsking the LLM to find the best answer...")
    llm_response = query_llm({"question": llm_prompt})
    print(f"LLM Response: '{llm_response}'")

    final_answer = {}
    try:
        # LLM is expected to return a number as a string (e.g., "1").
        best_choice_index = int(llm_response.strip()) - 1
        
        # Validate that the index is within the bounds of the contexts.
        if 0 <= best_choice_index < len(structured_contexts):
            final_answer = structured_contexts[best_choice_index]
        else:
            # Handle cases where the LLM returns 0 or an out-of-range number.
            final_answer = {
                "answer": "Sorry, I could not find a relevant answer in the provided lectures for your question.",
                "source": "N/A",
                "start_time": "N/A",
            }
    except (ValueError, IndexError):
        # Handle cases where the LLM response is not a valid number.
        print(f"Could not parse a valid index from LLM response: '{llm_response}'")
        # Return the raw LLM response so the user can see what happened.
        final_answer = {
            "answer": llm_response,
            "source": "Source not determined (LLM response parsing failed)",
            "start_time": "N/A",
        }
    
    return jsonify(final_answer)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <path_to_index.pkl>")
        sys.exit(1)

    index_path = sys.argv[1]
    if not os.path.exists(index_path):
        print(f"Error: Index file not found at '{index_path}'")
        sys.exit(1)

    load_model_and_index(index_path)
    
    # Use threaded=False for better compatibility with some ML models in debug mode.
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False) 