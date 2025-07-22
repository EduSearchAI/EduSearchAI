import os
import pickle
import sys
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from semantic_context_expansion import sliding_window_search
from scripts.llm.llm import query_llm

app = Flask(__name__, template_folder='templates')

# --- Global Variables ---
model = None
index = None

def load_model_and_index(index_path):
    """Loads the SentenceTransformer model and the search index."""
    global model, index
    try:
        print("Loading model and index... This may take a moment.")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if 'cuda' in sys.modules else 'cpu')
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        print("✅ Model and index loaded.")
    except Exception as e:
        print(f"Error loading model or index: {e}")
        sys.exit(1)

def search_engine(query: str, num_answers: int = 1):
    """
    Finds the most relevant windows of text segments for a given query.
    """
    print(f"\nSearching for: '{query}'...")
    query_embedding = model.encode(query)
    results = sliding_window_search(
        query_embedding,
        index,
        num_answers=num_answers
    )
    return results

def prepare_llm_prompt_and_contexts(query: str, results: list) -> tuple[str, list]:
    """
    Formats the user's query for the LLM and returns the prompt
    along with the structured context data.
    """
    context_texts = []
    structured_contexts = []

    for i, result_item in enumerate(results):
        segments = result_item['segments']
        lecture_name = result_item['lecture_name']
        
        text = " ".join(seg["text"] for seg in segments)
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
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('question')

    if not query:
        return jsonify({"error": "No question provided"}), 400

    results = search_engine(query, num_answers=3)
    if not results:
        return jsonify({"answer": "No relevant results found.", "source": "", "start_time": ""})

    llm_prompt, structured_contexts = prepare_llm_prompt_and_contexts(query, results)

    print("\nAsking the LLM to find the best answer...")
    llm_response = query_llm({"question": llm_prompt})
    print(f"LLM Response: {llm_response}")

    final_answer = {}
    try:
        # The LLM should return a number (1-based index).
        best_choice_index = int(llm_response.strip()) - 1
        
        # Check if the choice is valid (e.g., 1, 2, or 3 -> index 0, 1, or 2)
        if 0 <= best_choice_index < len(structured_contexts):
            final_answer = structured_contexts[best_choice_index]
        else:
            # This handles cases where LLM returns 0 or another out-of-range number.
            final_answer = {
                "answer": "Sorry, I could not find a relevant answer in the provided lectures for your question.",
                "source": "N/A",
                "start_time": "N/A",
            }
    except (ValueError, IndexError):
        # This handles cases where the LLM response is not a number.
        print(f"Could not parse a valid index from LLM response: '{llm_response}'")
        # Return the raw response to the user so they see what happened.
        final_answer = {
            "answer": llm_response,
            "source": "Source not determined",
            "start_time": "Time not determined",
        }
    
    return jsonify(final_answer)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <path_to_index.pkl>")
        sys.exit(1)

    index_path = sys.argv[1]
    if not os.path.exists(index_path):
        print(f"Error: Index file not found at {index_path}")
        sys.exit(1)

    load_model_and_index(index_path)
    
    app.run(debug=True, port=5000) 