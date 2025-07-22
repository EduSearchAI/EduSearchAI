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

def get_formatted_llm_question(query: str, results: list) -> dict:
    """
    Formats the user's query and search results into a question for the LLM.
    """
    context_texts = []
    for i, result_item in enumerate(results):
        segments = result_item['segments']
        lecture_name = result_item['lecture_name']
        
        text = " ".join(seg["text"] for seg in segments)
        start_time = segments[0].get('start') if segments else "N/A"
        
        if isinstance(start_time, (int, float)):
             start_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02}"
        else:
            start_str = "N/A"

        context_texts.append(
            f"Answer Context #{i+1}:\n"
            f"- Source Lecture: {lecture_name}\n"
            f"- Start Time: {start_str}\n"
            f"- Text: {text}\n"
        )
    
    all_contexts = "\n".join(context_texts)

    question = f"""
The user asked the following question: '{query}'

Here are three possible contextual answers from lecture recordings:

{all_contexts}
    
Please analyze these three answers and determine which one best answers the user's question.
Your response MUST be in the following format:
1. The full text of the best answer.
2. On a new line, provide the start time from the lecture, prefixed with "Start Time: ".
3. On a new line, provide the source lecture name, prefixed with "Source: ".

Example:
I recommend using cake flour because it has a lower protein content, which results in a lighter, more tender crumb.
Start Time: 00:02:15
Source: The Most AMAZING Vanilla Cake Recipe

Based on the context provided, which is the best answer?
"""
    return {"question": question}

def parse_llm_response(llm_response: str) -> dict:
    """
    Parses the LLM's response to extract the answer, start time, and source.
    """
    try:
        lines = llm_response.strip().split('\n')
        answer = lines[0]
        start_time_str = lines[1].replace('Start Time: ', '').strip()
        source = lines[2].replace('Source: ', '').strip()
        
        return {
            "answer": answer,
            "match": f"Source: {source}",
            "start": start_time_str,
            "end": ""
        }
    except (IndexError, AttributeError) as e:
        print(f"Error parsing LLM response: {e}\nResponse was: '{llm_response}'")
        return {
            "answer": "Could not parse the response from the language model.",
            "match": "No details available.",
            "start": "N/A",
            "end": ""
        }

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
        return jsonify({"answer": "No relevant results found.", "match": "", "start": "", "end": ""})

    llm_question = get_formatted_llm_question(query, results)
    print("\nAsking the LLM to find the best answer...")
    llm_response = query_llm(llm_question)
    print(f"LLM Response: {llm_response}")

    final_answer = parse_llm_response(llm_response)
    
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