import os
import pickle
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from semantic_context_expansion import sliding_window_search
from scripts.llm.llm import query_llm

def search_engine(query: str, index: dict, model: SentenceTransformer, num_answers: int = 1):
    """
    Finds the most relevant windows of text segments for a given query
    using a dynamic sliding window approach.
    """
    print(f"\nSearching for: '{query}'...")
    
    # 1. Vectorize the user's query
    query_embedding = model.encode(query)

    # 2. Use the sliding window search to find the best consecutive segments
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
        
        # Combine text and find the start time of the window
        text = " ".join(seg["text"] for seg in segments)
        start_time = segments[0].get('start') if segments else None
        
        if start_time is not None:
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python app.py <path_to_index.pkl>")
        sys.exit(1)

    index_path = sys.argv[1]
    if not os.path.exists(index_path):
        print(f"Error: Index file not found at {index_path}")
        sys.exit(1)

    print("Loading model and index... This may take a moment.")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if 'cuda' in sys.modules else 'cpu')
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
    except Exception as e:
        print(f"Failed to load model or index: {e}")
        sys.exit(1)
    
    print("✅ Model and index loaded. You can now ask questions.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            query = input("> ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue

            results = search_engine(query, index, model, num_answers=3)

            if not results:
                print("No relevant results found.")
            else:
                # Prepare the question for the LLM
                llm_question = get_formatted_llm_question(query, results)

                # Send to LLM and get the response
                print("\nAsking the LLM to find the best answer...")
                llm_response = query_llm(llm_question)

                # Print the final, formatted answer from the LLM
                print("\n==== Final Answer from LLM ====")
                print(llm_response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 