import os
import pickle
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from semantic_context_expansion import sliding_window_search

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
                for i, answer in enumerate(results):
                    print(f"\n==== Contextual Answer #{i+1} ====")
                    
                    # Print combined text block
                    combined = " ".join(seg["text"] for seg in answer)
                    print(combined)

                    print("\n---- Segment Details ----")
                    for j, res in enumerate(answer):
                        start_time = res.get('start')
                        if start_time is not None:
                            start_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02}"
                        else:
                            start_str = "N/A"

                        sim_str = f" | Score: {res.get('similarity'):.4f}" if j == 0 and res.get('similarity') else ""
                        print(f"[{j}] {start_str}{sim_str} | {res['text']}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 