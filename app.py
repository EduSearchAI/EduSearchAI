import os
import pickle
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

def search(query_vector, vectors):
    """Calculates cosine similarity between a query vector and a list of vectors."""
    # Normalize vectors to unit length to prepare for cosine similarity calculation
    query_vector_norm = np.linalg.norm(query_vector)
    if query_vector_norm == 0:
        return np.zeros(len(vectors))
        
    query_vector = query_vector / query_vector_norm
    
    vectors_norm = np.linalg.norm(vectors, axis=1)
    # Avoid division by zero for zero-vectors
    non_zero_norms = vectors_norm != 0
    vectors[non_zero_norms] = vectors[non_zero_norms] / vectors_norm[non_zero_norms, np.newaxis]
    
    # Calculate dot product (which is cosine similarity for normalized vectors)
    similarity = np.dot(vectors, query_vector)
    return similarity

def search_engine(query: str, index: dict, model: SentenceTransformer, top_k_lectures=3, top_k_segments=5):
    """
    Performs a two-stage search to find the most relevant segments for a query.
    """
    print(f"\nSearching for: '{query}'...")
    
    # 1. Vectorize the user's query
    query_embedding = model.encode(query)

    # --- STAGE 1: COARSE SEARCH (Find relevant lectures) ---
    lecture_names = list(index.keys())
    if not lecture_names:
        print("Index is empty.")
        return []
        
    average_embeddings = np.array([index[name]['average_embedding'] for name in lecture_names])
    
    # Calculate similarity between query and average lecture embeddings
    lecture_similarities = search(query_embedding, average_embeddings)
    
    # Get top N lectures
    # Ensure we don't request more lectures than available
    num_available_lectures = len(lecture_names)
    top_lecture_indices = np.argsort(lecture_similarities)[::-1][:min(top_k_lectures, num_available_lectures)]
    relevant_lectures = [lecture_names[i] for i in top_lecture_indices if lecture_similarities[i] > 0] # Filter out non-relevant lectures

    if not relevant_lectures:
        return []

    print(f"Found {len(relevant_lectures)} potentially relevant lectures: {', '.join(relevant_lectures)}")

    # --- STAGE 2: FINE-GRAINED SEARCH (Find relevant segments within those lectures) ---
    all_relevant_segments = []
    for lecture_name in relevant_lectures:
        lecture_data = index[lecture_name]
        segment_embeddings = np.array([seg['embedding'] for seg in lecture_data['segments']])
        
        segment_similarities = search(query_embedding, segment_embeddings)
        
        for i, segment in enumerate(lecture_data['segments']):
            all_relevant_segments.append({
                "lecture": lecture_name,
                "segment": segment,
                "similarity": segment_similarities[i]
            })

    # Sort all found segments by similarity
    all_relevant_segments.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return the top K segments overall
    return all_relevant_segments[:top_k_segments]

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

            results = search_engine(query, index, model)

            if not results:
                print("No relevant results found.")
            else:
                print("\n--- Top Results ---")
                for res in results:
                    start_time = res['segment'].get('start')
                    # A simple timestamp formatting
                    if start_time is not None:
                        start_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02}"
                    else:
                        start_str = "N/A"
                    print(f"Lecture: {res['lecture']}")
                    print(f"Time: {start_str}")
                    print(f"Text: {res['segment']['text']}")
                    print(f"Similarity: {res['similarity']:.4f}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 