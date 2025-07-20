import numpy as np
from itertools import product

def search(query_vector, vectors):
    """Calculates cosine similarity between a query vector and a list of vectors."""
    # Ensure vectors is a 2D array for consistent processing
    if vectors.ndim == 1:
        vectors = np.array([vectors])

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

def sliding_window_search(query_embedding: np.ndarray, index: dict, top_n_seeds: int = 100, context_window_size: int = 5, num_answers: int = 1):
    """
    Finds the best context windows using a "seed and expand" strategy.

    1. Seed Selection: Find the top N individual segments across all lectures.
    2. Context Expansion & Scoring: Build context windows around each seed and
       calculate a weighted score to find the best one.
    """
    all_segments = []
    for lecture_name, lecture_data in index.items():
        for i, segment in enumerate(lecture_data['segments']):
            all_segments.append({
                "lecture": lecture_name,
                "segment": segment,
                "index_in_lecture": i
            })

    if not all_segments:
        return []

    # Stage 1: Seed Selection
    segment_embeddings = np.array([item['segment']['embedding'] for item in all_segments])
    similarities = search(query_embedding, segment_embeddings)

    # Get the indices of the top N seeds
    top_seed_indices = np.argsort(similarities)[-top_n_seeds:][::-1]
    
    candidate_windows = []
    # Stage 2: Context Expansion & Scoring
    for seed_idx in top_seed_indices:
        seed_info = all_segments[seed_idx]
        lecture_name = seed_info['lecture']
        center_index = seed_info['index_in_lecture']
        
        lecture_segments = index[lecture_name]['segments']
        
        # Build the window around the seed
        half_window = (context_window_size - 1) // 2
        start_index = max(0, center_index - half_window)
        end_index = min(len(lecture_segments), start_index + context_window_size)
        
        # Adjust start_index again if end_index hit the boundary
        start_index = max(0, end_index - context_window_size)

        context_window = lecture_segments[start_index:end_index]
        
        if not context_window:
            continue

        window_embeddings = np.array([seg['embedding'] for seg in context_window])
        avg_similarity = np.mean(search(query_embedding, window_embeddings))
        
        # The seed's similarity is the max similarity in its window
        seed_similarity = similarities[seed_idx]

        # Weighted score - heavily prioritize the seed's direct similarity
        reranked_score = (0.95 * seed_similarity) + (0.05 * avg_similarity)

        candidate_windows.append({
            "segments": context_window,
            "score": reranked_score,
            "lecture_name": lecture_name  # Carry the lecture name forward
        })
    
    if not candidate_windows:
        return []

    # Sort all candidates by their final reranked score
    candidate_windows.sort(key=lambda x: x['score'], reverse=True)
    top_windows = candidate_windows[:num_answers]
    
    # Final processing for each of the top windows
    def strip_embedding(segment: dict) -> dict:
        seg_copy = segment.copy()
        seg_copy.pop("embedding", None)
        return seg_copy

    final_answers = []
    for window in top_windows:
        final_window = [strip_embedding(seg) for seg in window["segments"]]
        if final_window:
            final_window[0]['similarity'] = round(window["score"], 4)
        
        final_answers.append({
            "segments": final_window,
            "lecture_name": window["lecture_name"]
        })

    return final_answers 