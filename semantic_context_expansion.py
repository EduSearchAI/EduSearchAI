import numpy as np
from itertools import product

def search(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity. Assumes vectors are already normalized.
    The query vector is normalized within this function.
    """
    # Normalize the query vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        # If query vector is zero, similarity to all is zero.
        return np.zeros(vectors.shape[0])
    
    query_vector = query_vector / query_norm
    
    # Calculate dot product (cosine similarity for pre-normalized vectors)
    similarity = np.dot(vectors, query_vector)
    return similarity

def sliding_window_search(query_embedding: np.ndarray, index: dict, top_n_lectures: int = 3, top_n_seeds: int = 10, context_window_size: int = 5, num_answers: int = 1):
    """
    Finds the best context windows using a two-stage search:
    1. Lecture Search: Find the most relevant lectures using their average embeddings.
    2. Seed & Expand: Within those lectures, find top seed segments and expand them.
    """
    # Stage 1: Lecture Search
    lecture_names = list(index.keys())
    if not lecture_names:
        return []

    lecture_avg_embeddings = np.array([index[name]['average_embedding'] for name in lecture_names])
    lecture_similarities = search(query_embedding, lecture_avg_embeddings)
    
    # Get the indices of the top N most relevant lectures
    top_lecture_indices = np.argsort(lecture_similarities)[-top_n_lectures:][::-1]

    # Stage 2: Seed & Expand within the top lectures
    all_segments = []
    for lect_idx in top_lecture_indices:
        lecture_name = lecture_names[lect_idx]
        lecture_data = index[lecture_name]
        for i, segment in enumerate(lecture_data['segments']):
            all_segments.append({
                "lecture": lecture_name,
                "segment": segment,
                "index_in_lecture": i,
                # Carry lecture similarity forward to potentially influence scoring
                "lecture_similarity": lecture_similarities[lect_idx] 
            })

    if not all_segments:
        return []

    segment_embeddings = np.array([item['segment']['embedding'] for item in all_segments])
    similarities = search(query_embedding, segment_embeddings)

    # Get the indices of the top N seeds from the filtered segments
    top_seed_indices = np.argsort(similarities)[-top_n_seeds:][::-1]
    
    candidate_windows = []
    for seed_idx in top_seed_indices:
        seed_info = all_segments[seed_idx]
        lecture_name = seed_info['lecture']
        center_index = seed_info['index_in_lecture']
        
        lecture_segments = index[lecture_name]['segments']
        
        half_window = (context_window_size - 1) // 2
        start_index = max(0, center_index - half_window)
        end_index = min(len(lecture_segments), start_index + context_window_size)
        start_index = max(0, end_index - context_window_size)

        context_window = lecture_segments[start_index:end_index]
        
        if not context_window:
            continue

        # Since segment embeddings are pre-normalized, we can average their similarities directly.
        window_segment_indices = [
            all_segments.index(s) for s in all_segments if s['lecture'] == lecture_name and 
            start_index <= s['index_in_lecture'] < end_index
        ]
        
        # Ensure we found the corresponding indices in all_segments
        if not window_segment_indices:
             continue
        
        avg_similarity = np.mean(similarities[window_segment_indices])
        
        seed_similarity = similarities[seed_idx]

        reranked_score = (0.95 * seed_similarity) + (0.05 * avg_similarity)

        candidate_windows.append({
            "segments": context_window,
            "score": reranked_score,
            "lecture_name": lecture_name
        })
    
    if not candidate_windows:
        return []

    candidate_windows.sort(key=lambda x: x['score'], reverse=True)
    top_windows = candidate_windows[:num_answers]
    
    def strip_embedding(segment: dict) -> dict:
        seg_copy = segment.copy()
        seg_copy.pop("embedding", None)
        return seg_copy

    final_answers = []
    for window in top_windows:
        final_window = [strip_embedding(seg) for seg in window["segments"]]
        if final_window:
            # Add final score to the first segment for context
            final_window[0]['similarity'] = round(window["score"], 4)
        
        final_answers.append({
            "segments": final_window,
            "lecture_name": window["lecture_name"]
        })

    return final_answers 