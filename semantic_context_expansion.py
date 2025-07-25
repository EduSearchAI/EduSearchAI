from itertools import product
from typing import List, Dict, Any

import numpy as np


def search(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Finds the most similar vectors to a query vector using cosine similarity.

    This function calculates the cosine similarity between a given query vector and
    a collection of other vectors. It assumes that the input vectors (the search space)
    are already L2-normalized. The query vector is normalized within the function.

    Args:
        query_vector: The vector for which to find similar vectors.
        vectors: A 2D array of vectors to search through. Each row is expected
                 to be an L2-normalized vector.

    Returns:
        An array of cosine similarity scores between the query vector and each
        of the vectors in the search space.
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


def sliding_window_search(
    query_embedding: np.ndarray, 
    index: Dict[str, Any], 
    top_n_lectures: int = 3, 
    top_n_seeds: int = 10, 
    context_window_size: int = 5, 
    num_answers: int = 1
) -> List[Dict[str, Any]]:
    """Finds the best context windows using a two-stage search process.

    This function implements a search strategy by first identifying the most
    relevant lectures and then performing a detailed "seed and expand" search
    within those lectures to find the best contextual answer windows.

    Args:
        query_embedding: The embedding of the user's query.
        index: The search index, a dictionary containing lecture data and
               pre-computed embeddings.
        top_n_lectures: The number of top lectures to consider for the detailed search.
        top_n_seeds: The number of top "seed" segments to find within the
                     selected lectures.
        context_window_size: The number of segments to include in a context window.
        num_answers: The final number of answer windows to return.

    Returns:
        A list of dictionaries, where each dictionary represents a final answer
        containing the relevant segments and the lecture name. Returns an
        empty list if no suitable results are found.
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

        # Get the indices in `all_segments` corresponding to the current window
        window_segment_indices = [
            i for i, s in enumerate(all_segments) 
            if s['lecture'] == lecture_name and start_index <= s['index_in_lecture'] < end_index
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
    
    def strip_embedding(segment: Dict[str, Any]) -> Dict[str, Any]:
        """Removes the 'embedding' key from a segment dictionary for cleaner output."""
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