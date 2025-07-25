import os
import json
import pickle
import sys
from typing import Dict, Any, List

import numpy as np
from sentence_transformers import SentenceTransformer


def create_index(input_dir: str, output_path: str) -> None:
    """Processes JSON transcripts, generates embeddings, and saves an index file.

    This function iterates through a directory of JSON transcript files, extracts
    text segments, and generates sentence embeddings for each segment using a
    SentenceTransformer model. It also calculates an average embedding for each
    transcript. The resulting data, including normalized embeddings, is saved
    to a single index file using pickle.

    Args:
        input_dir: The path to the directory containing JSON transcript files.
        output_path: The path where the final index .pkl file will be saved.
    """
    # 1. Initialize the model and the main index dictionary
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if 'cuda' in sys.modules else 'cpu')
    index = {}

    print(f"Starting to process files in: {input_dir}")

    # 2. Iterate over all .json files in the input directory
    try:
        json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    except FileNotFoundError:
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    if not json_files:
        print("No JSON files found in the specified directory.")
        return
        
    for filename in json_files:
        input_path = os.path.join(input_dir, filename)
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Could not read or parse {filename}: {e}")
            continue

        segments_data = data.get("segments", [])
        if not segments_data:
            print(f"⚠️ No segments found in {filename}, skipping.")
            continue

        lecture_name = os.path.splitext(filename)[0]
        print(f"\nProcessing lecture: {lecture_name}...")

        # 3. Extract text and generate embeddings in a batch
        texts = [seg.get('text', '').strip() for seg in segments_data]
        if not any(texts):
            print(f"⚠️ All segments in {filename} are empty, skipping.")
            continue

        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

        # Normalize the segment embeddings before storing them for efficient cosine similarity
        embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(embeddings, embeddings_norm, out=np.zeros_like(embeddings), where=embeddings_norm != 0)

        # 4. Structure the data for the index
        segments_for_index = []
        for i, seg in enumerate(segments_data):
            segments_for_index.append({
                "id": seg.get("id", i),
                "text": texts[i],
                "start": seg.get("start"),
                "end": seg.get("end"),
                "embedding": embeddings[i]
            })

        # 5. Calculate the average embedding for the entire lecture (from pre-normalized segment embeddings)
        average_embedding = np.mean(embeddings, axis=0)

        # Also normalize the average embedding for efficient lecture-level search
        avg_norm = np.linalg.norm(average_embedding)
        if avg_norm > 0:
            average_embedding /= avg_norm

        # 6. Add the processed data to the main index
        index[lecture_name] = {
            "segments": segments_for_index,
            "average_embedding": average_embedding
        }
        print(f"✅ Finished processing {lecture_name}.")

    # 7. Save the final index to a pickle file
    try:
        with open(output_path, "wb") as f:
            pickle.dump(index, f)
        print(f"\n🎉 Index for {len(index)} lectures created successfully and saved to {output_path}")
    except IOError as e:
        print(f"❌ Failed to save index file: {e}")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python generate_segment_linkedlist.py <input_directory_with_transcripts> <output_index_file.pkl>")
        sys.exit(1)

    input_folder = sys.argv[1].strip('"')
    output_file = sys.argv[2].strip('"')

    if not os.path.isdir(input_folder):
        print(f"Error: Input directory not found at '{input_folder}'")
        sys.exit(1)

    # If the provided output path is a directory, append a default filename.
    if os.path.isdir(output_file):
        print(f"Warning: Output path '{output_file}' is a directory. Appending default filename 'index.pkl'.")
        output_file = os.path.join(output_file, "index.pkl")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    create_index(input_folder, output_file)


if __name__ == "__main__":
    main() 