import pickle
import sys
import pprint
import numpy as np

# Custom adapter for pprint to handle NumPy arrays gracefully
def numpy_array_representer(printer, obj):
    # Truncate the array representation for cleaner output
    # You can adjust max_line_width or other options as needed
    with np.printoptions(threshold=10, edgeitems=2):
        printer.text(repr(obj))

# Register the custom representer for ndarray class
# Use a more compatible way to register the representer
pprint.PrettyPrinter._dispatch[np.ndarray.__class__] = numpy_array_representer

def inspect_pickle_file(file_path):
    """
    Loads and prints the contents of a pickle file in a human-readable format.
    Handles NumPy arrays for cleaner printing.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Successfully loaded data from: {file_path}\n")
        print("--- Inspector Tool: Pickle File Content ---")
        
        # Use a PrettyPrinter instance to control indentation and width
        printer = pprint.PrettyPrinter(indent=2, width=100)
        printer.pprint(data)
        
        print("-------------------------------------------")
        print("\nℹ️ Note: NumPy embeddings are truncated for readability.")

    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py <path_to_your_index.pkl_file>")
        sys.exit(1)

    pickle_file = sys.argv[1].strip('"')
    inspect_pickle_file(pickle_file) 