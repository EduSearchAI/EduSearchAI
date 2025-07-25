import pickle
import sys
import pprint
from typing import Any

import numpy as np

# Custom adapter for pprint to handle NumPy arrays gracefully
def numpy_array_representer(printer: pprint.PrettyPrinter, obj: np.ndarray) -> None:
    """A custom representer for NumPy arrays to be used with pprint.

    This function formats NumPy arrays for pretty-printing by truncating them
    to ensure the output remains readable.

    Args:
        printer: The PrettyPrinter instance calling the representer.
        obj: The NumPy array object to represent.
    """
    # Truncate the array representation for cleaner output
    with np.printoptions(threshold=10, edgeitems=2):
        printer.text(repr(obj))

# Register the custom representer for the ndarray class.
# This makes pprint use our custom function for any NumPy arrays it encounters.
pprint.PrettyPrinter._dispatch[np.ndarray.__class__] = numpy_array_representer

def inspect_pickle_file(file_path: str) -> None:
    """Loads and prints the contents of a pickle file in a human-readable format.

    This function opens a specified .pkl file, loads its contents, and then
    pretty-prints the data to the console. It uses a custom representer to
    handle NumPy arrays gracefully, ensuring they don't flood the output.

    Args:
        file_path: The path to the .pkl file to be inspected.
    """
    try:
        with open(file_path, 'rb') as f:
            data: Any = pickle.load(f)
        
        print(f"✅ Successfully loaded data from: {file_path}\n")
        print("--- Inspector Tool: Pickle File Content ---")
        
        # Use a PrettyPrinter instance to control indentation and width
        printer = pprint.PrettyPrinter(indent=2, width=100)
        printer.pprint(data)
        
        print("-------------------------------------------")
        print("\nℹ️ Note: NumPy embeddings are truncated for readability.")

    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
    except (pickle.UnpicklingError, Exception) as e:
        print(f"❌ An unexpected error occurred: {e}")


def main() -> None:
    """Main function to run the pickle file inspector from the command line."""
    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py <path_to_your_index.pkl_file>")
        sys.exit(1)

    pickle_file = sys.argv[1].strip('"')
    inspect_pickle_file(pickle_file)


if __name__ == "__main__":
    main() 