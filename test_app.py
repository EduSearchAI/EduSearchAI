import itertools
import os
import pickle
import sys

import pytest

# Adjust path so we can import app
sys.path.append(os.path.dirname(__file__))

from app import search_engine, load_model_and_index  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_model_and_index():
    """
    Loads the model and index once for the entire test session.
    The 'autouse=True' flag ensures this fixture is automatically run.
    """
    index_path = os.path.join(os.path.dirname(__file__), "whisper_recording", "index.pkl")
    if not os.path.exists(index_path):
        pytest.skip(f"index.pkl not found at '{index_path}' – generate it before running tests.")

    # This will load the model and index into the global variables used by the app
    load_model_and_index(index_path)

# (query, list_of_expected_phrases)
CASES = [
    # Python Installation
    (
        "how to know if i have python on my computer",
        ["open command prompt", "type python"],
    ),
    (
        "how do i add python to my path during installation",
        ["add python.exe to path", "checkboxes"],
    ),
    (
        "how can i run a python script from the command prompt?",
        ["python and then the name of my file", "test.py"],
    ),
    (
        "what is python idle?",
        ["graphical user interface", "develop your python scripts"],
    ),
    (
        "where can i download python from?",
        ["python.org", "official website"],
    ),

    # Vanilla Cake Recipe
    (
        "what temperature should i set the oven when baking vanilla cake",
        ["set your oven to 350", "350"],
    ),
    (
        "how many grams of sugar are in the vanilla cake recipe",
        ["333 grams", "granulated sugar"],
    ),
    (
        "what happens if i over mix cake batter",
        ["gummy and tough", "contract"],
    ),
    (
        "why use cake strips for baking layers",
        ["perfectly flat", "wasted cake"],
    ),
    (
        "what kind of flour should i use for the vanilla cake?",
        ["all-purpose flour"],
    ),
    (
        "how should i measure flour correctly?",
        ["with a scale", "fluff the flour"],
    ),
    (
        "what should i do to prevent the cake from breaking?",
        ["parchment paper", "insurance"],
    ),
    (
        "what is a crumb coat?",
        ["crumb coat", "cakes can be crummy"],
    ),
]


def test_semantic_answers_formatted():
    """
    Runs a series of semantic search tests using the globally loaded model and index.
    The test fails if any of the scenarios do not return the expected context.
    """
    print("\n--- Running Semantic Search Scenarios ---")
    failed_cases = []

    for i, (query, expected_phrases) in enumerate(CASES):
        # Suppress verbose search output from the app for cleaner test results
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            # For testing, we check if any of the top 3 answers are correct
            results = search_engine(query, num_answers=3)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout  # Restore stdout

        print(f"\nQUESTION: {query}")

        if not results:
            summary = "-> FAILED: No results returned from search engine. [X]"
            failed_cases.append(f"Query '{query}' returned no results.")
            print(summary)
            continue

        is_correct = False
        all_returned_text = []
        successful_answer_text = ""

        for result_item in results:
            if not result_item or 'segments' not in result_item:
                continue
            
            combined_text = " ".join(seg["text"] for seg in result_item['segments'])
            all_returned_text.append(combined_text)

            if any(phrase.lower() in combined_text.lower() for phrase in expected_phrases):
                is_correct = True
                successful_answer_text = combined_text
                # Found a correct answer, no need to check further
                break
        
        if is_correct:
            # If correct, print the answer that passed the test
            print(f"RETURNED (Correct): {successful_answer_text}")
        else:
            # If incorrect, print all attempts that failed
            print("RETURNED (All attempts failed):")
            for idx, text in enumerate(all_returned_text):
                print(f"  -> Attempt #{idx+1}: {text[:150]}... [X]")
        
        if not is_correct:
            failed_cases.append(
                f"Query '{query}' did not return expected context in any of the top 3 answers.\n"
                f"  -> Expected one of: {expected_phrases}"
            )

    if failed_cases:
        pytest.fail(f"\n--- {len(failed_cases)}/{len(CASES)} tests failed ---\n" + "\n\n".join(failed_cases)) 