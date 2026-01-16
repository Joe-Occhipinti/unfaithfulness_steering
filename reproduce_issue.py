import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from hint_mention import extract_steered_response

def test_extraction():
    print("Testing extract_steered_response...")
    
    # Case 1: steered_response exists
    record1 = {'steered_response': 'Valid response'}
    res1 = extract_steered_response(record1)
    print(f"Case 1 (Standard): '{res1}'")
    assert res1 == 'Valid response', f"Failed Case 1. Got: {res1}"

    # Case 2: steered_response missing, prompt has Assistant:
    record2 = {'steered_prompt': 'Some prompt... Assistant: extracted part'}
    res2 = extract_steered_response(record2)
    print(f"Case 2 (Prompt fallback): '{res2}'")
    assert res2 == 'extracted part', f"Failed Case 2. Got: {res2}"

    # Case 3: steered_response missing, prompt has hints but NO Assistant:
    record3 = {'steered_prompt': 'Usage: Hint content...'}
    # Before fix, this would return the whole prompt. After fix, it should return empty string.
    res3 = extract_steered_response(record3)
    print(f"Case 3 (No response found): '{res3}'")
    assert res3 == '', f"Failed Case 3. Got '{res3}' instead of ''"

    # Case 4: steered_response missing, response itself contains "Assistant:"
    # Current implementation splits on last 'Assistant:', so it would return only " part."
    # Desired implementation splits on first 'Assistant:', so it returns "I am an Assistant part."
    record4 = {'steered_prompt': 'User: Hi\nAssistant: I am an Assistant part.'}
    res4 = extract_steered_response(record4)
    print(f"Case 4 (Assistant in response): '{res4}'")
    assert res4 == 'I am an Assistant part.', f"Failed Case 4. Got '{res4}'"

    print("\nAll tests passed! Fix is working.")

if __name__ == "__main__":
    test_extraction()
