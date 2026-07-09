import re
import signal
import inspect
import typing
import math
import collections
import itertools
import hashlib

# 1. Timeout Handling
class TimeoutException(Exception): 
    pass

def timeout_handler(signum, frame): 
    raise TimeoutException("Execution timed out!")

# 2. Code Extraction 
def extract_clean_code(raw_string, entry_point=None):
    """
    Parses the raw LLM output to find the valid Python block.
    Includes the 'Code Scavenger' if an entry_point is provided (HumanEval).
    """
    if "</think>" in raw_string:
        post_think_text = raw_string.split("</think>")[-1]
    else:
        post_think_text = raw_string

    clean_code = "No valid python block found."

    # Hunt 1: Standard ```python ... ``` blocks
    python_blocks = re.findall(r'```python(.*?)```', raw_string, re.DOTALL | re.IGNORECASE)
    if python_blocks:
        return python_blocks[-1].strip()

    # Hunt 2: Generic ``` ... ``` blocks
    generic_blocks = re.findall(r'```(.*?)```', raw_string, re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()

    # Hunt 3: The Naked Code Scavenger (Primarily for HumanEval)
    if entry_point:
        naked_code_pattern = rf'(def {entry_point}\(.*?:(?:\n\s+.*)+)'
        naked_blocks = re.findall(naked_code_pattern, raw_string)
        if naked_blocks:
            return naked_blocks[-1].strip()

    # Hunt 4: Just take whatever is left after </think>
    fallback_code = post_think_text.strip()
    if fallback_code:
        return fallback_code

    return clean_code

# 3. Creates a isolated environment 
def get_isolated_env():
    """Builds a fresh dictionary containing required modules for safe execution."""
    isolated_env = {}
    # Inject standard typing
    for t in dir(typing):
        if not t.startswith('_'):
            isolated_env[t] = getattr(typing, t)
            
    # Inject math and utility libraries
    isolated_env.update({
        'math': math, 
        'collections': collections, 
        'itertools': itertools, 
        're': re, 
        'hashlib': hashlib
    })
    return isolated_env

# 4. Evaluates HumanEval
def evaluate_humaneval(clean_code, test_string, entry_point, timeout_seconds=2):
    """Executes HumanEval logic utilizing the injected check() function."""
    passed_all_tests = False
    signal.signal(signal.SIGALRM, timeout_handler)

    try:
        isolated_env = get_isolated_env()
        signal.alarm(timeout_seconds)

        # 1. Load generated code
        exec(clean_code, isolated_env)
        # 2. Load OpenAI check() function
        exec(test_string, isolated_env)
        # 3. Execute test
        exec(f"check({entry_point})", isolated_env)

        passed_all_tests = True
    except Exception:
        passed_all_tests = False
    finally:
        signal.alarm(0)

    return passed_all_tests

# 5. Evaluates MBPP
def evaluate_mbpp(clean_code, hidden_tests, guiding_test, timeout_seconds=2):
    """Executes MBPP logic with dynamic function aliasing and argument wrapping."""
    passed_all_tests = True
    signal.signal(signal.SIGALRM, timeout_handler)

    try:
        isolated_env = get_isolated_env()
        signal.alarm(timeout_seconds)

        # 1. Load generated code
        exec(clean_code, isolated_env)

        # 2. Dynamic Function Aliasing & Argument Wrapper
        gen_match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', clean_code)
        test_match = re.search(r'assert\s+([a-zA-Z_]\w*)\s*\(', guiding_test)

        if gen_match and test_match:
            gen_name = gen_match.group(1)
            exp_name = test_match.group(1)

            if gen_name in isolated_env:
                original_func = isolated_env[gen_name]

                # Wrapper to swallow unexpected extra arguments from MBPP
                def arg_wrapper(*args, **kwargs):
                    sig = inspect.signature(original_func)
                    num_params = len(sig.parameters)
                    return original_func(*args[:num_params])

                isolated_env[exp_name] = arg_wrapper

        # 3. Run hidden tests individually
        for test_case in hidden_tests:
            exec(test_case, isolated_env)

    except Exception:
        passed_all_tests = False
    finally:
        signal.alarm(0)

    return passed_all_tests
