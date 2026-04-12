"""
HumanEval Baseline Evaluation

Script to evaluate a local llama.cpp server on HumanEval dataset.
Requires llama.cpp server running at http://127.0.0.1:8080/completion
"""

from modulefinder import test
import warnings
warnings.filterwarnings("ignore")

import requests
import sys
from datasets import load_dataset


def generate_with_llama(prompt):
    """
    Send prompt to local llama.cpp server and get generated code.
    
    Args:
        prompt: The input prompt for code generation
        
    Returns:
        Generated code as string
    """
    url = "http://127.0.0.1:8080/completion"
    
    payload = {
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0

    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to llama.cpp server: {e}")
        return ""


def execute_test(test_code, entry_point):
    try:
        namespace = {}
        exec(test_code, namespace)
        return True
    except Exception:
        return False


def main():
    """Main evaluation loop"""
    
    print("=" * 60)
    print("HumanEval Baseline Evaluation")
    print("=" * 60)
    
    # Load dataset
    try:
        dataset = load_dataset("openai_humaneval")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    test_set = dataset["test"]
    total_problems = len(test_set)
    
    print(f"\nTotal problems: {total_problems}")
    print(f"Connecting to llama.cpp server at http://127.0.0.1:8080/completion")
    print("-" * 60)
    
    passed = 0
    results = []
    
    # Iterate through each problem
    for i, problem in enumerate(test_set, 1):
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]
        test = problem["test"]
        
        # print(f"\n[{i}/{total_problems}] {entry_point}")
        
        # Generate code using llama.cpp
        generated_code = generate_with_llama(prompt)
        
        if not generated_code:
            print(f"  ✗ Failed to generate code")
            results.append({
                "problem": entry_point,
                "passed": False,
                "reason": "generation_failed"
            })
            continue
        
        # Combine prompt + generated code + test
        solution = prompt + generated_code
        complete_code = solution + "\n" + test
        
        # Execute test
        test_passed = execute_test(complete_code, entry_point)
        
        if test_passed:
            passed += 1
            print(f"[{i}/{total_problems}] {entry_point}  ✓ PASSED")
            results.append({
                "problem": entry_point,
                "passed": True
            })
        else:
            print(f"[{i}/{total_problems}] {entry_point}  ✗ FAILED")
            results.append({
                "problem": entry_point,
                "passed": False,
                "reason": "test_failed"
            })
    
    # Compute pass@1
    pass_at_1 = (passed / total_problems) * 100 if total_problems > 0 else 0
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total problems:  {total_problems}")
    print(f"Passed problems: {passed}")
    print(f"Pass@1 score:    {pass_at_1:.2f}%")
    print("=" * 60)
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 60)
    for result in results:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        reason = f" ({result.get('reason', '')})" if not result["passed"] else ""
        print(f"{status:8} {result['problem']}{reason}")
    
    return passed, total_problems, pass_at_1


if __name__ == "__main__":
    main()
