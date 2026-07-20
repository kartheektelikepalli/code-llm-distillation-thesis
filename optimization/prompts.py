def build_mbpp_prompt(sample):

    problem = sample["text"]

    return f"""### Instruction:
Write a Python function to solve the following problem.

### Problem:
{problem}

### Response:
```python
"""