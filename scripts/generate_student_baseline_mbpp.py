import os
import ast
import time
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_PATH = "/Users/telikepalli/code-llm-distillation-thesis/models/student/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)


# --------------------------------------------------
# Output
# --------------------------------------------------

OUTPUT_DIR = "data/student_baseline_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")

OUTPUT_PARQUET = os.path.join(
    OUTPUT_DIR,
    f"student_baseline_mbpp_{timestamp}.parquet"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = load_dataset("mbpp")["test"]


# --------------------------------------------------
# Prompt
# --------------------------------------------------

def build_prompt(problem):

    return f"""### Instruction:
Write a Python function to solve the following problem.

### Problem:
{problem}

### Response:
```python
"""


# --------------------------------------------------
# Code Extraction
# --------------------------------------------------

def extract_python_code(text):

    if "```python" in text:
        text = text.split("```python")[1]

    if "```" in text:
        text = text.split("```")[0]

    return text.strip()


# --------------------------------------------------
# Generation
# --------------------------------------------------

def generate_code(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][input_length:]

    decoded = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return decoded.strip()

# --------------------------------------------------
# Main Loop
# --------------------------------------------------

results = []

syntax_pass_count = 0
syntax_fail_count = 0

for sample in tqdm(dataset):

    prompt = build_prompt(sample["text"])

    start = time.time()

    raw_output = generate_code(prompt)

    latency = time.time() - start

    generated_code = extract_python_code(raw_output)

    syntax_passed = True

    try:
        ast.parse(generated_code)
        syntax_pass_count += 1

    except:
        syntax_passed = False
        syntax_fail_count += 1

    if syntax_passed:

        results.append(
            {
                "task_id": sample["task_id"],
                "prompt": prompt,
                "raw_output": raw_output,
                "generated_code": generated_code,
                "latency": latency,
                "passed": True
            }
        )

    print(
        f"[{sample['task_id']}] "
        f"Syntax Pass: {syntax_pass_count} | "
        f"Syntax Fail: {syntax_fail_count}"
    )

# --------------------------------------------------
# Save
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_parquet(OUTPUT_PARQUET)

print(f"\nSaved to:\n{OUTPUT_PARQUET}")