import warnings
warnings.filterwarnings("ignore")
import subprocess
import os
from datasets import load_dataset

def extract_code(output):
    lines = output.split("\n")
    code_lines = []

    for line in lines:
        if line.strip().startswith("def "):
            code_lines.append(line)
        elif code_lines:
            code_lines.append(line)

    return "\n".join(code_lines)


import requests

def generate_with_llama(prompt):

    url = "http://127.0.0.1:8080/completion"

    payload = {
        "prompt": f"[INST]\n{prompt}\n[/INST]",
        "n_predict": 128,
        "temperature": 0,
        "stop": ["\n\n\n", "if __name__", "\ndef "]
    }

    r = requests.post(url, json=payload)

    generated = r.json()["content"]

    # remove extra functions
    generated = generated.split("\n\n")[0]

    return generated

def main():
    dataset = load_dataset("openai_humaneval")

    print("Total problems:", len(dataset["test"]))

    sample = dataset["test"][0]
    prompt = sample["prompt"]

    print("\nPrompt:\n")
    print(prompt)

    print("\nGenerating code...\n")

    generated_code = generate_with_llama(prompt)

    print("Generated code:\n")
    print(generated_code)

if __name__ == "__main__":
    main()