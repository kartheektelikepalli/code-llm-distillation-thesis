import ast
import time
from tqdm import tqdm

def generate_dataset(
    backend,
    dataset,
    prompt_builder,
    extract_python_code,
    num_samples,
    max_tokens,
    temperature,
):
    """
    Generic generation engine.

    Parameters
    ----------
    backend
        MLXBackend or LlamaCppBackend

    dataset
        HuggingFace dataset

    prompt_builder
        Function that accepts one dataset sample and returns a prompt

    extract_python_code
        Function that extracts Python code from model output

    num_samples
        Number of generations per problem

    max_tokens
        Maximum tokens to generate

    Returns
    -------
    results : list
        List of generation dictionaries
    """

    results = []

    syntax_pass_count = 0
    syntax_fail_count = 0

    for sample in tqdm(
            dataset,
            desc="Generating",
        ):

        prompt = prompt_builder(sample)

        for sample_id in range(num_samples):

            generation_start = time.time()

            output = backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

            latency = time.time() - generation_start

            generated_code = extract_python_code(output)

            syntax_passed = True

            try:
                ast.parse(generated_code)
                syntax_pass_count += 1

            except Exception:

                syntax_passed = False
                syntax_fail_count += 1

            results.append(
                {
                    "task_id": sample["task_id"],
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "generated_output": output,
                    "generated_code": generated_code,
                    "latency": latency,
                    "syntax_passed": syntax_passed,
                }
            )

    return (
        results,
        syntax_pass_count,
        syntax_fail_count,
    )