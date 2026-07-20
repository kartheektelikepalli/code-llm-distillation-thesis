def extract_python_code(text):

    if "```python" in text:
        text = text.split("```python")[1]

    if "```" in text:
        text = text.split("```")[0]

    return text.strip()