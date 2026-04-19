import re
import pandas as pd

# load ONE real sample from your dataset
df = pd.read_parquet("data/teacher_mbpp.parquet")
row = df.iloc[0]

code = row["output"]
prompt = row["prompt"]

print("PROMPT:\n", prompt)
print("\nRAW CODE:\n", code)


def clean_code(code):
    # remove markdown
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)

    lines = code.split("\n")

    cleaned = []
    inside = False
    indent = None

    for line in lines:
        if line.strip().startswith("def "):
            inside = True
            indent = len(line) - len(line.lstrip())
            cleaned.append(line)
            continue

        if inside:
            cur = len(line) - len(line.lstrip())

            if line.strip() == "":
                cleaned.append(line)
                continue

            if cur > indent:
                cleaned.append(line)
            else:
                break

    return "\n".join(cleaned)


cleaned = clean_code(code)

print("\nCLEANED CODE:\n", cleaned)

# extract function name
match = re.search(r"def\s+(\w+)\(", cleaned)
if not match:
    print("\n❌ No function found")
    exit()

func_name = match.group(1)

# call with a safe input
test_input = "abcb"
exec_code = cleaned + f"\n__result = {func_name}({repr(test_input)})"

env = {}

try:
    exec(exec_code, {"__builtins__": __builtins__}, env)
    print("\nRESULT:", env.get("__result"))
except Exception as e:
    print("\n❌ EXECUTION ERROR:", str(e))