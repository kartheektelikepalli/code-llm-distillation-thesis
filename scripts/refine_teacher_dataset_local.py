import pandas as pd
import re

INPUT_PATH = "data/teacher_mbpp.parquet"
OUTPUT_PATH = "data/final_teacher_dataset.parquet"


def clean_code(code):
    # remove markdown
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)

    # extract first function
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

    if cleaned:
        return "\n".join(cleaned)

    return None


def main():
    df = pd.read_parquet(INPUT_PATH)

    # ✅ KEEP ONLY PASSED
    df = df[df["passed"] == True].copy()

    # ✅ CLEAN CODE
    df["cleaned_code"] = df["output"].apply(clean_code)

    # ✅ DROP BAD EXTRACTIONS
    df = df[df["cleaned_code"].notnull()]

    # keep only necessary columns
    final_df = df[["prompt", "cleaned_code"]].rename(
        columns={"cleaned_code": "output"}
    )

    final_df.to_parquet(OUTPUT_PATH)

    print("Final dataset size:", len(final_df))


if __name__ == "__main__":
    main()