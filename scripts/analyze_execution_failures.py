import pandas as pd
from collections import Counter

INPUT_PARQUET = "data/execution_validated_outputs/execution_validated_mbpp_20260523_021559.parquet"
df = pd.read_parquet(INPUT_PARQUET)

failed_df = df[
    df["execution_passed"] == False
]

failure_categories = []

for error in failed_df["execution_error"]:

    error = str(error)

    if "AssertionError" in error:
        failure_categories.append("assertion_failure")

    elif "RecursionError" in error:
        failure_categories.append("recursion_error")

    elif "NameError" in error:
        failure_categories.append("name_error")

    elif "TypeError" in error:
        failure_categories.append("type_error")

    elif "SyntaxError" in error:
        failure_categories.append("syntax_error")

    else:
        failure_categories.append("other_runtime_error")

counter = Counter(failure_categories)

print("\n" + "=" * 60)
print("FAILURE CATEGORY ANALYSIS")
print("=" * 60)

total_failures = len(failure_categories)

for category, count in counter.items():

    percentage = (
        count / total_failures
    ) * 100

    print(
        f"{category}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )

print("=" * 60)