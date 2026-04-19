import pandas as pd

# Path to your dataset
# FILE_PATH = "data/final_teacher_dataset.parquet"
FILE_PATH = "data/teacher_mbpp.parquet"

def main():
    df = pd.read_parquet(FILE_PATH)

    print("\n========== BASIC INFO ==========")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\n========== SAMPLE ROW ==========")
    print(df.iloc[0])

    print("\n========== FIRST 3 PROMPTS ==========")
    for i in range(min(3, len(df))):
        print(f"\n--- Sample {i+1} ---")
        print("PROMPT:\n", df.iloc[i]["prompt"])
        print("\nOUTPUT:\n", df.iloc[i]["output"])

    print("\n========== NULL CHECK ==========")
    print(df.isnull().sum())

    print("\n========== LENGTH STATS ==========")
    df["output_len"] = df["output"].apply(len)
    print(df["output_len"].describe())

if __name__ == "__main__":
    main()